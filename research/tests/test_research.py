from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from roy_research.analysis import paired_bootstrap_interval
from roy_research.baselines import evaluate_controlled_arms
from roy_research.cli import _compare_live_arms
from roy_research.controlled import (
    TERMINAL_SUCCESS_THRESHOLD,
    collect_group,
    generate_tasks,
    mechanism_diagnostics,
    task_event_graph,
)
from roy_research.grpo import clipped_policy_loss, hierarchical_advantages, masked_log_softmax
from roy_research.model import FrozenTextEncoder, StructuralPolicyNetwork, TEXT_DIMENSION
from roy_research.policy_server import PolicyServer
from roy_research.schema import require_schema
from roy_research.token_ledger import PersistentTokenLedger
from roy_research.training import evaluate_groups, train_groups
from roy_research.io import read_jsonl, write_jsonl
from roy_research.live_controlled import (
    ENVIRONMENT_REVISION_V2,
    build_live_problem,
    collect_forced_full_mas,
    collect_live_group,
    parse_answer,
    score_output,
)
from roy_research.providers import (
    Completion,
    DeepSeekClient,
    OpenAICompatibleClient,
    ProviderPaymentRequiredError,
    ProviderRetryExhaustedError,
)


class GRPOTests(unittest.TestCase):
    def test_masked_policy_and_clipped_objective(self) -> None:
        logits = torch.tensor([[1.0, 100.0, 2.0]])
        mask = torch.tensor([[True, False, True]])
        values = masked_log_softmax(logits, mask)
        self.assertLess(values[0, 1].item(), -1e20)
        loss = clipped_policy_loss(torch.log(torch.tensor([1.5])), torch.tensor([0.0]), torch.tensor([1.0]))
        self.assertAlmostEqual(loss.item(), -1.2, places=5)

    def test_zero_variance_and_hierarchical_branch_aggregation(self) -> None:
        result = hierarchical_advantages(
            {"CONTINUE": iter([0.5, 0.5]), "RETURN": iter([0.5, 0.5])},
            {"a": iter([0.2, 0.4]), "b": iter([0.8, 1.0])},
        )
        self.assertAlmostEqual(result.action_values["BRANCH"], 0.6)
        self.assertGreater(result.branch_advantages["b"], result.branch_advantages["a"])


class IOTests(unittest.TestCase):
    def test_gzip_jsonl_round_trip_and_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.jsonl.gz"
            self.assertEqual(write_jsonl(path, [{"id": 1}, {"id": 2}]), 2)
            self.assertEqual(path.read_bytes()[:2], b"\x1f\x8b")
            self.assertEqual(write_jsonl(path, [{"id": 3}], append=True), 1)
            self.assertEqual(list(read_jsonl(path)), [{"id": 1}, {"id": 2}, {"id": 3}])


class ControlledTests(unittest.TestCase):
    def test_dataset_splits_ood_and_mechanism_reconstruction(self) -> None:
        tasks = generate_tasks()
        self.assertEqual(len(tasks), 180)
        self.assertEqual(sum(task.split == "train" for task in tasks), 90)
        self.assertEqual(sum(task.split == "validation" for task in tasks), 30)
        self.assertEqual(sum(task.split == "test" for task in tasks), 60)
        self.assertEqual(sum(task.ood for task in tasks), 30)
        self.assertEqual(len({task.seed for task in tasks}), 180)
        mixed_ood = next(task for task in tasks if task.family == "mixed" and task.ood)
        self.assertNotEqual(list(mixed_ood.branch_values), sorted(mixed_ood.branch_values))
        self.assertLessEqual(max(mechanism_diagnostics(task)["reconstruction_error"] for task in tasks), 1e-6)
        self.assertEqual(mechanism_diagnostics(tasks[0])["remaining_budget_after_evidence"], 7.0)

    def test_unresolved_dependency_masks_return_and_traces_validate(self) -> None:
        task = next(task for task in generate_tasks() if task.family == "acquisition")
        group = collect_group(task)
        self.assertNotIn("RETURN", group["checkpoint"]["legal_actions"])
        self.assertTrue(any(node["kind"] == "dependency" for node in task_event_graph(task)["nodes"]))
        result = group["results"][0]
        trace = {
            "schema_version": 1, "action": result["action"], "utility": result["utility"],
        }
        require_schema(trace)

    def test_mixed_tasks_include_communication_and_artifact_events(self) -> None:
        task = next(task for task in generate_tasks() if task.family == "mixed")
        graph = task_event_graph(task)
        self.assertTrue(any(edge["kind"] == "communication" for edge in graph["edges"]))
        self.assertTrue(any(node["kind"] == "artifact" for node in graph["nodes"]))

    def test_baselines_report_paired_intervals(self) -> None:
        groups = [collect_group(task) for task in generate_tasks()[:6]]
        result = evaluate_controlled_arms(groups)
        self.assertEqual(result["arms"]["no_derivation"]["episodes"], 6)
        self.assertEqual(result["arms"]["no_derivation"]["display_name"], "direct (no_derivation)")
        self.assertEqual(result["arms"]["no_derivation"]["success_threshold"], TERMINAL_SUCCESS_THRESHOLD)
        self.assertEqual(result["arms"]["no_derivation"]["successes"], 2)
        self.assertAlmostEqual(result["arms"]["no_derivation"]["success_rate"], 2 / 6)
        self.assertIn("conclusion", result["arms"]["roy_heuristic"]["paired_success_vs_direct"])
        self.assertIn("conclusion", result["arms"]["roy_heuristic"]["paired_vs_no_derivation"])
        self.assertEqual(paired_bootstrap_interval([1, 1], [0, 0])["conclusion"], "positive")


class ModelTests(unittest.TestCase):
    def test_frozen_encoder_precache_deduplicates_and_batches(self) -> None:
        class RecordingModel:
            def __init__(self) -> None:
                self.calls = []

            def encode(self, texts, **_kwargs):
                self.calls.append(list(texts))
                return torch.arange(
                    len(texts) * TEXT_DIMENSION, dtype=torch.float32
                ).reshape(len(texts), TEXT_DIMENSION).numpy()

        encoder = FrozenTextEncoder.__new__(FrozenTextEncoder)
        encoder.model = RecordingModel()
        encoder._cache = {}
        encoder.precache(["a", "b", "a", "c"], batch_size=2)
        self.assertEqual(encoder.model.calls, [["a", "b"], ["c"]])
        encoder.precache(["a", "c"], batch_size=1)
        self.assertEqual(encoder.model.calls, [["a", "b"], ["c"]])
        self.assertEqual(tuple(encoder.encode(["c", "a"]).shape), (2, TEXT_DIMENSION))

    def test_pinned_encoder_output_dimension(self) -> None:
        encoder = FrozenTextEncoder(device="cpu", local_only=True)
        encoded = encoder.encode(["structural decision", "dependency evidence"])
        self.assertEqual(tuple(encoded.shape), (2, 384))

    def test_relational_policy_shapes_and_checkpoint_restore(self) -> None:
        model = StructuralPolicyNetwork()
        embeddings = torch.randn(3, TEXT_DIMENSION)
        node_types = torch.tensor([0, 1, 2])
        scalars = torch.zeros(3, 3)
        edges = torch.tensor([[0, 1], [1, 2]])
        edge_types = torch.tensor([0, 1])
        resources = torch.zeros(5)
        mask = torch.tensor([True, True, False])
        logits = model(embeddings, node_types, scalars, edges, edge_types, resources, mask)
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertLess(logits[2].item(), -1e20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save_checkpoint(path, {"epoch": 1})
            restored, metadata = StructuralPolicyNetwork.load_checkpoint(path)
            self.assertEqual(metadata["epoch"], 1)
            for left, right in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_training_checkpoint_resume(self) -> None:
        task = next(task for task in generate_tasks() if task.split == "train")
        group = collect_group(task)
        with tempfile.TemporaryDirectory() as directory:
            groups_path = Path(directory) / "groups.jsonl"
            model_path = Path(directory) / "model.pt"
            write_jsonl(groups_path, [group])
            first = train_groups(groups_path, model_path, epochs=1, device_name="cpu")
            resumed = train_groups(groups_path, model_path, epochs=2, device_name="cpu", resume=True)
            evaluation = evaluate_groups(groups_path, model_path, split="train", device_name="cpu")
            self.assertEqual(first["completed_epochs"], 1)
            self.assertEqual(resumed["resumed_from_epoch"], 1)
            self.assertEqual(resumed["completed_epochs"], 2)
            self.assertEqual(evaluation["direct"]["mean_utility"], task.direct_value)
            self.assertIn("conclusion", evaluation["paired_vs_direct"]["success"])


class RuntimeBoundaryTests(unittest.TestCase):
    def test_live_arm_comparison_uses_matched_repeat_means(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = root / "groups.jsonl"
            evaluation = root / "evaluation.json"
            mas = root / "mas.jsonl"
            write_jsonl(groups, [{
                "task": {"id": "task-1"},
                "action_values": {"CONTINUE": 0.6},
                "results": [
                    {"action": "CONTINUE", "repeat": 0, "utility": 0.2, "token_usage": 10, "duration_ms": 20},
                    {"action": "CONTINUE", "repeat": 1, "utility": 1.0, "token_usage": 30, "duration_ms": 40},
                ],
            }])
            evaluation.write_text(json.dumps({
                "split": "test",
                "task_results": [{
                    "task_id": "task-1", "selected_action": "CONTINUE",
                    "selected_child_specification": None, "utility": 0.6, "success": False,
                }],
            }), encoding="utf-8")
            write_jsonl(mas, [{
                "task_id": "task-1", "utility": 1.0, "success": True,
                "total_tokens": 100, "parallel_span_ms": 50,
                "work_latency_ms": 80, "child_agent_count": 3,
            }])

            result = _compare_live_arms(groups, evaluation, mas)
            direct = result["arms"]["single_agent_direct"]
            learned = result["arms"]["learned_full_policy"]
            self.assertEqual(direct["mean_utility"], 0.6)
            self.assertEqual(direct["success_rate"], 0.0)
            self.assertEqual(direct["mean_tokens"], 20)
            self.assertEqual(learned["mean_utility"], direct["mean_utility"])
            self.assertEqual(learned["success_rate"], direct["success_rate"])

    def test_persistent_ledger_hard_cap_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = PersistentTokenLedger(path, limit=10)
            ledger.reserve(8)
            conservative_restart = PersistentTokenLedger(path, limit=10)
            self.assertEqual(conservative_restart.snapshot()["remaining"], 2)
            with self.assertRaises(RuntimeError):
                ledger.reserve(3)
            ledger.settle(8, 6)
            resumed = PersistentTokenLedger(path, limit=10)
            self.assertEqual(resumed.snapshot()["remaining"], 4)

    def test_token_ledger_limit_can_only_be_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = PersistentTokenLedger(path, limit=10)
            ledger.reserve(6)
            ledger.settle(6, 4)

            state = PersistentTokenLedger.raise_existing_limit(path, 50)
            self.assertEqual(state["limit"], 50)
            self.assertEqual(state["used"], 4)
            self.assertEqual(state["remaining"], 46)
            with self.assertRaisesRegex(ValueError, "cannot be lowered"):
                PersistentTokenLedger.raise_existing_limit(path, 20)

    def test_policy_server_has_deterministic_legacy_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            server = PolicyServer()
        decision = server.decide({
            "legalActions": ["CONTINUE", "BRANCH"],
            "eventGraph": {"nodes": [{"kind": "dependency", "status": "unresolved"}]},
        })
        self.assertEqual(decision["action"], "BRANCH")

    def test_ambiguous_provider_failure_charges_full_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = PersistentTokenLedger(Path(directory) / "ledger.json", limit=10_000)
            events = Path(directory) / "events.jsonl"
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
                client = DeepSeekClient(ledger, event_log=events, max_retries=0)
            with patch("roy_research.providers.urllib.request.urlopen", side_effect=TimeoutError("ambiguous timeout")):
                with self.assertRaises(ProviderRetryExhaustedError):
                    client.complete([{"role": "user", "content": "bounded"}], max_tokens=64)
            snapshot = ledger.snapshot()
            self.assertEqual(snapshot["reserved"], 0)
            self.assertGreater(snapshot["used"], 64)
            event = json.loads(events.read_text(encoding="utf-8").strip())
            self.assertEqual(event["status"], "failed")
            self.assertEqual(event["reservation"], snapshot["used"])

    def test_provider_retries_429_without_charging_rejected_attempts(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }).encode()

        with tempfile.TemporaryDirectory() as directory:
            ledger = PersistentTokenLedger(Path(directory) / "ledger.json", limit=10_000)
            error = urllib.error.HTTPError(
                "https://api.deepseek.com", 429, "Too Many Requests", {}, None,
            )
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
                client = DeepSeekClient(
                    ledger, max_retries=2, retry_base_seconds=0, retry_max_seconds=0,
                )
            with patch(
                "roy_research.providers.urllib.request.urlopen",
                side_effect=[error, error, Response()],
            ) as opened:
                completion = client.complete(
                    [{"role": "user", "content": "bounded"}], max_tokens=64,
                )
            self.assertEqual(completion.content, "ok")
            self.assertEqual(opened.call_count, 3)
            self.assertEqual(ledger.snapshot()["used"], 3)
            self.assertEqual(ledger.snapshot()["reserved"], 0)

    def test_provider_402_opens_circuit_without_poisoning_token_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = PersistentTokenLedger(Path(directory) / "ledger.json", limit=10_000)
            error = urllib.error.HTTPError(
                "https://api.deepseek.com", 402, "Payment Required", {}, None,
            )
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
                client = DeepSeekClient(ledger)
            with patch("roy_research.providers.urllib.request.urlopen", side_effect=error):
                with self.assertRaises(ProviderPaymentRequiredError):
                    client.complete([{"role": "user", "content": "bounded"}], max_tokens=64)
            self.assertEqual(ledger.snapshot()["used"], 0)
            self.assertEqual(ledger.snapshot()["reserved"], 0)

    def test_openai_compatible_client_uses_standard_sdk_fields(self) -> None:
        class Response:
            def model_dump(self, mode="python"):
                self.mode = mode
                return {
                    "model": "qwen-compatible",
                    "choices": [{"message": {"content": '{"status":"ok"}'}}],
                    "usage": {
                        "prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11,
                    },
                }

        calls = []
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: calls.append(kwargs) or Response(),
        )))
        with tempfile.TemporaryDirectory() as directory:
            ledger = PersistentTokenLedger(Path(directory) / "ledger.json", limit=10_000)
            events = Path(directory) / "events.jsonl"
            with patch.dict(os.environ, {"TEST_OPENAI_KEY": "test-key"}, clear=True):
                client = OpenAICompatibleClient(
                    ledger,
                    model="qwen-compatible",
                    base_url="https://example.invalid/v1/",
                    api_key_env="TEST_OPENAI_KEY",
                    event_log=events,
                    max_output_tokens=32,
                    timeout=1800,
                    sdk_client=sdk,
                )
            completion = client.complete(
                [{"role": "user", "content": "status"}],
                max_tokens=64,
                temperature=0.0,
                json_mode=True,
                thinking="enabled",
            )
            self.assertEqual(completion.content, '{"status":"ok"}')
            self.assertEqual(ledger.snapshot()["used"], 11)
            self.assertEqual(calls[0]["max_tokens"], 32)
            self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
            self.assertEqual(calls[0]["timeout"], 1800)
            self.assertNotIn("thinking", calls[0])
            event = json.loads(events.read_text(encoding="utf-8"))
            self.assertEqual(event["provider"], "openai_compatible")
            self.assertEqual(event["requested_thinking"], "enabled")
            self.assertEqual(event["requested_max_tokens"], 64)
            self.assertEqual(event["request"]["timeout"], 1800)
            self.assertEqual(event["base_url"], "https://example.invalid/v1")


class LiveRolloutTests(unittest.TestCase):
    class FakeClient:
        model = "fake-deepseek"

        def __init__(self, answer: int, evidence_used: bool) -> None:
            self.answer = answer
            self.evidence_used = evidence_used
            self.calls = 0

        def complete(self, messages, max_tokens=1024, temperature=0.7, metadata=None):
            self.calls += 1
            content = json.dumps({
                "answer": self.answer,
                "checks": ["forward", "reverse"],
                "evidence_used": self.evidence_used,
            })
            return Completion(content, 10, 10, 20, 1, {"choices": []})

    def test_live_group_uses_real_completion_scores_and_masks_return(self) -> None:
        task = next(task for task in generate_tasks() if task.family == "acquisition")
        problem = build_live_problem(task)
        client = self.FakeClient(problem.gold_answer, True)
        group = collect_live_group(task, client, repeats=1, max_tokens=128)
        self.assertNotIn("RETURN", group["checkpoint"]["legal_actions"])
        self.assertEqual(client.calls, 7)
        self.assertEqual(len(group["results"]), 4)
        self.assertTrue(all(result["utility"] == 1.0 for result in group["results"]))
        self.assertEqual(group["provider"], "deepseek")
        self.assertEqual(group["checkpoint"]["environment_revision"], "live-controlled-arithmetic-v1")

    def test_v2_problems_are_deterministic_diverse_and_versioned(self) -> None:
        problems = {}
        for family in ("activation", "acquisition", "mixed"):
            task = next(task for task in generate_tasks() if task.family == family and task.ood)
            first = build_live_problem(task, version="v2")
            second = build_live_problem(task, version="v2")
            self.assertEqual(first, second)
            self.assertEqual(first.environment_revision, ENVIRONMENT_REVISION_V2)
            problems[family] = first
        self.assertEqual(problems["activation"].kind, "modular_recurrence")
        self.assertFalse(problems["activation"].evidence_required)
        self.assertEqual(problems["acquisition"].kind, "evidence_polynomial")
        self.assertTrue(problems["acquisition"].evidence_required)
        self.assertEqual(problems["mixed"].kind, "evidence_shortest_path")
        self.assertTrue(problems["mixed"].evidence_required)

    def test_v2_group_records_problem_revision(self) -> None:
        task = next(task for task in generate_tasks() if task.family == "acquisition" and task.ood)
        problem = build_live_problem(task, version="v2")
        group = collect_live_group(
            task, self.FakeClient(problem.gold_answer, True),
            repeats=1, max_tokens=128, problem_version="v2",
        )
        self.assertEqual(group["checkpoint"]["environment_revision"], ENVIRONMENT_REVISION_V2)
        self.assertEqual(group["task"]["live_problem"]["kind"], "evidence_polynomial")
        self.assertTrue(all(result["utility"] == 1.0 for result in group["results"]))

    def test_activation_checkpoint_draft_and_json_scoring(self) -> None:
        task = next(task for task in generate_tasks() if task.family == "activation")
        problem = build_live_problem(task)
        client = self.FakeClient(problem.gold_answer, False)
        group = collect_live_group(task, client, repeats=1, max_tokens=128)
        self.assertIn("RETURN", group["checkpoint"]["legal_actions"])
        self.assertEqual(client.calls, 8)
        self.assertIsNotNone(group["checkpoint"]["checkpoint_generation"])
        answer, parsed = parse_answer(f"```json\n{{\"answer\": {problem.gold_answer}, \"checks\": [\"a\", \"b\"], \"evidence_used\": false}}\n```")
        self.assertEqual(answer, problem.gold_answer)
        self.assertIsNotNone(parsed)
        self.assertEqual(score_output("not json", problem), 0.0)

    def test_forced_full_mas_aggregates_all_three_child_agents(self) -> None:
        task = next(task for task in generate_tasks() if task.family == "acquisition")
        problem = build_live_problem(task)
        client = self.FakeClient(problem.gold_answer, True)
        group = collect_live_group(task, client, repeats=1, max_tokens=128)
        child_content = json.dumps({
            "answer": problem.gold_answer,
            "checks": ["independent", "reverse"],
            "evidence_used": True,
        })
        events = [{
            "status": "completed",
            "metadata": {
                "task_id": task.id,
                "phase": "child",
                "action": "BRANCH",
                "child_id": specification["id"],
                "repeat": 0,
            },
            "response": {
                "choices": [{"message": {"content": child_content}}],
                "usage": {"total_tokens": 20},
            },
            "latency_ms": 5,
        } for specification in group["branch_specifications"]]
        result = collect_forced_full_mas(group, events, client, max_tokens=128)
        self.assertEqual(result["child_agent_count"], 3)
        self.assertEqual(result["child_tokens"], 60)
        self.assertEqual(result["total_tokens"], 80)
        self.assertEqual(result["utility"], 1.0)


if __name__ == "__main__":
    unittest.main()
