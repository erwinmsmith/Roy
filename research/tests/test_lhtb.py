from __future__ import annotations

import json
import tempfile
import unittest
import sys
import asyncio
from types import SimpleNamespace
from pathlib import Path

import torch

from roy_research.lhtb import (
    LHTB_COMMIT,
    build_lhtb_split,
    load_lhtb_manifest,
    require_training_task,
)
from roy_research.process_state import (
    FrozenDeepSeekSemanticClient,
    GlobalEpistemicState,
    SemanticStateBuilder,
    append_state,
    embedding_candidate_pairs,
    _semantic_payload_projection,
)
from roy_research.semantic_server import SemanticServer
from roy_research.value_model import (
    EpistemicValueModel,
    equal_trajectory_value_loss,
    make_ema_target,
    process_credit,
    trajectory_weighted_advantages,
    update_ema,
)
from roy_research.harbor_agent import PersistentNodeRPC, RoyHarborAgent
from roy_research.lhtb_experiment import (
    build_training_schedule,
    select_dev_checkpoint,
    summarize_test,
    write_harbor_group_config,
)
from roy_research.lhtb_results import import_harbor_group, official_lhtb_reward, sample_audit
from roy_research.lhtb_transitions import (
    build_decision_transition_samples,
    build_state_transition_samples,
)
from roy_research.model import epistemic_state_graph
from roy_research.organization_model import InformationRealizationPolicy
from roy_research.lhtb_training import (
    LHTB_POLICY_INTERFACE_REVISION,
    LHTBProcessGRPOTrainer,
)
from roy_research.lhtb_native import (
    audit_native_tasks,
    native_task_id_from_harbor,
    native_environment_digest,
    native_proot_launcher_environment,
    native_session_uids,
    normalize_native_task_id,
    oci_repository_reference,
    provision_native_task,
    resolve_native_task_source,
    tree_digest,
)
from roy_research.organization_replay import (
    _hierarchical_candidate_log_probs,
    organization_candidate_distribution,
    sample_organization_decision,
)


class FakeEncoder:
    def encode(self, texts):
        values = []
        for index, _ in enumerate(texts):
            vector = [0.0] * max(2, len(texts))
            vector[index % len(vector)] = 1.0
            values.append(vector)
        return torch.tensor(values)


class FakeEncoder384:
    def encode(self, texts):
        values = torch.zeros((len(texts), 384), dtype=torch.float32)
        if len(texts):
            values[:, 0] = 1
        return values


class FakeSemanticClient:
    def __init__(self):
        self.requests = []

    def complete_json(self, prompt_name, payload):
        self.requests.append((prompt_name, payload))
        if prompt_name == "epistemic_extractor_v1":
            return {"claims": [{"id": "c1", "statement": "The service is available"}]}
        # Deliberately contradict high-overlap wording. The embedding score has no vote.
        return {"label": "contradict", "probabilities": {
            "entail": 0.01, "contradict": 0.98, "unknown": 0.01,
        }, "model": "mock", "request_id": "r1"}


class LHTBProtocolTests(unittest.TestCase):
    def test_oci_repository_reference_removes_tag_but_preserves_registry_port(self) -> None:
        self.assertEqual(
            oci_repository_reference("mirror.example:5000/team/image:revision"),
            "mirror.example:5000/team/image",
        )
        self.assertEqual(
            oci_repository_reference("docker.io/team/image@sha256:abc"),
            "docker.io/team/image",
        )

    def test_frozen_semantic_client_retries_invalid_json_with_strict_json_mode(self) -> None:
        class Client:
            model = "deepseek-v4-flash"

            def __init__(self) -> None:
                self.calls = []
                self.responses = ["not-json", json.dumps({
                    "requirements": [], "claims": [], "assumptions": [], "evidence": [],
                    "external_observations": [], "blind_spots": [],
                })]

            def complete(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return SimpleNamespace(content=self.responses.pop(0))

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            semantic = FrozenDeepSeekSemanticClient(
                client, Path(directory) / "cache.jsonl", "frozen", max_attempts=2
            )
            result = semantic.complete_json("epistemic_extractor_v1", {"event": {"id": "e"}})
            self.assertEqual(result["claims"], [])
            self.assertEqual(len(client.calls), 2)
            self.assertTrue(all(call[1]["json_mode"] for call in client.calls))
            self.assertTrue(all(call[1]["thinking"] == "disabled" for call in client.calls))
            failures = (Path(directory) / "semantic-failures.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(failures), 1)
            self.assertEqual(len(client.calls[1][0][0]), 3)
            self.assertNotIn("not-json", json.dumps(client.calls[1][0][0]))

    def test_semantic_projection_preserves_raw_event_and_bounds_model_text(self) -> None:
        raw = "head" + ("x" * 50_000) + "tail"
        payload = {"event": {"id": "event", "output": raw}}
        projected = _semantic_payload_projection(payload)
        self.assertEqual(payload["event"]["output"], raw)
        self.assertLess(len(projected["event"]["output"]), len(raw))
        self.assertIn("semantic projection omitted", projected["event"]["output"])

    def test_semantic_extractor_failure_is_audited_without_killing_rollout(self) -> None:
        class Builder:
            def extract(self, event):
                raise TimeoutError("provider unavailable")

        with tempfile.TemporaryDirectory() as directory:
            server = SemanticServer.__new__(SemanticServer)
            server.root = Path(directory)
            server.builder = Builder()
            result = server.process({"id": "event-1"}, {})
            self.assertEqual(result["event_id"], "event-1")
            self.assertEqual(result["claims"], [])
            self.assertEqual(
                result["provenance"]["status"], "unresolved_semantic_event"
            )
            audit = (Path(directory) / "semantic-fallbacks.jsonl").read_text()
            self.assertIn("no_relation_or_entity_fabricated", audit)

    def test_existing_requirement_must_entail_new_requirement_before_suppression(self) -> None:
        class Builder:
            calls = []

            def extract(self, event):
                return {"requirements": [{"id": "new", "description": "new requirement"}],
                        "claims": [], "assumptions": [], "evidence": [],
                        "external_observations": [], "blind_spots": []}

            def verify_candidates(self, left, right):
                self.calls.append((left, right))
                return [{"left_id": left[0]["id"], "right_id": right[0]["id"],
                         "label": "entail", "probabilities": {
                             "entail": 1.0, "contradict": 0.0, "unknown": 0.0,
                         }}]

        with tempfile.TemporaryDirectory() as directory:
            server = SemanticServer.__new__(SemanticServer)
            server.root = Path(directory)
            server.builder = Builder()
            result = server.process({"id": "event-1"}, {
                "requirements": [{"id": "existing", "description": "existing requirement"}]
            })
            self.assertEqual(server.builder.calls[0][0][0]["id"], "existing")
            self.assertEqual(server.builder.calls[0][1][0]["id"], "new")
            self.assertEqual(result["requirements"], [])
            self.assertEqual(
                result["provenance"]["semantically_suppressed_requirement_ids"], ["new"]
            )

    def test_json_rpc_process_restarts_and_restores_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = Path(directory) / "server.py"
            server.write_text("""import json, sys
snapshot = None
for line in sys.stdin:
    request = json.loads(line)
    method = request['method']
    if method == 'run':
        snapshot = {'sequence': 0}
        result = {'status': 'ready', 'snapshot': snapshot}
    elif method == 'restore':
        snapshot = request['params']['snapshot']
        result = {'status': 'restored', 'snapshot': snapshot}
    else:
        result = {'status': 'ready', 'snapshot': snapshot}
    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)
""", encoding="utf-8")
            client = PersistentNodeRPC([sys.executable, "-u", str(server)], timeout=1)
            try:
                client.request("run", {})
                client.restart()
                self.assertEqual(client.request("snapshot", {})["snapshot"]["sequence"], 0)
            finally:
                client.close()

    def test_fake_harbor_terminal_round_trip_and_partial_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = Path(directory) / "server.py"
            server.write_text("""import json, sys
snapshot = {'processStates': [{'sequence': 0}]}
for line in sys.stdin:
    request = json.loads(line)
    if request['method'] == 'run':
        result = {'status': 'terminal_request', 'request': {
            'id': 'one', 'command': 'pwd', 'cwd': '/workspace', 'timeoutMs': 1000},
            'snapshot': snapshot}
    else:
        snapshot = {'processStates': [{'sequence': 0}, {'sequence': 1}]}
        result = {'status': 'ready', 'snapshot': snapshot}
    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)
""", encoding="utf-8")

            class Environment:
                environment_name = "fake-task"
                calls = []

                async def exec(self, command, cwd=None, timeout_sec=None):
                    self.calls.append((command, cwd, timeout_sec))
                    return SimpleNamespace(return_code=0, stdout="/workspace\n", stderr="")

            agent = RoyHarborAgent(
                logs_dir=Path(directory), model_name="deepseek/deepseek-v4-flash",
                node_command=f"{sys.executable} -u {server}", rpc_timeout=1,
                track_file_changes=False,
            )
            context = SimpleNamespace(metadata=None)
            environment = Environment()
            try:
                asyncio.run(agent.run("solve", environment, context))
                continuation = SimpleNamespace(metadata=None)
                asyncio.run(agent.resume_after_verifier_rejection(
                    user_prompt="Verifier rejected phase one", context=continuation
                ))
            finally:
                agent.close()
            self.assertEqual(environment.calls, [("pwd", "/workspace", 1)])
            self.assertTrue((Path(directory) / "roy-partial-trajectory.json").exists())
            self.assertIn("roy_trajectory_id", context.metadata)
            self.assertEqual(continuation.metadata["same_conversation_continuations"], 1)
            self.assertEqual(
                continuation.metadata["roy_trajectory_id"], context.metadata["roy_trajectory_id"]
            )

    def test_harbor_agent_returns_normally_before_outer_timeout_for_final_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = Path(directory) / "deadline_server.py"
            server.write_text("""import json, sys, time
snapshot = {'processStates': [{'sequence': 0, 'usage': {}}],
            'pendingTerminalRequest': {'id': 'pending'}}
for line in sys.stdin:
    request = json.loads(line)
    method = request['method']
    if method == 'run':
        result = {'status': 'continue', 'snapshot': snapshot}
    elif method == 'advance':
        time.sleep(0.25)
        result = {'status': 'continue', 'snapshot': snapshot}
    elif method == 'restore':
        snapshot = request['params']['snapshot']
        result = {'status': 'restored', 'snapshot': snapshot}
    elif method == 'rollout_deadline':
        snapshot = {**snapshot, 'pendingTerminalRequest': None,
                    'processStates': [*snapshot['processStates'],
                                      {'sequence': 1, 'usage': {},
                                       'runtimeEvents': [{'kind': 'verifier'}]}]}
        result = {'status': 'completed', 'snapshot': snapshot}
    else:
        result = {'status': 'completed', 'snapshot': snapshot}
    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)
""", encoding="utf-8")

            class Environment:
                environment_name = "fake-task"

            agent = RoyHarborAgent(
                logs_dir=Path(directory), model_name="deepseek/deepseek-v4-flash",
                node_command=f"{sys.executable} -u {server}", rpc_timeout=1,
                rollout_timeout_sec=0.08, track_file_changes=False,
            )
            context = SimpleNamespace(metadata=None)
            try:
                asyncio.run(agent.run("solve", Environment(), context))
            finally:
                agent.close()
            self.assertEqual(context.metadata["termination_reason"], "rollout_deadline")
            partial = json.loads(
                (Path(directory) / "roy-partial-trajectory.json").read_text()
            )
            self.assertIsNone(partial["pendingTerminalRequest"])
            self.assertEqual(len(partial["processStates"]), 2)

    def test_internal_rollout_deadline_with_official_reward_is_trainable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = root / "trial"
            agent_dir = trial / "agent"
            agent_dir.mkdir(parents=True)
            result = {
                "id": "trial", "task_name": "task", "task_checksum": "checksum",
                "started_at": "2026-08-25T00:00:00+00:00",
                "finished_at": "2026-08-25T01:00:00+00:00",
                "verifier_result": {"rewards": {"reward": 0.375}},
                "exception_info": None,
                "agent_result": {"metadata": {"termination_reason": "rollout_deadline"}},
            }
            (trial / "result.json").write_text(json.dumps(result))
            snapshot = {
                "organizationSeed": 1, "initialSnapshotFingerprint": "same",
                "runtime": {}, "policyRecords": [],
                "processStates": [{
                    "sequence": 0, "fingerprint": "m0", "runtimeEvents": [],
                    "nodes": [{"id": "root", "depth": 0, "status": "running"}],
                    "dagEdges": [], "usage": {},
                }],
            }
            (agent_dir / "roy-partial-trajectory.json").write_text(json.dumps(snapshot))
            records = import_harbor_group(
                root, root / "rollouts.jsonl", "group", "task", "category", "train",
                0, 0, "native:digest", {"timeout": 3600}, expected=1,
                environment_backend="native",
            )
            self.assertTrue(records[0]["complete"])
            self.assertTrue(records[0]["accepted_for_training"])
            self.assertEqual(records[0]["termination_type"], "timeout_with_reward")
            self.assertEqual(records[0]["terminal_reward"], 0.375)
            self.assertEqual(len(records[0]["process_states"]), 2)

    def test_clean_policy_dead_end_with_reward_is_not_trainable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = root / "trial"
            agent_dir = trial / "agent"
            agent_dir.mkdir(parents=True)
            result = {
                "id": "trial", "task_name": "task", "task_checksum": "checksum",
                "started_at": "2026-08-25T00:00:00+00:00",
                "finished_at": "2026-08-25T00:01:00+00:00",
                "verifier_result": {"rewards": {"reward": 0.25}},
                "exception_info": None,
                "agent_result": {"metadata": {}},
            }
            (trial / "result.json").write_text(json.dumps(result))
            snapshot = {
                "organizationSeed": 1, "initialSnapshotFingerprint": "same",
                "runtime": {"finalOutput": {"status": "policy_dead_end",
                    "reasons": ["proposal_json_malformed"]}},
                "policyRecords": [],
                "processStates": [{
                    "sequence": 0, "fingerprint": "m0", "runtimeEvents": [],
                    "nodes": [{"id": "root", "depth": 0, "status": "completed"}],
                    "dagEdges": [], "usage": {},
                }],
            }
            (agent_dir / "roy-partial-trajectory.json").write_text(json.dumps(snapshot))
            records = import_harbor_group(
                root, root / "rollouts.jsonl", "group", "task", "category", "train",
                0, 0, "native:digest", {"timeout": 3600}, expected=1,
                environment_backend="native",
            )
            self.assertFalse(records[0]["complete"])
            self.assertFalse(records[0]["accepted_for_training"])
            self.assertFalse(records[0]["environment_failure"])
            self.assertEqual(records[0]["termination_type"], "policy_dead_end")

    def test_harbor_agent_closes_rpc_process_after_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = Path(directory) / "server.py"
            server.write_text("""import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'],
                      'error': {'code': -32000, 'message': 'failed'}}), flush=True)
""", encoding="utf-8")

            class Environment:
                environment_name = "fake-task"

            agent = RoyHarborAgent(
                logs_dir=Path(directory), model_name="deepseek/deepseek-v4-flash",
                node_command=f"{sys.executable} -u {server}", rpc_timeout=1,
                track_file_changes=False,
            )
            with self.assertRaisesRegex(RuntimeError, "failed"):
                asyncio.run(agent.run("solve", Environment(), SimpleNamespace(metadata=None)))
            self.assertIsNone(agent.rpc.process)

    def test_pinned_split_is_exact_and_trainer_rejects_dev_test(self) -> None:
        records = build_lhtb_split()
        self.assertEqual(len(records), 46)
        self.assertEqual(len({value.task_id for value in records}), 46)
        self.assertEqual(
            {split: sum(value.split == split for value in records)
             for split in ("train", "dev", "test")},
            {"train": 30, "dev": 8, "test": 8},
        )
        manifest = [value.to_dict() for value in records]
        require_training_task(next(value.task_id for value in records if value.split == "train"), manifest)
        with self.assertRaisesRegex(ValueError, "rejects non-train"):
            require_training_task(next(value.task_id for value in records if value.split == "test"), manifest)

        checked_in = Path(__file__).parents[1] / "config" / "lhtb_split.json"
        checked_value = json.loads(checked_in.read_text())
        self.assertEqual(checked_value["commit"], LHTB_COMMIT)
        self.assertEqual(len(load_lhtb_manifest(checked_in)), 46)
        self.assertEqual(checked_value["tasks"], [value.to_dict() for value in records])
        self.assertEqual(len(build_training_schedule(manifest)), 120)

    def test_harbor_config_and_official_reward_are_unambiguous(self) -> None:
        self.assertEqual(official_lhtb_reward({
            "verifier_result": {"rewards": {"reward": 0.75}}
        }), 0.75)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            official_lhtb_reward({"verifier_result": {"rewards": {"a": 0.2, "b": 0.3}}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            write_harbor_group_config(path, "task", Path(directory) / "jobs",
                                      "single_agent_direct", "fingerprint", 1,
                                      attempts=3, official_timeout=True)
            value = json.loads(path.read_text())
            self.assertEqual(value["n_attempts"], 3)
            self.assertNotIn("override_timeout_sec", value["agents"][0])
            self.assertNotIn("rollout_timeout_sec", value["agents"][0]["kwargs"])

            learned = Path(directory) / "learned.json"
            write_harbor_group_config(
                learned, "task", Path(directory) / "jobs",
                "learned_information_realization", "fingerprint", 1,
                attempts=8, official_timeout=True,
            )
            learned_value = json.loads(learned.read_text())
            self.assertEqual(learned_value["n_attempts"], 1)
            self.assertEqual(len(learned_value["agents"]), 8)
            self.assertTrue(all(
                "ROY_LHTB_TOPOLOGY_PROFILE" not in agent["env"]
                for agent in learned_value["agents"]
            ))

            native = Path(directory) / "native.json"
            write_harbor_group_config(
                native, "task", Path(directory) / "jobs", "single_agent_direct",
                "fingerprint", 1, attempts=1, environment_backend="native",
                native_runtime_root=Path(directory) / "runtime",
                native_template_root=Path(directory) / "templates",
                max_retries=0,
            )
            native_value = json.loads(native.read_text())
            self.assertEqual(
                native_value["environment"]["import_path"],
                "roy_research.native_environment:NativeProcessEnvironment",
            )
            self.assertEqual(
                native_value["agents"][0]["env"]["ROY_LHTB_ENVIRONMENT_BACKEND"],
                "native",
            )
            self.assertEqual(
                native_value["agents"][0]["env"]["HB_CONTINUE_MODE"],
                "same_conversation",
            )
            self.assertEqual(native_value["retry"]["max_retries"], 0)
            self.assertEqual(native_value["agents"][0]["kwargs"]["rpc_timeout"], 720)
            self.assertEqual(native_value["agents"][0]["override_timeout_sec"], 21_600)
            self.assertEqual(native_value["agents"][0]["kwargs"]["rollout_timeout_sec"], 21_570)

    def test_native_proot_uses_the_uid_owned_session_tmp_before_guest_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = native_proot_launcher_environment({
                "PATH": "/usr/bin:/bin", "PROOT_TMP_DIR": "/untrusted",
            }, root / "tmp")
            trusted = f"PROOT_TMP_DIR={root / 'tmp'}"
            self.assertIn(trusted, values)
            self.assertNotIn("PROOT_TMP_DIR=/untrusted", values)

    def test_native_audit_is_fail_closed_and_uses_environment_digest(self) -> None:
        self.assertEqual(
            normalize_native_task_id(
                "long-horizon-terminal-bench/great-expectations-audit"
            ),
            "great-expectations-audit",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lhtb = root / "LHTB"
            templates = root / "templates"
            split_path = root / "split.json"
            records = build_lhtb_split()
            split_path.write_text(json.dumps({
                "commit": LHTB_COMMIT,
                "tasks": [value.to_dict() for value in records],
            }))
            for value in records:
                task = lhtb / "tasks" / value.task_id
                (task / "environment").mkdir(parents=True)
                (task / "environment" / "Dockerfile").write_text("FROM python:3.11-slim\n")
                (task / "task.toml").write_text(
                    "[environment]\nallow_internet = true\ngpus = 0\n"
                )
            selected = records[0]
            task_root = lhtb / "tasks" / selected.task_id
            native_root = templates / selected.task_id
            native_root.mkdir(parents=True)
            (native_root / "native-manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "lhtb_commit": LHTB_COMMIT,
                "task_digest": tree_digest(task_root),
                "environment_digest": "sha256:native",
            }))
            audit = audit_native_tasks(lhtb, split_path, templates)
            self.assertEqual(audit["counts"], {"compatible": 1, "needs_provisioning": 45})
            self.assertEqual(native_environment_digest(audit, selected.task_id), "sha256:native")
            (task_root / "environment" / "docker-compose.yaml").write_text(
                "services:\n  game: {}\n"
            )
            (native_root / "native-manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "lhtb_commit": LHTB_COMMIT,
                "task_digest": tree_digest(task_root),
                "environment_digest": "sha256:native-service",
                "services": [{"name": "game"}],
            }))
            isolated = audit_native_tasks(lhtb, split_path, templates)
            selected_isolated = next(
                value for value in isolated["tasks"]
                if value["task_id"] == selected.task_id
            )
            self.assertEqual(selected_isolated["status"], "incompatible")
            degraded = audit_native_tasks(
                lhtb, split_path, templates, allow_network_degraded=True
            )
            selected_degraded = next(
                value for value in degraded["tasks"]
                if value["task_id"] == selected.task_id
            )
            self.assertEqual(selected_degraded["status"], "degraded")
            missing = next(
                value for value in records if value.task_id != selected.task_id
            )
            with self.assertRaisesRegex(ValueError, "not runnable"):
                native_environment_digest(audit, missing.task_id)

    def test_reviewed_native_provisioning_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "LHTB" / "tasks" / "fixture"
            environment = task / "environment"
            (environment / "project").mkdir(parents=True)
            (environment / "project" / "input.txt").write_text("fixture")
            (environment / "Dockerfile").write_text("FROM python:3.11-slim\n")
            (task / "task.toml").write_text("[environment]\nallow_internet = true\n")
            specs = root / "specs.json"
            specs.write_text(json.dumps({"schema_version": 1, "tasks": {
                "fixture": {
                    "copies": [{"source": "project", "target": "app"}],
                    "required_commands": ["bash"],
                    "path_permissions": [{
                        "path": "/app/input.txt", "owner": "service",
                        "mode": "0400", "recursive": False,
                    }],
                    "wrappers": [{"name": "fixture-tool", "content": "#!/bin/sh\nexit 0\n"}],
                }
            }}))
            result = provision_native_task(
                root / "LHTB", root / "templates", specs, "fixture",
                minimum_free_fraction=0.0,
            )
            self.assertTrue(str(result["environment_digest"]).startswith("sha256:"))
            self.assertEqual(result["path_permissions"][0]["owner"], "service")
            self.assertEqual(
                (root / "templates" / "fixture" / "app" / "input.txt").read_text(),
                "fixture",
            )
            self.assertTrue(
                (root / "templates" / "fixture" / "bin" / "fixture-tool").stat().st_mode
                & 0o111
            )

    def test_native_template_digest_can_exclude_pinned_rootfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rootfs" / "usr" / "bin").mkdir(parents=True)
            (root / "rootfs" / "usr" / "bin" / "tool").write_text("image-v1")
            (root / "app").mkdir()
            mutable = root / "app" / "state.json"
            mutable.write_text("one")
            first = tree_digest(root, excluded_prefixes=("rootfs",))
            (root / "rootfs" / "usr" / "bin" / "tool").write_text("image-v2")
            self.assertEqual(first, tree_digest(root, excluded_prefixes=("rootfs",)))
            mutable.write_text("two")
            self.assertNotEqual(first, tree_digest(root, excluded_prefixes=("rootfs",)))

    def test_native_agent_and_service_use_disjoint_kernel_identities(self) -> None:
        agent, service = native_session_uids("identity-test", 210_000, 20_000)
        self.assertTrue(210_000 <= agent < 230_000)
        self.assertTrue(230_000 <= service < 250_000)
        self.assertNotEqual(agent, service)

    def test_native_task_id_uses_dataset_directory_not_display_name(self) -> None:
        self.assertEqual(
            native_task_id_from_harbor(
                "/dataset/tasks/snake_maze_campaign/environment",
                "custom/snake-obstacle-campaign",
            ),
            "snake_maze_campaign",
        )

    def test_native_control_overlay_resolves_only_matching_official_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "LHTB" / "tasks" / "fixture"
            (source / "environment").mkdir(parents=True)
            (source / "task.toml").write_text("[agent]\n", encoding="utf-8")
            overlay = root / "overlay" / "fixture"
            overlay.mkdir(parents=True)
            (overlay / ".roy-native-source-task").write_text(
                str(source), encoding="utf-8"
            )
            self.assertEqual(
                resolve_native_task_source(overlay, "fixture"), source.resolve()
            )
            with self.assertRaisesRegex(ValueError, "invalid native source"):
                resolve_native_task_source(overlay, "different")

    def test_dev_selection_and_test_report_follow_locked_rules(self) -> None:
        dev = []
        for epoch, reward, mae, tokens in ((0, 0.4, 0.1, 100), (1, 0.5, 0.2, 200)):
            dev.extend({"split": "dev", "epoch": epoch, "task_id": f"task-{index}",
                        "checkpoint": f"epoch-{epoch}.pt", "reward": reward,
                        "value_mae": mae, "tokens": tokens} for index in range(8))
        self.assertEqual(select_dev_checkpoint(dev)["epoch"], 1)
        test = []
        for arm, reward in (("single_agent_direct", 0.2),
                            ("roy_runtime_heuristic", 0.3),
                            ("learned_information_realization", 0.4)):
            for index in range(3):
                test.append({"split": "test", "arm": arm, "task_id": "task",
                             "repeat": index, "reward": reward, "tokens": 1,
                             "wall_time_seconds": 1})
        summary = summarize_test(test)
        self.assertEqual(summary["arms"]["learned_information_realization"]["success_rate"], 0)

    def test_process_state_is_append_only_and_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.jsonl"
            first = GlobalEpistemicState("trajectory", 0, "task")
            fingerprint = append_state(path, first)
            second = GlobalEpistemicState(
                "trajectory", 1, "task", previous_fingerprint=fingerprint,
                runtime_events=({"kind": "terminal_result", "exit_code": 0},),
            )
            append_state(path, second)
            self.assertEqual(len(path.read_text().splitlines()), 2)
            with self.assertRaisesRegex(ValueError, "append-only"):
                append_state(path, second)

    def test_embedding_only_recalls_and_llm_labels_contradiction(self) -> None:
        left = [{"id": "a", "statement": "The service is available"}]
        right = [{"id": "b", "statement": "The service is not available"}]
        self.assertEqual(len(embedding_candidate_pairs(left, right, FakeEncoder(), 8)), 1)
        many_left = [{"id": f"left-{index}", "statement": str(index)} for index in range(3)]
        many_right = [{"id": f"right-{index}", "statement": str(index)} for index in range(3)]
        self.assertEqual(
            len(embedding_candidate_pairs(many_left, many_right, FakeEncoder(), 2)), 2
        )
        with tempfile.TemporaryDirectory() as directory:
            client = FakeSemanticClient()
            builder = SemanticStateBuilder(client, FakeEncoder(), Path(directory) / "audit.jsonl")
            relations = builder.verify_candidates(left, right)
            self.assertEqual(relations[0]["label"], "contradict")
            audit = json.loads((Path(directory) / "audit.jsonl").read_text())
            self.assertEqual(audit["candidates"][0]["candidate_source"], "minilm_top_k")

    def test_delta_value_telescopes_and_weighting_is_per_trajectory(self) -> None:
        process, returns = process_credit([[0.2, 0.3, 0.6], [0.2, 0.1]], [0.8, 0.4])
        self.assertAlmostEqual(sum(process[0]), 0.6, places=7)
        self.assertAlmostEqual(sum(process[1]), 0.2, places=7)
        self.assertAlmostEqual(returns[0][0], 0.6, places=7)
        self.assertEqual([returns[0][1], returns[1][0]], [0.5, 0.2])
        advantages, mean, deviation = trajectory_weighted_advantages(returns)
        self.assertAlmostEqual(mean, (0.55 + 0.2) / 2, places=7)
        self.assertGreater(deviation, 0)
        self.assertEqual([len(value) for value in advantages], [2, 1])

    def test_topology_edits_save_adjacent_positive_and_negative_reward_samples(self) -> None:
        states = [{
            "sequence": 0, "fingerprint": "m0",
            "nodes": [{"id": "root", "depth": 0, "status": "ready"}],
            "dagEdges": [], "runtimeEvents": [],
        }, {
            "sequence": 1, "fingerprint": "m1",
            "nodes": [{"id": "root", "depth": 0, "status": "ready"},
                      {"id": "child", "parentId": "root", "depth": 1,
                       "status": "ready"}],
            "dagEdges": [{"kind": "derivation", "from": "root", "to": "child"}],
            "runtimeEvents": [{"id": "derive", "kind": "organization_action",
                               "attributes": {"action": {"kind": "DERIVE"}}}],
        }, {
            "sequence": 2, "fingerprint": "m2",
            "nodes": [{"id": "root", "depth": 0, "status": "completed"},
                      {"id": "child", "parentId": "root", "depth": 1,
                       "status": "returned"}],
            "dagEdges": [{"kind": "derivation", "from": "root", "to": "child"},
                         {"kind": "communication", "from": "child", "to": "root"}],
            "runtimeEvents": [{"id": "derive", "kind": "organization_action",
                               "attributes": {"action": {"kind": "DERIVE"}}},
                              {"id": "stop", "kind": "organization_action",
                               "attributes": {"action": {"kind": "STOP"}}}],
        }]
        values = [0.5, 0.7, 0.4]
        samples = build_state_transition_samples(states, values, 0.2)
        self.assertEqual([value["reward_sign"] for value in samples],
                         ["positive", "negative"])
        self.assertEqual(samples[0]["topology_delta"]["node_count_delta"], 1)
        self.assertEqual(samples[1]["topology_delta"]["edge_count_delta"], 1)
        self.assertAlmostEqual(sum(value["process_reward"] for value in samples), -0.3)
        decisions = build_decision_transition_samples(states, [{
            "stateFingerprint": "m0", "selectedAction": "DERIVE", "candidateId": "derive",
            "policyState": {"sampling_profile": {"id": "compact"}},
        }, {
            "stateFingerprint": "m1", "selectedAction": "STOP", "candidateId": "stop",
            "policyState": {"sampling_profile": {"id": "compact"}},
        }], values, 0.2)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0]["from_state_fingerprint"], "m0")
        self.assertEqual(decisions[0]["to_state_fingerprint"], "m1")

    def test_outer_action_probability_is_not_multiplied_by_derive_candidate_count(self) -> None:
        joint, outer, inner = _hierarchical_candidate_log_probs(
            ["DERIVE", "DERIVE", "DERIVE", "EXECUTE"], torch.zeros(4),
            torch.ones(4, dtype=torch.bool), 1.0,
        )
        self.assertAlmostEqual(float(outer["DERIVE"].exp()), 0.5, places=6)
        self.assertAlmostEqual(float(outer["EXECUTE"].exp()), 0.5, places=6)
        self.assertAlmostEqual(float(joint[:3].exp().sum()), 0.5, places=6)
        self.assertAlmostEqual(float(joint[3].exp()), 0.5, places=6)
        self.assertAlmostEqual(float(inner[:3].exp().sum()), 1.0, places=6)

    def test_topology_sampling_bias_is_soft_and_part_of_exact_old_policy(self) -> None:
        model = InformationRealizationPolicy()
        policy_state = {
            "interface_revision": LHTB_POLICY_INTERFACE_REVISION,
            "state_fingerprint": "m0",
            "event_graph": {"nodes": [{"id": "root", "kind": "agent",
                "timestamp": 0, "text": "solve", "status": "ready"}], "edges": []},
            "active_node_ids": ["root"], "active_node_legal": [True],
            "active_node_sampling_logit_bias": [0.0],
            "candidates": [{"id": "derive", "kind": "DERIVE",
                "actor_node_id": "root", "description": "derive another child",
                "scheduler_complexity": 1, "legal": True, "sampling_logit_bias": -100.0},
                {"id": "execute", "kind": "EXECUTE", "actor_node_id": "root",
                 "description": "execute current plan", "scheduler_complexity": 1,
                 "legal": True, "sampling_logit_bias": 0.0}],
            "envelope": {"id": "open", "minimum_nodes": 0, "maximum_nodes": 1_000_000,
                "minimum_depth": 0, "maximum_depth": 1_000_000, "mode": "expansive"},
            "node_count": 8, "maximum_depth_reached": 2,
            "unresolved_gap_exists": True, "unbounded_structure": True,
            "resources": {"llm_calls_remaining_fraction": 1,
                "tool_calls_remaining_fraction": 1, "nodes_remaining_fraction": 1,
                "depth_remaining_fraction": 1, "decisions_remaining_fraction": 1},
            "organization_temperature": 1,
        }
        generator = torch.Generator(device="cpu")
        generator.manual_seed(3)
        candidate, record = sample_organization_decision(
            model, FakeEncoder384(), policy_state, generator=generator,
            device=torch.device("cpu")
        )
        self.assertEqual(candidate["id"], "execute")
        self.assertGreater(record["masked_probabilities"]["DERIVE"], 0.0)
        self.assertLess(record["masked_probabilities"]["DERIVE"], 1e-30)

        distribution = organization_candidate_distribution(
            model, FakeEncoder384(), policy_state, device=torch.device("cpu")
        )
        self.assertAlmostEqual(sum(distribution["candidate_priors"].values()), 1.0, places=6)
        self.assertAlmostEqual(sum(distribution["action_priors"].values()), 1.0, places=6)
        self.assertGreater(distribution["candidate_priors"]["execute"], 1 - 1e-6)

    def test_value_graph_contains_epistemic_progress_as_well_as_agents(self) -> None:
        graph = epistemic_state_graph({
            "nodes": [{"id": "root", "localObjective": "solve", "status": "ready"}],
            "requirements": [{"id": "gap", "description": "find evidence", "status": "open",
                              "parentNodeId": "root"}],
            "claims": [{"id": "claim", "statement": "answer", "status": "supported",
                        "originNodeId": "root"}],
            "evidence": [{"id": "evidence", "content": "proof", "supports": ["claim"]}],
            "blindSpots": ["the failure mode is still unknown"],
            "activeSubtree": ["root"],
            "runtimeEvents": [
                {"id": "inspect", "kind": "terminal_command", "nodeId": "root",
                 "command": "pytest -q", "at": 1},
                {"id": "result", "kind": "terminal_result", "nodeId": "root",
                 "exitCode": 1, "output": "one assertion failed", "at": 2},
            ],
            "usage": {"inputTokens": 120, "outputTokens": 30, "wallTimeMs": 2000},
        })
        node_ids = {value["id"] for value in graph["nodes"]}
        self.assertTrue({"root", "gap", "claim", "evidence"}.issubset(node_ids))
        self.assertTrue({"blind-spot:0", "runtime:inspect", "runtime:result",
                         "state:active-subtree", "metric:input_tokens",
                         "metric:wall_time_ms"}.issubset(node_ids))
        self.assertIn({"kind": "dependency", "from": "root", "to": "gap"}, graph["edges"])
        self.assertIn({"kind": "temporal", "from": "runtime:inspect",
                       "to": "runtime:result"}, graph["edges"])

        model = EpistemicValueModel(text_dim=4, hidden_dim=8, node_type_dim=2, layers=1)
        self.assertEqual(model.value_head[0].in_features, 24)

    def test_constant_value_is_terminal_grpo_and_value_updates_on_zero_variance(self) -> None:
        model = EpistemicValueModel(text_dim=4, hidden_dim=8, node_type_dim=2, layers=1)
        target = make_ema_target(model)
        text = torch.zeros((1, 4))
        kinds = torch.zeros(1, dtype=torch.long)
        scalars = torch.zeros((1, 3))
        edges = torch.zeros((2, 0), dtype=torch.long)
        edge_types = torch.zeros(0, dtype=torch.long)
        self.assertEqual(float(model(text, kinds, scalars, edges, edge_types).detach()), 0.5)
        _, returns = process_credit([[0.5, 0.5], [0.5, 0.5]], [0.2, 0.8])
        advantages, _, _ = trajectory_weighted_advantages(returns)
        self.assertLess(float(advantages[0][0]), float(advantages[1][0]))

        predictions = [model(text, kinds, scalars, edges, edge_types).unsqueeze(0) for _ in range(2)]
        loss = equal_trajectory_value_loss(predictions, [0.4, 0.4])
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        before = [value.clone() for value in target.parameters()]
        update_ema(target, model, 0.99)
        self.assertEqual(len(before), len(list(target.parameters())))

    def test_mcts_behavior_probability_is_exact_and_revision_locked(self) -> None:
        record = {
            "behaviorPolicy": "mcts_puct", "candidateId": "derive",
            "mctsBehaviorProbabilities": {"derive": 0.75, "execute": 0.25},
            "mctsVisitCounts": {"derive": 18, "execute": 6},
            "maskedOldLogProbability": float(torch.log(torch.tensor(0.75))),
            "targetValueRevision": 2,
            "mctsSearchTrace": [{"phase": "selection"}, {"phase": "backup"}],
        }
        LHTBProcessGRPOTrainer._validate_mcts_behavior(record, 2)
        with self.assertRaisesRegex(ValueError, "stale target-value"):
            LHTBProcessGRPOTrainer._validate_mcts_behavior(record, 3)
        invalid = {**record, "maskedOldLogProbability": 0.0}
        with self.assertRaisesRegex(ValueError, "exact old-policy"):
            LHTBProcessGRPOTrainer._validate_mcts_behavior(invalid, 2)

    def test_sample_audit_exposes_step_rewards_topology_and_mcts(self) -> None:
        states = [{"sequence": 0, "fingerprint": "m0", "nodes": [{"id": "root",
            "depth": 0}], "dagEdges": [], "runtimeEvents": []},
            {"sequence": 1, "fingerprint": "m1", "nodes": [{"id": "root", "depth": 0},
                {"id": "child", "depth": 1}], "dagEdges": [{"kind": "derivation",
                    "from": "root", "to": "child"}], "runtimeEvents": []}]
        transitions = build_state_transition_samples(states, [0.4, 0.6], 0.7)
        result = sample_audit([{"id": "one", "rollout_index": 0, "organization_seed": 1,
            "terminal_reward": 0.7, "process_states": states, "state_transitions": transitions,
            "shaped_returns": [0.3],
            "policy_records": [{"behaviorPolicy": "mcts_puct", "selectedAction": "DERIVE",
                "policyState": {"sampling_profile": {"id": "recursive"}},
                "selectedProcessReward": 0.2,
                "mctsSearchTrace": [{"phase": "selection"}, {"phase": "backup"}]}],
            "complete": True, "environment_failure": False, "tokens": 10,
            "wall_time_seconds": 1}])
        self.assertTrue(result["all_step_rewards_complete"])
        self.assertTrue(result["all_mcts_traces_complete"])
        self.assertEqual(result["trajectories"][0]["terminal_node_count"], 2)
        self.assertEqual(result["trajectories"][0]["process_reward_signs"]["positive"], 1)
        self.assertTrue(result["value_training_available"])
        self.assertFalse(result["actor_dense_signal_available"])

    def test_zero_variance_group_updates_value_and_full_checkpoint_restores(self) -> None:
        manifest = [value.to_dict() for value in build_lhtb_split()]
        task_id = next(str(value["task_id"]) for value in manifest if value["split"] == "train")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            encoder = FakeEncoder384()
            trainer = LHTBProcessGRPOTrainer(checkpoint, manifest, encoder=encoder)
            policy_state = {
                "interface_revision": LHTB_POLICY_INTERFACE_REVISION,
                "topology_search": {"mode": "mcts_unconstrained"},
                "state_fingerprint": "m0",
                "event_graph": {"nodes": [{"id": "root", "kind": "agent",
                    "timestamp": 0, "text": "solve", "status": "ready"}], "edges": []},
                "active_node_ids": ["root"], "active_node_legal": [True],
                "candidates": [{"id": "stop", "kind": "STOP", "actor_node_id": "root",
                    "description": "finish", "scheduler_complexity": 0, "legal": True}],
                "envelope": {"id": "open", "minimum_nodes": 0, "maximum_nodes": 100,
                    "minimum_depth": 0, "maximum_depth": 100, "mode": "expansive"},
                "node_count": 1, "maximum_depth_reached": 0,
                "unresolved_gap_exists": False,
                "resources": {"llm_calls_remaining_fraction": 1,
                    "tool_calls_remaining_fraction": 1, "nodes_remaining_fraction": 1,
                    "depth_remaining_fraction": 1, "decisions_remaining_fraction": 1},
                "organization_temperature": 1,
            }
            records = []
            for rollout_index in range(8):
                rollout_policy_state = dict(policy_state)
                _, policy_record = sample_organization_decision(
                    trainer.actor, encoder, rollout_policy_state, device=torch.device("cpu")
                )
                self.assertEqual(policy_record["selected_action"], "STOP")
                self.assertAlmostEqual(
                    sum(policy_record["raw_probabilities"].values()), 1.0, places=6
                )
                self.assertAlmostEqual(
                    sum(policy_record["masked_probabilities"].values()), 1.0, places=6
                )
                states = [{"sequence": 0, "fingerprint": "m0", "nodes": [{"id": "root",
                    "localObjective": "solve", "createdAt": 0, "status": "ready"}],
                    "dagEdges": [], "runtimeEvents": []}]
                for node_index in range(rollout_index % 3):
                    prior_nodes = list(states[-1]["nodes"])
                    child_id = f"child-{node_index}"
                    states.append({"sequence": len(states),
                        "fingerprint": f"m{len(states)}-{rollout_index}",
                        "nodes": [*prior_nodes, {"id": child_id, "parentId": "root",
                            "localObjective": "subtask", "createdAt": node_index + 1,
                            "depth": 1, "status": "ready"}],
                        "dagEdges": [*states[-1]["dagEdges"],
                            {"kind": "derivation", "from": "root", "to": child_id}],
                        "runtimeEvents": []})
                terminal = dict(states[-1])
                terminal["sequence"] = len(states)
                terminal["fingerprint"] = f"terminal-{rollout_index}"
                terminal["nodes"] = [{**node, "status": "completed"}
                                     if node["id"] == "root" else node
                                     for node in terminal["nodes"]]
                states.append(terminal)
                records.append({
                    "group_id": f"lhtb:0:{task_id}", "benchmark": "lhtb",
                    "task_id": task_id, "split": "train", "epoch": 0,
                    "rollout_index": rollout_index, "policy_revision": 0,
                    "organization_seed": rollout_index,
                    "complete": True, "environment_failure": False,
                    "terminal_reward": 0.4, "initial_snapshot_fingerprint": "same",
                    "task_checksum": "checksum", "docker_digest": "sha256:image",
                    "runtime_config": {"timeout": 3600}, "policy_records": [policy_record],
                    "policy_interface_revision": LHTB_POLICY_INTERFACE_REVISION,
                    "process_states": states,
                    "state_transitions": build_state_transition_samples(states),
                })
            update = trainer.update_group(records)
            self.assertFalse(update["actor_updated"])
            self.assertGreater(update["transition_reward_summary"]["topology_transitions"], 0)
            self.assertGreater(update["transition_reward_summary"]["negative_topology_rewards"], 0)
            self.assertTrue(update["transition_samples"])
            self.assertEqual(trainer.value_steps, 1)
            restored = LHTBProcessGRPOTrainer(
                checkpoint, manifest, encoder=encoder, resume=True
            )
            self.assertEqual(restored.groups, 1)
            self.assertEqual(restored.value_steps, 1)
            self.assertEqual(restored.actor_steps, 0)


if __name__ == "__main__":
    unittest.main()
