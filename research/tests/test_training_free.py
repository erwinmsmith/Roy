from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest
from roy_research.training_free.aflow import AFlowDataset, AFlowEvaluator
from roy_research.training_free.engine import (
    RoyTrainingFreeEngine,
    TrainingFreeConfig,
    select_transition,
)
from roy_research.training_free.information import (
    AnswerDistributionMeasure,
    PairwiseStateMeasure,
)
from roy_research.training_free.harness import AgentHarness, AgentHarnessConfig
from roy_research.training_free.llm import (
    CallAudit,
    CandidateXRealizer,
    JsonLLM,
    WorkerModel,
    parse_json_object,
)
from roy_research.training_free.mia import (
    MIAObjectiveEvaluator,
    SemanticInformationLandscape,
)
from roy_research.training_free.matrix import (
    InformationMatrix,
    MatrixSearchResult,
    SemanticRolloutEvaluator,
    admissible_matrix_count,
    expand_matrix,
    matrix_neighbors,
)
from roy_research.training_free.tools import (
    TaskToolRegistry,
    ToolRequest,
    macos_readonly_sandbox_prefix,
)
from roy_research.training_free.types import (
    AgentState,
    AgentStatus,
    BenchmarkTask,
    CandidateDependency,
    CandidateGraph,
    ContextState,
    MemoryState,
    ResultState,
)


@dataclass
class FakeCompletion:
    content: str
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30
    latency_ms: int = 1


class ScriptedClient:
    model = "scripted-model"

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def complete(self, messages, **kwargs):
        purpose = kwargs["metadata"]["purpose"]
        payload = json.loads(messages[1]["content"])
        self.calls.append({"purpose": purpose, "payload": payload, **kwargs})
        candidate_id = "r0_A0_c1"
        if purpose == "root_worker":
            value = {
                "result": {
                    "candidate_answer": "120", "claims": ["initial attempt"],
                    "evidence": [], "assumptions": [], "unresolved": ["angle sum check"],
                    "reasoning_summary": "An uncertain initial computation.", "confidence": 0.55,
                },
                "memory_entries": [],
            }
        elif purpose == "candidate_proposal":
            value = {
                "candidate_dependency_graph": {
                    "nodes": [{
                        "candidate_id": candidate_id, "parent_id": "A0",
                        "direction": "independently recompute the polygon angle sum",
                        "why_needed": "the root arithmetic is uncertain",
                        "required_inputs": ["A0.current_derivation"],
                        "requested_tools": ["symbolic_math"],
                        "expected_output": "a checked numeric angle",
                        "stop_condition": "the angle sum has been independently verified",
                    }],
                    "dependencies": [{
                        "source": "A0", "target": candidate_id, "kind": "hard",
                        "artifact": "current_derivation",
                    }],
                },
            }
        elif purpose == "global_selector":
            value = {"selected_subgraphs": [{
                "candidate_ids": [candidate_id], "selection_reason": "resolves the only gap",
            }]}
        elif purpose == "candidate_x_realization":
            candidate_id = payload["selected_candidate_subgraph"]["nodes"][0]["candidate_id"]
            value = {
                "configuration_reasoning_summary": "Use an independent arithmetic verifier.",
                "risks": ["may repeat the root approach"],
                "agents": [{
                    # Runtime binds the immutable parent from the dependency graph.
                    "agent_id": candidate_id, "parent_id": None,
                    "objective": "Recompute the pentagon angle sum and verify the large angle.",
                    "role": "Independent geometry arithmetic verifier.",
                    "context": {
                        "original_task": payload["original_task"],
                        "mandatory_inputs": ["A0.current_derivation"],
                        "weighted_inputs": {"A0": 0.5}, "received_messages": [],
                        "public_tests": [],
                    },
                    "memory": {
                        "namespace": f"memory/{candidate_id}",
                        "inherited_refs": ["A0.current_derivation"], "entries": [],
                    },
                    "tools": ["symbolic_math"],
                    "result": {
                        "candidate_answer": "", "claims": [], "evidence": [],
                        "assumptions": [], "unresolved": ["large angle"],
                        "reasoning_summary": "Not executed yet.", "confidence": 0,
                    },
                    "status": "ready", "expected_output": "verified large angle",
                    "stop_condition": "a numeric angle is derived and checked",
                }],
            }
        elif purpose == "provisional_worker":
            value = {
                "result": {
                    "candidate_answer": "135", "claims": ["8x=540"],
                    "evidence": ["pentagon interior sum is 540"], "assumptions": [],
                    "unresolved": [], "reasoning_summary": "The large angle is 135.",
                    "confidence": 0.98,
                },
                "memory_entries": ["independent result: 135"],
            }
        elif purpose == "channelizer":
            value = {"message": "Independent verification gives 8x=540 and large angle 135."}
        elif purpose == "receiver_update":
            agent_id = payload["agent"]["contract"]["agent_id"]
            answer = "135" if agent_id == "A0" else payload["agent"]["result"]["candidate_answer"]
            value = {
                "result": {
                    "candidate_answer": answer, "claims": ["cross-checked"],
                    "evidence": payload["inbound_messages"], "assumptions": [],
                    "unresolved": [], "reasoning_summary": "Integrated inbound evidence.",
                    "confidence": 0.95,
                },
                "memory_entries": ["received cross-check"],
            }
        elif purpose == "worker_result_reconciliation":
            value = {
                "candidate_answer": payload["result"]["candidate_answer"],
                "ambiguous": False,
                "basis": "the supplied derivation",
            }
        elif purpose == "semantic_information_judge":
            agent_ids = payload["agent_ids"]
            size = len(agent_ids)
            g = [[0.0 for _ in agent_ids] for _ in agent_ids]
            if size > 1:
                root = agent_ids.index("A0")
                for index in range(size):
                    if index != root:
                        g[index][root] = 0.9
                        g[root][index] = 0.2
            value = {
                "agent_ids": agent_ids,
                "directional_potential": g,
                "redundancy": [[0.0 for _ in agent_ids] for _ in agent_ids],
                "conversion_fidelity": [1.0 for _ in agent_ids],
                "root_relations": {
                    agent_id: ("supports" if agent_id == "A0" else "contradicts")
                    for agent_id in agent_ids
                },
                "root_uncertainty": 0.1,
                "calibration_summary": "Independent verification is novel for the root.",
            }
        elif purpose == "posterior_probe":
            support = payload["hypothesis_support"]
            answer = payload["root_state"]["result"]["candidate_answer"]
            preferred = "135" if answer == "135" and "135" in support else "120"
            probabilities = {item: 0.1 / max(1, len(support) - 1) for item in support}
            probabilities[preferred] = 0.9
            value = {"probabilities": probabilities}
        else:  # pragma: no cover
            raise AssertionError(purpose)
        return FakeCompletion(json.dumps(value))


def test_engine_commits_externally_realized_candidate_x() -> None:
    client = ScriptedClient()
    config = TrainingFreeConfig(
        maximum_selected_subgraphs=1,
        maximum_nodes_per_subgraph=1,
        matrix_iterations=1,
        matrix_beam_width=2,
        maximum_matrix_evaluations=4,
        communication_rounds=1,
        maximum_organization_rounds=1,
        information_gain_epsilon=0.001,
    )
    task = BenchmarkTask(
        "math-0", "MATH", "A pentagon problem", [], {"solution": "\\boxed{135}"},
    )
    run = RoyTrainingFreeEngine(client, client, config=config).run(task)

    assert run.rounds[0].committed is True
    assert run.final_answer == "135"
    realized = run.rounds[0].realized_candidates["round-0-subgraph-0"]
    assert realized.agents["r0_A0_c1"].memory.namespace == "memory/r0_A0_c1"
    expensive = [call for call in client.calls if call["purpose"] == "candidate_x_realization"]
    assert len(expensive) == 1
    assert expensive[0]["thinking"] == "enabled"
    assert expensive[0]["max_tokens"] == config.candidate_realizer_max_tokens
    candidate_execution = [
        call for call in client.calls if call["purpose"] == "provisional_worker"
    ]
    assert candidate_execution
    assert all(call["thinking"] == "enabled" for call in candidate_execution)
    assert all(call["max_tokens"] == config.candidate_worker_max_tokens for call in candidate_execution)
    assert config.candidate_worker_max_tokens == 16_384
    value = run.to_dict()
    assert value["schema_version"] == 5
    assert value["initial_root_answer"] == "120"
    assert value["final_answer"] == "135"
    assert value["agent_harness"]["schema_version"] == 1
    assert value["agent_harness"]["config"]["maximum_memory_entries"] == 64
    assert [checkpoint["phase"] for checkpoint in value["checkpoints"]] == [
        "initial", "search_state", "committed",
    ]
    assert value["rounds"][0]["transition_kind"] == "expand"
    assert value["rounds"][0]["topology_drift"]["agent_expansion"] == 1
    assert value["rounds"][0]["provisional_agents"]["round-0-subgraph-0"][
        "r0_A0_c1"
    ]["result"]["candidate_answer"] == "135"
    assert {item["relation"] for item in value["dependency_ledger"]} == {
        "derivation", "hard",
    }
    assert value["organization_summary"] == {
        "rounds": 1,
        "candidates_proposed": 3,
        "candidate_subgraphs_realized": 1,
        "candidate_subgraphs_rejected": 0,
        "committed_expansions": 1,
        "committed_reorganizations": 0,
        "terminal_stops": 0,
        "committed_derivation_dependencies": 1,
    }
    worker_payload = next(
        call["payload"] for call in client.calls if call["purpose"] == "root_worker"
    )
    proposal_payload = next(
        call["payload"] for call in client.calls if call["purpose"] == "candidate_proposal"
    )
    organization = proposal_payload["current_organization_context"]
    assert organization["current_matrix"]["agent_ids"] == ["A0"]
    assert organization["recent_history"][0]["kind"] == "state_initialized"
    assert worker_payload["agent"]["harness_schema_version"] == 1
    assert [item["name"] for item in worker_payload["available_tool_schemas"]] == [
        "symbolic_math"
    ]
    selector_payload = next(
        call["payload"] for call in client.calls if call["purpose"] == "global_selector"
    )
    assert "entries" not in selector_payload["current_agents"]["A0"]
    assert "memory_entry_count" in selector_payload["current_agents"]["A0"]
    realizer_payload = next(
        call["payload"] for call in client.calls if call["purpose"] == "candidate_x_realization"
    )
    assert "entries" not in realizer_payload["current_mas"]["A0"]
    judge_calls = [call for call in client.calls if call["purpose"] == "semantic_information_judge"]
    assert len(judge_calls) == 1
    assert not {
        "posterior_probe", "pairwise_information_probe"
    } & {call["purpose"] for call in client.calls}
    assert value["search_architecture"]["candidate_matrix_llm_rollouts"] == 0
    assert value["call_audit"]["calls"]["semantic_information_judge"] == 1
    assert sum(
        event["kind"] == "winner_matrix_executed" for event in value["event_ledger"]
    ) == 1
    searched = value["rounds"][0]["frontiers"]["round-0-subgraph-0"]["search_space"]
    assert searched["evaluated_count"] > 1
    assert all(
        item["information_observations"][0]["llm_rollouts"] == 0
        for item in searched["evaluated_matrices"]
    )


def test_matched_single_agent_direct_skips_all_organization_calls() -> None:
    client = ScriptedClient()
    run = RoyTrainingFreeEngine(client, config=TrainingFreeConfig()).run_direct(
        BenchmarkTask("math-direct", "MATH", "pentagon", [], {"solution": "135"})
    )
    value = run.to_dict()
    assert value["method"] == "single_agent_direct"
    assert value["initial_root_answer"] == value["final_answer"]
    assert value["final_answer"] == "120"
    assert value["rounds"] == []
    assert [call["purpose"] for call in client.calls] == [
        "root_worker", "worker_result_reconciliation",
    ]
    assert value["call_audit"]["calls"] == {
        "root_worker": 1, "worker_result_reconciliation": 1,
    }


def test_roy_and_direct_use_the_identical_root_execution_request() -> None:
    task = BenchmarkTask("same-root", "MATH", "pentagon", [], {"solution": "135"})
    direct_client = ScriptedClient()
    roy_client = ScriptedClient()
    RoyTrainingFreeEngine(direct_client).run_direct(task)
    RoyTrainingFreeEngine(
        roy_client,
        config=TrainingFreeConfig(maximum_agents=1, maximum_organization_rounds=1),
    ).run(task)
    direct_root = next(call for call in direct_client.calls if call["purpose"] == "root_worker")
    roy_root = next(call for call in roy_client.calls if call["purpose"] == "root_worker")
    assert direct_root["payload"] == roy_root["payload"]
    assert direct_root["max_tokens"] == roy_root["max_tokens"]


def test_initial_candidate_graph_has_generic_epistemic_coverage() -> None:
    run = RoyTrainingFreeEngine(
        ScriptedClient(),
        config=TrainingFreeConfig(maximum_agents=4, maximum_organization_rounds=1),
    ).run(BenchmarkTask("coverage", "MATH", "pentagon", [], {"solution": "135"}))
    directions = [
        node["direction"] for node in run.rounds[0].candidate_graph["nodes"]
    ]
    operations = {
        node["epistemic_operation"] for node in run.rounds[0].candidate_graph["nodes"]
    }
    assert len(directions) >= 3
    assert any("specification" in direction for direction in directions)
    assert any("falsify" in direction for direction in directions)
    assert {"specification_audit", "adversarial_falsification"} <= operations
    assert (
        len(run.rounds[0].realized_candidates)
        + len(run.rounds[0].rejected_candidates)
    ) == 3


def test_math_worker_reconciles_candidate_answer_with_its_own_derivation() -> None:
    class InconsistentClient:
        model = "inconsistent"

        def complete(self, messages, **kwargs):
            purpose = kwargs["metadata"]["purpose"]
            if purpose == "root_worker":
                value = {
                    "result": {
                        "candidate_answer": "\\boxed{144}",
                        "claims": ["8x=540, so the requested angle is 135"],
                        "evidence": ["540/4=135"], "assumptions": [], "unresolved": [],
                        "reasoning_summary": "Therefore the requested angle is 135 degrees.",
                        "confidence": 1.0,
                    },
                    "memory_entries": [],
                }
            elif purpose == "worker_result_reconciliation":
                value = {
                    "candidate_answer": "\\boxed{135}", "ambiguous": False,
                    "basis": "claims and reasoning both conclude 135",
                }
            else:  # pragma: no cover
                raise AssertionError(purpose)
            return FakeCompletion(json.dumps(value))

    run = RoyTrainingFreeEngine(InconsistentClient()).run_direct(
        BenchmarkTask("math", "MATH", "angle", [], {"solution": "135"})
    )
    assert run.final_answer == "\\boxed{135}"


def test_worker_fails_closed_when_provider_echoes_the_request_payload() -> None:
    class EchoClient:
        model = "echo"

        def complete(self, messages, **kwargs):
            return FakeCompletion(messages[1]["content"])

    with pytest.raises(ValueError, match="top-level result"):
        RoyTrainingFreeEngine(EchoClient()).run_direct(
            BenchmarkTask("math", "MATH", "solve this", [], {"solution": "1"})
        )


def test_worker_recovers_echoed_payload_with_reasoning_schema_retry() -> None:
    class EchoThenRecoverClient:
        model = "echo-then-recover"

        def __init__(self):
            self.calls = []

        def complete(self, messages, **kwargs):
            purpose = kwargs["metadata"]["purpose"]
            self.calls.append((purpose, kwargs.get("thinking")))
            if purpose == "root_worker":
                return FakeCompletion(messages[1]["content"])
            if purpose == "root_worker_schema_retry":
                return FakeCompletion(json.dumps({
                    "result": {
                        "candidate_answer": "\\boxed{1}", "claims": ["the answer is 1"],
                        "evidence": [], "assumptions": [], "unresolved": [],
                        "reasoning_summary": "Solved after a schema retry.", "confidence": 1.0,
                    },
                    "memory_entries": [],
                }))
            if purpose == "worker_result_reconciliation":
                return FakeCompletion(json.dumps({
                    "candidate_answer": "\\boxed{1}", "ambiguous": False,
                    "basis": "the supplied result",
                }))
            raise AssertionError(purpose)

    client = EchoThenRecoverClient()
    run = RoyTrainingFreeEngine(client).run_direct(
        BenchmarkTask("math", "MATH", "solve this", [], {"solution": "1"})
    )
    assert run.final_answer == "\\boxed{1}"
    assert ("root_worker_schema_retry", "enabled") in client.calls


def test_json_llm_retries_reasoning_that_exhausts_budget_before_json() -> None:
    class EmptyThenValidClient:
        model = "empty-then-valid"

        def __init__(self):
            self.calls = []

        def complete(self, messages, **kwargs):
            self.calls.append(kwargs["metadata"]["purpose"])
            if len(self.calls) == 1:
                return FakeCompletion("", completion_tokens=8192, total_tokens=8202)
            return FakeCompletion(json.dumps({"answer": "complete"}))

    client = EmptyThenValidClient()
    audit = CallAudit()
    value = JsonLLM(client, audit).call(
        "candidate_x_realization", "configure X", {},
        max_tokens=16_384, thinking="enabled",
    )
    assert value == {"answer": "complete"}
    assert client.calls == [
        "candidate_x_realization", "candidate_x_realization_empty_retry",
    ]


def test_engine_fails_closed_when_selected_candidate_x_never_materializes() -> None:
    class EmptyCandidateClient(ScriptedClient):
        def complete(self, messages, **kwargs):
            if kwargs["metadata"]["purpose"].startswith("candidate_x_realization"):
                return FakeCompletion("", completion_tokens=8192, total_tokens=8202)
            return super().complete(messages, **kwargs)

    with pytest.raises(RuntimeError, match="portfolio was not fully realized"):
        RoyTrainingFreeEngine(
            EmptyCandidateClient(),
            config=TrainingFreeConfig(
                maximum_selected_subgraphs=1, maximum_nodes_per_subgraph=1,
                maximum_organization_rounds=1,
            ),
        ).run(BenchmarkTask("math", "MATH", "solve", [], {"solution": "1"}))


def test_json_llm_regenerates_malformed_json_once_with_reasoning() -> None:
    class MalformedThenValidClient:
        model = "malformed-then-valid"

        def __init__(self):
            self.calls = []

        def complete(self, messages, **kwargs):
            self.calls.append((kwargs["metadata"]["purpose"], kwargs.get("thinking")))
            if len(self.calls) == 1:
                return FakeCompletion('{"answer": "broken"')
            payload = json.loads(messages[1]["content"])
            assert payload["_protocol_retry"]["reason"].startswith("invalid JSON")
            return FakeCompletion('{"answer": "valid"}')

    client = MalformedThenValidClient()
    audit = CallAudit()
    value = JsonLLM(client, audit).call(
        "unit", "Return JSON.", {"required_schema": {"answer": "string"}}, max_tokens=64,
    )
    assert value == {"answer": "valid"}
    assert client.calls == [("unit", "disabled"), ("unit_json_retry", "enabled")]
    assert audit.calls == {"unit": 1, "unit_json_retry": 1}


def test_runtime_preserves_immutable_original_task_from_candidate_realizer() -> None:
    agent = AgentState.from_dict(
        {
            "agent_id": "candidate", "parent_id": "A0", "objective": "verify",
            "role": "verifier", "tools": [],
            "context": {"original_task": "a paraphrase owned by the model"},
            "memory": {"namespace": "memory/candidate"},
            "result": {}, "status": "ready", "expected_output": "evidence",
            "stop_condition": "evidence produced",
        },
        original_task="the exact benchmark instruction",
        available_tools=[],
    )
    assert agent.context.original_task == "the exact benchmark instruction"


def test_next_round_proposals_receive_committed_path_state() -> None:
    class TwoRoundClient(ScriptedClient):
        def complete(self, messages, **kwargs):
            purpose = kwargs["metadata"]["purpose"]
            payload = json.loads(messages[1]["content"])
            if purpose == "candidate_proposal" and payload["round_index"] == 1:
                self.calls.append({"purpose": purpose, "payload": payload, **kwargs})
                return FakeCompletion(json.dumps({
                    "candidate_dependency_graph": {"nodes": [], "dependencies": []},
                }))
            return super().complete(messages, **kwargs)

    client = TwoRoundClient()
    config = TrainingFreeConfig(
        maximum_selected_subgraphs=1,
        maximum_nodes_per_subgraph=1,
        matrix_iterations=1,
        matrix_beam_width=2,
        maximum_matrix_evaluations=4,
        communication_rounds=1,
        maximum_organization_rounds=2,
        maximum_agents=2,
        information_gain_epsilon=0.001,
    )
    task = BenchmarkTask("math", "MATH", "pentagon", [], {"solution": "135"})
    run = RoyTrainingFreeEngine(client, client, config=config).run(task)
    assert len(run.rounds) == 2
    assert run.rounds[0].transition_kind == "expand"
    assert run.rounds[1].transition_kind == "stop"
    second_round = [
        call["payload"] for call in client.calls
        if call["purpose"] == "candidate_proposal"
        and call["payload"].get("round_index") == 1
    ]
    assert second_round
    assert run.final_answer == "135"
    assert sum(call["purpose"] == "root_worker" for call in client.calls) == 1
    assert second_round[0]["committed_agent"]["result"]["candidate_answer"] == "135"
    context = second_round[0]["current_organization_context"]
    assert {item["relation"] for item in context["dependency_state"]} == {
        "derivation", "hard",
    }
    assert "organization_transition_committed" in {
        item["kind"] for item in context["recent_history"]
    }
    assert len(run.to_dict()["matrix_trajectory"]) == 3
    assert run.to_dict()["call_audit"]["calls"]["semantic_information_judge"] == 2
    assert sum(event.kind == "winner_matrix_executed" for event in run.event_ledger) == 1


def test_json_parser_repairs_unescaped_latex_without_masking_truncation() -> None:
    assert parse_json_object(r'{"answer":"\boxed{2\sqrt{3}}"}') == {
        "answer": r"\boxed{2\sqrt{3}}",
    }
    with pytest.raises(json.JSONDecodeError):
        parse_json_object('{"answer":"unterminated')


def test_candidate_graph_closes_hard_predecessors_and_rejects_cycles() -> None:
    graph = CandidateGraph.from_dict({
        "nodes": [
            {"candidate_id": "c1", "direction": "one", "why_needed": "gap",
             "required_inputs": [], "requested_tools": [], "expected_output": "one",
             "stop_condition": "done"},
            {"candidate_id": "c2", "direction": "two", "why_needed": "gap",
             "required_inputs": [], "requested_tools": [], "expected_output": "two",
             "stop_condition": "done"},
        ],
        "dependencies": [
            {"source": "c1", "target": "c2", "kind": "hard", "artifact": "result"},
        ],
    }, "A0")
    assert graph.hard_closure(["c2"], ["A0"]) == frozenset(("c1", "c2"))

    with pytest.raises(ValueError, match="cycle"):
        CandidateGraph.from_dict({
            "nodes": [
                {"candidate_id": "c1", "direction": "one", "why_needed": "gap",
                 "expected_output": "one", "stop_condition": "done"},
                {"candidate_id": "c2", "direction": "two", "why_needed": "gap",
                 "expected_output": "two", "stop_condition": "done"},
            ],
            "dependencies": [
                {"source": "c1", "target": "c2", "kind": "hard"},
                {"source": "c2", "target": "c1", "kind": "hard"},
            ],
        }, "A0")


def test_x_realizer_rejects_incomplete_external_configuration() -> None:
    class IncompleteClient(ScriptedClient):
        def complete(self, messages, **kwargs):
            return FakeCompletion(json.dumps({
                "configuration_reasoning_summary": "draft",
                "agents": [{"agent_id": "c1", "objective": "check"}],
            }))

    graph = CandidateGraph.from_dict({
        "nodes": [{
            "candidate_id": "c1", "direction": "check", "why_needed": "gap",
            "required_inputs": [], "requested_tools": [], "expected_output": "answer",
            "stop_condition": "checked",
        }],
        "dependencies": [],
    }, "A0")
    root = AgentState(
        "A0", None, "solve", "solver", ContextState("task"), MemoryState("memory/A0"),
        [], ResultState(), AgentStatus.DONE, "answer", "done",
    )
    realizer = CandidateXRealizer(JsonLLM(IncompleteClient(), CallAudit()))
    with pytest.raises(ValueError, match="incomplete|explicitly configure"):
        realizer.realize(
            "s0", ["c1"], graph, {"A0": root}, benchmark="MATH",
            original_task="task", public_tests=[], available_tools=[],
        )


def test_x_realizer_binds_graph_required_inputs_before_freezing_contract() -> None:
    graph = CandidateGraph.from_dict({
        "nodes": [{
            "candidate_id": "r0_A0_c1", "direction": "check", "why_needed": "gap",
            "required_inputs": ["committed answer"], "requested_tools": [],
            "expected_output": "answer", "stop_condition": "checked",
        }],
        "dependencies": [],
    }, "A0")
    root = AgentState(
        "A0", None, "solve", "solver", ContextState("task"), MemoryState("memory/A0"),
        [], ResultState(candidate_answer="1"), AgentStatus.DONE, "answer", "done",
    )
    realized = CandidateXRealizer(JsonLLM(ScriptedClient(), CallAudit())).realize(
        "s0", ["r0_A0_c1"], graph, {"A0": root}, benchmark="MATH",
        original_task="task", public_tests=[], available_tools=["symbolic_math"],
    )
    assert "committed answer" in realized.agents["r0_A0_c1"].context.mandatory_inputs


def test_aflow_loader_exposes_public_but_not_hidden_humaneval_tests(tmp_path: Path) -> None:
    root = tmp_path / "AFlow"
    data = root / "data" / "datasets"
    data.mkdir(parents=True)
    validation = data / "humaneval_validate.jsonl"
    public = data / "humaneval_public_test.jsonl"
    validation.write_text(json.dumps({
        "task_id": "HumanEval/0", "prompt": "def f(): ...", "entry_point": "f",
        "canonical_solution": "    return 1", "test": "def check(candidate): assert candidate()==1",
    }) + "\n", encoding="utf-8")
    public.write_text(json.dumps({
        "problem_id": "HumanEval/0", "entry_point": "f", "test": ["assert candidate()==1"],
    }) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "source": {"revision": "ignored"},
        "benchmarks": {"HumanEval": {
            "optimization": {"path": str(validation.relative_to(root)),
                             "sha256": hashlib.sha256(validation.read_bytes()).hexdigest()},
            "public_tests": {"path": str(public.relative_to(root)),
                             "sha256": hashlib.sha256(public.read_bytes()).hexdigest()},
        }},
    }), encoding="utf-8")
    dataset = AFlowDataset(root, manifest)
    dataset.verify = lambda benchmark, split: None  # type: ignore[method-assign]
    task = dataset.load("HumanEval", "optimization")[0]
    assert task.public_tests == ["assert candidate()==1"]
    assert "def check" not in task.instruction
    assert "def check" in task.evaluator_payload["test"]


def test_humaneval_evaluator_requires_process_isolation(tmp_path: Path) -> None:
    task = BenchmarkTask(
        "HumanEval/0", "HumanEval", "def f(): ...", [],
        {"entry_point": "f", "canonical_solution": "return 1", "test": ""},
    )
    evaluator = AFlowEvaluator(tmp_path, Path("/usr/bin/python3"))
    with pytest.raises(RuntimeError, match="process-isolation"):
        evaluator.score(task, "def f(): return 1")


def test_aflow_evaluator_preserves_virtual_environment_launcher(tmp_path: Path) -> None:
    launcher = tmp_path / "python"
    launcher.symlink_to(Path(sys.executable))
    evaluator = AFlowEvaluator(tmp_path, launcher)
    assert evaluator.python == launcher.absolute()


def test_aflow_math_scorer_fails_closed_without_symbolic_equivalence(
    tmp_path: Path,
) -> None:
    evaluator = AFlowEvaluator(tmp_path, Path(sys.executable))
    evaluator._invoke = lambda *args, **kwargs: {  # type: ignore[method-assign]
        "symbolic_equivalence": False,
    }
    task = BenchmarkTask("math", "MATH", "solve", [], {"solution": "\\boxed{1}"})
    with pytest.raises(RuntimeError, match="antlr4-python3-runtime"):
        evaluator.score(task, "\\boxed{1}")


def test_aflow_scorer_runs_from_disposable_writable_directory(tmp_path: Path) -> None:
    evaluator = AFlowEvaluator(tmp_path, Path(sys.executable))
    value = evaluator._invoke(
        """
import json
from pathlib import Path
Path('logs').mkdir()
Path('logs/evaluator.log').write_text('ok')
print(json.dumps({'score': 1.0}))
""",
        {},
    )
    assert value == {"score": 1.0}
    assert not (tmp_path / "logs").exists()


def test_symbolic_math_tool_solves_equation_and_rejects_code_execution() -> None:
    task = BenchmarkTask("math", "MATH", "solve", [], {"solution": "2"})
    agent = AgentState(
        "A0", None, "solve", "solver", ContextState("solve"), MemoryState("memory/A0"),
        ["symbolic_math"], ResultState(), AgentStatus.READY, "answer", "done",
    )
    registry = TaskToolRegistry(task)
    solved = registry.execute(agent, ToolRequest(
        "symbolic_math",
        {"operation": "solve", "expression": " 2*x+1 = 5 ", "variable": "x"},
        "verify arithmetic",
    ))
    rejected = registry.execute(agent, ToolRequest(
        "symbolic_math",
        {"operation": "evaluate", "expression": "__import__('os').getcwd()"},
        "invalid request",
    ))
    assert solved.success is True
    assert json.loads(solved.output)["result"] == "[2]"
    assert rejected.success is False
    assert registry.audit.to_dict()["total_calls"] == 2


@pytest.mark.skipif(not macos_readonly_sandbox_prefix(), reason="macOS sandbox-exec unavailable")
def test_public_test_tool_runs_only_agent_visible_tests() -> None:
    task = BenchmarkTask(
        "HumanEval/x", "HumanEval", "def increment", ["assert candidate(2) == 3"],
        {
            "entry_point": "increment",
            "canonical_solution": "return x + 1",
            "test": "def check(candidate): assert False  # hidden and must not run",
        },
    )
    agent = AgentState(
        "A0", None, "implement", "coder",
        ContextState(task.instruction, public_tests=list(task.public_tests)),
        MemoryState("memory/A0"), ["public_tests"], ResultState(), AgentStatus.READY,
        "code", "public tests pass",
    )
    registry = TaskToolRegistry(task, code_sandbox_prefix=macos_readonly_sandbox_prefix())
    result = registry.execute(agent, ToolRequest(
        "public_tests", {"code": "def increment(x):\n    return x + 1"}, "check code",
    ))
    assert result.success is True
    assert json.loads(result.output) == {"passed": 1, "total": 1, "all_passed": True}
    assert "hidden" not in registry.audit.records[0].output


def test_worker_reasons_again_after_tool_observation() -> None:
    class ToolLoopClient:
        model = "tool-loop"

        def __init__(self) -> None:
            self.calls = []

        def complete(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            self.calls.append(payload)
            if "tool_observations" not in payload:
                value = {
                    "result": {"candidate_answer": "", "unresolved": ["solve equation"]},
                    "memory_entries": [],
                    "tool_requests": [{
                        "tool_name": "symbolic_math",
                        "arguments": {
                            "operation": "solve", "expression": "2*x+1=5", "variable": "x",
                        },
                        "reason": "obtain a checked solution",
                    }],
                }
            else:
                value = {
                    "result": {
                        "candidate_answer": "2", "claims": ["x=2"], "evidence": ["tool"],
                        "assumptions": [], "unresolved": [], "reasoning_summary": "Used solver.",
                        "confidence": 0.99,
                    },
                    "memory_entries": ["symbolic solver returned x=2"],
                    "tool_requests": [],
                }
            return FakeCompletion(json.dumps(value))

    task = BenchmarkTask("math", "MATH", "solve", [], {"solution": "2"})
    agent = AgentState(
        "A0", None, "solve equation", "solver", ContextState("solve"),
        MemoryState("memory/A0"), ["symbolic_math"], ResultState(), AgentStatus.READY,
        "answer", "done",
    )
    client = ToolLoopClient()
    registry = TaskToolRegistry(task)
    worker = WorkerModel(
        JsonLLM(client, CallAudit()), max_tool_rounds=2, max_tool_calls=2,
        result_reconciler_max_tokens=0,
    )
    worker.configure_tools(registry)
    updated = worker.execute_local(agent, "MATH")
    assert updated.result.candidate_answer == "2"
    assert len(client.calls) == 2
    assert json.loads(client.calls[1]["tool_observations"][0]["output"])["result"] == "[2]"
    assert registry.audit.to_dict()["successful_calls"] == 1
    assert registry.audit.records[0].scope == "counterfactual"


def test_worker_recovers_from_malformed_tool_request() -> None:
    class MalformedToolClient:
        model = "malformed-tool"

        def __init__(self) -> None:
            self.calls = []

        def complete(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            self.calls.append(payload)
            if "tool_observations" not in payload:
                value = {"result": {}, "tool_requests": ["not an object"]}
            else:
                value = {
                    "result": {"candidate_answer": "\\boxed{2}", "confidence": 1},
                    "tool_requests": [],
                }
            return FakeCompletion(json.dumps(value))

    task = BenchmarkTask("math", "MATH", "solve", [], {"solution": "2"})
    agent = AgentState(
        "A0", None, "solve", "solver", ContextState("solve"), MemoryState("memory/A0"),
        ["symbolic_math"], ResultState(), AgentStatus.READY, "answer", "done",
    )
    client = MalformedToolClient()
    worker = WorkerModel(JsonLLM(client, CallAudit()), max_tool_rounds=1, max_tool_calls=1)
    worker.configure_tools(TaskToolRegistry(task))
    updated = worker.execute_local(agent, "MATH")
    assert updated.result.candidate_answer == "\\boxed{2}"
    assert client.calls[1]["tool_observations"][0]["tool_name"] == "INVALID_TOOL_REQUEST"


def test_agent_harness_fixes_contract_and_bounds_private_runtime_state() -> None:
    state = AgentState(
        "A7", "A0", "verify equation", "algebra verifier",
        ContextState("solve the equation"), MemoryState("memory/A7"),
        ["symbolic_math"], ResultState(unresolved=["equation"]),
        AgentStatus.READY, "checked root", "root is verified",
    )
    config = AgentHarnessConfig(
        maximum_memory_entries=2,
        maximum_memory_entry_chars=20,
        retrieved_memory_entries=1,
        maximum_received_messages=2,
        maximum_context_chars=24_000,
    )
    harness = AgentHarness(state, config=config)
    fingerprint = harness.contract_fingerprint()
    harness.remember(["geometry note", "equation x equals two", "irrelevant recent note"])
    harness.receive_messages(["first", "second", "third", "third"])
    harness.apply_model_update(ResultState(candidate_answer="2"), ["equation verified"])

    view = harness.execution_view()
    assert harness.contract_fingerprint() == fingerprint
    assert state.memory.entries == ["irrelevant recent no", "equation verified"]
    assert view["private_memory"]["namespace"] == "memory/A7"
    assert view["private_memory"]["retrieved_entries"] == ["equation verified"]
    assert view["context"]["received_messages"] == ["second", "third"]
    assert "entries" not in harness.public_summary()

    snapshot = harness.snapshot()
    restored = AgentHarness.restore(snapshot, available_tools=["symbolic_math"])
    assert restored.snapshot() == snapshot
    state.role = "silently changed role"
    with pytest.raises(ValueError, match="immutable contract"):
        harness.execution_view()


def test_agent_harness_enforces_tool_capabilities_before_registry_execution() -> None:
    task = BenchmarkTask("math", "MATH", "solve", [], {"solution": "2"})
    state = AgentState(
        "A0", None, "solve equation", "solver", ContextState("solve"),
        MemoryState("memory/A0"), ["symbolic_math"], ResultState(), AgentStatus.READY,
        "answer", "done",
    )
    registry = TaskToolRegistry(task)
    harness = AgentHarness(state, registry)
    assert [item["name"] for item in harness.tool_catalog] == ["symbolic_math"]

    denied = harness.execute_tool(
        ToolRequest("python", {"code": "print(2)"}, "try unassigned capability"),
        scope="committed",
    )
    allowed = harness.execute_tool(
        ToolRequest(
            "symbolic_math",
            {"operation": "solve", "expression": "x=2", "variable": "x"},
            "verify",
        ),
        scope="committed",
    )
    assert denied.success is False
    assert "outside" in denied.error
    assert allowed.success is True
    assert registry.audit.to_dict()["total_calls"] == 1


def test_agent_harness_public_summary_does_not_expose_private_memory() -> None:
    source = AgentState(
        "A1", "A0", "derive checksum", "verifier", ContextState("task"),
        MemoryState("memory/A1", entries=["checksum is 42", "private unrelated note"]),
        [], ResultState(candidate_answer="42"), AgentStatus.DONE, "checksum", "derived",
    )
    receiver = AgentState(
        "A2", "A0", "check checksum", "reviewer", ContextState("task"),
        MemoryState("memory/A2", entries=["receiver-only secret"]), [], ResultState(),
        AgentStatus.READY, "review", "checked",
    )
    source_harness = AgentHarness(source)
    public = source_harness.public_summary()
    channel_source = source_harness.channel_source_view(receiver.objective)

    assert "checksum is 42" not in json.dumps(public)
    assert channel_source["relevant_private_memory"] == [
        "checksum is 42", "private unrelated note",
    ]
    assert "receiver-only secret" not in json.dumps(channel_source)


def test_agent_harness_execution_context_has_a_hard_serialized_size_limit() -> None:
    state = AgentState(
        "A0", None, "solve", "solver", ContextState("x" * 60_000),
        MemoryState("memory/A0", entries=["memory" * 5_000]), [],
        ResultState(reasoning_summary="reason" * 5_000), AgentStatus.READY,
        "answer", "done",
    )
    config = AgentHarnessConfig(maximum_context_chars=8_000)
    view = AgentHarness(state, config=config).execution_view()
    assert len(json.dumps(view, ensure_ascii=False)) <= config.maximum_context_chars
    assert view["context"]["original_task"].endswith("...[truncated]")


def test_matrix_search_neighbors_are_local_and_expansion_preserves_old_block() -> None:
    current = InformationMatrix.zero(["A0", "A1"])
    current.set_weight("A0", "A1", 0.5)
    neighbors = matrix_neighbors(current, (0.0, 0.5, 1.0), (), 0.0)
    for neighbor in neighbors:
        changes = [
            abs(neighbor.values[row][column] - current.values[row][column])
            for row in range(2) for column in range(2)
        ]
        assert max(changes) == 0.5
        assert sum(change > 0 for change in changes) == 1

    child = AgentState(
        "c1", "A1", "verify", "verifier",
        ContextState("task", weighted_inputs={"A1": 0.5}),
        MemoryState("memory/c1"), [], ResultState(), AgentStatus.READY,
        "proof", "verified",
    )
    expanded = expand_matrix(current, {"c1": child}, [])
    assert expanded.agent_ids == ["A0", "A1", "c1"]
    assert expanded.weight("A0", "A1") == current.weight("A0", "A1")

    saturated = InformationMatrix.zero(["A0", "A1"])
    saturated.set_weight("A0", "A1", 1.0)
    saturated_expansion = expand_matrix(saturated, {"c1": child}, [])
    saturated_expansion.validate()
    assert saturated_expansion.weight("A0", "A1") == 1.0
    assert saturated_expansion.weight("c1", "A1") == 0.0

    hard = [CandidateDependency("A1", "c1", "hard", "proof")]
    constrained = matrix_neighbors(expanded, (0.0, 0.5, 1.0), hard, 0.5)
    assert all(matrix.weight("A1", "c1") >= 0.5 for matrix in constrained)
    assert admissible_matrix_count(["A0", "A1"], (0.0, 0.5, 1.0)) == 9


def test_zero_communication_cannot_create_information_gain_from_probe_noise() -> None:
    class ProbeMustNotRun:
        def compare(self, *args, **kwargs):
            raise AssertionError("unchanged state must reuse the supplied prior")

    root = AgentState(
        "A0", None, "solve", "solver", ContextState("task"), MemoryState("memory/A0"),
        [], ResultState(candidate_answer="\\boxed{2}"), AgentStatus.DONE, "answer", "done",
    )
    evaluator = SemanticRolloutEvaluator(
        worker=None,  # type: ignore[arg-type]
        channelizer=None,  # type: ignore[arg-type]
        information_measure=PairwiseStateMeasure(ProbeMustNotRun()),  # type: ignore[arg-type]
        benchmark="MATH", root_id="A0", inbound_token_budget=100,
        communication_rounds=1, information_rollouts=2,
    )
    score, observations = evaluator.evaluate(
        {"A0": root}, InformationMatrix.zero(["A0"]),
    )
    assert score == 0.0
    assert all(item["score"] == 0.0 for item in observations)


def test_mia_functional_scores_direct_multihop_and_redundancy_without_llm() -> None:
    landscape = SemanticInformationLandscape(
        ["A0", "A1", "A2"],
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.7],
            [0.2, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ],
        [1.0, 0.5, 1.0], "A0", 0.6, "A1 has direct and relayed evidence",
    )
    matrix = InformationMatrix.zero(landscape.agent_ids)
    matrix.set_weight("A1", "A0", 0.5)
    matrix.set_weight("A1", "A2", 0.5)
    matrix.set_weight("A2", "A0", 0.5)
    objective = MIAObjectiveEvaluator(landscape, path_horizon=3).objective(matrix)
    assert objective.direct_delivery == pytest.approx(0.5)
    assert objective.multi_hop_delivery == pytest.approx(0.035)
    assert objective.redundancy_correction > 0
    assert objective.objective < objective.direct_delivery + objective.multi_hop_delivery

    certain = SemanticInformationLandscape(
        landscape.agent_ids, landscape.directional_potential, landscape.redundancy,
        landscape.conversion_fidelity, "A0", 0.0, "the root has no removable uncertainty",
    )
    certain_objective = MIAObjectiveEvaluator(certain).objective(matrix)
    assert certain_objective.usable_delivery > 0
    assert certain_objective.objective == 0.0


def test_mia_gain_is_relative_to_current_committed_matrix() -> None:
    landscape = SemanticInformationLandscape(
        ["A0", "A1"], [[0.0, 0.0], [0.8, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0],
        "A0", 0.2, "A1 can verify the root",
    )
    current = InformationMatrix.zero(landscape.agent_ids)
    current.set_weight("A1", "A0", 1.0)
    evaluator = MIAObjectiveEvaluator(landscape, reference_matrix=current)
    gain, observations = evaluator.evaluate({}, current)
    assert gain == pytest.approx(0.0)
    assert observations[0]["llm_rollouts"] == 0


def test_mia_uses_intersection_not_product_of_uncertainty_and_delivery() -> None:
    landscape = SemanticInformationLandscape(
        ["A0", "A1"], [[0.0, 0.0], [0.2, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0],
        "A0", 0.1, "A1 has a distinct check for the uncertain root",
    )
    matrix = InformationMatrix.zero(landscape.agent_ids)
    matrix.set_weight("A1", "A0", 1.0)
    objective = MIAObjectiveEvaluator(landscape).objective(matrix)
    assert objective.usable_delivery == pytest.approx(0.2)
    assert objective.objective == pytest.approx(0.1 * 0.2 / 0.3)


def test_mia_smooth_intersection_preserves_delivery_order_below_uncertainty_ceiling() -> None:
    landscape = SemanticInformationLandscape(
        ["A0", "weak", "strong"],
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.8, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [1.0, 1.0, 1.0], "A0", 0.2, "strong has more novel evidence",
    )
    weak = InformationMatrix.zero(landscape.agent_ids)
    weak.set_weight("weak", "A0", 1.0)
    strong = InformationMatrix.zero(landscape.agent_ids)
    strong.set_weight("strong", "A0", 1.0)
    evaluator = MIAObjectiveEvaluator(landscape)
    assert evaluator.objective(strong).objective > evaluator.objective(weak).objective
    assert evaluator.objective(strong).objective < landscape.root_uncertainty


def test_semantic_landscape_reorders_judge_axes_and_symmetrizes_redundancy() -> None:
    landscape = SemanticInformationLandscape.from_dict(
        {
            "agent_ids": ["A1", "A0"],
            "directional_potential": [[0.0, 0.7], [0.2, 0.0]],
            "redundancy": [[0.0, 0.8], [0.6, 0.0]],
            "conversion_fidelity": [0.9, 0.5],
            "root_relations": {"A1": "contradicts", "A0": "supports"},
            "root_uncertainty": 0.25,
            "calibration_summary": "ordered semantic estimates",
        },
        expected_agent_ids=["A0", "A1"], root_id="A0",
    )
    assert landscape.agent_ids == ["A0", "A1"]
    assert landscape.directional_potential == [[0.0, 0.2], [0.7, 0.0]]
    assert landscape.redundancy[0][1] == pytest.approx(0.7)
    assert landscape.conversion_fidelity == [0.5, 0.9]


def test_semantic_landscape_accepts_agent_id_maps_from_judge() -> None:
    landscape = SemanticInformationLandscape.from_dict(
        {
            "agent_ids": ["A0", "A1"],
            "directional_potential": {
                "A0": {"A0": 0.0, "A1": 0.1},
                "A1": {"A0": 0.8, "A1": 0.0},
            },
            "redundancy": {
                "A0": {"A0": 0.0, "A1": 0.3},
                "A1": {"A0": 0.3, "A1": 0.0},
            },
            "conversion_fidelity": {"A0": 0.9, "A1": 0.7},
            "root_relations": {"A0": "supports", "A1": "complements"},
            "root_uncertainty": 0.2,
            "calibration_summary": "mapped output",
        },
        expected_agent_ids=["A0", "A1"], root_id="A0",
    )
    assert landscape.directional_potential[1][0] == 0.8
    assert landscape.conversion_fidelity == [0.9, 0.7]


def test_semantic_landscape_repairs_zero_gain_for_explicit_contradiction() -> None:
    landscape = SemanticInformationLandscape.from_dict(
        {
            "agent_ids": ["A0", "A1"],
            "directional_potential": [[0.0, 0.0], [0.0, 0.0]],
            "redundancy": [[0.0, 0.2], [0.2, 0.0]],
            "conversion_fidelity": [1.0, 1.0],
            "root_relations": {"A0": "supports", "A1": "contradicts"},
            "root_uncertainty": 0.4,
            "calibration_summary": "A1 gives a conflicting answer that A0 must inspect.",
        },
        expected_agent_ids=["A0", "A1"], root_id="A0",
    )
    assert landscape.directional_potential[1][0] == pytest.approx(0.8)
    assert landscape.coherence_adjustments


def test_rollout_promotes_new_root_answer_into_dynamic_hypothesis_support() -> None:
    class ExactAnswerProbe:
        def posterior(self, root, support, benchmark, state_context):
            answer = root.result.candidate_answer
            selected = answer if answer in support else "OTHER"
            return {item: float(item == selected) for item in support}

    root = AgentState(
        "A0", None, "solve", "solver", ContextState("task"), MemoryState("memory/A0"),
        [], ResultState(candidate_answer="A"), AgentStatus.DONE, "answer", "done",
    )
    after = copy.deepcopy(root)
    after.result.candidate_answer = "B"
    measure = AnswerDistributionMeasure(ExactAnswerProbe(), [{"A0": root}])
    observation = measure.compare(root, after, "MATH")
    assert measure.support == ["A", "B", "OTHER"]
    assert observation.after["distribution"] == {"A": 0.0, "B": 1.0, "OTHER": 0.0}
    assert observation.score > 20.0


def test_pairwise_information_measure_requires_no_hypothesis_support() -> None:
    class PairwiseProbe:
        def compare(self, before, after, benchmark, state_context):
            return {
                "information_gain": 0.7,
                "before_uncertainty": 0.8,
                "after_uncertainty": 0.1,
                "new_information": ["public test now passes"],
                "rationale": "Execution evidence resolved the defect.",
            }

    before = AgentState(
        "A0", None, "implement", "coder", ContextState("task"), MemoryState("memory/A0"),
        [], ResultState(candidate_answer="code-v1", confidence=0.2), AgentStatus.DONE,
        "code", "tests pass",
    )
    after = copy.deepcopy(before)
    after.result.candidate_answer = "code-v2"
    after.result.confidence = 0.9
    measure = PairwiseStateMeasure(PairwiseProbe())  # type: ignore[arg-type]
    observation = measure.compare(before, after, "HumanEval")
    assert observation.score == 0.7
    assert "support" not in json.dumps(observation.to_dict())
    assert observation.details["new_information"] == ["public test now passes"]


def test_semantic_landscape_is_benchmark_agnostic_and_factory_injectable() -> None:
    engine = RoyTrainingFreeEngine(ScriptedClient(), config=TrainingFreeConfig())
    root = AgentState(
        "A0", None, "solve", "worker", ContextState("task"), MemoryState("memory/A0"),
        [], ResultState(), AgentStatus.READY, "artifact", "done",
    )
    estimated = engine._estimate_semantic_landscape(
        "FutureBenchmark", {"A0": root}, {},
    )
    assert estimated.agent_ids == ["A0"]

    custom = SemanticInformationLandscape(
        ["A0"], [[0.0]], [[0.0]], [0.8], "A0", 0.4, "custom estimate",
    )
    custom_engine = RoyTrainingFreeEngine(
        ScriptedClient(),
        semantic_landscape_factory=lambda benchmark, agents, context: custom,
    )
    assert custom_engine._estimate_semantic_landscape("Anything", {"A0": root}, {}) is custom


def test_positive_baseline_reorganization_is_committed_without_expansion() -> None:
    initial = InformationMatrix.zero(["A0", "A1"])
    reorganized = initial.clone()
    reorganized.set_weight("A1", "A0", 0.5)
    baseline = MatrixSearchResult(initial, reorganized, 0.2, 2, [{"x": 1.0}])
    candidate = MatrixSearchResult(initial, reorganized, 0.22, 2, [{"x": 1.0}])
    decision = select_transition(baseline, {"candidate": candidate}, epsilon=0.05)
    assert decision.kind == "reorganize"
    assert decision.commit is True
    assert decision.subgraph_id is None
    assert decision.information_gain == 0.2
