from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping
import numpy as np
from copy import deepcopy

from .process_state import canonical_fingerprint
from .lhtb_transitions import build_state_transition_samples
from .lhtb_native import normalize_native_task_id
from .organization import LHTB_POLICY_INTERFACE_REVISION
from .lhtb_value_metrics import annotate_value_traces

from .io import write_jsonl


def official_lhtb_reward(result: Mapping[str, Any]) -> float:
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    if not isinstance(rewards, Mapping) or not rewards:
        raise ValueError("Harbor result has no official verifier rewards")
    if "reward" in rewards:
        value = rewards["reward"]
    elif len(rewards) == 1:
        value = next(iter(rewards.values()))
    else:
        raise ValueError(f"ambiguous official verifier rewards: {sorted(rewards)}")
    reward = float(value)
    if not 0.0 <= reward <= 1.0:
        raise ValueError("official LHTB reward is outside [0,1]")
    return reward


def import_harbor_group(
    job_dir: Path, output: Path, group_id: str, task_id: str, category: str,
    split: str, epoch: int, policy_revision: int, environment_digest: str,
    runtime_config: Mapping[str, Any], expected: int = 8,
    arm: str = "learned_information_realization",
    environment_backend: str = "docker",
    value_checkpoint: Path | None = None,
    device_name: str = "cpu",
) -> List[Dict[str, Any]]:
    result_paths = []
    for candidate in sorted(job_dir.rglob("result.json")):
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(value, Mapping) and "task_name" in value and "task_checksum" in value:
            result_paths.append(candidate)
    records: List[Dict[str, Any]] = []
    for rollout_index, result_path in enumerate(result_paths):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        partial_candidates = list(result_path.parent.rglob("roy-partial-trajectory.json"))
        if not partial_candidates:
            partial_candidates = list(result_path.parent.parent.rglob("roy-partial-trajectory.json"))
        snapshot: Mapping[str, Any] = {}
        if partial_candidates:
            snapshot = json.loads(partial_candidates[0].read_text(encoding="utf-8"))
        exception = result.get("exception_info")
        reward_available = True
        try:
            reward = official_lhtb_reward(result)
        except ValueError:
            reward_available = False
            reward = 0.0
            exception = exception or {"exception_type": "MissingVerifierReward"}
        started = result.get("started_at")
        finished = result.get("finished_at")
        wall_seconds = 0.0
        if started and finished:
            wall_seconds = (
                datetime.fromisoformat(str(finished)).timestamp()
                - datetime.fromisoformat(str(started)).timestamp()
            )
        timeout_types = {"TimeoutError", "AgentTimeoutError", "AgentTimeout"}
        exception_type = str((exception or {}).get("exception_type", ""))
        exception_message = str((exception or {}).get("exception_message", ""))
        agent_metadata = _agent_metadata(result)
        internal_deadline = agent_metadata.get("termination_reason") == "rollout_deadline"
        normal_deadline = reward_available and (
            exception_type in timeout_types or internal_deadline
        )
        if normal_deadline and snapshot:
            snapshot = _deadline_terminal_snapshot(snapshot, reward)
        final_output = (snapshot.get("runtime") or {}).get("finalOutput") \
            if isinstance(snapshot.get("runtime"), Mapping) else None
        policy_dead_end = (
            isinstance(final_output, Mapping)
            and final_output.get("status") == "policy_dead_end"
        ) or "policy_dead_end:" in exception_message
        environment_invalid = "environment_invalid:" in exception_message
        sampling_invalid = "sampling_invalid:" in exception_message
        complete = bool(partial_candidates) and reward_available \
            and (exception is None or normal_deadline) \
            and not policy_dead_end and not environment_invalid and not sampling_invalid
        if normal_deadline:
            termination_type = "timeout_with_reward"
        elif policy_dead_end and reward_available:
            termination_type = "policy_dead_end"
        elif exception is None and reward_available:
            termination_type = "completed_with_reward"
        elif environment_invalid:
            termination_type = "environment_invalid"
        elif sampling_invalid:
            termination_type = "sampling_invalid"
        else:
            termination_type = "environment_failure"
        process_states = snapshot.get("processStates", [])
        record = {
            "schema_version": 2, "id": str(result.get("id", result_path.parent.name)),
            "group_id": group_id, "benchmark": "lhtb", "task_id": task_id,
            "category": category, "split": split, "epoch": epoch,
            "arm": arm,
            "rollout_index": rollout_index, "policy_revision": policy_revision,
            "organization_seed": snapshot.get("organizationSeed"),
            "task_checksum": result.get("task_checksum"),
            "environment_backend": environment_backend,
            "environment_digest": environment_digest,
            "docker_digest": environment_digest if environment_backend == "docker" else None,
            "runtime_config": dict(runtime_config),
            "initial_snapshot_fingerprint": snapshot.get("initialSnapshotFingerprint"),
            "actions": _actions(snapshot), "policy_records": snapshot.get("policyRecords", []),
            "runtime_events": snapshot.get("runtimeEvents", []),
            "policy_interface_revision": _policy_interface_revision(snapshot),
            "process_states": process_states,
            "state_transitions": build_state_transition_samples(process_states),
            "terminal_reward": reward, "complete": complete,
            "environment_failure": termination_type in {
                "environment_invalid", "sampling_invalid", "environment_failure"
            },
            "environment_invalid": environment_invalid,
            "sampling_invalid": sampling_invalid,
            "accepted_for_training": complete,
            "termination_type": termination_type,
            "harbor_result": result, "harbor_result_path": str(result_path),
            "wall_time_seconds": wall_seconds,
            "tokens": _token_total(result),
            "semantic_audit_path": agent_metadata.get("roy_semantic_audit_root"),
            "runtime_audit_path": agent_metadata.get("roy_runtime_audit_root"),
        }
        records.append(record)
    if len(records) != expected:
        raise ValueError(f"expected {expected} Harbor results, found {len(records)}")
    if value_checkpoint is not None:
        records = [dict(value) for value in annotate_value_traces(
            records, str(value_checkpoint), device_name
        )]
    write_jsonl(output, records, append=output.exists())
    return records


def sample_audit(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize topology, MCTS and dense-credit completeness without updating models."""
    trajectories = []
    for record in records:
        states = list(record.get("process_states", []))
        transitions = list(record.get("state_transitions", []))
        policy = list(record.get("policy_records", []))
        terminal = states[-1] if states else {}
        rewards = [value.get("process_reward") for value in transitions]
        shaped_returns = [float(value) for value in record.get("shaped_returns", [])]
        mcts_records = [value for value in policy if isinstance(value, Mapping)
                        and value.get("behaviorPolicy", value.get("behavior_policy")) == "mcts_puct"]
        profiles = sorted({str(((value.get("policyState", value.get("policy_state")) or {})
                                .get("sampling_profile") or {}).get("id"))
                           for value in policy if isinstance(value, Mapping)})
        profiles = [value for value in profiles if value != "None"]
        trajectories.append({
            "id": record.get("id"), "rollout_index": record.get("rollout_index"),
            "organization_seed": record.get("organization_seed"),
            "official_terminal_reward": record.get("terminal_reward"),
            "sampling_profiles": profiles,
            "state_count": len(states), "decision_count": len(policy),
            "terminal_node_count": len(terminal.get("nodes", [])),
            "terminal_maximum_depth": max(
                (int(value.get("depth", 0)) for value in terminal.get("nodes", [])), default=0
            ),
            "derive_count": sum(value.get("selectedAction", value.get("selected_action"))
                                == "DERIVE" for value in policy if isinstance(value, Mapping)),
            "connect_count": sum(value.get("selectedAction", value.get("selected_action"))
                                 == "CONNECT" for value in policy if isinstance(value, Mapping)),
            "dependency_edge_count": sum(value.get("kind") == "dependency"
                                         for value in terminal.get("dagEdges", [])),
            "communication_edge_count": sum(value.get("kind") == "communication"
                                             for value in terminal.get("dagEdges", [])),
            "transition_chain_complete": bool(states)
                and len(transitions) == max(0, len(states) - 1),
            "step_node_deltas": [
                int((value.get("topology_delta") or {}).get("node_count_delta", 0))
                for value in transitions
            ],
            "process_rewards": rewards,
            "shaped_returns": shaped_returns,
            "process_reward_complete": bool(transitions)
                and all(value is not None for value in rewards),
            "process_reward_signs": {
                sign: sum(value.get("reward_sign") == sign for value in transitions)
                for sign in ("positive", "zero", "negative")
            },
            "mcts_decision_count": len(mcts_records),
            "mcts_selected_process_rewards": [
                value.get("selectedProcessReward", value.get("selected_process_reward"))
                for value in mcts_records
            ],
            "mcts_trace_complete": len(mcts_records) == len(policy) and all(
                any(item.get("phase") == "backup" for item in
                    value.get("mctsSearchTrace", value.get("mcts_search_trace", [])))
                for value in mcts_records
            ),
            "tokens": record.get("tokens"), "wall_time_seconds": record.get("wall_time_seconds"),
            "complete": bool(record.get("complete")),
            "environment_failure": bool(record.get("environment_failure")),
        })
    terminal_rewards = [float(value["official_terminal_reward"]) for value in trajectories
                        if value["official_terminal_reward"] is not None]
    shaped_returns = [value for trajectory in trajectories
                      for value in trajectory["shaped_returns"]]
    node_counts = [int(value["terminal_node_count"]) for value in trajectories]
    profile_set = sorted({profile for value in trajectories for profile in value["sampling_profiles"]})
    return {
        "trajectory_count": len(trajectories),
        "official_reward_std": float(np.std(terminal_rewards)) if terminal_rewards else None,
        "sampling_profiles": profile_set,
        "terminal_node_span": max(node_counts) - min(node_counts) if node_counts else None,
        "all_transition_chains_complete": all(value["transition_chain_complete"]
                                               for value in trajectories),
        "all_step_rewards_complete": all(value["process_reward_complete"]
                                         for value in trajectories),
        "all_mcts_traces_complete": all(value["mcts_trace_complete"]
                                        for value in trajectories),
        "value_training_available": len(terminal_rewards) == len(trajectories)
            and len(trajectories) > 0,
        "actor_dense_signal_available": len(shaped_returns) > 1
            and float(np.std(shaped_returns)) > 1e-8,
        "shaped_return_std": float(np.std(shaped_returns)) if shaped_returns else None,
        "preconditions_for_training": len(trajectories) == 8
            and all(value["complete"] and not value["environment_failure"]
                    for value in trajectories)
            and len({value["organization_seed"] for value in trajectories}) == 8
            and len(profile_set) >= 3
            and bool(node_counts) and max(node_counts) - min(node_counts) >= 2
            and all(value["transition_chain_complete"] for value in trajectories)
            and all(value["process_reward_complete"] for value in trajectories)
            and all(value["mcts_trace_complete"] for value in trajectories)
            and len(terminal_rewards) == 8,
        "trajectories": trajectories,
    }


def _deadline_terminal_snapshot(snapshot: Mapping[str, Any], reward: float) -> Mapping[str, Any]:
    value = deepcopy(dict(snapshot))
    states = list(value.get("processStates", []))
    if not states:
        return value
    terminal = deepcopy(states[-1])
    previous = str(terminal.get("fingerprint"))
    terminal["sequence"] = int(terminal.get("sequence", len(states) - 1)) + 1
    terminal["previousFingerprint"] = previous
    terminal_event = {
        "id": f"deadline-{terminal['sequence']}", "kind": "verifier",
        "at": 0, "attributes": {"officialFinalReward": reward,
                                  "termination": "training_deadline"},
    }
    terminal["runtimeEvents"] = [*terminal.get("runtimeEvents", [])[-23:], terminal_event]
    previous_range = terminal.get("runtimeEventRange") or {}
    previous_total = int(previous_range.get("total", len(value.get("runtimeEvents", []))))
    terminal["runtimeEventRange"] = {
        "start": max(0, previous_total + 1 - 24),
        "endExclusive": previous_total + 1,
        "total": previous_total + 1,
    }
    terminal.pop("fingerprint", None)
    terminal["fingerprint"] = canonical_fingerprint(terminal)
    states.append(terminal)
    value["processStates"] = states
    value["runtimeEvents"] = [*value.get("runtimeEvents", []), terminal_event]
    return value


def _token_total(result: Mapping[str, Any]) -> int:
    contexts = []
    if isinstance(result.get("agent_result"), Mapping):
        contexts.append(result["agent_result"])
    for step in result.get("step_results") or []:
        if isinstance(step, Mapping) and isinstance(step.get("agent_result"), Mapping):
            contexts.append(step["agent_result"])
    return sum(int(value.get("n_input_tokens") or 0)
               + int(value.get("n_output_tokens") or 0) for value in contexts)


def _agent_metadata(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(result.get("agent_result"), Mapping):
        return result["agent_result"].get("metadata") or {}
    for step in reversed(result.get("step_results") or []):
        if isinstance(step, Mapping) and isinstance(step.get("agent_result"), Mapping):
            return step["agent_result"].get("metadata") or {}
    return {}


def _actions(snapshot: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    states = list(snapshot.get("processStates", []))
    result = []
    for state in states:
        for event in state.get("runtimeEvents", []):
            if event.get("kind") == "organization_action":
                action = (event.get("attributes") or {}).get("action")
                if action and action not in result:
                    result.append(action)
    return result


def _policy_interface_revision(snapshot: Mapping[str, Any]) -> str:
    for record in snapshot.get("policyRecords", []):
        state = record.get("policyState", record.get("policy_state"))
        if isinstance(state, Mapping) and state.get("interface_revision"):
            return str(state["interface_revision"])
    return "legacy-full-growing-context"


def validate_smoke(root: Path, task_ids: tuple[str, ...] = (
    "great-expectations-audit", "poc-exploit-craft",
    "opensees-seismic-structural-regression-audit",
), max_input_tokens: int = 15_000_000) -> Dict[str, Any]:
    by_task: Dict[str, List[tuple[Path, Mapping[str, Any]]]] = {}
    for path in root.rglob("result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        agent = value.get("agent_info") or {}
        raw_task_name = value.get("task_name")
        if not isinstance(raw_task_name, str) or not raw_task_name:
            continue
        task_name = normalize_native_task_id(raw_task_name)
        if (agent.get("name") == "roy-lhtb-agent"
                and task_name in task_ids):
            by_task.setdefault(task_name, []).append((path, value))
    groups = {}
    for task_id, values in by_task.items():
        valid = []
        for item in values:
            result = item[1]
            exception = result.get("exception_info") or {}
            exception_type = str(exception.get("exception_type", ""))
            try:
                official_lhtb_reward(result)
                reward_available = True
            except ValueError:
                reward_available = False
            if reward_available and (not exception or exception_type in {
                "TimeoutError", "AgentTimeoutError", "AgentTimeout"
            }):
                valid.append(item)
        selected = sorted(valid, key=lambda item: item[0].stat().st_mtime)[-8:]
        if len(selected) != 8:
            continue
        rewards = []
        fingerprints = set()
        seeds = set()
        state_counts = []
        terminal_node_counts = []
        terminal_depths = []
        sampling_profiles = set()
        topology_transition_count = 0
        input_tokens = 0
        policy_diagnostics = True
        compact_interface = True
        raw_terminal_events = True
        derive_available = False
        derive_selected = False
        for result_path, result in selected:
            rewards.append(official_lhtb_reward(result))
            partials = list(result_path.parent.rglob("roy-partial-trajectory.json")) \
                or list(result_path.parent.parent.rglob("roy-partial-trajectory.json"))
            if not partials:
                raise ValueError(f"smoke trajectory is missing for {task_id}")
            snapshot = json.loads(partials[0].read_text(encoding="utf-8"))
            fingerprints.add(str(snapshot.get("initialSnapshotFingerprint") or ""))
            seeds.add(int(snapshot.get("organizationSeed", -1)))
            states = list(snapshot.get("processStates", []))
            if not states or any(not value.get("fingerprint") for value in states):
                raise ValueError(f"smoke M_0...M_T is incomplete for {task_id}")
            state_counts.append(len(states))
            transitions = build_state_transition_samples(states)
            if len(transitions) != len(states) - 1:
                raise ValueError(f"smoke transition chain is incomplete for {task_id}")
            if any(abs(int(value["topology_delta"]["node_count_delta"])) > 1
                   for value in transitions):
                raise ValueError(f"smoke topology was not derived one node at a time for {task_id}")
            topology_transition_count += sum(bool(value["topology_changed"])
                                             for value in transitions)
            terminal_node_counts.append(len(states[-1].get("nodes", [])))
            terminal_depths.append(max(
                (int(node.get("depth", 0)) for node in states[-1].get("nodes", [])),
                default=0,
            ))
            input_tokens += int((states[-1].get("usage") or {}).get("inputTokens", 0))
            records = list(snapshot.get("policyRecords", []))
            trajectory_profiles = {
                str(((record.get("policyState") or {}).get("sampling_profile") or {}).get("id"))
                for record in records
                if ((record.get("policyState") or {}).get("sampling_profile") or {}).get("id")
            }
            if len(trajectory_profiles) != 1:
                raise ValueError(f"smoke trajectory has ambiguous sampling profile for {task_id}")
            trajectory_profile = next(iter(trajectory_profiles))
            preferred_upper = {"compact": 3, "branching": 5,
                               "recursive": 7, "connected": 8}[trajectory_profile]
            if terminal_node_counts[-1] > preferred_upper:
                raise ValueError(
                    f"smoke {trajectory_profile} topology exceeded its sampling range "
                    f"for {task_id}: {terminal_node_counts[-1]} > {preferred_upper}"
                )
            if trajectory_profile in {"recursive", "connected"} and terminal_depths[-1] < 2:
                raise ValueError(
                    f"smoke {trajectory_profile} topology did not derive subsub for {task_id}"
                )
            sampling_profiles.update(
                str(((value.get("policyState") or {}).get("sampling_profile") or {}).get("id"))
                for value in records
                if ((value.get("policyState") or {}).get("sampling_profile") or {}).get("id")
            )
            policy_diagnostics = policy_diagnostics and bool(records) and all(
                isinstance(value.get("rawProbabilities"), Mapping)
                and isinstance(value.get("maskedProbabilities"), Mapping)
                and value.get("selectedAction")
                and math.isfinite(float(value.get("maskedOldLogProbability", float("nan"))))
                and math.isfinite(float(value.get(
                    "maskedOldActionLogProbability", float("nan")
                )))
                and math.isfinite(float(value.get(
                    "maskedOldCandidateConditionalLogProbability", float("nan")
                )))
                and abs(sum(float(probability) for probability in
                            value["rawProbabilities"].values()) - 1.0) <= 1e-4
                and abs(sum(float(probability) for probability in
                            value["maskedProbabilities"].values()) - 1.0) <= 1e-4
                for value in records
            )
            compact_interface = compact_interface and all(
                (value.get("policyState") or {}).get("interface_revision")
                == LHTB_POLICY_INTERFACE_REVISION
                and "terminal_result_count" in (value.get("policyState") or {})
                and "organization_action_count" in (value.get("policyState") or {})
                for value in records
            )
            derive_available = derive_available or any(
                "DERIVE" in value.get("availableActions", []) for value in records
            )
            derive_selected = derive_selected or any(
                value.get("selectedAction") == "DERIVE" for value in records
            )
            raw_terminal_events = raw_terminal_events and any(
                event.get("kind") == "terminal_result" and "output" in event
                for state in states for event in state.get("runtimeEvents", [])
            )
        if len(fingerprints) != 1 or "" in fingerprints:
            raise ValueError(f"smoke initial fingerprints do not match for {task_id}")
        if len(seeds) != 8:
            raise ValueError(f"smoke organization seeds are not unique for {task_id}")
        if input_tokens > max_input_tokens:
            raise ValueError(
                f"smoke input-token gate exceeded for {task_id}: "
                f"{input_tokens} > {max_input_tokens}"
            )
        if not policy_diagnostics:
            raise ValueError(f"smoke policy diagnostics are incomplete for {task_id}")
        if not compact_interface:
            raise ValueError(f"smoke policy interface is stale for {task_id}")
        if not raw_terminal_events:
            raise ValueError(f"smoke raw terminal ledger is incomplete for {task_id}")
        if not derive_available or not derive_selected:
            raise ValueError(f"smoke did not preserve and sample DERIVE for {task_id}")
        if len(sampling_profiles) < 3:
            raise ValueError(f"smoke lacks simple-to-complex topology coverage for {task_id}")
        if max(terminal_node_counts) - min(terminal_node_counts) < 2:
            raise ValueError(f"smoke terminal topologies lack complexity variance for {task_id}")
        groups[task_id] = {"rewards": rewards, "state_counts": state_counts,
                           "reward_std": float(np.std(rewards)),
                           "input_tokens": input_tokens,
                           "input_token_gate": max_input_tokens,
                           "policy_diagnostics": True,
                           "compact_interface": True,
                           "raw_terminal_ledger": True,
                           "derive_available": True,
                           "derive_selected": True,
                           "sampling_profiles": sorted(sampling_profiles),
                           "terminal_node_counts": terminal_node_counts,
                           "terminal_depths": terminal_depths,
                           "topology_transition_count": topology_transition_count,
                           "stepwise_topology": True}
    if len(groups) != len(task_ids):
        raise ValueError(f"expected {len(task_ids)} complete smoke groups, found {len(groups)}")
    if not any(value["reward_std"] > 1e-8 for value in groups.values()):
        raise ValueError("smoke has no within-group continuous reward variance")
    for audit in root.rglob("semantic-audit.jsonl"):
        for line in audit.read_text(encoding="utf-8").splitlines():
            if line and _contains_forbidden_benchmark_field(json.loads(line)):
                raise ValueError(f"benchmark keyword field leaked into semantic audit: {audit}")
    return {"tasks": len(groups), "trajectories": sum(len(value["rewards"])
            for value in groups.values()), "groups": groups,
            "semantic_field_leakage": False}


def _contains_forbidden_benchmark_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "keywords" in value:
            return True
        return any(_contains_forbidden_benchmark_field(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_benchmark_field(item) for item in value)
    return False
