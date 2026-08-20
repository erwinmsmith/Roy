from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping
import numpy as np
from copy import deepcopy

from .process_state import canonical_fingerprint

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
    split: str, epoch: int, policy_revision: int, docker_digest: str,
    runtime_config: Mapping[str, Any], expected: int = 8,
    arm: str = "learned_information_realization",
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
        normal_deadline = reward_available and exception_type in timeout_types
        if normal_deadline and snapshot:
            snapshot = _deadline_terminal_snapshot(snapshot, reward)
        complete = bool(partial_candidates) and (exception is None or normal_deadline)
        record = {
            "schema_version": 1, "id": str(result.get("id", result_path.parent.name)),
            "group_id": group_id, "benchmark": "lhtb", "task_id": task_id,
            "category": category, "split": split, "epoch": epoch,
            "arm": arm,
            "rollout_index": rollout_index, "policy_revision": policy_revision,
            "organization_seed": snapshot.get("organizationSeed"),
            "task_checksum": result.get("task_checksum"), "docker_digest": docker_digest,
            "runtime_config": dict(runtime_config),
            "initial_snapshot_fingerprint": snapshot.get("initialSnapshotFingerprint"),
            "actions": _actions(snapshot), "policy_records": snapshot.get("policyRecords", []),
            "process_states": snapshot.get("processStates", []),
            "terminal_reward": reward, "complete": complete,
            "environment_failure": exception is not None and not normal_deadline,
            "accepted_for_training": complete,
            "termination_type": "official_deadline_verifier" if normal_deadline else
                "completed" if exception is None else "environment_failure",
            "harbor_result": result, "harbor_result_path": str(result_path),
            "wall_time_seconds": wall_seconds,
            "tokens": _token_total(result),
            "semantic_audit_path": _agent_metadata(result).get("roy_semantic_audit_root"),
            "runtime_audit_path": _agent_metadata(result).get("roy_runtime_audit_root"),
        }
        records.append(record)
    if len(records) != expected:
        raise ValueError(f"expected {expected} Harbor results, found {len(records)}")
    write_jsonl(output, records, append=output.exists())
    return records


def _deadline_terminal_snapshot(snapshot: Mapping[str, Any], reward: float) -> Mapping[str, Any]:
    value = deepcopy(dict(snapshot))
    states = list(value.get("processStates", []))
    if not states:
        return value
    terminal = deepcopy(states[-1])
    previous = str(terminal.get("fingerprint"))
    terminal["sequence"] = int(terminal.get("sequence", len(states) - 1)) + 1
    terminal["previousFingerprint"] = previous
    terminal["runtimeEvents"] = [*terminal.get("runtimeEvents", []), {
        "id": f"deadline-{terminal['sequence']}", "kind": "verifier",
        "at": 0, "attributes": {"officialFinalReward": reward,
                                  "termination": "training_deadline"},
    }]
    terminal.pop("fingerprint", None)
    terminal["fingerprint"] = canonical_fingerprint(terminal)
    states.append(terminal)
    value["processStates"] = states
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


def validate_smoke(root: Path, task_ids: tuple[str, ...] = (
    "great-expectations-audit", "poc-exploit-craft",
    "opensees-seismic-structural-regression-audit",
)) -> Dict[str, Any]:
    by_task: Dict[str, List[tuple[Path, Mapping[str, Any]]]] = {}
    for path in root.rglob("result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        agent = value.get("agent_info") or {}
        if (agent.get("name") == "roy-lhtb-agent"
                and str(value.get("task_name")) in task_ids):
            by_task.setdefault(str(value["task_name"]), []).append((path, value))
    groups = {}
    for task_id, values in by_task.items():
        selected = sorted(values, key=lambda item: item[0].stat().st_mtime)[-8:]
        if len(selected) != 8:
            continue
        rewards = []
        fingerprints = set()
        state_counts = []
        for result_path, result in selected:
            rewards.append(official_lhtb_reward(result))
            partials = list(result_path.parent.rglob("roy-partial-trajectory.json")) \
                or list(result_path.parent.parent.rglob("roy-partial-trajectory.json"))
            if not partials:
                raise ValueError(f"smoke trajectory is missing for {task_id}")
            snapshot = json.loads(partials[0].read_text(encoding="utf-8"))
            fingerprints.add(str(snapshot.get("initialSnapshotFingerprint") or ""))
            states = list(snapshot.get("processStates", []))
            if not states or any(not value.get("fingerprint") for value in states):
                raise ValueError(f"smoke M_0...M_T is incomplete for {task_id}")
            state_counts.append(len(states))
        if len(fingerprints) != 1 or "" in fingerprints:
            raise ValueError(f"smoke initial fingerprints do not match for {task_id}")
        groups[task_id] = {"rewards": rewards, "state_counts": state_counts,
                           "reward_std": float(np.std(rewards))}
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
