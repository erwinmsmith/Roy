from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple

from .grpo import hierarchical_advantages

TERMINAL_SUCCESS_THRESHOLD = 0.8


@dataclass(frozen=True)
class ControlledTask:
    id: str
    family: str
    split: str
    ood: bool
    seed: int
    hidden_evidence_value: float
    direct_value: float
    return_value: float
    branch_values: Tuple[float, float, float]
    required_depth: int
    unseen_tool: bool
    expected_action: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["branch_values"] = list(self.branch_values)
        return value


def generate_tasks(seed: int = 20260815) -> List[ControlledTask]:
    tasks: List[ControlledTask] = []
    for family_offset, family in enumerate(("activation", "acquisition", "mixed")):
        for index in range(60):
            task_seed = seed + family_offset * 10_000 + index
            randomizer = random.Random(task_seed)
            split = "train" if index < 30 else "validation" if index < 40 else "test"
            ood = split == "test" and index >= 50
            hidden = 0.0 if family == "activation" else randomizer.uniform(0.25, 0.65)
            direct = randomizer.uniform(0.2, 0.55)
            return_value = randomizer.uniform(0.2, 0.55)
            branch_base = randomizer.uniform(0.38, 0.58)
            branches = tuple(max(0.0, min(1.0, branch_base + delta)) for delta in (-0.08, 0.0, 0.08))
            if family == "activation" and index % 3 == 0:
                direct = randomizer.uniform(0.82, 0.96)
            elif family == "activation" and index % 3 == 2:
                return_value = randomizer.uniform(0.82, 0.96)
            elif family != "activation" and index % 2 == 0:
                direct = randomizer.uniform(0.82, 0.96)
            else:
                branch_base = randomizer.uniform(0.78, 0.88)
                branches = tuple(max(0.0, min(1.0, branch_base + delta)) for delta in (-0.06, 0.0, 0.06))
            if ood:
                branches = (branches[2], branches[0], branches[1])
            values = {"CONTINUE": direct, "BRANCH": sum(branches) / len(branches)}
            if family == "activation":
                values["RETURN"] = return_value
            expected = max(values, key=values.get)
            tasks.append(ControlledTask(
                id=f"{family}-{index:03d}",
                family=family,
                split=split,
                ood=ood,
                seed=task_seed,
                hidden_evidence_value=hidden,
                direct_value=direct,
                return_value=return_value,
                branch_values=branches,
                required_depth=(5 + index % 4) if ood else (1 + index % 4),
                unseen_tool=ood and family != "activation",
                expected_action=expected,
            ))
    return tasks


def task_event_graph(task: ControlledTask) -> Dict[str, Any]:
    nodes = [
        {"id": "agent:root", "kind": "agent", "timestamp": 0, "actorId": "root", "text": "root solver", "status": "running"},
        {"id": "task", "kind": "subtask", "timestamp": 1, "actorId": "root", "text": f"{task.family} controlled structural task; direct confidence {task.direct_value:.3f}", "status": "active", "attributes": {"signal": task.direct_value}},
        {"id": "resource", "kind": "resource", "timestamp": 2, "actorId": "root", "text": f"required depth {task.required_depth}; branch signal {sum(task.branch_values) / 3:.3f}", "status": "available", "attributes": {"signal": sum(task.branch_values) / 3}},
    ]
    if task.family == "activation":
        nodes.append({
            "id": "artifact:draft", "kind": "artifact", "timestamp": 3, "actorId": "root",
            "text": f"draft answer confidence {task.return_value:.3f}", "status": "available",
            "attributes": {"outputContractSatisfied": True, "signal": task.return_value},
        })
    if task.family != "activation":
        nodes.append({
            "id": "dependency:evidence", "kind": "dependency", "timestamp": 3,
            "actorId": "root", "text": "hidden evidence must be acquired", "status": "unresolved",
        })
    if task.family == "mixed":
        nodes.extend([
            {"id": "agent:verifier", "kind": "agent", "timestamp": 4, "actorId": "root", "text": "derived verifier", "status": "waiting"},
            {"id": "message:verification", "kind": "message", "timestamp": 5, "actorId": "root", "text": "verification request", "status": "queued"},
            {"id": "artifact:claim", "kind": "artifact", "timestamp": 6, "actorId": "root", "text": "candidate claim", "status": "available"},
        ])
    edges = [
        {"id": "temporal:1", "kind": "temporal", "from": "agent:root", "to": "task"},
        {"id": "consumes:1", "kind": "consumes", "from": "task", "to": "resource"},
    ]
    if task.family != "activation":
        edges.append({
            "id": "dependency:1", "kind": "dependency", "from": "dependency:evidence", "to": "task", "required": True,
        })
    if task.family == "mixed":
        edges.extend([
            {"id": "communication:1", "kind": "communication", "from": "task", "to": "message:verification", "required": False},
            {"id": "communication:2", "kind": "communication", "from": "message:verification", "to": "agent:verifier", "required": True},
            {"id": "produces:1", "kind": "produces", "from": "agent:verifier", "to": "artifact:claim", "required": True},
        ])
    return {"parentId": "root", "nodes": nodes, "edges": edges, "observedAt": 0}


def child_specifications(task: ControlledTask) -> List[Dict[str, Any]]:
    return [{
        "id": f"{task.id}-branch-{index}",
        "task": f"Resolve candidate computation branch {index} for {task.id}",
        "context": [task.family],
        "tools": ["fixture.lookup"] if task.family != "activation" else [],
        "resources": {"computeTokens": 1, "parallelSlots": 1, "toolCalls": 1 if task.family != "activation" else 0},
        "outputContract": {"format": "json", "requiredFields": ["answer"], "groundingRequired": task.family != "activation"},
        "dependencies": ["dependency:evidence"] if task.family != "activation" else [],
    } for index in range(3)]


def collect_group(task: ControlledTask, repeats: int = 2) -> Dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    branches = child_specifications(task)
    results: List[Dict[str, Any]] = []
    legal_actions = ["CONTINUE", "BRANCH"]
    if task.family == "activation":
        legal_actions.append("RETURN")
    non_branch: Dict[str, List[float]] = {"CONTINUE": []}
    if "RETURN" in legal_actions:
        non_branch["RETURN"] = []
    branch_returns: Dict[str, List[float]] = {branch["id"]: [] for branch in branches}
    for repeat in range(repeats):
        non_branch["CONTINUE"].append(task.direct_value)
        results.append(_result("CONTINUE", task.direct_value, repeat, wait_ms=1 if task.family != "activation" else 0))
        if "RETURN" in legal_actions:
            return_utility = task.return_value
            non_branch["RETURN"].append(return_utility)
            results.append(_result("RETURN", return_utility, repeat, terminal=True))
        for index, branch in enumerate(branches):
            value = task.branch_values[index]
            branch_returns[branch["id"]].append(value)
            results.append(_result(
                "BRANCH", value, repeat, child=branch,
                wait_ms=1 if task.family != "activation" else 0,
                communication_edges=1 if task.family == "mixed" else 0,
            ))
    advantages = hierarchical_advantages(non_branch, branch_returns)
    checkpoint_id = f"checkpoint-{task.id}"
    fingerprint = hashlib.sha256(f"{task.id}:{task.seed}".encode()).hexdigest()
    return {
        "schema_version": 1,
        "task": task.to_dict(),
        "checkpoint": {
            "id": checkpoint_id,
            "fingerprint": fingerprint,
            "parent_id": "root",
            "event_graph": task_event_graph(task),
            "resources": {"computeTokens": 8, "wallClockMs": 1000, "parallelSlots": 3, "communicationEdges": 8, "toolCalls": 3},
            "legal_actions": legal_actions,
            "environment_revision": "controlled-derivation-v1",
            "environment_seed": task.seed,
        },
        "branch_specifications": branches,
        "results": results,
        "action_values": advantages.action_values,
        "outer_advantages": advantages.outer_advantages,
        "branch_values": advantages.branch_values,
        "branch_advantages": advantages.branch_advantages,
    }


def mechanism_diagnostics(task: ControlledTask) -> Dict[str, float]:
    no_child = 0.0 if task.family != "activation" else task.direct_value * 0.25
    evidence_only = min(1.0, no_child + task.hidden_evidence_value)
    full_child = min(1.0, evidence_only + max(task.branch_values) * (1.0 - evidence_only))
    acquisition = evidence_only - no_child
    activation = full_child - evidence_only
    return {
        "no_child": no_child,
        "evidence_only": evidence_only,
        "full_child": full_child,
        "acquisition": acquisition,
        "activation": activation,
        "remaining_budget_after_evidence": 7.0,
        "reconstruction_error": abs((full_child - no_child) - (acquisition + activation)),
    }


def _result(
    action: str,
    utility: float,
    repeat: int,
    child: Dict[str, Any] | None = None,
    terminal: bool = False,
    wait_ms: int = 0,
    communication_edges: int = 0,
) -> Dict[str, Any]:
    return {
        "action": action,
        "child_specification": child,
        "utility": utility,
        "duration_ms": 0,
        "wait_ms": wait_ms,
        "communication_edges": communication_edges,
        "resources_before": {"computeTokens": 8, "parallelSlots": 3, "toolCalls": 3},
        "resources_after": {"computeTokens": 7, "parallelSlots": 2 if action == "BRANCH" else 3, "toolCalls": 2 if action == "BRANCH" else 3},
        "terminal": terminal,
        "repeat": repeat,
    }
