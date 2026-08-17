from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Protocol, Tuple

from .controlled import ControlledTask
from .grpo import hierarchical_advantages
from .providers import Completion


ENVIRONMENT_REVISION = "live-controlled-arithmetic-v1"


class CompletionClient(Protocol):
    model: str

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        metadata: Dict[str, Any] | None = None,
    ) -> Completion: ...


@dataclass(frozen=True)
class LiveProblem:
    task_id: str
    public_prompt: str
    evidence: str
    gold_answer: int
    evidence_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_live_problem(task: ControlledTask) -> LiveProblem:
    randomizer = random.Random(task.seed ^ 0x5A17C0DE)
    units_a = randomizer.randint(17, 49)
    price_a = randomizer.randint(11, 37)
    units_b = randomizer.randint(13, 43)
    price_b = randomizer.randint(7, 29)
    adjustment = randomizer.randint(31, 119)
    multiplier = randomizer.randint(2, 5)
    gold = (units_a * price_a + units_b * price_b - adjustment) * multiplier
    evidence_required = task.family != "activation"
    adjustment_text = "EVIDENCE_VALUE" if evidence_required else str(adjustment)
    prompt = (
        "Compute the audited inventory total using ordinary integer arithmetic. "
        f"Batch A has {units_a} units at {price_a} credits each. "
        f"Batch B has {units_b} units at {price_b} credits each. "
        f"Subtract the audit adjustment ({adjustment_text}), then multiply the result by {multiplier}. "
        "Order is: multiply each batch, add both products, subtract the adjustment, then apply the multiplier."
    )
    evidence = f"The deterministic audit tool returned EVIDENCE_VALUE={adjustment}."
    return LiveProblem(task.id, prompt, evidence, gold, evidence_required)


def live_event_graph(task: ControlledTask, problem: LiveProblem, draft: str | None) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = [
        {"id": "agent:root", "kind": "agent", "timestamp": 0, "actorId": "root", "text": "root solver", "status": "running"},
        {"id": "task", "kind": "subtask", "timestamp": 1, "actorId": "root", "text": problem.public_prompt, "status": "active"},
        {"id": "resource", "kind": "resource", "timestamp": 2, "actorId": "root", "text": "bounded live DeepSeek rollout", "status": "available"},
    ]
    edges: List[Dict[str, Any]] = [
        {"id": "temporal:1", "kind": "temporal", "from": "agent:root", "to": "task"},
        {"id": "consumes:1", "kind": "consumes", "from": "task", "to": "resource"},
    ]
    if problem.evidence_required:
        nodes.append({
            "id": "dependency:evidence", "kind": "dependency", "timestamp": 3,
            "actorId": "root", "text": "audit adjustment must be acquired", "status": "unresolved",
        })
        edges.append({
            "id": "dependency:1", "kind": "dependency", "from": "dependency:evidence", "to": "task", "required": True,
        })
    elif draft is not None:
        nodes.append({
            "id": "artifact:draft", "kind": "artifact", "timestamp": 3, "actorId": "root",
            "text": draft, "status": "available", "attributes": {"outputContractSatisfied": True},
        })
    if task.family == "mixed":
        nodes.extend([
            {"id": "agent:verifier", "kind": "agent", "timestamp": 4, "actorId": "root", "text": "derived verifier", "status": "waiting"},
            {"id": "message:verification", "kind": "message", "timestamp": 5, "actorId": "root", "text": "verification request", "status": "queued"},
        ])
        edges.append({
            "id": "communication:1", "kind": "communication", "from": "task", "to": "message:verification", "required": False,
        })
    return {"parentId": "root", "nodes": nodes, "edges": edges, "observedAt": 0}


def live_child_specifications(task: ControlledTask, max_tokens: int = 384) -> List[Dict[str, Any]]:
    roles = (
        "Solve the arithmetic independently and show both batch products.",
        "Act as a verifier: recompute in reverse and identify any arithmetic inconsistency.",
        "Audit evidence use and produce a clean candidate answer with at least two checks.",
    )
    return [{
        "id": f"{task.id}-live-branch-{index}",
        "task": role,
        "context": [task.family, ENVIRONMENT_REVISION],
        "tools": ["fixture.audit.lookup"] if task.family != "activation" else [],
        "resources": {"computeTokens": max_tokens, "parallelSlots": 1, "toolCalls": 1 if task.family != "activation" else 0},
        "outputContract": {"format": "json", "requiredFields": ["answer", "checks", "evidence_used"]},
        "dependencies": ["dependency:evidence"] if task.family != "activation" else [],
    } for index, role in enumerate(roles)]


def collect_live_group(
    task: ControlledTask,
    client: CompletionClient,
    repeats: int = 1,
    max_tokens: int = 384,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if max_tokens < 64:
        raise ValueError("max_tokens must be at least 64")
    problem = build_live_problem(task)
    draft_completion: Completion | None = None
    if not problem.evidence_required:
        draft_completion = _ask(
            client, problem, "Create the checkpoint draft answer.", max_tokens, temperature,
            {"task_id": task.id, "phase": "checkpoint_draft"},
        )
    draft = draft_completion.content if draft_completion else None
    event_graph = live_event_graph(task, problem, draft)
    legal_actions = ["CONTINUE", "BRANCH"]
    if not problem.evidence_required:
        legal_actions.append("RETURN")
    branches = live_child_specifications(task, max_tokens)
    results: List[Dict[str, Any]] = []
    non_branch: Dict[str, List[float]] = {"CONTINUE": []}
    if "RETURN" in legal_actions:
        assert draft_completion is not None
        return_score = score_output(draft_completion.content, problem)
        non_branch["RETURN"] = [return_score for _ in range(repeats)]
    branch_returns: Dict[str, List[float]] = {branch["id"]: [] for branch in branches}

    for repeat in range(repeats):
        direct = _ask(
            client, problem, "Continue as the parent and finish the task directly.", max_tokens, temperature,
            {"task_id": task.id, "action": "CONTINUE", "repeat": repeat, "phase": "rollout"},
        )
        direct_score = score_output(direct.content, problem)
        non_branch["CONTINUE"].append(direct_score)
        results.append(_live_result("CONTINUE", direct_score, repeat, [direct], max_tokens))

        if "RETURN" in legal_actions:
            assert draft_completion is not None
            results.append(_live_result(
                "RETURN", non_branch["RETURN"][repeat], repeat, [], max_tokens,
                raw_output=draft_completion.content, terminal=True,
            ))

        for branch in branches:
            child = _ask(
                client, problem, branch["task"], max_tokens, temperature,
                {"task_id": task.id, "action": "BRANCH", "child_id": branch["id"], "repeat": repeat, "phase": "child"},
            )
            synthesis_instruction = (
                "Synthesize the final answer as the parent. Check the child proposal; do not trust it blindly. "
                f"Child proposal: {child.content}"
            )
            synthesis = _ask(
                client, problem, synthesis_instruction, max_tokens, temperature,
                {"task_id": task.id, "action": "BRANCH", "child_id": branch["id"], "repeat": repeat, "phase": "synthesis"},
            )
            branch_score = score_output(synthesis.content, problem)
            branch_returns[branch["id"]].append(branch_score)
            results.append(_live_result(
                "BRANCH", branch_score, repeat, [child, synthesis], max_tokens,
                child=branch, raw_output=synthesis.content,
                communication_edges=1 if task.family == "mixed" else 0,
            ))

    advantages = hierarchical_advantages(non_branch, branch_returns)
    checkpoint_resources = {
        "computeTokens": max_tokens * 8,
        "wallClockMs": 120_000,
        "parallelSlots": 3,
        "communicationEdges": 8,
        "toolCalls": 3,
    }
    randomness = {
        "temperature": temperature,
        "provider_seed_supported": False,
        "repeat_count": repeats,
    }
    checkpoint_material = json.dumps({
        "task": task.id,
        "seed": task.seed,
        "problem": problem.to_dict(),
        "draft": draft,
        "event_graph": event_graph,
        "legal_actions": legal_actions,
        "resources": checkpoint_resources,
        "randomness": randomness,
        "environment_revision": ENVIRONMENT_REVISION,
    }, sort_keys=True).encode("utf-8")
    fingerprint = hashlib.sha256(checkpoint_material).hexdigest()
    action_values = dict(advantages.action_values)
    oracle_values = dict(action_values)
    oracle_values["BRANCH"] = max(advantages.branch_values.values())
    oracle_utility = max(oracle_values.values())
    expected_actions = [
        action for action in ("CONTINUE", "BRANCH", "RETURN")
        if action in oracle_values and abs(oracle_values[action] - oracle_utility) <= 1e-9
    ]
    live_task = task.to_dict()
    live_task["fixture_expected_action"] = live_task["expected_action"]
    live_task["expected_action"] = expected_actions[0]
    live_task["expected_actions"] = expected_actions
    live_task["live_problem"] = problem.to_dict()
    return {
        "schema_version": 1,
        "task": live_task,
        "checkpoint": {
            "id": f"live-checkpoint-{task.id}",
            "fingerprint": fingerprint,
            "parent_id": "root",
            "event_graph": event_graph,
            "resources": checkpoint_resources,
            "legal_actions": legal_actions,
            "environment_revision": ENVIRONMENT_REVISION,
            "environment_seed": task.seed,
            "randomness": randomness,
            "checkpoint_generation": _completion_summary(draft_completion) if draft_completion else None,
        },
        "branch_specifications": branches,
        "results": results,
        "action_values": advantages.action_values,
        "outer_advantages": advantages.outer_advantages,
        "branch_values": advantages.branch_values,
        "branch_advantages": advantages.branch_advantages,
        "provider": "deepseek",
        "model": client.model,
    }


def parse_answer(content: str) -> Tuple[int | None, Dict[str, Any] | None]:
    stripped = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return None, None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, None
    if not isinstance(value, dict):
        return None, None
    answer = value.get("answer")
    if isinstance(answer, bool):
        return None, value
    try:
        return int(answer), value
    except (TypeError, ValueError):
        return None, value


def score_output(content: str, problem: LiveProblem) -> float:
    answer, value = parse_answer(content)
    if value is None:
        return 0.0
    score = 0.05
    if answer == problem.gold_answer:
        score += 0.80
    checks = value.get("checks")
    if isinstance(checks, list) and len(checks) >= 2:
        score += 0.10
    if value.get("evidence_used") is problem.evidence_required:
        score += 0.05
    return min(1.0, score)


def _ask(
    client: CompletionClient,
    problem: LiveProblem,
    instruction: str,
    max_tokens: int,
    temperature: float,
    metadata: Dict[str, Any],
) -> Completion:
    evidence = f"\n{problem.evidence}" if problem.evidence_required else ""
    return client.complete([
        {
            "role": "system",
            "content": (
                "You are a bounded downstream rollout policy. Return exactly one JSON object with integer "
                "answer, a checks array containing at least two short checks, and boolean evidence_used. No prose."
            ),
        },
        {"role": "user", "content": f"{instruction}\n\nTask: {problem.public_prompt}{evidence}"},
    ], max_tokens=max_tokens, temperature=temperature, metadata=metadata)


def _live_result(
    action: str,
    utility: float,
    repeat: int,
    completions: List[Completion],
    max_tokens: int,
    child: Dict[str, Any] | None = None,
    raw_output: str | None = None,
    terminal: bool = False,
    communication_edges: int = 0,
) -> Dict[str, Any]:
    used = sum(completion.total_tokens for completion in completions)
    rollout_budget = max_tokens * 8
    return {
        "action": action,
        "child_specification": child,
        "utility": utility,
        "duration_ms": sum(completion.latency_ms for completion in completions),
        "wait_ms": 0,
        "communication_edges": communication_edges,
        "resources_before": {"computeTokens": rollout_budget, "parallelSlots": 3, "toolCalls": 3},
        "resources_after": {"computeTokens": rollout_budget - used, "parallelSlots": 3, "toolCalls": 2 if child else 3},
        "terminal": terminal,
        "repeat": repeat,
        "provider": "deepseek",
        "token_usage": used,
        "prompt_tokens": sum(completion.prompt_tokens for completion in completions),
        "completion_tokens": sum(completion.completion_tokens for completion in completions),
        "request_count": len(completions),
        "raw_output": raw_output if raw_output is not None else (completions[-1].content if completions else ""),
    }


def _completion_summary(completion: Completion | None) -> Dict[str, Any] | None:
    if completion is None:
        return None
    return {
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "total_tokens": completion.total_tokens,
        "latency_ms": completion.latency_ms,
        "raw_output": completion.content,
    }
