from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence


class AgentStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting_for_dependencies"
    DORMANT = "dormant"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ResultState:
    candidate_answer: str = ""
    claims: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ResultState":
        value = value or {}
        confidence = min(1.0, max(0.0, float(value.get("confidence", 0.0))))
        return cls(
            candidate_answer=str(value.get("candidate_answer", "")),
            claims=_strings(value.get("claims")),
            evidence=_strings(value.get("evidence")),
            assumptions=_strings(value.get("assumptions")),
            unresolved=_strings(value.get("unresolved")),
            reasoning_summary=str(value.get("reasoning_summary", "")),
            confidence=confidence,
        )


@dataclass
class ContextState:
    original_task: str
    mandatory_inputs: List[str] = field(default_factory=list)
    weighted_inputs: Dict[str, float] = field(default_factory=dict)
    received_messages: List[str] = field(default_factory=list)
    public_tests: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], original_task: str) -> "ContextState":
        weighted = {
            str(key): min(1.0, max(0.0, float(weight)))
            for key, weight in dict(value.get("weighted_inputs", {})).items()
        }
        return cls(
            # The external realizer may describe or paraphrase the task, but it
            # does not own this immutable harness field. Always bind C_i to the
            # runtime-supplied source text.
            original_task=original_task,
            mandatory_inputs=_strings(value.get("mandatory_inputs")),
            weighted_inputs=weighted,
            received_messages=_strings(value.get("received_messages")),
            public_tests=_strings(value.get("public_tests")),
        )


@dataclass
class MemoryState:
    namespace: str
    inherited_refs: List[str] = field(default_factory=list)
    entries: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], agent_id: str) -> "MemoryState":
        namespace = str(value.get("namespace") or f"memory/{agent_id}")
        if namespace != f"memory/{agent_id}":
            raise ValueError(f"Agent {agent_id} must use private namespace memory/{agent_id}")
        return cls(
            namespace=namespace,
            inherited_refs=_strings(value.get("inherited_refs")),
            entries=_strings(value.get("entries")),
        )


@dataclass
class AgentState:
    """The complete semantic basis element X_i = (Q, R, C, M, T, Z, Sigma)."""

    agent_id: str
    parent_id: str | None
    objective: str
    role: str
    context: ContextState
    memory: MemoryState
    tools: List[str]
    result: ResultState
    status: AgentStatus
    expected_output: str
    stop_condition: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        original_task: str,
        available_tools: Iterable[str],
        expected_agent_id: str | None = None,
    ) -> "AgentState":
        agent_id = str(value.get("agent_id") or expected_agent_id or "").strip()
        if not agent_id:
            raise ValueError("realized X is missing agent_id")
        if expected_agent_id is not None and agent_id != expected_agent_id:
            raise ValueError(f"realized agent_id {agent_id!r} != candidate {expected_agent_id!r}")
        objective = str(value.get("objective", "")).strip()
        role = str(value.get("role", "")).strip()
        expected_output = str(value.get("expected_output", "")).strip()
        stop_condition = str(value.get("stop_condition", "")).strip()
        if not all((objective, role, expected_output, stop_condition)):
            raise ValueError(f"Agent {agent_id} has an incomplete Q/R/output/stop configuration")
        requested_tools = _strings(value.get("tools"))
        allowed = set(available_tools)
        unknown = sorted(set(requested_tools) - allowed)
        if unknown:
            raise ValueError(f"Agent {agent_id} requested unavailable tools: {unknown}")
        raw_status = str(value.get("status", AgentStatus.READY.value)).lower()
        try:
            status = AgentStatus(raw_status)
        except ValueError as error:
            raise ValueError(f"Agent {agent_id} has invalid status {raw_status!r}") from error
        context_value = value.get("context")
        memory_value = value.get("memory")
        if not isinstance(context_value, Mapping) or not isinstance(memory_value, Mapping):
            raise ValueError(f"Agent {agent_id} must explicitly configure C and M")
        return cls(
            agent_id=agent_id,
            parent_id=_optional_string(value.get("parent_id")),
            objective=objective,
            role=role,
            context=ContextState.from_dict(context_value, original_task),
            memory=MemoryState.from_dict(memory_value, agent_id),
            tools=requested_tools,
            result=ResultState.from_dict(value.get("result")),
            status=status,
            expected_output=expected_output,
            stop_condition=stop_condition,
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


DependencyKind = Literal["hard", "soft"]


@dataclass(frozen=True)
class CandidateDependency:
    source: str
    target: str
    kind: DependencyKind
    artifact: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateDependency":
        kind = str(value.get("kind", "soft"))
        if kind not in ("hard", "soft"):
            raise ValueError(f"invalid dependency kind: {kind}")
        source, target = str(value.get("source", "")), str(value.get("target", ""))
        if not source or not target or source == target:
            raise ValueError("dependency requires distinct source and target")
        return cls(source, target, kind, str(value.get("artifact", "result")))


@dataclass(frozen=True)
class CandidateNode:
    candidate_id: str
    parent_id: str
    epistemic_operation: str
    direction: str
    why_needed: str
    required_inputs: List[str]
    requested_tools: List[str]
    expected_output: str
    stop_condition: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], parent_id: str) -> "CandidateNode":
        candidate_id = str(value.get("candidate_id", "")).strip()
        direction = str(value.get("direction", "")).strip()
        why_needed = str(value.get("why_needed", "")).strip()
        expected_output = str(value.get("expected_output", "")).strip()
        stop_condition = str(value.get("stop_condition", "")).strip()
        operation = str(value.get("epistemic_operation", "targeted_gap")).strip()
        allowed_operations = {
            "independent_reconstruction", "specification_audit",
            "adversarial_falsification", "targeted_gap",
        }
        if operation not in allowed_operations:
            raise ValueError(f"invalid epistemic operation: {operation}")
        if not all((candidate_id, direction, why_needed, expected_output, stop_condition)):
            raise ValueError("candidate proposal is incomplete")
        return cls(
            candidate_id=candidate_id,
            parent_id=str(value.get("parent_id") or parent_id),
            epistemic_operation=operation,
            direction=direction,
            why_needed=why_needed,
            required_inputs=_strings(value.get("required_inputs")),
            requested_tools=_strings(value.get("requested_tools")),
            expected_output=expected_output,
            stop_condition=stop_condition,
        )


@dataclass
class CandidateGraph:
    parent_id: str
    nodes: Dict[str, CandidateNode]
    dependencies: List[CandidateDependency]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], parent_id: str) -> "CandidateGraph":
        nodes = {}
        for raw in list(value.get("nodes", [])):
            node = CandidateNode.from_dict(raw, parent_id)
            if node.candidate_id in nodes:
                raise ValueError(f"duplicate candidate id: {node.candidate_id}")
            nodes[node.candidate_id] = node
        dependencies = [CandidateDependency.from_dict(raw) for raw in value.get("dependencies", [])]
        for dependency in dependencies:
            if dependency.target not in nodes:
                raise ValueError(f"dependency target is not a candidate: {dependency.target}")
        graph = cls(parent_id, nodes, dependencies)
        graph.topological_order(set(nodes), hard_only=True)
        return graph

    def hard_closure(self, candidate_ids: Iterable[str], existing_ids: Iterable[str]) -> frozenset[str]:
        selected = set(candidate_ids)
        existing = set(existing_ids)
        unknown = selected - self.nodes.keys()
        if unknown:
            raise ValueError(f"unknown candidate ids: {sorted(unknown)}")
        changed = True
        while changed:
            changed = False
            for edge in self.dependencies:
                if edge.kind == "hard" and edge.target in selected:
                    if edge.source in self.nodes and edge.source not in selected:
                        selected.add(edge.source)
                        changed = True
                    elif edge.source not in self.nodes and edge.source not in existing:
                        raise ValueError(f"missing hard dependency source: {edge.source}")
        return frozenset(selected)

    def topological_order(self, selected: set[str], hard_only: bool = True) -> List[str]:
        incoming = {node_id: 0 for node_id in selected}
        outgoing = {node_id: [] for node_id in selected}
        for edge in self.dependencies:
            if hard_only and edge.kind != "hard":
                continue
            if edge.source in selected and edge.target in selected:
                incoming[edge.target] += 1
                outgoing[edge.source].append(edge.target)
        ready = sorted(node_id for node_id, count in incoming.items() if count == 0)
        result: List[str] = []
        while ready:
            node_id = ready.pop(0)
            result.append(node_id)
            for target in outgoing[node_id]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(result) != len(selected):
            raise ValueError("hard candidate dependencies contain a cycle")
        return result

    def subset(self, selected: Iterable[str]) -> Dict[str, Any]:
        selected_set = set(selected)
        return {
            "parent_id": self.parent_id,
            "nodes": [asdict(self.nodes[node_id]) for node_id in sorted(selected_set)],
            "dependencies": [
                asdict(edge) for edge in self.dependencies
                if edge.target in selected_set and (edge.source in selected_set or edge.source not in self.nodes)
            ],
        }


@dataclass
class RealizedSubgraph:
    subgraph_id: str
    candidate_ids: List[str]
    agents: Dict[str, AgentState]
    dependencies: List[CandidateDependency]
    configuration_reasoning_summary: str
    risks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgraph_id": self.subgraph_id,
            "candidate_ids": self.candidate_ids,
            "agents": {key: value.to_dict() for key, value in self.agents.items()},
            "dependencies": [asdict(edge) for edge in self.dependencies],
            "configuration_reasoning_summary": self.configuration_reasoning_summary,
            "risks": self.risks,
        }


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    benchmark: str
    instruction: str
    public_tests: List[str]
    evaluator_payload: Dict[str, Any]


def _strings(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("expected a list of strings")
    return [str(item) for item in value]


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
