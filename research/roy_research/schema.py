from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

SCHEMA_VERSION = 1
ACTIONS = ("CONTINUE", "BRANCH", "RETURN")
Action = Literal["CONTINUE", "BRANCH", "RETURN"]


@dataclass(frozen=True)
class ResourceEnvelope:
    compute_tokens: Optional[int] = None
    wall_clock_ms: Optional[int] = None
    parallel_slots: Optional[int] = None
    communication_edges: Optional[int] = None
    tool_calls: Optional[int] = None

    def vector(self) -> List[float]:
        values = (
            self.compute_tokens,
            self.wall_clock_ms,
            self.parallel_slots,
            self.communication_edges,
            self.tool_calls,
        )
        return [0.0 if value is None else float(value) for value in values]


@dataclass(frozen=True)
class EventNode:
    id: str
    kind: str
    timestamp: int
    text: str = ""
    actor_id: Optional[str] = None
    status: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventEdge:
    id: str
    kind: str
    source: str
    target: str
    required: bool = False
    active: bool = True


@dataclass(frozen=True)
class EventGraph:
    parent_id: str
    nodes: List[EventNode]
    edges: List[EventEdge]
    observed_at: int


@dataclass(frozen=True)
class ChildSpecification:
    id: str
    task: str
    context: List[str]
    tools: List[str]
    resources: ResourceEnvelope
    output_contract: Dict[str, Any]
    dependencies: List[str]


@dataclass(frozen=True)
class TraceRecord:
    task_id: str
    checkpoint_id: str
    checkpoint_fingerprint: str
    parent_id: str
    event_graph: Dict[str, Any]
    legal_actions: List[Action]
    action: Action
    resources_before: Dict[str, Any]
    resources_after: Dict[str, Any]
    utility: float
    provider: str
    model: str
    token_usage: int
    latency_ms: int
    repeat: int
    environment_revision: str
    child_specification: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def require_schema(record: Dict[str, Any]) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported trace schema: {record.get('schema_version')}")
    if record.get("action") not in ACTIONS:
        raise ValueError(f"Unknown structural action: {record.get('action')}")
    utility = record.get("utility")
    if not isinstance(utility, (int, float)) or not 0 <= utility <= 1:
        raise ValueError("Trace utility must be in [0, 1]")
