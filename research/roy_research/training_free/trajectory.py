from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Mapping

from .matrix import InformationMatrix
from .types import AgentState, CandidateDependency

DependencyRelation = Literal["derivation", "hard", "soft"]
DependencyStatus = Literal["active", "satisfied", "dormant", "invalidated"]


@dataclass
class DependencyRecord:
    dependency_id: str
    source: str
    target: str
    relation: DependencyRelation
    artifact: str
    introduced_round: int
    status: DependencyStatus
    subgraph_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryEvent:
    event_id: str
    round_index: int
    kind: str
    details: Dict[str, Any]
    scope: Literal["committed", "counterfactual", "audit"] = "committed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StateCheckpoint:
    checkpoint_id: str
    parent_checkpoint_id: str | None
    round_index: int
    phase: Literal["initial", "search_state", "committed", "terminal"]
    agents: Dict[str, Dict[str, Any]]
    matrix: Dict[str, Any]
    dependencies: List[Dict[str, Any]]
    information_state: Dict[str, Any]
    history_event_count: int
    history_fingerprint: str
    fingerprint: str

    @classmethod
    def capture(
        cls,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        round_index: int,
        phase: Literal["initial", "search_state", "committed", "terminal"],
        agents: Mapping[str, AgentState],
        matrix: InformationMatrix,
        dependencies: List[DependencyRecord],
        information_state: Mapping[str, Any] | None,
        history_events: List[TrajectoryEvent],
    ) -> "StateCheckpoint":
        agent_value = {key: agent.to_dict() for key, agent in agents.items()}
        matrix_value = matrix.to_dict()
        dependency_value = [dependency.to_dict() for dependency in dependencies]
        information_state_value = dict(information_state or {})
        history_value = [event.to_dict() for event in history_events]
        history_fingerprint = hashlib.sha256(
            json.dumps(history_value, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        fingerprint_payload = {
            "agents": agent_value,
            "matrix": matrix_value,
            "dependencies": dependency_value,
            "information_state": information_state_value,
            "history_fingerprint": history_fingerprint,
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return cls(
            checkpoint_id=checkpoint_id,
            parent_checkpoint_id=parent_checkpoint_id,
            round_index=round_index,
            phase=phase,
            agents=agent_value,
            matrix=matrix_value,
            dependencies=dependency_value,
            information_state=information_state_value,
            history_event_count=len(history_events),
            history_fingerprint=history_fingerprint,
            fingerprint=fingerprint,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TopologyDrift:
    agent_ids: List[str]
    delta_matrix: List[List[float]]
    agent_expansion: int
    edge_l1: float
    edge_frobenius: float
    added_edges: int
    removed_edges: int
    changed_edges: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def topology_drift(previous: InformationMatrix, current: InformationMatrix) -> TopologyDrift:
    agent_ids = list(previous.agent_ids)
    agent_ids.extend(agent_id for agent_id in current.agent_ids if agent_id not in agent_ids)
    delta: List[List[float]] = []
    added = removed = changed = 0
    squared = 0.0
    absolute = 0.0
    for source in agent_ids:
        row: List[float] = []
        for target in agent_ids:
            old = _weight_or_zero(previous, source, target)
            new = _weight_or_zero(current, source, target)
            difference = new - old
            row.append(difference)
            absolute += abs(difference)
            squared += difference * difference
            if abs(difference) > 1e-9:
                changed += 1
            if old <= 1e-9 < new:
                added += 1
            elif new <= 1e-9 < old:
                removed += 1
        delta.append(row)
    return TopologyDrift(
        agent_ids=agent_ids,
        delta_matrix=delta,
        agent_expansion=len(current.agent_ids) - len(previous.agent_ids),
        edge_l1=absolute,
        edge_frobenius=squared ** 0.5,
        added_edges=added,
        removed_edges=removed,
        changed_edges=changed,
    )


def committed_dependency_records(
    round_index: int,
    subgraph_id: str,
    agents: Mapping[str, AgentState],
    dependencies: List[CandidateDependency],
) -> List[DependencyRecord]:
    records = [
        DependencyRecord(
            dependency_id=f"d-{round_index}-{agent_id}-parent",
            source=str(agent.parent_id),
            target=agent_id,
            relation="derivation",
            artifact="agent_derivation",
            introduced_round=round_index,
            status="active",
            subgraph_id=subgraph_id,
        )
        for agent_id, agent in agents.items()
    ]
    for index, edge in enumerate(dependencies):
        records.append(DependencyRecord(
            dependency_id=f"d-{round_index}-{index}-{edge.source}-{edge.target}",
            source=edge.source,
            target=edge.target,
            relation=edge.kind,
            artifact=edge.artifact,
            introduced_round=round_index,
            status="satisfied" if edge.kind == "hard" else "active",
            subgraph_id=subgraph_id,
        ))
    return records


def _weight_or_zero(matrix: InformationMatrix, source: str, target: str) -> float:
    if source not in matrix.agent_ids or target not in matrix.agent_ids:
        return 0.0
    return matrix.weight(source, target)
