from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ORGANIZATION_ACTIONS = (
    "DERIVE",
    "ACQUIRE",
    "CONNECT",
    "EXECUTE",
    "RETURN",
    "PRUNE",
    "STOP",
)


@dataclass(frozen=True)
class ExplorationEnvelope:
    id: str
    minimum_nodes: int
    maximum_nodes: int
    minimum_depth: int
    maximum_depth: int
    mode: str

    def __post_init__(self) -> None:
        if self.minimum_nodes < 1 or self.maximum_nodes < self.minimum_nodes:
            raise ValueError("invalid exploration node envelope")
        if self.minimum_depth < 0 or self.maximum_depth < self.minimum_depth:
            raise ValueError("invalid exploration depth envelope")
        if self.mode not in {"shallow", "medium", "deep", "expansive"}:
            raise ValueError(f"unknown exploration mode: {self.mode}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_EXPLORATION_GROUP: tuple[ExplorationEnvelope, ...] = (
    ExplorationEnvelope("shallow-1", 1, 4, 1, 2, "shallow"),
    ExplorationEnvelope("shallow-2", 2, 5, 1, 3, "shallow"),
    ExplorationEnvelope("medium-1", 4, 7, 2, 3, "medium"),
    ExplorationEnvelope("medium-2", 4, 7, 2, 4, "medium"),
    ExplorationEnvelope("deep-1", 6, 9, 3, 4, "deep"),
    ExplorationEnvelope("deep-2", 6, 9, 3, 4, "deep"),
    ExplorationEnvelope("expansive-1", 6, 12, 3, 5, "expansive"),
    ExplorationEnvelope("expansive-2", 6, 12, 4, 5, "expansive"),
)


def validate_exploration_group(envelopes: Sequence[ExplorationEnvelope]) -> None:
    if len(envelopes) != 8:
        raise ValueError("organization GRPO groups require exactly eight exploration trajectories")
    identifiers = {envelope.id for envelope in envelopes}
    if len(identifiers) != len(envelopes):
        raise ValueError("exploration envelope ids must be unique")


def envelope_legal_actions(
    candidates: Sequence[Mapping[str, Any]],
    envelope: ExplorationEnvelope,
    node_count: int,
    maximum_depth_reached: int,
) -> List[bool]:
    """Apply training exploration structure without introducing a reward term.

    Minimum node/depth requirements only affect whether terminal actions are
    available during exploration. Maximum values prevent the sampler from
    proposing further derivation; they are organization-search bounds, not
    resource-budget constraints or utility components.
    """

    terminal_allowed = (
        node_count >= envelope.minimum_nodes
        and maximum_depth_reached >= envelope.minimum_depth
    )
    derivation_allowed = node_count < envelope.maximum_nodes
    result: List[bool] = []
    for candidate in candidates:
        legal = bool(candidate.get("legal", True))
        kind = str(candidate.get("kind"))
        if kind == "STOP" and not terminal_allowed:
            legal = False
        if kind == "DERIVE":
            resulting_depth = int(candidate.get("resulting_depth", maximum_depth_reached + 1))
            if not derivation_allowed or resulting_depth > envelope.maximum_depth:
                legal = False
        result.append(legal)
    if not any(result):
        raise ValueError("exploration envelope removed every organization action")
    return result


def require_single_terminal_utility(trajectory: Mapping[str, Any]) -> float:
    """Read the only optimization signal accepted by organization training."""

    forbidden = {
        "cost_reward",
        "node_reward",
        "communication_reward",
        "rationality_reward",
        "redundancy_reward",
        "teacher_reward",
        "auxiliary_reward",
    }
    present = sorted(key for key in forbidden if key in trajectory)
    if present:
        raise ValueError(f"organization trajectory contains forbidden reward components: {present}")
    if "terminal_utility" not in trajectory:
        raise ValueError("organization trajectory is missing terminal_utility")
    return float(trajectory["terminal_utility"])


def group_trajectories(records: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        group_id = str(record.get("group_id") or "")
        if not group_id:
            raise ValueError("organization trajectory is missing group_id")
        groups.setdefault(group_id, []).append(record)
    return groups
