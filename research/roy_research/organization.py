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
        if self.minimum_nodes < 0 or self.maximum_nodes < max(1, self.minimum_nodes):
            raise ValueError("invalid exploration node envelope")
        if self.minimum_depth < 0 or self.maximum_depth < self.minimum_depth:
            raise ValueError("invalid exploration depth envelope")
        if self.mode not in {"shallow", "medium", "deep", "expansive"}:
            raise ValueError(f"unknown exploration mode: {self.mode}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ORGANIZATION_GROUP_SIZE = 8


@dataclass(frozen=True)
class RuntimeBudget:
    """Hard limits on observable runtime events, never reward components."""

    maximum_llm_calls: int = 32
    maximum_tool_calls: int = 8
    maximum_nodes: int = 12
    maximum_depth: int = 5
    maximum_decisions: int = 64

    def __post_init__(self) -> None:
        if min(
            self.maximum_llm_calls,
            self.maximum_nodes,
            self.maximum_decisions,
        ) < 1:
            raise ValueError("llm-call, node, and decision budgets must be positive")
        if self.maximum_tool_calls < 0 or self.maximum_depth < 0:
            raise ValueError("tool-call and depth budgets must be non-negative")

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def training_envelope(epoch: int, epochs: int) -> ExplorationEnvelope:
    """Return one shared group envelope with annealed, conditional floors."""

    if epochs < 1 or epoch < 0 or epoch >= epochs:
        raise ValueError("epoch must be within the configured training range")
    fraction = epoch / max(1, epochs - 1)
    minimum_nodes = round(6 * (1.0 - fraction))
    minimum_depth = round(3 * (1.0 - fraction))
    return ExplorationEnvelope(
        id=f"epoch-{epoch}",
        minimum_nodes=minimum_nodes,
        maximum_nodes=12,
        minimum_depth=minimum_depth,
        maximum_depth=5,
        mode="expansive",
    )


def validate_exploration_group(envelopes: Sequence[ExplorationEnvelope]) -> None:
    if len(envelopes) != ORGANIZATION_GROUP_SIZE:
        raise ValueError("organization GRPO groups require exactly eight exploration trajectories")
    if any(envelope != envelopes[0] for envelope in envelopes[1:]):
        raise ValueError("all counterfactual trajectories must share one exploration envelope")


def envelope_legal_actions(
    candidates: Sequence[Mapping[str, Any]],
    envelope: ExplorationEnvelope,
    node_count: int,
    maximum_depth_reached: int,
    unresolved_gap_exists: bool,
) -> List[bool]:
    """Apply training exploration structure without introducing a reward term.

    Floors delay STOP only while a genuine model-reported residual requirement
    exists. They never synthesize a gap or a child proposal. Ceilings are hard
    runtime feasibility bounds, not utility components.
    """

    floors_reached = (
        node_count >= envelope.minimum_nodes
        and maximum_depth_reached >= envelope.minimum_depth
    )
    terminal_allowed = not unresolved_gap_exists or floors_reached
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
