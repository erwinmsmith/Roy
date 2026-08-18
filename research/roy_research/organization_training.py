from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .grpo import standardized_advantages
from .organization import group_trajectories, require_single_terminal_utility


@dataclass(frozen=True)
class TrajectoryAdvantage:
    trajectory_id: str
    group_id: str
    terminal_utility: float
    advantage: float


def organization_group_advantages(
    records: Iterable[Mapping[str, object]],
) -> List[TrajectoryAdvantage]:
    """Compute advantages only from the single terminal task utility."""

    result: List[TrajectoryAdvantage] = []
    for group_id, trajectories in group_trajectories(records).items():
        utilities = [require_single_terminal_utility(value) for value in trajectories]
        advantages = standardized_advantages(utilities)
        for trajectory, utility, advantage in zip(trajectories, utilities, advantages.tolist()):
            result.append(
                TrajectoryAdvantage(
                    trajectory_id=str(trajectory.get("id") or ""),
                    group_id=group_id,
                    terminal_utility=utility,
                    advantage=float(advantage),
                )
            )
    return result


def single_objective_organization_grpo_loss(
    new_log_probabilities: Sequence[Tensor],
    reference_log_probabilities: Sequence[Tensor],
    trajectory_advantages: Sequence[float],
    clip_epsilon: float = 0.2,
) -> Tensor:
    """Clipped on-policy objective with no teacher or auxiliary objective.

    Each sequence entry contains the joint active-node/action log-probabilities
    for one complete trajectory. Length normalization prevents deeper sampled
    organizations from receiving a mechanically larger gradient solely because
    they contain more organization decisions.
    """

    if not (
        len(new_log_probabilities)
        == len(reference_log_probabilities)
        == len(trajectory_advantages)
    ):
        raise ValueError("trajectory policy inputs must have the same length")
    if not new_log_probabilities:
        raise ValueError("organization GRPO requires at least one trajectory")

    objectives: List[Tensor] = []
    for current, behavior, advantage in zip(
        new_log_probabilities, reference_log_probabilities, trajectory_advantages
    ):
        if current.shape != behavior.shape or current.numel() == 0:
            raise ValueError("each trajectory requires matched non-empty step log-probabilities")
        ratio = torch.exp(current - behavior.detach())
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        value = torch.tensor(float(advantage), dtype=current.dtype, device=current.device)
        objectives.append(torch.minimum(ratio * value, clipped * value).mean())
    return -torch.stack(objectives).mean()


def mean_realized_resource(records: Iterable[Mapping[str, object]], key: str) -> float:
    """Report resource expectation without converting it into a reward."""

    values = []
    for record in records:
        resources = record.get("realized_resources", record.get("resources", {}))
        if isinstance(resources, Mapping):
            values.append(float(resources.get(key, 0.0)))
    return float(np.mean(values)) if values else 0.0
