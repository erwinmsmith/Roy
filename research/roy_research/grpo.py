from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


def standardized_advantages(values: Sequence[float], epsilon: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    deviation = float(array.std())
    if deviation <= epsilon:
        return np.zeros_like(array)
    return (array - array.mean()) / (deviation + epsilon)


@dataclass(frozen=True)
class HierarchicalAdvantages:
    action_values: Dict[str, float]
    outer_advantages: Dict[str, float]
    branch_values: Dict[str, float]
    branch_advantages: Dict[str, float]


def hierarchical_advantages(
    non_branch_returns: Mapping[str, Iterable[float]],
    branch_returns: Mapping[str, Iterable[float]],
) -> HierarchicalAdvantages:
    branch_values = _non_empty_means(branch_returns)
    action_values = _non_empty_means(non_branch_returns)
    if branch_values:
        action_values["BRANCH"] = float(np.mean(list(branch_values.values())))
    action_keys = list(action_values)
    outer = standardized_advantages([action_values[key] for key in action_keys])
    branch_keys = list(branch_values)
    inner = standardized_advantages([branch_values[key] for key in branch_keys])
    return HierarchicalAdvantages(
        action_values=action_values,
        outer_advantages=dict(zip(action_keys, outer.tolist())),
        branch_values=branch_values,
        branch_advantages=dict(zip(branch_keys, inner.tolist())),
    )


def _non_empty_means(values_by_key: Mapping[str, Iterable[float]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, values in values_by_key.items():
        materialized = list(values)
        if materialized:
            result[key] = float(np.mean(materialized))
    return result


def masked_log_softmax(logits: Tensor, legal_mask: Tensor) -> Tensor:
    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have the same shape")
    if not torch.all(legal_mask.any(dim=-1)):
        raise ValueError("every policy row must contain a legal action")
    masked = logits.masked_fill(~legal_mask.bool(), torch.finfo(logits.dtype).min)
    return torch.log_softmax(masked, dim=-1)


def clipped_policy_loss(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantage: Tensor,
    clip_epsilon: float = 0.2,
) -> Tensor:
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -torch.minimum(ratio * advantage, clipped * advantage).mean()


def select_action_log_probs(logits: Tensor, mask: Tensor, actions: Tensor) -> Tensor:
    return masked_log_softmax(logits, mask).gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
