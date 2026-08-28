from __future__ import annotations

from copy import deepcopy
from typing import List, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import EDGE_KINDS, NODE_KINDS, RelationalMessagePassing, TEXT_DIMENSION


LHTB_VALUE_MODEL_REVISION = "relational-attention-mean-max-v2"


class EpistemicValueModel(nn.Module):
    """Independent relational critic; only the frozen text encoder is shared."""

    def __init__(self, text_dim: int = TEXT_DIMENSION, hidden_dim: int = 256,
                 node_type_dim: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.node_types = nn.Embedding(len(NODE_KINDS), node_type_dim)
        self.input_projection = nn.Linear(text_dim + node_type_dim + 3, hidden_dim)
        self.layers = nn.ModuleList(
            RelationalMessagePassing(hidden_dim, len(EDGE_KINDS)) for _ in range(layers)
        )
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(self, text: Tensor, kinds: Tensor, scalars: Tensor,
                edges: Tensor, edge_types: Tensor) -> Tensor:
        return self.forward_batch([(text, kinds, scalars, edges, edge_types)])[0]

    def forward_batch(
        self,
        graphs: Sequence[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    ) -> Tensor:
        """Evaluate independent variable-size graphs in one disjoint GNN pass."""
        if not graphs:
            return torch.zeros(0, dtype=torch.float32,
                               device=self.value_head[-1].weight.device)
        counts = [int(graph[0].shape[0]) for graph in graphs]
        if any(count <= 0 for count in counts):
            return torch.stack([self._forward_single(*graph) for graph in graphs])
        text = torch.cat([graph[0] for graph in graphs], dim=0)
        kinds = torch.cat([graph[1] for graph in graphs], dim=0)
        scalars = torch.cat([graph[2] for graph in graphs], dim=0)
        edge_parts = []
        edge_type_parts = []
        offset = 0
        for graph, count in zip(graphs, counts):
            if graph[3].numel():
                edge_parts.append(graph[3] + offset)
                edge_type_parts.append(graph[4])
            offset += count
        edges = torch.cat(edge_parts, dim=1) if edge_parts else torch.zeros(
            (2, 0), dtype=torch.long, device=text.device
        )
        edge_types = torch.cat(edge_type_parts) if edge_type_parts else torch.zeros(
            0, dtype=torch.long, device=text.device
        )
        states = F.gelu(self.input_projection(torch.cat(
            [text, self.node_types(kinds.long()), scalars], dim=-1
        )))
        for layer in self.layers:
            states = layer(states, edges, edge_types)
        pooled = [self._pool(chunk) for chunk in states.split(counts)]
        return torch.sigmoid(self.value_head(torch.stack(pooled))).squeeze(-1)

    def _forward_single(self, text: Tensor, kinds: Tensor, scalars: Tensor,
                        edges: Tensor, edge_types: Tensor) -> Tensor:
        states = F.gelu(self.input_projection(torch.cat(
            [text, self.node_types(kinds.long()), scalars], dim=-1
        )))
        for layer in self.layers:
            states = layer(states, edges, edge_types)
        return torch.sigmoid(self.value_head(self._pool(states))).squeeze(-1)

    def _pool(self, states: Tensor) -> Tensor:
        if states.shape[0]:
            attention = torch.softmax(self.pool_gate(states).squeeze(-1), dim=0)
            attentive = torch.sum(states * attention.unsqueeze(-1), dim=0)
            return torch.cat([
                attentive, states.mean(dim=0), states.max(dim=0).values
            ])
        return torch.zeros(
            self.input_projection.out_features * 3,
            dtype=self.value_head[-1].weight.dtype,
            device=self.value_head[-1].weight.device,
        )


def make_ema_target(model: EpistemicValueModel) -> EpistemicValueModel:
    target = deepcopy(model).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target


def constant_value_output(model: EpistemicValueModel) -> float | None:
    """Return the exact scalar output when the final value layer ignores its input."""
    final = model.value_head[-1]
    weight = getattr(final, "weight", None)
    bias = getattr(final, "bias", None)
    if weight is None or bias is None or bias.numel() != 1:
        return None
    if int(torch.count_nonzero(weight.detach()).item()) != 0:
        return None
    return float(torch.sigmoid(bias.detach()).item())


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, decay: float = 0.99) -> None:
    for target_value, source_value in zip(target.parameters(), source.parameters()):
        target_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
    for target_value, source_value in zip(target.buffers(), source.buffers()):
        target_value.copy_(source_value)


def process_credit(target_values: Sequence[Sequence[float]], rewards: Sequence[float]) -> Tuple[List[List[float]], List[List[float]]]:
    """Compute ΔV rewards and their telescoping return-to-go."""
    if len(target_values) != len(rewards):
        raise ValueError("target values and rewards must have equal group size")
    process_rewards: List[List[float]] = []
    returns: List[List[float]] = []
    for values, reward in zip(target_values, rewards):
        if len(values) < 2:
            raise ValueError("a trajectory needs M_0 and at least one successor state")
        step_rewards = [float(values[index + 1] - values[index])
                        for index in range(len(values) - 2)]
        step_rewards.append(float(reward - values[-2]))
        process_rewards.append(step_rewards)
        returns.append([float(reward - values[index]) for index in range(len(values) - 1)])
        if abs(sum(step_rewards) - (float(reward) - float(values[0]))) > 1e-6:
            raise ValueError("process rewards do not telescope")
    return process_rewards, returns


def trajectory_weighted_advantages(
    returns: Sequence[Sequence[float]], epsilon: float = 1e-8
) -> Tuple[List[Tensor], float, float]:
    if not returns or any(not values for values in returns):
        raise ValueError("advantages require non-empty trajectories")
    flat = torch.tensor([value for values in returns for value in values], dtype=torch.float64)
    weights = torch.tensor([1.0 / len(values) for values in returns for _ in values], dtype=torch.float64)
    weights /= weights.sum()
    mean = float((flat * weights).sum())
    variance = float((((flat - mean) ** 2) * weights).sum())
    deviation = variance ** 0.5
    advantages: List[Tensor] = []
    for values in returns:
        if deviation <= epsilon:
            advantages.append(torch.zeros(len(values), dtype=torch.float32))
        else:
            advantages.append(torch.tensor(
                [(value - mean) / (deviation + epsilon) for value in values], dtype=torch.float32
            ))
    return advantages, mean, deviation


def equal_trajectory_value_loss(
    predictions: Sequence[Tensor], rewards: Sequence[float]
) -> Tensor:
    if len(predictions) != len(rewards) or not predictions:
        raise ValueError("value predictions and rewards must have equal non-zero length")
    losses = []
    for values, reward in zip(predictions, rewards):
        if values.numel() == 0:
            raise ValueError("each trajectory needs value states")
        target = torch.full_like(values, float(reward))
        losses.append(F.huber_loss(values, target, reduction="mean"))
    return torch.stack(losses).mean()
