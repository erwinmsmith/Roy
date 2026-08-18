from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import EDGE_KINDS, NODE_KINDS, RelationalMessagePassing, TEXT_DIMENSION
from .organization import ORGANIZATION_ACTIONS


class InformationRealizationPolicy(nn.Module):
    """Joint active-node and open organization-candidate policy.

    Agent specifications are not selected from predefined roles. An executing
    node emits residual requirements and open child proposals as part of its
    epistemic report. This policy embeds those current proposals and chooses
    among them jointly with the fixed organization grammar.
    """

    def __init__(
        self,
        text_dim: int = TEXT_DIMENSION,
        hidden_dim: int = 256,
        node_type_dim: int = 32,
        action_type_dim: int = 32,
        resource_dim: int = 5,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.node_types = nn.Embedding(len(NODE_KINDS), node_type_dim)
        self.action_types = nn.Embedding(len(ORGANIZATION_ACTIONS), action_type_dim)
        self.input_projection = nn.Linear(text_dim + node_type_dim + 3, hidden_dim)
        self.layers = nn.ModuleList(
            RelationalMessagePassing(hidden_dim, len(EDGE_KINDS)) for _ in range(layers)
        )
        self.resource_projection = nn.Sequential(
            nn.Linear(resource_dim, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.active_node_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + text_dim + action_type_dim + 64 + 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def encode_graph(
        self,
        text_embeddings: Tensor,
        node_types: Tensor,
        scalar_features: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        states = torch.cat(
            [text_embeddings, self.node_types(node_types.long()), scalar_features], dim=-1
        )
        states = F.gelu(self.input_projection(states))
        for layer in self.layers:
            states = layer(states, edge_index, edge_types)
        pooled = states.mean(dim=0) if states.shape[0] else torch.zeros(
            self.input_projection.out_features,
            device=states.device,
            dtype=states.dtype,
        )
        return states, pooled

    def active_node_logits(
        self,
        node_states: Tensor,
        graph_state: Tensor,
        resources: Tensor,
        active_mask: Tensor,
    ) -> Tensor:
        if node_states.shape[0] == 0:
            raise ValueError("organization policy requires at least one active-node candidate")
        resource_state = self.resource_projection(resources.float())
        repeated_graph = graph_state.unsqueeze(0).expand(node_states.shape[0], -1)
        repeated_resources = resource_state.unsqueeze(0).expand(node_states.shape[0], -1)
        logits = self.active_node_head(
            torch.cat([node_states, repeated_graph, repeated_resources], dim=-1)
        ).squeeze(-1)
        return _masked_logits(logits, active_mask)

    def candidate_logits(
        self,
        graph_state: Tensor,
        active_node_state: Tensor,
        candidate_embeddings: Tensor,
        candidate_action_types: Tensor,
        candidate_features: Tensor,
        resources: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        count = candidate_embeddings.shape[0]
        if count == 0:
            raise ValueError("organization policy requires at least one action candidate")
        resource_state = self.resource_projection(resources.float())
        values = torch.cat(
            [
                graph_state.unsqueeze(0).expand(count, -1),
                active_node_state.unsqueeze(0).expand(count, -1),
                candidate_embeddings,
                self.action_types(candidate_action_types.long()),
                resource_state.unsqueeze(0).expand(count, -1),
                candidate_features.float(),
            ],
            dim=-1,
        )
        return _masked_logits(self.candidate_head(values).squeeze(-1), legal_mask)

    def save_checkpoint(self, path: Path, metadata: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load_checkpoint(
        cls, path: Path, map_location: str = "cpu"
    ) -> Tuple["InformationRealizationPolicy", Dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls()
        model.load_state_dict(payload["state_dict"])
        return model, dict(payload.get("metadata", {}))


def expected_resource_log_probs(
    logits: Tensor,
    legal_mask: Tensor,
    expected_costs: Tensor,
    expected_budget: float,
    tolerance: float = 1e-6,
    iterations: int = 80,
) -> Tensor:
    """KL-project an action distribution onto an expected-cost constraint.

    Resource use remains a feasibility constraint. It is not added to task
    utility, the GRPO advantage, or the policy loss. Individual sampled
    trajectories may exceed the expectation; only the action distribution is
    constrained.
    """

    if logits.shape != legal_mask.shape or logits.shape != expected_costs.shape:
        raise ValueError("resource projection tensors must have identical shapes")
    if not bool(legal_mask.any()):
        raise ValueError("resource projection requires a legal action")
    legal_costs = expected_costs[legal_mask.bool()].float()
    if expected_budget + tolerance < float(legal_costs.min().item()):
        raise ValueError("expected resource constraint is infeasible for the legal action set")

    base_logits = _masked_logits(logits, legal_mask)
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    base_expected = float((base_log_probs.exp() * expected_costs).sum().item())
    if base_expected <= expected_budget + tolerance:
        return base_log_probs

    lower = 0.0
    upper = 1.0
    for _ in range(60):
        projected = torch.log_softmax(
            _masked_logits(logits - upper * expected_costs, legal_mask), dim=-1
        )
        if float((projected.exp() * expected_costs).sum().item()) <= expected_budget:
            break
        upper *= 2.0
    else:
        raise ValueError("unable to satisfy expected resource constraint")

    for _ in range(iterations):
        multiplier = (lower + upper) / 2.0
        projected = torch.log_softmax(
            _masked_logits(logits - multiplier * expected_costs, legal_mask), dim=-1
        )
        expected = float((projected.exp() * expected_costs).sum().item())
        if expected > expected_budget:
            lower = multiplier
        else:
            upper = multiplier
    return torch.log_softmax(
        _masked_logits(logits - upper * expected_costs, legal_mask), dim=-1
    )


def mixed_behavior_log_probs(
    policy_log_probs: Tensor,
    exploration_log_probs: Tensor,
    policy_weight: float,
) -> Tensor:
    """Exact log-probability of the design's exploration/policy mixture."""

    if policy_log_probs.shape != exploration_log_probs.shape:
        raise ValueError("mixture distributions must have identical shapes")
    if not 0.0 <= policy_weight <= 1.0:
        raise ValueError("policy_weight must be in [0, 1]")
    if policy_weight == 0.0:
        return exploration_log_probs
    if policy_weight == 1.0:
        return policy_log_probs
    weights = torch.tensor(
        [math.log(policy_weight), math.log1p(-policy_weight)],
        dtype=policy_log_probs.dtype,
        device=policy_log_probs.device,
    )
    return torch.logsumexp(
        torch.stack([policy_log_probs + weights[0], exploration_log_probs + weights[1]]),
        dim=0,
    )


def uniform_exploration_log_probs(legal_mask: Tensor) -> Tensor:
    logits = torch.zeros(legal_mask.shape, dtype=torch.float32, device=legal_mask.device)
    return torch.log_softmax(_masked_logits(logits, legal_mask), dim=-1)


def action_type_indices(kinds: Sequence[str], device: torch.device | None = None) -> Tensor:
    unknown = [kind for kind in kinds if kind not in ORGANIZATION_ACTIONS]
    if unknown:
        raise ValueError(f"unknown organization actions: {unknown}")
    return torch.tensor(
        [ORGANIZATION_ACTIONS.index(kind) for kind in kinds], dtype=torch.long, device=device
    )


def _masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError("logits and legal mask must have identical shapes")
    if not bool(mask.any()):
        raise ValueError("every organization decision requires a legal option")
    return logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
