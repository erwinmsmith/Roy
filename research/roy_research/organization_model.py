from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import EDGE_KINDS, NODE_KINDS, RelationalMessagePassing, TEXT_DIMENSION
from .organization import ORGANIZATION_ACTIONS


LHTB_ACTOR_MODEL_REVISION = "scheduler-context-relational-attention-policy-20260829"


class InformationRealizationPolicy(nn.Module):
    """Structural action policy for a scheduler-selected context node.

    The Runtime scheduler supplies the node that currently owns execution.  It
    is observed context, never a learned routing action.  The policy chooses an
    outer organization action and, conditionally, its open child specification
    or connection payload from the candidates generated for that one node.
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
        self.graph_attention = nn.Linear(hidden_dim, 1)
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.resource_projection = nn.Sequential(
            nn.Linear(resource_dim, 64), nn.GELU(), nn.LayerNorm(64)
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
        pooled = self._pool_graph(states)
        return states, pooled

    def _pool_graph(self, states: Tensor) -> Tensor:
        if not states.shape[0]:
            return torch.zeros(
                self.input_projection.out_features,
                device=states.device,
                dtype=states.dtype,
            )
        attention = torch.softmax(self.graph_attention(states).squeeze(-1), dim=0)
        attended = (states * attention.unsqueeze(-1)).sum(dim=0)
        return self.graph_projection(torch.cat(
            [attended, states.mean(dim=0), states.max(dim=0).values], dim=-1
        ))

    def encode_graph_batch(
        self,
        graphs: Sequence[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    ) -> Sequence[Tuple[Tensor, Tensor]]:
        """Encode independent variable-size graphs as one disjoint graph."""
        if not graphs:
            return []
        counts = [int(graph[0].shape[0]) for graph in graphs]
        if any(count <= 0 for count in counts):
            return [self.encode_graph(*graph) for graph in graphs]
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
        chunks = states.split(counts)
        return [(chunk, self._pool_graph(chunk)) for chunk in chunks]

    def candidate_logits(
        self,
        graph_state: Tensor,
        context_node_state: Tensor,
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
                context_node_state.unsqueeze(0).expand(count, -1),
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
