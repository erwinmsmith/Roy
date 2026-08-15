from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
TEXT_DIMENSION = 384

NODE_KINDS = (
    "agent", "message", "subtask", "tool_call", "tool_result",
    "child_result", "dependency", "artifact", "resource",
)
EDGE_KINDS = (
    "temporal", "derivation", "dependency", "communication",
    "tool_use", "return", "produces", "consumes",
)


class FrozenTextEncoder:
    def __init__(self, device: str | None = None, local_only: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        resolved = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SentenceTransformer(
            MINILM_MODEL_ID,
            revision=MINILM_REVISION,
            device=resolved,
            local_files_only=local_only,
        )

    def encode(self, texts: Sequence[str]) -> Tensor:
        if not texts:
            return torch.zeros((0, TEXT_DIMENSION), dtype=torch.float32)
        values = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return torch.from_numpy(np.asarray(values, dtype=np.float32))


class RelationalMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, relation_count: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.relation_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(relation_count)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, states: Tensor, edge_index: Tensor, edge_types: Tensor) -> Tensor:
        aggregate = torch.zeros_like(states)
        degree = torch.ones((states.shape[0], 1), device=states.device, dtype=states.dtype)
        if edge_index.numel() > 0:
            sources, targets = edge_index.long()
            for relation, projection in enumerate(self.relation_projections):
                selected = edge_types.long() == relation
                if not selected.any():
                    continue
                messages = projection(states[sources[selected]])
                aggregate.index_add_(0, targets[selected], messages)
                degree.index_add_(0, targets[selected], torch.ones(
                    (int(selected.sum().item()), 1), device=states.device, dtype=states.dtype
                ))
        return self.norm(torch.relu(self.self_projection(states) + aggregate / degree))


class StructuralPolicyNetwork(nn.Module):
    def __init__(
        self,
        text_dim: int = TEXT_DIMENSION,
        hidden_dim: int = 128,
        node_type_dim: int = 24,
        resource_dim: int = 5,
    ) -> None:
        super().__init__()
        self.node_types = nn.Embedding(len(NODE_KINDS), node_type_dim)
        self.input_projection = nn.Linear(text_dim + node_type_dim + 3, hidden_dim)
        self.layers = nn.ModuleList([
            RelationalMessagePassing(hidden_dim, len(EDGE_KINDS)),
            RelationalMessagePassing(hidden_dim, len(EDGE_KINDS)),
        ])
        self.resource_projection = nn.Sequential(
            nn.Linear(resource_dim, 32), nn.ReLU(), nn.LayerNorm(32)
        )
        self.node_head = nn.Sequential(nn.Linear(hidden_dim + 32, 64), nn.ReLU(), nn.Linear(64, 3))
        self.derive_head = nn.Sequential(
            nn.Linear(hidden_dim + text_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.communication_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def encode_graph(
        self,
        text_embeddings: Tensor,
        node_types: Tensor,
        scalar_features: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        states = torch.cat([
            text_embeddings,
            self.node_types(node_types.long()),
            scalar_features,
        ], dim=-1)
        states = torch.relu(self.input_projection(states))
        for layer in self.layers:
            states = layer(states, edge_index, edge_types)
        pooled = states.mean(dim=0) if states.shape[0] else torch.zeros(
            states.shape[-1], device=states.device, dtype=states.dtype
        )
        return states, pooled

    def forward(
        self,
        text_embeddings: Tensor,
        node_types: Tensor,
        scalar_features: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
        resources: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        _, pooled = self.encode_graph(
            text_embeddings, node_types, scalar_features, edge_index, edge_types
        )
        resource_state = self.resource_projection(resources.float())
        logits = self.node_head(torch.cat([pooled, resource_state], dim=-1))
        return logits.masked_fill(~legal_mask.bool(), torch.finfo(logits.dtype).min)

    def score_derivations(self, graph_state: Tensor, candidate_embeddings: Tensor) -> Tensor:
        repeated = graph_state.unsqueeze(0).expand(candidate_embeddings.shape[0], -1)
        return self.derive_head(torch.cat([repeated, candidate_embeddings], dim=-1)).squeeze(-1)

    def score_communication_edges(self, node_states: Tensor, candidate_edges: Tensor) -> Tensor:
        source, target = candidate_edges.long()
        return self.communication_head(torch.cat([node_states[source], node_states[target]], dim=-1)).squeeze(-1)

    def save_checkpoint(self, path: Path, metadata: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str = "cpu") -> Tuple["StructuralPolicyNetwork", Dict[str, object]]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls()
        model.load_state_dict(payload["state_dict"])
        return model, dict(payload.get("metadata", {}))


def graph_tensors(graph: Dict[str, object], encoder: FrozenTextEncoder) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    node_index = {str(node["id"]): index for index, node in enumerate(nodes)}
    texts = [str(node.get("text") or node.get("kind") or "") for node in nodes]
    embeddings = encoder.encode(texts)
    kinds = torch.tensor([
        NODE_KINDS.index(str(node.get("kind"))) if str(node.get("kind")) in NODE_KINDS else 1
        for node in nodes
    ], dtype=torch.long)
    scalars = torch.tensor([
        [
            float(node.get("timestamp", 0)) / 1e12,
            1.0 if node.get("status") == "failed" else 0.0,
            float(node.get("attributes", {}).get("signal", 1.0 if node.get("status") in ("completed", "done") else 0.0)),
        ]
        for node in nodes
    ], dtype=torch.float32) if nodes else torch.zeros((0, 3), dtype=torch.float32)
    valid = [edge for edge in edges if edge.get("from", edge.get("source")) in node_index
             and edge.get("to", edge.get("target")) in node_index]
    edge_index = torch.tensor([
        [node_index[str(edge.get("from", edge.get("source")))] for edge in valid],
        [node_index[str(edge.get("to", edge.get("target")))] for edge in valid],
    ], dtype=torch.long) if valid else torch.zeros((2, 0), dtype=torch.long)
    edge_types = torch.tensor([
        EDGE_KINDS.index(str(edge.get("kind"))) if str(edge.get("kind")) in EDGE_KINDS else 0
        for edge in valid
    ], dtype=torch.long)
    return embeddings, kinds, scalars, edge_index, edge_types
