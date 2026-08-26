from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
        self._cache: Dict[str, Tensor] = {}

    def encode(self, texts: Sequence[str]) -> Tensor:
        if not texts:
            return torch.zeros((0, TEXT_DIMENSION), dtype=torch.float32)
        keys = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in texts]
        missing: Dict[str, str] = {}
        for key, value in zip(keys, texts):
            if key not in self._cache:
                missing[key] = value
        if missing:
            missing_keys = list(missing)
            values = self.model.encode(
                [missing[key] for key in missing_keys],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            encoded = torch.from_numpy(np.asarray(values, dtype=np.float32))
            for key, value in zip(missing_keys, encoded):
                self._cache[key] = value.detach().cpu()
        return torch.stack([self._cache[key] for key in keys])


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


def epistemic_state_graph(state: Mapping[str, Any]) -> Dict[str, object]:
    """Project a full M_t into typed relational nodes without semantic heuristics."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add_node(identifier: Any, kind: str, text: Any, status: Any = None,
                 timestamp: Any = 0, signal: float = 0.0) -> str | None:
        if identifier is None:
            return None
        node_id = str(identifier)
        if not node_id or node_id in seen:
            return node_id if node_id else None
        seen.add(node_id)
        nodes.append({"id": node_id, "kind": kind, "text": str(text or ""),
                      "status": status, "timestamp": timestamp,
                      "attributes": {"signal": signal}})
        return node_id

    for value in state.get("nodes", []):
        if isinstance(value, Mapping):
            status = str(value.get("status", ""))
            add_node(value.get("id"), "agent",
                     value.get("localObjective", value.get("local_objective", "")),
                     status, value.get("createdAt", value.get("created_at", 0)),
                     1.0 if status in ("completed", "returned") else 0.0)
    for value in state.get("requirements", []):
        if not isinstance(value, Mapping):
            continue
        status = str(value.get("status", "open"))
        node_id = add_node(value.get("id"), "subtask",
                           value.get("description", value.get("requiredInformation", "")),
                           status, signal=1.0 if status == "resolved" else 0.0)
        parent = value.get("parentNodeId", value.get("parent_node_id"))
        if node_id and parent is not None:
            edges.append({"kind": "dependency", "from": str(parent), "to": node_id})
    for value in state.get("claims", []):
        if not isinstance(value, Mapping):
            continue
        status = str(value.get("status", "tentative"))
        node_id = add_node(value.get("id"), "child_result", value.get("statement", ""),
                           status, signal=1.0 if status == "supported" else 0.0)
        origin = value.get("originNodeId", value.get("origin_node_id"))
        if node_id and origin is not None:
            edges.append({"kind": "produces", "from": str(origin), "to": node_id})
    for index, value in enumerate(state.get("assumptions", [])):
        if isinstance(value, Mapping):
            status = str(value.get("status", "unverified"))
            add_node(value.get("id", f"assumption-{index}"), "message",
                     value.get("statement", ""), status,
                     signal=1.0 if status == "verified" else 0.0)
    for index, value in enumerate(state.get("evidence", [])):
        if not isinstance(value, Mapping):
            continue
        node_id = add_node(value.get("id", f"evidence-{index}"), "artifact",
                           value.get("content", ""), "observed", signal=1.0)
        for claim_id in value.get("supports", []):
            if node_id:
                edges.append({"kind": "produces", "from": node_id, "to": str(claim_id)})
    for index, value in enumerate(state.get(
        "externalObservations", state.get("external_observations", [])
    )):
        if not isinstance(value, Mapping):
            continue
        node_id = add_node(value.get("id", f"observation-{index}"), "tool_result",
                           value.get("observation", ""), "observed", signal=1.0)
        for claim_id in value.get("supports", []):
            if node_id:
                edges.append({"kind": "produces", "from": node_id, "to": str(claim_id)})
    for index, value in enumerate(state.get("blindSpots", state.get("blind_spots", []))):
        text = value.get("description", value.get("statement", "")) \
            if isinstance(value, Mapping) else value
        add_node(f"blind-spot:{index}", "message", text, "open", signal=-1.0)

    runtime_node_ids: List[str] = []
    runtime_kind_map = {
        "task_instruction": "subtask",
        "organization_action": "message",
        "terminal_command": "tool_call",
        "terminal_result": "tool_result",
        "file_change": "artifact",
        "failure": "tool_result",
        "usage": "resource",
        "verifier": "tool_result",
    }
    for index, value in enumerate(state.get("runtimeEvents", state.get("runtime_events", []))):
        if not isinstance(value, Mapping):
            continue
        event_kind = str(value.get("kind", "message"))
        attributes = value.get("attributes", {})
        attributes = attributes if isinstance(attributes, Mapping) else {}
        action = attributes.get("action", {})
        action = action if isinstance(action, Mapping) else {}
        text_parts = [
            event_kind,
            str(action.get("kind", "")),
            str(value.get("command", "")),
            str(value.get("output", "")),
        ]
        event_text = " ".join(part for part in text_parts if part)
        if len(event_text) > 2_000:
            event_text = event_text[:1_000] + " [event text omitted] " + event_text[-1_000:]
        failed = event_kind == "failure" or (
            event_kind == "terminal_result" and int(value.get("exitCode", value.get("exit_code", 0)) or 0) != 0
        )
        event_id = add_node(
            f"runtime:{value.get('id', index)}",
            runtime_kind_map.get(event_kind, "message"),
            event_text,
            "failed" if failed else "observed",
            value.get("at", 0),
            -1.0 if failed else 1.0 if event_kind in {"terminal_result", "file_change", "verifier"} else 0.0,
        )
        if event_id is None:
            continue
        runtime_node_ids.append(event_id)
        actor_id = value.get("nodeId", value.get("node_id"))
        if actor_id is not None:
            edges.append({
                "kind": "tool_use" if event_kind in {"terminal_command", "terminal_result"}
                else "produces",
                "from": str(actor_id),
                "to": event_id,
            })
    for before, after in zip(runtime_node_ids, runtime_node_ids[1:]):
        edges.append({"kind": "temporal", "from": before, "to": after})

    active_ids = [str(value) for value in state.get(
        "activeSubtree", state.get("active_subtree", [])
    )]
    active_summary = add_node(
        "state:active-subtree", "resource", "active subtree " + " ".join(active_ids),
        "observed", signal=math.log1p(len(active_ids)),
    )
    if active_summary:
        for node_id in active_ids:
            edges.append({"kind": "consumes", "from": node_id, "to": active_summary})

    usage = state.get("usage", {})
    usage = usage if isinstance(usage, Mapping) else {}
    state_nodes = [value for value in state.get("nodes", []) if isinstance(value, Mapping)]
    dag_edges = [value for value in state.get(
        "dagEdges", state.get("dag_edges", [])
    ) if isinstance(value, Mapping)]
    metrics = {
        "node_count": len(state_nodes),
        "edge_count": len(dag_edges),
        "maximum_depth": max((int(value.get("depth", 0)) for value in state_nodes), default=0),
        "active_node_count": len(active_ids),
        "blind_spot_count": len(state.get("blindSpots", state.get("blind_spots", []))),
        "input_tokens": float(usage.get("inputTokens", usage.get("input_tokens", 0)) or 0),
        "output_tokens": float(usage.get("outputTokens", usage.get("output_tokens", 0)) or 0),
        "wall_time_ms": float(usage.get("wallTimeMs", usage.get("wall_time_ms", 0)) or 0),
    }
    for name, value in metrics.items():
        add_node(
            f"metric:{name}", "resource", f"{name} {value}", "observed",
            signal=math.log1p(max(0.0, float(value))),
        )
    for value in state.get("dagEdges", state.get("dag_edges", [])):
        if isinstance(value, Mapping):
            edges.append(dict(value))
    for value in state.get("semanticRelations", state.get("semantic_relations", [])):
        if isinstance(value, Mapping):
            edges.append({"kind": "dependency",
                          "from": str(value.get("leftId", value.get("left_id", ""))),
                          "to": str(value.get("rightId", value.get("right_id", "")))})
    return {"nodes": nodes, "edges": edges}
