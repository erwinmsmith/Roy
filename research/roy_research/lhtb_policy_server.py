from __future__ import annotations

import json
import hashlib
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import torch

from .model import FrozenTextEncoder, graph_tensors
from .organization_model import InformationRealizationPolicy, LHTB_ACTOR_MODEL_REVISION
from .organization_replay import (
    _candidate_text,
    organization_candidate_distribution,
    sample_organization_decision,
)
from .value_model import EpistemicValueModel, LHTB_VALUE_MODEL_REVISION


class LHTBPolicyServer:
    def __init__(self) -> None:
        checkpoint = os.environ.get("ROY_LHTB_MODEL")
        if not checkpoint:
            raise RuntimeError("ROY_LHTB_MODEL is required; learned mode has no heuristic fallback")
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        checkpoint_revision = payload.get("metadata", {}).get("value_model_revision")
        if checkpoint_revision != LHTB_VALUE_MODEL_REVISION:
            raise ValueError(
                "LHTB value checkpoint is incompatible: expected "
                f"{LHTB_VALUE_MODEL_REVISION}, found {checkpoint_revision or 'legacy'}; "
                "initialize a fresh checkpoint and keep the legacy file for audit"
            )
        actor_revision = payload.get("metadata", {}).get("actor_model_revision")
        if actor_revision != LHTB_ACTOR_MODEL_REVISION:
            raise ValueError(
                "LHTB actor checkpoint is incompatible: expected "
                f"{LHTB_ACTOR_MODEL_REVISION}, found "
                f"{actor_revision or 'legacy-active-node-routing'}; initialize a fresh checkpoint"
            )
        self.model = InformationRealizationPolicy()
        state = payload.get("actor_state_dict", payload.get("state_dict"))
        if not isinstance(state, dict):
            raise ValueError("checkpoint contains no LHTB actor state")
        self.model.load_state_dict(state)
        self.model.eval()
        self.target = EpistemicValueModel()
        target_state = payload.get("target_state_dict")
        if not isinstance(target_state, dict):
            raise ValueError("checkpoint contains no frozen target value state")
        self.target.load_state_dict(target_state)
        self.target.eval()
        self.target_revision = int(payload.get("metadata", {}).get("groups", 0))
        self.encoder = FrozenTextEncoder(device="cpu", local_only=True)
        self.generator = torch.Generator(device="cpu")
        cache_size = max(1, int(os.environ.get("ROY_LHTB_VALUE_CACHE_SIZE", "4096")))
        self.value_cache = _LRUCache(cache_size)
        self.analysis_cache = _LRUCache(cache_size)

    def decide(self, policy_state: Dict[str, Any], seed: int) -> Dict[str, Any]:
        self.generator.manual_seed(seed)
        with torch.no_grad():
            candidate, record = sample_organization_decision(
                self.model, self.encoder, policy_state, self.generator, torch.device("cpu")
            )
        return {"candidate_id": candidate["id"], "policy_record": record}

    def analyze(self, policy_state: Dict[str, Any]) -> Dict[str, Any]:
        graph = policy_state.get("event_graph")
        if not isinstance(graph, dict):
            raise ValueError("organization policy state is missing event_graph")
        key = _stable_digest(policy_state)
        cached = self.analysis_cache.get(key)
        if cached is not None:
            return dict(cached)
        self.encoder.precache([
            *_event_graph_texts(graph),
            *(_candidate_text(candidate)
              for candidate in policy_state.get("candidates", [])
              if isinstance(candidate, dict)),
        ])
        target_value = self._target_value(graph)
        with torch.no_grad():
            distribution = organization_candidate_distribution(
                self.model, self.encoder, policy_state, torch.device("cpu")
            )
        result = {
            "target_value": target_value,
            "target_revision": self.target_revision,
            **distribution,
        }
        self.analysis_cache.put(key, result)
        return result

    def value(self, event_graph: Dict[str, Any]) -> Dict[str, Any]:
        target_value = self._target_value(event_graph)
        return {"target_value": target_value, "target_revision": self.target_revision}

    def values(self, event_graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(event_graphs, list) or not all(
            isinstance(graph, dict) for graph in event_graphs
        ):
            raise ValueError("event_graphs must be a list of graph objects")
        keys = [_stable_digest(graph) for graph in event_graphs]
        results: List[float | None] = [None] * len(event_graphs)
        missing_by_key: Dict[str, Dict[str, Any]] = {}
        for index, (key, graph) in enumerate(zip(keys, event_graphs)):
            cached = self.value_cache.get(key)
            if cached is not None:
                results[index] = float(cached)
            elif key not in missing_by_key:
                missing_by_key[key] = graph
        if missing_by_key:
            missing_keys = list(missing_by_key)
            self.encoder.precache(
                text
                for key in missing_keys
                for text in _event_graph_texts(missing_by_key[key])
            )
            tensors = [graph_tensors(missing_by_key[key], self.encoder)
                       for key in missing_keys]
            with torch.no_grad():
                predictions = self.target.forward_batch(tensors).detach().cpu().tolist()
            for key, prediction in zip(missing_keys, predictions):
                self.value_cache.put(key, float(prediction))
        for index, key in enumerate(keys):
            if results[index] is None:
                cached = self.value_cache.get(key)
                if cached is None:
                    raise RuntimeError("batched target value was not cached")
                results[index] = float(cached)
        return {"target_values": results, "target_revision": self.target_revision}

    def _target_value(self, event_graph: Dict[str, Any]) -> float:
        key = _stable_digest(event_graph)
        cached = self.value_cache.get(key)
        if cached is not None:
            return float(cached)
        with torch.no_grad():
            tensors = graph_tensors(event_graph, self.encoder)
            target_value = float(self.target(*tensors))
        self.value_cache.put(key, target_value)
        return target_value


class _LRUCache:
    def __init__(self, maximum_size: int) -> None:
        self.maximum_size = maximum_size
        self.values: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self.values:
            return None
        value = self.values.pop(key)
        self.values[key] = value
        return value

    def put(self, key: str, value: Any) -> None:
        self.values.pop(key, None)
        self.values[key] = value
        while len(self.values) > self.maximum_size:
            self.values.popitem(last=False)


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _event_graph_texts(event_graph: Dict[str, Any]) -> List[str]:
    return [
        str(node.get("text") or node.get("label") or node.get("id") or "")
        for node in event_graph.get("nodes", [])
        if isinstance(node, dict)
    ]


def main() -> None:
    server = LHTBPolicyServer()
    for line in sys.stdin:
        request: Dict[str, Any] = {}
        try:
            request = json.loads(line)
            operation = str(request.get("operation", "select"))
            if operation == "analyze":
                result = server.analyze(request["policy_state"])
            elif operation == "value":
                result = server.value(request["event_graph"])
            elif operation == "values":
                result = server.values(request["event_graphs"])
            elif operation == "select":
                result = server.decide(request["policy_state"], int(request["seed"]))
            else:
                raise ValueError(f"unsupported policy operation: {operation}")
            response = {"id": request["id"], **result}
        except Exception as error:
            response = {"id": request.get("id", "unknown"), "error": str(error)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
