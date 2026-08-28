from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from .model import FrozenTextEncoder, epistemic_state_graph, graph_tensors
from .lhtb_transitions import (
    build_decision_transition_samples,
    build_state_transition_samples,
)
from .value_model import EpistemicValueModel, constant_value_output, process_credit


def value_metrics(records: Sequence[Mapping[str, Any]], checkpoint: str,
                  device_name: str = "cpu") -> Dict[str, float]:
    if not records:
        raise ValueError("value metrics require trajectories")
    device = torch.device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = EpistemicValueModel().to(device)
    model.load_state_dict(payload["value_state_dict"])
    model.eval()
    constant = constant_value_output(model)
    encoder = None if constant is not None else FrozenTextEncoder(
        device=device_name, local_only=True
    )
    predictions = []
    targets = []
    with torch.no_grad():
        for record in records:
            reward = float(record["terminal_reward"])
            states = list(record["process_states"])
            predictions.extend(
                [constant] * len(states) if constant is not None
                else _predict_states(model, states, encoder, device)
            )
            targets.extend([reward] * len(states))
    prediction = np.asarray(predictions)
    target = np.asarray(targets)
    return {
        "value_mae": float(np.mean(np.abs(prediction - target))),
        "value_spearman": _spearman(prediction, target),
    }


def annotate_value_traces(records: Sequence[Mapping[str, Any]], checkpoint: str,
                          device_name: str = "cpu") -> Sequence[Mapping[str, Any]]:
    device = torch.device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    target_revision = int(payload.get("metadata", {}).get("groups", 0))
    value = EpistemicValueModel().to(device)
    target = EpistemicValueModel().to(device)
    value.load_state_dict(payload["value_state_dict"])
    target.load_state_dict(payload["target_state_dict"])
    value.eval(); target.eval()
    value_constant = constant_value_output(value)
    target_constant = constant_value_output(target)
    encoder = None if value_constant is not None and target_constant is not None \
        else FrozenTextEncoder(device=device_name, local_only=True)
    result = []
    with torch.no_grad():
        for record in records:
            states = list(record.get("process_states", []))
            value_predictions = [value_constant] * len(states) if value_constant is not None \
                else _predict_states(value, states, encoder, device)
            target_predictions = [target_constant] * len(states) if target_constant is not None \
                else _predict_states(target, states, encoder, device)
            by_fingerprint = {str(state.get("fingerprint")): index for index, state in enumerate(states)}
            indices = [by_fingerprint[str(item.get("state_fingerprint",
                       item.get("stateFingerprint")))] for item in record.get("policy_records", [])]
            decision_targets = [target_predictions[index] for index in indices] + [target_predictions[-1]]
            process_rewards, returns = process_credit(
                [decision_targets], [float(record.get("terminal_reward", record.get("reward")))]
            ) if indices else ([[]], [[]])
            enriched = dict(record)
            enriched["value_trace"] = value_predictions
            enriched["target_value_trace"] = target_predictions
            enriched["target_value_revision"] = target_revision
            enriched["process_rewards"] = process_rewards[0]
            enriched["shaped_returns"] = returns[0]
            metadata = {"trajectory_id": record.get("id"),
                        "task_id": record.get("task_id"),
                        "rollout_index": record.get("rollout_index"),
                        "target_value_revision": target_revision}
            enriched["state_transitions"] = build_state_transition_samples(
                states, target_predictions,
                float(record.get("terminal_reward", record.get("reward"))), metadata
            )
            enriched["decision_transitions"] = build_decision_transition_samples(
                states, list(record.get("policy_records", [])), target_predictions,
                float(record.get("terminal_reward", record.get("reward"))), metadata
            ) if indices else []
            result.append(enriched)
    return result


def _predict_state(model: EpistemicValueModel, state: Mapping[str, Any],
                   encoder: FrozenTextEncoder | None, device: torch.device) -> float:
    if encoder is None:
        raise ValueError("non-constant value prediction requires a text encoder")
    graph = _state_graph(state)
    tensors = [value.to(device) for value in graph_tensors(graph, encoder)]
    return float(model(*tensors))


def _predict_states(
    model: EpistemicValueModel,
    states: Sequence[Mapping[str, Any]],
    encoder: FrozenTextEncoder | None,
    device: torch.device,
    batch_size: int = 64,
) -> list[float]:
    """Score exact graph projections in batches, reusing identical M_t graphs."""
    if encoder is None:
        raise ValueError("non-constant value prediction requires a text encoder")
    if batch_size <= 0:
        raise ValueError("value inference batch_size must be positive")
    graphs: list[Dict[str, object]] = []
    unique: Dict[str, int] = {}
    state_indices: list[int] = []
    for state in states:
        graph = _state_graph(state)
        key = hashlib.sha256(json.dumps(
            graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        index = unique.get(key)
        if index is None:
            index = len(graphs)
            unique[key] = index
            graphs.append(graph)
        state_indices.append(index)
    precache = getattr(encoder, "precache", None)
    if callable(precache):
        precache(
            str(node.get("text") or node.get("kind") or "")
            for graph in graphs for node in graph.get("nodes", [])
            if isinstance(node, Mapping)
        )
    unique_values: list[float] = []
    for start in range(0, len(graphs), batch_size):
        tensors = [tuple(
            value.to(device) for value in graph_tensors(graph, encoder)
        ) for graph in graphs[start:start + batch_size]]
        unique_values.extend(
            float(value) for value in model.forward_batch(tensors).detach().cpu()
        )
    return [unique_values[index] for index in state_indices]


def _state_graph(state: Mapping[str, Any]) -> Dict[str, object]:
    graph = state.get("event_graph")
    return dict(graph) if isinstance(graph, Mapping) else epistemic_state_graph(state)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(left.std()) == 0 or float(right.std()) == 0:
        return 0.0
    left_rank = np.argsort(np.argsort(left))
    right_rank = np.argsort(np.argsort(right))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])
