from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from .model import FrozenTextEncoder, epistemic_state_graph, graph_tensors
from .lhtb_transitions import (
    build_decision_transition_samples,
    build_state_transition_samples,
)
from .value_model import EpistemicValueModel, process_credit


def value_metrics(records: Sequence[Mapping[str, Any]], checkpoint: str,
                  device_name: str = "cpu") -> Dict[str, float]:
    if not records:
        raise ValueError("value metrics require trajectories")
    device = torch.device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = EpistemicValueModel().to(device)
    model.load_state_dict(payload["value_state_dict"])
    model.eval()
    encoder = FrozenTextEncoder(device=device_name, local_only=True)
    predictions = []
    targets = []
    with torch.no_grad():
        for record in records:
            reward = float(record["terminal_reward"])
            for state in record["process_states"]:
                graph = epistemic_state_graph(state)
                tensors = [value.to(device) for value in graph_tensors(graph, encoder)]
                predictions.append(float(model(*tensors)))
                targets.append(reward)
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
    value = EpistemicValueModel().to(device)
    target = EpistemicValueModel().to(device)
    value.load_state_dict(payload["value_state_dict"])
    target.load_state_dict(payload["target_state_dict"])
    value.eval(); target.eval()
    encoder = FrozenTextEncoder(device=device_name, local_only=True)
    result = []
    with torch.no_grad():
        for record in records:
            states = list(record.get("process_states", []))
            value_predictions = [_predict_state(value, state, encoder, device) for state in states]
            target_predictions = [_predict_state(target, state, encoder, device) for state in states]
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
            enriched["process_rewards"] = process_rewards[0]
            enriched["shaped_returns"] = returns[0]
            metadata = {"trajectory_id": record.get("id"),
                        "task_id": record.get("task_id"),
                        "rollout_index": record.get("rollout_index")}
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
                   encoder: FrozenTextEncoder, device: torch.device) -> float:
    graph = epistemic_state_graph(state)
    tensors = [value.to(device) for value in graph_tensors(graph, encoder)]
    return float(model(*tensors))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(left.std()) == 0 or float(right.std()) == 0:
        return 0.0
    left_rank = np.argsort(np.argsort(left))
    right_rank = np.argsort(np.argsort(right))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])
