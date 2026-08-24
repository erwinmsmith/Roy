from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .value_model import process_credit


def topology_projection(state: Mapping[str, Any]) -> Dict[str, Any]:
    nodes = []
    for value in state.get("nodes", []):
        if not isinstance(value, Mapping):
            continue
        nodes.append({
            "id": str(value.get("id")),
            "parent_id": value.get("parentId", value.get("parent_id")),
            "depth": int(value.get("depth", 0)),
            "status": str(value.get("status", "unknown")),
        })
    edges = []
    for value in state.get("dagEdges", state.get("dag_edges", [])):
        if not isinstance(value, Mapping):
            continue
        edges.append({
            "kind": str(value.get("kind", "unknown")),
            "from": str(value.get("from", value.get("source", ""))),
            "to": str(value.get("to", value.get("target", ""))),
            "artifact_id": value.get("artifactId", value.get("artifact_id")),
            "resolved": value.get("resolved"),
            "active": value.get("active"),
        })
    return {
        "nodes": sorted(nodes, key=lambda item: item["id"]),
        "edges": sorted(edges, key=_edge_key),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "maximum_depth": max((value["depth"] for value in nodes), default=0),
    }


def topology_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    left = topology_projection(before)
    right = topology_projection(after)
    left_nodes = {value["id"]: value for value in left["nodes"]}
    right_nodes = {value["id"]: value for value in right["nodes"]}
    left_edges = {_edge_key(value): value for value in left["edges"]}
    right_edges = {_edge_key(value): value for value in right["edges"]}
    status_changes = [
        {"id": node_id, "before": left_nodes[node_id]["status"],
         "after": right_nodes[node_id]["status"]}
        for node_id in sorted(set(left_nodes).intersection(right_nodes))
        if left_nodes[node_id]["status"] != right_nodes[node_id]["status"]
    ]
    edge_state_changes = [
        {"edge": right_edges[key], "before": {
            "resolved": left_edges[key].get("resolved"),
            "active": left_edges[key].get("active"),
        }, "after": {
            "resolved": right_edges[key].get("resolved"),
            "active": right_edges[key].get("active"),
        }}
        for key in sorted(set(left_edges).intersection(right_edges))
        if (left_edges[key].get("resolved"), left_edges[key].get("active"))
        != (right_edges[key].get("resolved"), right_edges[key].get("active"))
    ]
    return {
        "added_nodes": [right_nodes[key] for key in sorted(set(right_nodes) - set(left_nodes))],
        "removed_nodes": [left_nodes[key] for key in sorted(set(left_nodes) - set(right_nodes))],
        "status_changes": status_changes,
        "added_edges": [right_edges[key] for key in sorted(set(right_edges) - set(left_edges))],
        "removed_edges": [left_edges[key] for key in sorted(set(left_edges) - set(right_edges))],
        "edge_state_changes": edge_state_changes,
        "node_count_delta": right["node_count"] - left["node_count"],
        "edge_count_delta": right["edge_count"] - left["edge_count"],
        "depth_delta": right["maximum_depth"] - left["maximum_depth"],
    }


def build_state_transition_samples(
    states: Sequence[Mapping[str, Any]], target_values: Sequence[float] | None = None,
    terminal_reward: float | None = None, metadata: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    if len(states) < 2:
        return []
    rewards: Sequence[float | None] = [None] * (len(states) - 1)
    returns: Sequence[float | None] = [None] * (len(states) - 1)
    if target_values is not None or terminal_reward is not None:
        if target_values is None or terminal_reward is None or len(target_values) != len(states):
            raise ValueError("scored state transitions require one value per state and a reward")
        scored_rewards, scored_returns = process_credit([target_values], [terminal_reward])
        rewards, returns = scored_rewards[0], scored_returns[0]
    result = []
    for index, (before, after) in enumerate(zip(states, states[1:])):
        delta = topology_delta(before, after)
        new_events = _new_events(before, after)
        action = next((event.get("attributes", {}).get("action") for event in reversed(new_events)
                       if event.get("kind") == "organization_action"), None)
        process_reward = rewards[index]
        result.append({
            **dict(metadata or {}), "sample_type": "state_transition",
            "transition_index": index,
            "from_sequence": before.get("sequence", index),
            "to_sequence": after.get("sequence", index + 1),
            "from_state_fingerprint": before.get("fingerprint"),
            "to_state_fingerprint": after.get("fingerprint"),
            "event_kinds": [str(value.get("kind")) for value in new_events]
                or ["semantic_state_update"],
            "organization_action": action,
            "topology_before": topology_projection(before),
            "topology_after": topology_projection(after),
            "topology_delta": delta,
            "topology_changed": _topology_changed(delta),
            "target_value_before": target_values[index] if target_values is not None else None,
            "target_value_after": target_values[index + 1] if target_values is not None else None,
            "process_reward": process_reward,
            "reward_sign": _reward_sign(process_reward),
            "return_to_go": returns[index],
            "terminal_reward": terminal_reward,
            "reward_target": "official_terminal" if index == len(states) - 2
                and target_values is not None else "ema_delta_value"
                if target_values is not None else None,
        })
    return result


def build_decision_transition_samples(
    states: Sequence[Mapping[str, Any]], policy_records: Sequence[Mapping[str, Any]],
    target_values: Sequence[float], terminal_reward: float,
    metadata: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    if not policy_records:
        return []
    by_fingerprint = {str(value.get("fingerprint")): index
                      for index, value in enumerate(states)}
    from_indices = []
    for record in policy_records:
        fingerprint = str(record.get("state_fingerprint")
                          or record.get("stateFingerprint") or "")
        if fingerprint not in by_fingerprint:
            raise ValueError("policy decision fingerprint is absent from process states")
        from_indices.append(by_fingerprint[fingerprint])
    to_indices = [*from_indices[1:], len(states) - 1]
    decision_values = [target_values[index] for index in from_indices] + [target_values[-1]]
    rewards, returns = process_credit([decision_values], [terminal_reward])
    result = []
    for index, (record, from_index, to_index) in enumerate(zip(
        policy_records, from_indices, to_indices
    )):
        delta = topology_delta(states[from_index], states[to_index])
        result.append({
            **dict(metadata or {}), "sample_type": "decision_transition",
            "decision_index": index,
            "from_sequence": states[from_index].get("sequence", from_index),
            "to_sequence": states[to_index].get("sequence", to_index),
            "from_state_fingerprint": states[from_index].get("fingerprint"),
            "to_state_fingerprint": states[to_index].get("fingerprint"),
            "selected_action": record.get("selected_action", record.get("selectedAction")),
            "candidate_id": record.get("candidate_id", record.get("candidateId")),
            "sampling_profile": (record.get("policy_state", record.get("policyState")) or {})
                .get("sampling_profile"),
            "topology_before": topology_projection(states[from_index]),
            "topology_after": topology_projection(states[to_index]),
            "topology_delta": delta,
            "topology_changed": _topology_changed(delta),
            "target_value_before": target_values[from_index],
            "target_value_after": target_values[to_index],
            "process_reward": rewards[0][index],
            "reward_sign": _reward_sign(rewards[0][index]),
            "return_to_go": returns[0][index],
            "terminal_reward": terminal_reward,
            "reward_target": "official_terminal" if index == len(policy_records) - 1
                else "ema_delta_value",
        })
    return result


def _new_events(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    known = {str(value.get("id")) for value in before.get("runtimeEvents", [])
             if isinstance(value, Mapping)}
    return [value for value in after.get("runtimeEvents", [])
            if isinstance(value, Mapping) and str(value.get("id")) not in known]


def _edge_key(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (str(value.get("kind", "unknown")), str(value.get("from", "")),
            str(value.get("to", "")), str(value.get("artifact_id") or ""))


def _topology_changed(delta: Mapping[str, Any]) -> bool:
    return any(delta.get(key) for key in (
        "added_nodes", "removed_nodes", "status_changes", "added_edges",
        "removed_edges", "edge_state_changes",
    ))


def _reward_sign(value: float | None, epsilon: float = 1e-8) -> str | None:
    if value is None:
        return None
    if value > epsilon:
        return "positive"
    if value < -epsilon:
        return "negative"
    return "zero"
