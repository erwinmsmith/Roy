from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from .model import FrozenTextEncoder, graph_tensors
from .organization import (
    ExplorationEnvelope,
    ORGANIZATION_GROUP_SIZE,
    envelope_legal_actions,
    validate_exploration_group,
)
from .organization_model import (
    InformationRealizationPolicy,
    action_type_indices,
)
from .organization_training import (
    organization_group_advantages,
    single_objective_organization_grpo_loss,
)
class OrganizationGRPOTrainer:
    """Stateful on-policy trainer used by the tau3 sample-and-update loop."""

    def __init__(
        self,
        output_path: Path,
        learning_rate: float = 1e-4,
        device_name: str | None = None,
        seed: int = 20260818,
        resume: bool = False,
    ) -> None:
        torch.manual_seed(seed)
        self.output_path = output_path
        self.seed = seed
        self.device = torch.device(
            device_name or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.encoder = FrozenTextEncoder(device=str(self.device), local_only=True)
        self.model = InformationRealizationPolicy().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.groups = 0
        self.trajectories = 0
        self.optimizer_steps = 0
        self.zero_variance_groups = 0
        self.losses: List[float] = []
        self.loss_sum = 0.0
        self.updated_group_ids: set[str] = set()
        if resume and output_path.exists():
            payload = torch.load(output_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(payload["state_dict"])
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            metadata = payload.get("metadata", {})
            self.groups = int(metadata.get("groups", 0))
            self.trajectories = int(metadata.get("trajectories", 0))
            self.optimizer_steps = int(metadata.get("optimizer_steps", 0))
            self.zero_variance_groups = int(
                metadata.get("zero_variance_groups", 0)
            )
            self.loss_sum = float(
                metadata.get(
                    "loss_sum", float(metadata.get("mean_loss", 0.0)) * self.optimizer_steps
                )
            )
            self.updated_group_ids = set(metadata.get("updated_group_ids", []))

    def update_group(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if len(records) != ORGANIZATION_GROUP_SIZE:
            raise ValueError("on-policy tau3 updates require exactly eight trajectories")
        group_ids = {str(value.get("group_id") or "") for value in records}
        if len(group_ids) != 1 or "" in group_ids:
            raise ValueError("all on-policy trajectories must belong to one task group")
        group_id = next(iter(group_ids))
        if group_id in self.updated_group_ids:
            raise ValueError(f"organization group was already optimized: {group_id}")
        if any(
            value.get("benchmark") != "tau3"
            or value.get("split") != "train"
            or not value.get("terminal")
            for value in records
        ):
            raise ValueError("on-policy updates accept only complete tau3 train trajectories")
        if any(bool(value.get("truncated")) for value in records):
            raise ValueError("on-policy updates reject censored or truncated trajectories")
        envelopes = tuple(
            ExplorationEnvelope(**dict(value.get("envelope") or {})) for value in records
        )
        validate_exploration_group(envelopes)
        if {int(value.get("rollout_index", -1)) for value in records} != set(
            range(ORGANIZATION_GROUP_SIZE)
        ):
            raise ValueError("on-policy tau3 group requires rollout indices zero through seven")
        if len({int(value.get("environment_seed", -1)) for value in records}) != 1:
            raise ValueError("counterfactual trajectories must share one environment seed")
        snapshot_fingerprints = {
            str(value.get("initial_snapshot_fingerprint") or "") for value in records
        }
        if len(snapshot_fingerprints) != 1 or "" in snapshot_fingerprints:
            raise ValueError("counterfactual trajectories must share one initial snapshot")
        if len({
            json.dumps(value.get("runtime_budget") or {}, sort_keys=True) for value in records
        }) != 1:
            raise ValueError("counterfactual trajectories must share one runtime budget")

        advantages = {
            value.trajectory_id: value.advantage
            for value in organization_group_advantages(records)
        }
        utilities = [float(value["terminal_utility"]) for value in records]
        utility_mean = sum(utilities) / len(utilities)
        utility_std = math.sqrt(
            sum((value - utility_mean) ** 2 for value in utilities) / len(utilities)
        )
        if utility_std <= 1e-8:
            self.groups += 1
            self.trajectories += len(records)
            self.zero_variance_groups += 1
            self.updated_group_ids.add(group_id)
            self.save()
            return {
                **self.metadata(),
                "update_applied": False,
                "utility_std": utility_std,
            }
        current_probabilities: List[Tensor] = []
        behavior_probabilities: List[Tensor] = []
        trajectory_advantages: List[float] = []
        for trajectory in records:
            policy_records = trajectory.get("policy_records")
            if not isinstance(policy_records, Sequence) or not policy_records:
                raise ValueError(f"trajectory {trajectory.get('id')} has no policy records")
            if any(not isinstance(record, Mapping) for record in policy_records):
                raise ValueError("invalid organization policy record")
            current_probabilities.append(torch.stack([
                replay_joint_log_probability(self.model, self.encoder, record, self.device)
                for record in policy_records
            ]))
            behavior_probabilities.append(torch.tensor(
                [float(record["masked_old_log_probability"]) for record in policy_records],
                dtype=torch.float32,
                device=self.device,
            ))
            trajectory_advantages.append(advantages[str(trajectory["id"])])

        loss = single_objective_organization_grpo_loss(
            current_probabilities, behavior_probabilities, trajectory_advantages
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.groups += 1
        self.trajectories += len(records)
        self.optimizer_steps += 1
        self.updated_group_ids.add(group_id)
        loss_value = float(loss.detach().cpu())
        self.losses.append(loss_value)
        self.loss_sum += loss_value
        self.save()
        return {
            **self.metadata(),
            "update_applied": True,
            "utility_std": utility_std,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "roy_information_realization",
            "benchmark": "tau3",
            "objective": "terminal_task_utility_only",
            "policy_source": "autonomous_on_policy_sampling",
            "training_protocol": "on_policy_sample_and_update",
            "resource_constraint": "observable_runtime_action_mask",
            "groups": self.groups,
            "trajectories": self.trajectories,
            "optimizer_steps": self.optimizer_steps,
            "zero_variance_groups": self.zero_variance_groups,
            "mean_loss": self.loss_sum / max(1, self.optimizer_steps),
            "loss_sum": self.loss_sum,
            "seed": self.seed,
            "updated_group_ids": sorted(self.updated_group_ids),
        }

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": self.metadata(),
        }, self.output_path)


def sample_organization_decision(
    model: InformationRealizationPolicy,
    encoder: FrozenTextEncoder,
    policy_state: Mapping[str, Any],
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sample from the exact masked old policy used by GRPO replay."""

    resolved_device = device or next(model.parameters()).device
    context = _policy_context(model, encoder, policy_state, resolved_device)
    active_index = int(
        torch.multinomial(context["active_probabilities"], 1, generator=generator).item()
    )
    active_id = context["active_node_ids"][active_index]
    values = _candidate_distribution(model, policy_state, context, active_id, resolved_device)
    candidate_index = int(
        torch.multinomial(values["candidate_probabilities"], 1, generator=generator).item()
    )
    candidate = values["candidates"][candidate_index]
    joint_log_probability = (
        context["active_log_probs"][active_index]
        + values["candidate_log_probs"][candidate_index]
    )
    record = {
        "state_fingerprint": str(policy_state.get("state_fingerprint") or ""),
        "active_node_id": active_id,
        "candidate_id": str(candidate["id"]),
        "masked_old_log_probability": float(joint_log_probability.detach().cpu()),
        "envelope_id": str(policy_state["envelope"]["id"]),
        "policy_state": dict(policy_state),
    }
    return dict(candidate), record


def replay_joint_log_probability(
    model: InformationRealizationPolicy,
    encoder: FrozenTextEncoder,
    record: Mapping[str, Any],
    device: torch.device,
) -> Tensor:
    policy_state = record.get("policy_state", record.get("policyState"))
    if not isinstance(policy_state, Mapping):
        raise ValueError("policy record is missing policy_state")
    context = _policy_context(model, encoder, policy_state, device)
    try:
        active_id = str(record.get("active_node_id", record.get("activeNodeId")))
        active_index = context["active_node_ids"].index(active_id)
        values = _candidate_distribution(model, policy_state, context, active_id, device)
        candidate_index = [str(value["id"]) for value in values["candidates"]].index(
            str(record.get("candidate_id", record.get("candidateId")))
        )
    except ValueError as error:
        raise ValueError("recorded organization choice is not available during replay") from error
    return (
        context["active_log_probs"][active_index]
        + values["candidate_log_probs"][candidate_index]
    )


def _policy_context(
    model: InformationRealizationPolicy,
    encoder: FrozenTextEncoder,
    policy_state: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    graph = policy_state.get("event_graph")
    if not isinstance(graph, Mapping):
        raise ValueError("organization policy state is missing event_graph")
    tensors = tuple(value.to(device) for value in graph_tensors(dict(graph), encoder))
    node_states, graph_state = model.encode_graph(*tensors)
    graph_nodes = list(graph.get("nodes", []))
    node_index = {str(value["id"]): index for index, value in enumerate(graph_nodes)}
    active_node_ids = [str(value) for value in policy_state.get("active_node_ids", [])]
    if not active_node_ids or any(value not in node_index for value in active_node_ids):
        raise ValueError("active-node candidates must reference event graph nodes")
    active_indices = torch.tensor([node_index[value] for value in active_node_ids], device=device)
    active_states = node_states[active_indices]
    active_mask = torch.tensor(
        list(policy_state.get("active_node_legal", [True] * len(active_node_ids))),
        dtype=torch.bool,
        device=device,
    )
    resources = _organization_resource_tensor(dict(policy_state.get("resources", {})), device)
    temperature = float(policy_state.get("organization_temperature", 1.0))
    if temperature <= 0:
        raise ValueError("organization temperature must be positive during replay")
    active_logits = model.active_node_logits(active_states, graph_state, resources, active_mask)
    active_log_probs = torch.log_softmax(active_logits / temperature, dim=-1)

    candidates = list(policy_state.get("candidates", []))
    if not candidates:
        raise ValueError("organization policy state contains no candidates")
    candidate_embeddings = encoder.encode([
        str(value.get("description") or value.get("kind") or "") for value in candidates
    ]).to(device)
    return {
        "node_states": node_states,
        "graph_state": graph_state,
        "node_index": node_index,
        "active_node_ids": active_node_ids,
        "active_log_probs": active_log_probs,
        "active_probabilities": active_log_probs.exp(),
        "candidate_embeddings": candidate_embeddings,
        "resources": resources,
    }


def _candidate_distribution(
    model: InformationRealizationPolicy,
    policy_state: Mapping[str, Any],
    context: Mapping[str, Any],
    actor_node_id: str,
    device: torch.device,
) -> Dict[str, Any]:
    node_index = context["node_index"]
    if actor_node_id not in node_index:
        raise ValueError("candidate actor node is absent from event graph")
    candidates = list(policy_state.get("candidates", []))
    if not candidates:
        raise ValueError("organization policy state contains no candidates")
    envelope = ExplorationEnvelope(**dict(policy_state["envelope"]))
    if bool(policy_state.get("unbounded_structure", False)):
        legal_values = [bool(value.get("legal", True)) for value in candidates]
    else:
        legal_values = envelope_legal_actions(
            candidates,
            envelope,
            int(policy_state.get("node_count", 1)),
            int(policy_state.get("maximum_depth_reached", 0)),
            bool(policy_state.get("unresolved_gap_exists", False)),
        )
    legal_values = [
        legal and (
            not value.get("actor_node_id")
            or str(value.get("actor_node_id")) == actor_node_id
        )
        for value, legal in zip(candidates, legal_values)
    ]
    if not any(legal_values):
        raise ValueError(f"active node {actor_node_id} has no legal organization candidates")
    legal_mask = torch.tensor(legal_values, dtype=torch.bool, device=device)
    candidate_features = torch.tensor(
        [_candidate_features(value) for value in candidates], dtype=torch.float32, device=device
    )
    candidate_logits = model.candidate_logits(
        context["graph_state"],
        context["node_states"][node_index[actor_node_id]],
        context["candidate_embeddings"],
        action_type_indices([str(value["kind"]) for value in candidates], device),
        candidate_features,
        context["resources"],
        legal_mask,
    )
    temperature = float(policy_state.get("organization_temperature", 1.0))
    if temperature <= 0:
        raise ValueError("organization temperature must be positive during replay")
    candidate_log_probs = torch.log_softmax(candidate_logits / temperature, dim=-1)
    return {
        "candidates": candidates,
        "candidate_log_probs": candidate_log_probs,
        "candidate_probabilities": candidate_log_probs.exp(),
    }


def _candidate_features(value: Mapping[str, Any]) -> List[float]:
    return [
        math.log1p(max(0.0, float(value.get("scheduler_complexity", 0.0)))),
        float(bool(value.get("external_access", False))),
        float(bool(value.get("resolves_gap", False))),
        float(value.get("depth_delta", 0.0)),
    ]


def _organization_resource_tensor(resources: Mapping[str, Any], device: torch.device) -> Tensor:
    return torch.tensor(
        [
            float(resources.get("llm_calls_remaining_fraction", 0.0)),
            float(resources.get("tool_calls_remaining_fraction", 0.0)),
            float(resources.get("nodes_remaining_fraction", 0.0)),
            float(resources.get("depth_remaining_fraction", 0.0)),
            float(resources.get("decisions_remaining_fraction", 0.0)),
        ],
        dtype=torch.float32,
        device=device,
    )
