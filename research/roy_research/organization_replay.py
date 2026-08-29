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
    ORGANIZATION_ACTIONS,
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
    values = _candidate_distribution(model, policy_state, context, resolved_device)
    candidate_index = int(
        torch.multinomial(values["candidate_probabilities"], 1, generator=generator).item()
    )
    candidate = values["candidates"][candidate_index]
    selected_kind = str(candidate.get("kind"))
    joint_log_probability = values["candidate_log_probs"][candidate_index]
    record = {
        "behavior_policy": "actor",
        "state_fingerprint": str(policy_state.get("state_fingerprint") or ""),
        "context_node_id": context["context_node_id"],
        "candidate_id": str(candidate["id"]),
        "masked_old_log_probability": float(joint_log_probability.detach().cpu()),
        "envelope_id": str(policy_state["envelope"]["id"]),
        "policy_state": dict(policy_state),
        "available_actions": sorted({str(value.get("kind"))
                                     for value in values["candidates"]}),
        "raw_probabilities": _action_probability_summary(
            values["candidates"], values["candidate_raw_probabilities"]
        ),
        "masked_probabilities": _action_probability_summary(
            values["candidates"], values["candidate_probabilities"]
        ),
        "selected_action": str(candidate.get("kind")),
        "selected_spawn_mode": candidate.get("realization_mode"),
        "spawn_mode_probabilities": _spawn_mode_probability_summary(
            values["candidates"], values["candidate_probabilities"]
        ),
        "masked_old_action_log_probability": float(
            values["masked_outer_action_log_probabilities"][selected_kind].detach().cpu()
        ),
        "masked_old_candidate_conditional_log_probability": float(
            values["masked_inner_candidate_log_probabilities"][candidate_index].detach().cpu()
        ),
        "num_real_residual_gaps": int(policy_state.get("num_real_residual_gaps", 0)),
        "num_child_proposals": int(policy_state.get("num_child_proposals", 0)),
        "stop_legal_reason": str(policy_state.get("stop_legal_reason", "unspecified")),
        "exploration_stop_masked": bool(
            policy_state.get("exploration_stop_masked", False)
        ),
    }
    return dict(candidate), record


def organization_candidate_distribution(
    model: InformationRealizationPolicy,
    encoder: FrozenTextEncoder,
    policy_state: Mapping[str, Any],
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Return the actor prior for structural choices at one scheduled node."""

    resolved_device = device or next(model.parameters()).device
    context = _policy_context(model, encoder, policy_state, resolved_device)
    candidates = list(policy_state.get("candidates", []))
    candidate_ids = [str(value["id"]) for value in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("organization candidate ids must be unique")
    values = _candidate_distribution(model, policy_state, context, resolved_device)
    probabilities = values["candidate_probabilities"]
    paths = [{
        "candidate_id": candidate_id,
        "context_node_id": context["context_node_id"],
        "probability": float(probability.detach().cpu()),
    } for candidate_id, probability in zip(candidate_ids, probabilities)
        if bool(probability > 0)]
    return {
        "candidate_priors": {
            candidate_id: float(probability.detach().cpu())
            for candidate_id, probability in zip(candidate_ids, probabilities)
        },
        "actor_paths": paths,
        "action_priors": _action_probability_summary(candidates, probabilities),
    }


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
        context_id = str(record.get("context_node_id", record.get("contextNodeId")))
        if context_id != context["context_node_id"]:
            raise ValueError("recorded context node differs from scheduler context")
        values = _candidate_distribution(model, policy_state, context, device)
        candidate_index = [str(value["id"]) for value in values["candidates"]].index(
            str(record.get("candidate_id", record.get("candidateId")))
        )
    except ValueError as error:
        raise ValueError("recorded organization choice is not available during replay") from error
    return values["candidate_log_probs"][candidate_index]


def replay_joint_log_probabilities(
    model: InformationRealizationPolicy,
    encoder: FrozenTextEncoder,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> Tensor:
    """Exact replay for a microbatch using one disjoint-graph actor pass."""
    if not records:
        return torch.zeros(0, dtype=torch.float32, device=device)
    policy_states = []
    graph_inputs = []
    for record in records:
        policy_state = record.get("policy_state", record.get("policyState"))
        if not isinstance(policy_state, Mapping):
            raise ValueError("policy record is missing policy_state")
        graph = policy_state.get("event_graph")
        if not isinstance(graph, Mapping):
            raise ValueError("organization policy state is missing event_graph")
        policy_states.append(policy_state)
        graph_inputs.append(tuple(
            value.to(device) for value in graph_tensors(dict(graph), encoder)
        ))
    encoded = model.encode_graph_batch(graph_inputs)
    results = []
    for record, policy_state, (node_states, graph_state) in zip(
        records, policy_states, encoded
    ):
        context = _policy_context_from_encoding(
            model, encoder, policy_state, device, node_states, graph_state
        )
        try:
            context_id = str(record.get("context_node_id", record.get("contextNodeId")))
            if context_id != context["context_node_id"]:
                raise ValueError("recorded context node differs from scheduler context")
            values = _candidate_distribution(model, policy_state, context, device)
            candidate_index = [str(value["id"]) for value in values["candidates"]].index(
                str(record.get("candidate_id", record.get("candidateId")))
            )
        except ValueError as error:
            raise ValueError(
                "recorded organization choice is not available during replay"
            ) from error
        results.append(values["candidate_log_probs"][candidate_index])
    return torch.stack(results)


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
    return _policy_context_from_encoding(
        model, encoder, policy_state, device, node_states, graph_state
    )


def _policy_context_from_encoding(
    model: InformationRealizationPolicy,
    encoder: FrozenTextEncoder,
    policy_state: Mapping[str, Any],
    device: torch.device,
    node_states: Tensor,
    graph_state: Tensor,
) -> Dict[str, Any]:
    graph = policy_state.get("event_graph")
    if not isinstance(graph, Mapping):
        raise ValueError("organization policy state is missing event_graph")
    graph_nodes = list(graph.get("nodes", []))
    node_index = {str(value["id"]): index for index, value in enumerate(graph_nodes)}
    context_node_id = str(policy_state.get("context_node_id") or "")
    if not context_node_id or context_node_id not in node_index:
        raise ValueError("scheduler context node must reference an event graph node")
    context_node = policy_state.get("context_node")
    if not isinstance(context_node, Mapping) or str(context_node.get("id") or "") != context_node_id:
        raise ValueError("organization policy state must include the exact node-local context")
    resources = _organization_resource_tensor(dict(policy_state.get("resources", {})), device)
    temperature = float(policy_state.get("organization_temperature", 1.0))
    if temperature <= 0:
        raise ValueError("organization temperature must be positive during replay")
    candidates = list(policy_state.get("candidates", []))
    if not candidates:
        raise ValueError("organization policy state contains no candidates")
    kinds = [str(value.get("kind")) for value in candidates]
    if len(set(kinds)) != len(kinds):
        raise ValueError("shared Controller requires at most one candidate per structural action")
    if any(str(value.get("actor_node_id") or "") != context_node_id for value in candidates):
        raise ValueError("all organization candidates must belong to the scheduler context node")
    candidate_embeddings = encoder.encode([_candidate_text(value) for value in candidates]).to(device)
    return {
        "node_states": node_states,
        "graph_state": graph_state,
        "node_index": node_index,
        "context_node_id": context_node_id,
        "context_node_state": node_states[node_index[context_node_id]],
        "candidate_embeddings": candidate_embeddings,
        "resources": resources,
    }


def _candidate_distribution(
    model: InformationRealizationPolicy,
    policy_state: Mapping[str, Any],
    context: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    node_index = context["node_index"]
    context_node_id = str(context["context_node_id"])
    candidates = list(policy_state.get("candidates", []))
    if not candidates:
        raise ValueError("organization policy state contains no candidates")
    kinds = [str(value.get("kind")) for value in candidates]
    if len(set(kinds)) != len(kinds):
        raise ValueError("shared Controller requires at most one candidate per structural action")
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
    actor_values = [str(value.get("actor_node_id") or "") == context_node_id
                    for value in candidates]
    legal_values = [legal and actor for legal, actor in zip(legal_values, actor_values)]
    if not any(legal_values):
        raise ValueError(f"context node {context_node_id} has no legal organization candidates")
    legal_mask = torch.tensor(legal_values, dtype=torch.bool, device=device)
    actor_mask = torch.tensor(actor_values, dtype=torch.bool, device=device)
    candidate_features = torch.tensor(
        [_candidate_features(value) for value in candidates], dtype=torch.float32, device=device
    )
    candidate_logits = model.candidate_logits(
        context["graph_state"],
        context["context_node_state"],
        context["candidate_embeddings"],
        action_type_indices([str(value["kind"]) for value in candidates], device),
        candidate_features,
        context["resources"],
        actor_mask,
    )
    temperature = float(policy_state.get("organization_temperature", 1.0))
    if temperature <= 0:
        raise ValueError("organization temperature must be positive during replay")
    candidate_raw_log_probs, raw_outer, raw_inner = _categorical_action_log_probs(
        kinds, candidate_logits, actor_mask, temperature)
    candidate_log_probs, masked_outer, masked_inner = _categorical_action_log_probs(
        kinds, candidate_logits, legal_mask, temperature)
    return {
        "candidates": candidates,
        "candidate_raw_probabilities": candidate_raw_log_probs.exp(),
        "candidate_log_probs": candidate_log_probs,
        "candidate_probabilities": candidate_log_probs.exp(),
        "raw_outer_action_log_probabilities": raw_outer,
        "masked_outer_action_log_probabilities": masked_outer,
        "raw_inner_candidate_log_probabilities": raw_inner,
        "masked_inner_candidate_log_probabilities": masked_inner,
    }


def _categorical_action_log_probs(
    kinds: Sequence[str], logits: Tensor, mask: Tensor, temperature: float,
) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
    """One exact masked categorical distribution over the six Controller actions."""
    if len(set(kinds)) != len(kinds):
        raise ValueError("Controller action candidates must be unique")
    scaled = logits / temperature
    masked = scaled.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
    if not bool(mask.any()):
        raise ValueError("categorical Controller policy has no legal action")
    joint = torch.log_softmax(masked, dim=0)
    outer = {kind: joint[index] for index, kind in enumerate(kinds) if bool(mask[index])}
    inner = torch.full_like(logits, torch.finfo(logits.dtype).min)
    inner[mask.bool()] = 0.0
    return joint, outer, inner


def _action_probability_summary(
    candidates: Sequence[Mapping[str, Any]], probabilities: Tensor
) -> Dict[str, float]:
    result = {kind: 0.0 for kind in ORGANIZATION_ACTIONS}
    for candidate, probability in zip(candidates, probabilities.detach().cpu()):
        kind = str(candidate.get("kind"))
        if kind in result:
            result[kind] += float(probability)
    return result


def _spawn_mode_probability_summary(
    candidates: Sequence[Mapping[str, Any]], probabilities: Tensor
) -> Dict[str, float]:
    result = {"acquire_external": 0.0, "organize_knowledge": 0.0}
    for candidate, probability in zip(candidates, probabilities.detach().cpu()):
        mode = str(candidate.get("realization_mode") or "")
        if mode in result:
            result[mode] += float(probability)
    return result


def _candidate_features(value: Mapping[str, Any]) -> List[float]:
    kind = str(value.get("kind") or "")
    return [
        0.0,
        float(kind == "DERIVE_INFO"),
        float(kind in {"RETURN", "FINISH"}),
        float(kind in {"DERIVE_INFO", "DERIVE_ORG"}),
    ]


def _candidate_text(value: Mapping[str, Any]) -> str:
    # Semantic descriptions and payloads belong to the frozen Worker.  Encoding
    # only the action token prevents the Controller from selecting child prompts,
    # commands, reports, connection targets, or prune targets indirectly.
    return str(value.get("kind") or "")


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
