from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .lhtb import require_training_task
from .lhtb_transitions import (
    build_decision_transition_samples,
    build_state_transition_samples,
)
from .model import FrozenTextEncoder, epistemic_state_graph, graph_tensors
from .organization import LHTB_POLICY_INTERFACE_REVISION
from .organization_model import InformationRealizationPolicy, LHTB_ACTOR_MODEL_REVISION
from .organization_replay import _candidate_text, replay_joint_log_probabilities
from .value_model import (
    EpistemicValueModel,
    LHTB_VALUE_MODEL_REVISION,
    constant_value_output,
    make_ema_target,
    process_credit,
    trajectory_weighted_advantages,
    update_ema,
)


LHTB_GROUP_SIZE = 8
LHTB_GROUP_INPUT_TOKEN_WARNING = 15_000_000


def dense_clipped_policy_loss(
    current: Sequence[Tensor], behavior: Sequence[Tensor], advantages: Sequence[Tensor],
    clip_epsilon: float = 0.2,
) -> Tensor:
    if not (len(current) == len(behavior) == len(advantages)) or not current:
        raise ValueError("dense policy inputs must have equal non-zero trajectory count")
    values: List[Tensor] = []
    for new, old, advantage in zip(current, behavior, advantages):
        advantage = advantage.to(device=new.device, dtype=new.dtype)
        if new.shape != old.shape or new.shape != advantage.shape or new.numel() == 0:
            raise ValueError("dense policy inputs must align at every decision")
        ratio = torch.exp(new - old.detach())
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        values.append(torch.minimum(ratio * advantage, clipped * advantage).mean())
    return -torch.stack(values).mean()


class LHTBProcessGRPOTrainer:
    """Single-pass LHTB update: frozen EMA credit, actor, critic, then EMA."""

    def __init__(self, checkpoint: Path, manifest: Sequence[Mapping[str, object]],
                 learning_rate: float = 1e-4, value_learning_rate: float = 1e-4,
                 ema_decay: float = 0.99, device_name: str = "cpu", seed: int = 20260820,
                 encoder: Any | None = None, resume: bool = False,
                 actor_microbatch: int = 64, value_microbatch: int = 64,
                 value_state_sample_limit: int | None = 256) -> None:
        if actor_microbatch <= 0 or value_microbatch <= 0:
            raise ValueError("LHTB microbatch sizes must be positive")
        if value_state_sample_limit is not None and value_state_sample_limit <= 0:
            raise ValueError("LHTB value state sample limit must be positive or None")
        torch.manual_seed(seed)
        self.checkpoint = checkpoint
        self.manifest = list(manifest)
        self.device = torch.device(device_name)
        self.encoder = encoder or FrozenTextEncoder(device=device_name, local_only=True)
        self.actor = InformationRealizationPolicy().to(self.device)
        self.value = EpistemicValueModel().to(self.device)
        self.target = make_ema_target(self.value).to(self.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=value_learning_rate)
        self.ema_decay = ema_decay
        self.actor_microbatch = actor_microbatch
        self.value_microbatch = value_microbatch
        self.value_state_sample_limit = value_state_sample_limit
        self.groups = 0
        self.actor_steps = 0
        self.value_steps = 0
        self.updated_group_ids: set[str] = set()
        self.history: List[Mapping[str, Any]] = []
        if resume and checkpoint.exists():
            self._restore()

    def update_group(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        self._validate(records)
        self._precache_group_text(records)
        group_id = str(records[0]["group_id"])
        rewards = [float(value["terminal_reward"]) for value in records]
        state_sequences = [list(value["process_states"]) for value in records]
        decision_states = [self._decision_states(states, list(value["policy_records"]))
                           for states, value in zip(state_sequences, records)]

        # The target is frozen for this entire group. Every adjacent M_t -> M_{t+1}
        # transition is scored for audit; actor credit sums the same potential
        # differences over event-driven decision spans.
        target_constant = constant_value_output(self.target)
        with torch.no_grad():
            full_target_values = [
                [target_constant] * len(states) if target_constant is not None
                else [
                    float(value) for start in range(0, len(states), 32)
                    for value in self._predict_batch(
                        self.target, states[start:start + 32]
                    ).detach().cpu()
                ]
                for states in state_sequences
            ]
        target_values = [self._values_for_states(states, values, selected)
                         for states, values, selected in zip(
                             state_sequences, full_target_values, decision_states
                         )]
        process_rewards, returns = process_credit(target_values, rewards)
        advantages, return_mean, shaped_std = trajectory_weighted_advantages(returns)

        actor_updated = False
        actor_loss_value: float | None = None
        actor_trajectory_losses: List[Tensor] = []
        search_training_summary = {
            "states": 0, "edges": 0, "positive": 0, "negative": 0, "zero": 0,
            "official_selected_edge_overrides": 0,
        }
        for trajectory, raw_returns, step_advantages in zip(records, returns, advantages):
            policy_records = list(trajectory["policy_records"])
            if len(policy_records) != len(step_advantages):
                raise ValueError("policy decisions must align with process credit")
            decision_losses: List[Tensor] = []
            for record, actual_return, actual_advantage in zip(
                policy_records, raw_returns, step_advantages.tolist()
            ):
                local_losses = self._saved_search_policy_losses(
                    record, float(actual_return), search_training_summary
                )
                if local_losses:
                    decision_losses.append(torch.stack(local_losses).mean())
                    continue
                if abs(float(actual_advantage)) <= 1e-8:
                    continue
                current = replay_joint_log_probabilities(
                    self.actor, self.encoder, [record], self.device
                )[0]
                old = torch.tensor(float(record.get(
                    "masked_old_log_probability", record.get("maskedOldLogProbability")
                )), dtype=current.dtype, device=self.device)
                advantage = torch.tensor(float(actual_advantage), dtype=current.dtype,
                                         device=self.device)
                ratio = torch.exp(current - old.detach())
                clipped = torch.clamp(ratio, 0.8, 1.2)
                decision_losses.append(-torch.minimum(ratio * advantage, clipped * advantage))
            if decision_losses:
                actor_trajectory_losses.append(torch.stack(decision_losses).mean())
        if actor_trajectory_losses:
            actor_updated = True
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(actor_trajectory_losses).mean()
            loss.backward()
            actor_loss_value = float(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self.actor_steps += 1

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss_value = 0.0
        value_sampling = []
        for trajectory_index, (states, reward) in enumerate(zip(state_sequences, rewards)):
            selected_indices = self._value_state_indices(
                len(states), group_id, trajectory_index
            )
            selected_states = [states[index] for index in selected_indices]
            value_sampling.append({
                "trajectory_id": records[trajectory_index].get("id"),
                "total_states": len(states), "sampled_states": len(selected_states),
                "sampled_indices": selected_indices,
            })
            # A uniform sample mean is an unbiased estimator of the original
            # per-trajectory mean. Every trajectory still has total weight 1/G.
            scale = 1.0 / (len(records) * len(selected_states))
            for start in range(0, len(selected_states), self.value_microbatch):
                predictions = self._predict_batch(
                    self.value, selected_states[start:start + self.value_microbatch]
                )
                target = torch.full_like(predictions, reward)
                loss = F.huber_loss(predictions, target, reduction="sum") * scale
                loss.backward()
                value_loss_value += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_optimizer.step()
        self.value_steps += 1
        update_ema(self.target, self.value, self.ema_decay)

        transition_samples: List[Mapping[str, Any]] = []
        for trajectory, states, values, reward in zip(
            records, state_sequences, full_target_values, rewards
        ):
            metadata = {
                "group_id": group_id, "trajectory_id": trajectory.get("id"),
                "task_id": trajectory.get("task_id"),
                "rollout_index": trajectory.get("rollout_index"),
                "epoch": trajectory.get("epoch"), "split": trajectory.get("split"),
                "policy_revision": trajectory.get("policy_revision"),
                "target_value_revision": self.groups,
            }
            transition_samples.extend(build_state_transition_samples(
                states, values, reward, metadata
            ))
            transition_samples.extend(build_decision_transition_samples(
                states, list(trajectory["policy_records"]), values, reward, metadata
            ))
        transition_reward_summary = self._transition_reward_summary(transition_samples)
        group_input_tokens = sum(int(
            (list(value["process_states"])[-1].get("usage") or {}).get("inputTokens", 0)
        ) for value in records)

        self.groups += 1
        self.updated_group_ids.add(group_id)
        result: Dict[str, Any] = {
            "group_id": group_id, "actor_updated": actor_updated,
            "actor_skip_reason": None if actor_updated
            else "zero_real_and_saved_search_advantage_variance",
            "actor_loss": actor_loss_value, "value_loss": value_loss_value,
            "mcts_saved_search_training": search_training_summary,
            "terminal_reward_std": float(torch.tensor(rewards).std(unbiased=False)),
            "shaped_return_mean": return_mean, "shaped_return_std": shaped_std,
            "target_revision": self.groups - 1, "ema_decay": self.ema_decay,
            "process_rewards": process_rewards, "target_values": target_values,
            "transition_reward_summary": transition_reward_summary,
            "value_state_sampling": {
                "estimator": "uniform_without_replacement_equal_trajectory_unbiased",
                "sample_limit": self.value_state_sample_limit,
                "total_states": sum(value["total_states"] for value in value_sampling),
                "sampled_states": sum(value["sampled_states"] for value in value_sampling),
                "trajectories": value_sampling,
            },
            "group_input_tokens": group_input_tokens,
            "input_token_warning_threshold": LHTB_GROUP_INPUT_TOKEN_WARNING,
            "input_token_warning": group_input_tokens > LHTB_GROUP_INPUT_TOKEN_WARNING,
        }
        self.history.append(result)
        self.save()
        return {**result, "transition_samples": transition_samples}

    def _saved_search_policy_losses(
        self, record: Mapping[str, Any], actual_return: float,
        summary: Dict[str, int],
    ) -> List[Tensor]:
        """Replay MCTS edges saved by the sampler; never execute search here."""
        behavior_kind = record.get("behavior_policy", record.get("behaviorPolicy"))
        if behavior_kind != "mcts_puct":
            return []
        raw_samples = record.get("mcts_search_samples", record.get("mctsSearchSamples"))
        if not isinstance(raw_samples, Sequence) or not raw_samples:
            raise ValueError("MCTS sampling record has no saved structural-edge samples")
        search_states = record.get("mcts_search_states", record.get("mctsSearchStates"))
        if not isinstance(search_states, Mapping) or not search_states:
            raise ValueError("MCTS sampling record has no deduplicated search states")
        grouped: Dict[tuple[str, int], List[Mapping[str, Any]]] = {}
        for sample in raw_samples:
            if not isinstance(sample, Mapping):
                raise ValueError("invalid saved MCTS structural-edge sample")
            state = str(sample.get("state_fingerprint", sample.get("stateFingerprint", "")))
            revision = int(sample.get(
                "target_value_revision", sample.get("targetValueRevision", -1)
            ))
            grouped.setdefault((state, revision), []).append(sample)
        selected_state = str(record.get("state_fingerprint", record.get("stateFingerprint", "")))
        selected_candidate = str(record.get("candidate_id", record.get("candidateId", "")))
        losses: List[Tensor] = []
        for (state_fingerprint, _revision), samples in grouped.items():
            priors = torch.tensor([
                float(value.get("actor_prior", value.get("actorPrior", 0.0)))
                for value in samples
            ], dtype=torch.float32, device=self.device)
            if bool((priors <= 0).any()) or not bool(torch.isfinite(priors).all()):
                raise ValueError("saved MCTS actor priors must be finite and positive")
            priors = priors / priors.sum()
            action_values = []
            replay_records = []
            old_logs = []
            for sample in samples:
                candidate_id = str(sample.get("candidate_id", sample.get("candidateId", "")))
                value = float(sample.get(
                    "backed_up_advantage", sample.get("backedUpAdvantage", 0.0)
                ))
                if state_fingerprint == selected_state and candidate_id == selected_candidate:
                    value = actual_return
                    summary["official_selected_edge_overrides"] += 1
                action_values.append(value)
                policy_state_fingerprint = str(sample.get(
                    "policy_state_fingerprint", sample.get("policyStateFingerprint", "")
                ))
                policy_state = search_states.get(policy_state_fingerprint)
                if not isinstance(policy_state, Mapping):
                    raise ValueError("saved MCTS edge references a missing search policy state")
                context_node = sample.get("context_node_id", sample.get("contextNodeId"))
                replay_records.append({
                    "policy_state": policy_state, "context_node_id": context_node,
                    "candidate_id": candidate_id,
                })
                old_logs.append(float(sample.get(
                    "old_actor_log_probability", sample.get("oldActorLogProbability")
                )))
            values = torch.tensor(action_values, dtype=torch.float32, device=self.device)
            mean = (values * priors).sum()
            variance = ((values - mean).square() * priors).sum()
            deviation = torch.sqrt(variance)
            summary["states"] += 1
            summary["edges"] += len(samples)
            summary["positive"] += sum(value > 1e-8 for value in action_values)
            summary["negative"] += sum(value < -1e-8 for value in action_values)
            summary["zero"] += sum(abs(value) <= 1e-8 for value in action_values)
            if float(deviation.detach().cpu()) <= 1e-8:
                continue
            advantages = (values - mean) / (deviation + 1e-8)
            current = replay_joint_log_probabilities(
                self.actor, self.encoder, replay_records, self.device
            )
            old = torch.tensor(old_logs, dtype=current.dtype, device=self.device)
            ratio = torch.exp(current - old.detach())
            clipped = torch.clamp(ratio, 0.8, 1.2)
            objective = torch.minimum(ratio * advantages, clipped * advantages)
            losses.append(-(objective * priors).sum())
        return losses

    def _value_state_indices(
        self, state_count: int, group_id: str, trajectory_index: int
    ) -> List[int]:
        if state_count <= 0:
            raise ValueError("value training requires at least one process state")
        limit = self.value_state_sample_limit
        if limit is None or state_count <= limit:
            return list(range(state_count))
        digest = hashlib.sha256(
            f"{group_id}:{self.groups}:{trajectory_index}".encode("utf-8")
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "big") % (2 ** 63 - 1))
        return sorted(torch.randperm(
            state_count, generator=generator
        )[:limit].tolist())

    def _predict(self, model: EpistemicValueModel, state: Mapping[str, Any]) -> Tensor:
        graph = state.get("event_graph")
        if not isinstance(graph, Mapping):
            graph = epistemic_state_graph(state)
        tensors = [value.to(self.device) for value in graph_tensors(dict(graph), self.encoder)]
        return model(*tensors)

    def _predict_batch(
        self, model: EpistemicValueModel, states: Sequence[Mapping[str, Any]]
    ) -> Tensor:
        graphs = []
        for state in states:
            graph = state.get("event_graph")
            if not isinstance(graph, Mapping):
                graph = epistemic_state_graph(state)
            graphs.append(tuple(
                value.to(self.device) for value in graph_tensors(dict(graph), self.encoder)
            ))
        return model.forward_batch(graphs)

    def _precache_group_text(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Batch frozen text encoding for all actor and critic replay inputs."""
        precache = getattr(self.encoder, "precache", None)
        if not callable(precache):
            return

        def texts() -> Any:
            for trajectory in records:
                for state in trajectory.get("process_states", []):
                    graph = state.get("event_graph") if isinstance(state, Mapping) else None
                    if not isinstance(graph, Mapping):
                        graph = epistemic_state_graph(state)
                    for node in graph.get("nodes", []):
                        if isinstance(node, Mapping):
                            yield str(node.get("text") or node.get("kind") or "")
                for record in trajectory.get("policy_records", []):
                    policy_state = record.get("policy_state", record.get("policyState"))
                    if not isinstance(policy_state, Mapping):
                        continue
                    graph = policy_state.get("event_graph")
                    if isinstance(graph, Mapping):
                        for node in graph.get("nodes", []):
                            if isinstance(node, Mapping):
                                yield str(node.get("text") or node.get("kind") or "")
                    for candidate in policy_state.get("candidates", []):
                        if isinstance(candidate, Mapping):
                            yield _candidate_text(candidate)

        precache(texts())

    def _decision_states(self, states: Sequence[Mapping[str, Any]],
                         policy_records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        by_fingerprint = {str(value.get("fingerprint")): value for value in states}
        selected = []
        for record in policy_records:
            fingerprint = str(record.get("state_fingerprint") or record.get("stateFingerprint") or "")
            if fingerprint not in by_fingerprint:
                raise ValueError("policy decision fingerprint is absent from M_0...M_T")
            selected.append(by_fingerprint[fingerprint])
        selected.append(states[-1])
        return selected

    @staticmethod
    def _values_for_states(states: Sequence[Mapping[str, Any]], values: Sequence[float],
                           selected: Sequence[Mapping[str, Any]]) -> List[float]:
        by_fingerprint = {str(state.get("fingerprint")): value
                          for state, value in zip(states, values)}
        return [by_fingerprint[str(state.get("fingerprint"))] for state in selected]

    @staticmethod
    def _transition_reward_summary(samples: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        topology = [value for value in samples
                    if value.get("sample_type") == "state_transition"
                    and value.get("topology_changed")]
        return {
            "all_state_transitions": sum(value.get("sample_type") == "state_transition"
                                         for value in samples),
            "decision_transitions": sum(value.get("sample_type") == "decision_transition"
                                        for value in samples),
            "topology_transitions": len(topology),
            "positive_topology_rewards": sum(value.get("reward_sign") == "positive"
                                             for value in topology),
            "negative_topology_rewards": sum(value.get("reward_sign") == "negative"
                                             for value in topology),
            "zero_topology_rewards": sum(value.get("reward_sign") == "zero"
                                         for value in topology),
        }

    def _validate(self, records: Sequence[Mapping[str, Any]]) -> None:
        if len(records) != LHTB_GROUP_SIZE:
            raise ValueError("LHTB on-policy groups require exactly G=8 trajectories")
        group_ids = {str(value.get("group_id", "")) for value in records}
        if len(group_ids) != 1 or "" in group_ids:
            raise ValueError("all trajectories must share one group id")
        group_id = next(iter(group_ids))
        if group_id in self.updated_group_ids:
            raise ValueError(f"group was already optimized: {group_id}")
        task_ids = {str(value.get("task_id", "")) for value in records}
        if len(task_ids) != 1:
            raise ValueError("counterfactual group must share one LHTB task")
        require_training_task(next(iter(task_ids)), self.manifest)
        if any(value.get("benchmark") != "lhtb" or value.get("split") != "train"
               for value in records):
            raise ValueError("trainer accepts only LHTB train trajectories")
        if any(not value.get("complete") or value.get("environment_failure") for value in records):
            raise ValueError("crashed or incomplete trajectories are audit-only")
        if any(value.get("policy_interface_revision") != LHTB_POLICY_INTERFACE_REVISION
               for value in records):
            raise ValueError("LHTB group uses a stale policy-state interface")
        if any(int(value.get("policy_revision", -1)) != self.groups for value in records):
            raise ValueError("LHTB group is stale; sampling must use the current actor revision")
        if any(int(value.get("epoch", -1)) not in range(4) for value in records):
            raise ValueError("formal LHTB training uses epochs zero through three")
        if any(not 0.0 <= float(value.get("terminal_reward", -1)) <= 1.0 for value in records):
            raise ValueError("official LHTB reward must be in [0,1]")
        if {int(value.get("rollout_index", -1)) for value in records} != set(range(8)):
            raise ValueError("rollout indices must be zero through seven")
        if len({int(value.get("organization_seed", -1)) for value in records}) != 8:
            raise ValueError("each LHTB rollout requires a distinct organization seed")
        immutable = ("initial_snapshot_fingerprint", "task_checksum", "runtime_config")
        for key in immutable:
            values = {json.dumps(value.get(key), sort_keys=True) for value in records}
            if len(values) != 1 or values == {"null"}:
                raise ValueError(f"counterfactual trajectories must share {key}")
        environment_values = {
            json.dumps(value.get("environment_digest", value.get("docker_digest")), sort_keys=True)
            for value in records
        }
        if len(environment_values) != 1 or environment_values == {"null"}:
            raise ValueError("counterfactual trajectories must share environment_digest")
        for value in records:
            states = value.get("process_states")
            policy = value.get("policy_records")
            if not isinstance(states, Sequence) or not isinstance(policy, Sequence):
                raise ValueError("trajectory is missing process states or policy records")
            if len(states) < len(policy) + 1:
                raise ValueError("trajectory must save all decision states plus a terminal state")
            self._decision_states(states, policy)
            transitions = value.get("state_transitions")
            if not isinstance(transitions, Sequence) or len(transitions) != len(states) - 1:
                raise ValueError("trajectory must save every adjacent M_t -> M_t+1 transition")
            if any(abs(int(item.get("topology_delta", {}).get("node_count_delta", 0))) > 1
                   for item in transitions if isinstance(item, Mapping)):
                raise ValueError("topology must be derived one node at a time")
            for record in policy:
                if not isinstance(record, Mapping):
                    raise ValueError("policy records must be mappings")
                old_log = float(record.get(
                    "masked_old_log_probability", record.get("maskedOldLogProbability")
                ))
                if not math.isfinite(old_log):
                    raise ValueError("policy record requires a finite saved behavior log-probability")
                behavior_kind = record.get("behavior_policy", record.get("behaviorPolicy"))
                if behavior_kind == "mcts_puct":
                    self._validate_saved_search_samples(record, int(value["policy_revision"]))

    @staticmethod
    def _validate_saved_search_samples(record: Mapping[str, Any], policy_revision: int) -> None:
        """Validate sampler output only; this method never runs or imports MCTS."""
        probabilities = record.get(
            "mcts_behavior_probabilities", record.get("mctsBehaviorProbabilities")
        )
        visits = record.get("mcts_visit_counts", record.get("mctsVisitCounts"))
        samples = record.get("mcts_search_samples", record.get("mctsSearchSamples"))
        search_states = record.get("mcts_search_states", record.get("mctsSearchStates"))
        selected = str(record.get("candidate_id", record.get("candidateId", "")))
        if not isinstance(probabilities, Mapping) or not isinstance(visits, Mapping):
            raise ValueError("MCTS policy record is missing behavior probabilities or visits")
        values = {str(key): float(value) for key, value in probabilities.items()}
        if selected not in values or values[selected] <= 0:
            raise ValueError("MCTS selected candidate has no positive behavior probability")
        if abs(sum(values.values()) - 1.0) > 1e-6:
            raise ValueError("MCTS behavior probabilities must sum to one")
        old_log = float(record.get(
            "masked_old_log_probability", record.get("maskedOldLogProbability")
        ))
        if abs(math.log(values[selected]) - old_log) > 1e-6:
            raise ValueError("MCTS exact old-policy log probability does not match visits")
        if sum(int(value) for value in visits.values()) <= 0:
            raise ValueError("MCTS visit ledger is empty")
        if int(record.get("target_value_revision",
                          record.get("targetValueRevision", -1))) != policy_revision:
            raise ValueError("MCTS search used a stale target-value revision")
        if not isinstance(samples, Sequence) or not samples:
            raise ValueError("MCTS sampler saved no structural-edge training samples")
        if not isinstance(search_states, Mapping) or not search_states:
            raise ValueError("MCTS sampler saved no deduplicated policy-state table")
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError("invalid MCTS structural-edge training sample")
            if sample.get("reward_source", sample.get("rewardSource")) != "frozen_value_bootstrap":
                raise ValueError("counterfactual MCTS edges must identify frozen value bootstrap")
            sample_revision = int(sample.get(
                "target_value_revision", sample.get("targetValueRevision", -1)
            ))
            if sample_revision != policy_revision:
                raise ValueError("saved MCTS edge used a stale target-value revision")
            state_ref = str(sample.get(
                "policy_state_fingerprint", sample.get("policyStateFingerprint", "")
            ))
            if state_ref not in search_states or not isinstance(search_states[state_ref], Mapping):
                raise ValueError("saved MCTS edge is missing replayable policy state")

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "learned_information_realization", "benchmark": "lhtb",
            "objective": "official_terminal_reward_with_ema_delta_value_credit",
            "actor_model_revision": LHTB_ACTOR_MODEL_REVISION,
            "search_role": "sampling_only_saved_edges_replayed_without_search",
            "value_model_revision": LHTB_VALUE_MODEL_REVISION,
            "groups": self.groups, "actor_steps": self.actor_steps,
            "value_steps": self.value_steps, "ema_decay": self.ema_decay,
            "updated_group_ids": sorted(self.updated_group_ids), "history": self.history,
        }

    def save(self) -> None:
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "metadata": self.metadata(),
        }, self.checkpoint)

    def _restore(self) -> None:
        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        actor_revision = payload.get("metadata", {}).get("actor_model_revision")
        if actor_revision != LHTB_ACTOR_MODEL_REVISION:
            raise ValueError(
                "LHTB actor checkpoint is incompatible: expected "
                f"{LHTB_ACTOR_MODEL_REVISION}, found {actor_revision or 'legacy-active-node-routing'}; "
                "initialize a fresh structural-policy checkpoint"
            )
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.value.load_state_dict(payload["value_state_dict"])
        self.target.load_state_dict(payload["target_state_dict"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(payload["value_optimizer_state_dict"])
        metadata = payload.get("metadata", {})
        self.groups = int(metadata.get("groups", 0))
        self.actor_steps = int(metadata.get("actor_steps", 0))
        self.value_steps = int(metadata.get("value_steps", 0))
        self.updated_group_ids = set(metadata.get("updated_group_ids", []))
        self.history = list(metadata.get("history", []))
