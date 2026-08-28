from __future__ import annotations

import json
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
from .organization_model import InformationRealizationPolicy
from .organization_replay import replay_joint_log_probabilities
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
                 actor_microbatch: int = 64, value_microbatch: int = 64) -> None:
        if actor_microbatch <= 0 or value_microbatch <= 0:
            raise ValueError("LHTB microbatch sizes must be positive")
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

        actor_updated = shaped_std > 1e-8
        actor_loss_value: float | None = None
        if actor_updated:
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss_value = 0.0
            for trajectory, step_advantages in zip(records, advantages):
                policy_records = list(trajectory["policy_records"])
                if len(policy_records) != len(step_advantages):
                    raise ValueError("policy decisions must align with process credit")
                behavior = torch.tensor(
                    [float(record.get("masked_old_log_probability",
                                      record.get("maskedOldLogProbability")))
                     for record in policy_records],
                    dtype=torch.float32, device=self.device,
                )
                step_advantages = step_advantages.to(self.device)
                for start in range(0, len(policy_records), self.actor_microbatch):
                    stop = start + self.actor_microbatch
                    current = replay_joint_log_probabilities(
                        self.actor, self.encoder, policy_records[start:stop], self.device
                    )
                    old = behavior[start:stop]
                    advantage = step_advantages[start:stop].to(dtype=current.dtype)
                    ratio = torch.exp(current - old.detach())
                    clipped = torch.clamp(ratio, 0.8, 1.2)
                    loss = -torch.minimum(
                        ratio * advantage, clipped * advantage
                    ).sum() / (len(records) * len(policy_records))
                    loss.backward()
                    actor_loss_value += float(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self.actor_steps += 1

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss_value = 0.0
        for states, reward in zip(state_sequences, rewards):
            scale = 1.0 / (len(records) * len(states))
            for start in range(0, len(states), self.value_microbatch):
                predictions = self._predict_batch(
                    self.value, states[start:start + self.value_microbatch]
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
            "actor_skip_reason": None if actor_updated else "zero_shaped_return_variance",
            "actor_loss": actor_loss_value, "value_loss": value_loss_value,
            "terminal_reward_std": float(torch.tensor(rewards).std(unbiased=False)),
            "shaped_return_mean": return_mean, "shaped_return_std": shaped_std,
            "target_revision": self.groups - 1, "ema_decay": self.ema_decay,
            "process_rewards": process_rewards, "target_values": target_values,
            "transition_reward_summary": transition_reward_summary,
            "group_input_tokens": group_input_tokens,
            "input_token_warning_threshold": LHTB_GROUP_INPUT_TOKEN_WARNING,
            "input_token_warning": group_input_tokens > LHTB_GROUP_INPUT_TOKEN_WARNING,
        }
        self.history.append(result)
        self.save()
        return {**result, "transition_samples": transition_samples}

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
                            yield str(candidate.get("description") or candidate.get("kind") or "")

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
                behavior_kind = record.get("behavior_policy", record.get("behaviorPolicy"))
                if behavior_kind == "mcts_puct":
                    self._validate_mcts_behavior(record, int(value["policy_revision"]))
        search_modes = {
            str(((record.get("policy_state") or record.get("policyState") or {})
                 .get("topology_search") or {}).get("mode"))
            for value in records for record in list(value["policy_records"])
        }
        if search_modes != {"mcts_unconstrained"}:
            raise ValueError(
                f"formal LHTB training requires unconstrained MCTS topology search; "
                f"observed {sorted(search_modes)}"
            )

    @staticmethod
    def _validate_mcts_behavior(record: Mapping[str, Any], policy_revision: int) -> None:
        probabilities = record.get(
            "mcts_behavior_probabilities", record.get("mctsBehaviorProbabilities")
        )
        visits = record.get("mcts_visit_counts", record.get("mctsVisitCounts"))
        trace = record.get("mcts_search_trace", record.get("mctsSearchTrace"))
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
        if not isinstance(trace, Sequence) or not any(
            isinstance(value, Mapping) and value.get("phase") == "backup" for value in trace
        ):
            raise ValueError("MCTS search trace has no PUCT backup")

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "learned_information_realization", "benchmark": "lhtb",
            "objective": "official_terminal_reward_with_ema_delta_value_credit",
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
