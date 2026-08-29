from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from .lhtb import require_training_task
from .lhtb_transitions import build_state_transition_samples
from .model import FrozenTextEncoder
from .organization import LHTB_POLICY_INTERFACE_REVISION, ORGANIZATION_ACTIONS
from .organization_model import InformationRealizationPolicy, LHTB_ACTOR_MODEL_REVISION
from .organization_replay import _candidate_text, replay_joint_log_probabilities


LHTB_GROUP_SIZE = 8
LHTB_GROUP_INPUT_TOKEN_WARNING = 15_000_000


class LHTBProcessGRPOTrainer:
    """On-policy node-action GRPO using only one official terminal reward per rollout."""

    def __init__(self, checkpoint: Path, manifest: Sequence[Mapping[str, object]],
                 learning_rate: float = 1e-4, device_name: str = "cpu", seed: int = 20260820,
                 encoder: Any | None = None, resume: bool = False,
                 actor_microbatch: int = 64) -> None:
        if actor_microbatch <= 0:
            raise ValueError("LHTB actor microbatch size must be positive")
        torch.manual_seed(seed)
        self.checkpoint = checkpoint
        self.manifest = list(manifest)
        self.device = torch.device(device_name)
        self.encoder = encoder or FrozenTextEncoder(device=device_name, local_only=True)
        self.actor = InformationRealizationPolicy().to(self.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=learning_rate)
        self.actor_microbatch = actor_microbatch
        self.groups = 0
        self.actor_steps = 0
        self.updated_group_ids: set[str] = set()
        self.history: List[Mapping[str, Any]] = []
        if resume and checkpoint.exists():
            self._restore()

    def update_group(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        self._validate(records)
        self._precache_group_text(records)
        group_id = str(records[0]["group_id"])
        rewards = [float(value["terminal_reward"]) for value in records]
        # Group statistics are computed in float64 so a constant decimal score
        # (for example eight copies of 0.4) cannot acquire a machine-dependent
        # float32 standard deviation and trigger a spurious actor update.
        reward_tensor = torch.tensor(rewards, dtype=torch.float64, device=self.device)
        reward_mean = reward_tensor.mean()
        reward_std = reward_tensor.std(unbiased=False)
        has_preference_signal = max(rewards) - min(rewards) > 1e-12
        trajectory_advantages = (
            torch.zeros_like(reward_tensor) if not has_preference_signal
            else (reward_tensor - reward_mean) / (reward_std + 1e-8)
        ).to(dtype=torch.float32)

        actor_updated = False
        actor_loss_value: float | None = None
        if has_preference_signal:
            actor_updated = True
            actor_loss_value = self._update_actor(records, trajectory_advantages)
            self.actor_steps += 1

        transition_samples: List[Mapping[str, Any]] = []
        for trajectory, reward in zip(records, rewards):
            states = list(trajectory["process_states"])
            metadata = {
                "group_id": group_id, "trajectory_id": trajectory.get("id"),
                "task_id": trajectory.get("task_id"),
                "rollout_index": trajectory.get("rollout_index"),
                "epoch": trajectory.get("epoch"), "split": trajectory.get("split"),
                "policy_revision": trajectory.get("policy_revision"),
                "terminal_reward": reward,
            }
            transition_samples.extend(build_state_transition_samples(states, metadata=metadata))
        group_input_tokens = sum(int(
            (list(value["process_states"])[-1].get("usage") or {}).get("inputTokens", 0)
        ) for value in records)

        self.groups += 1
        self.updated_group_ids.add(group_id)
        result: Dict[str, Any] = {
            "group_id": group_id, "actor_updated": actor_updated,
            "actor_skip_reason": None if actor_updated
            else "zero_official_terminal_reward_variance",
            "actor_loss": actor_loss_value,
            "terminal_reward_std": float(reward_std.detach().cpu()),
            "terminal_reward_mean": float(reward_mean.detach().cpu()),
            "trajectory_advantages": trajectory_advantages.detach().cpu().tolist(),
            "reward_source": "official_lhtb_terminal_verifier_only",
            "credit_assignment": "same_group_relative_advantage_for_every_node_decision",
            "group_input_tokens": group_input_tokens,
            "input_token_warning_threshold": LHTB_GROUP_INPUT_TOKEN_WARNING,
            "input_token_warning": group_input_tokens > LHTB_GROUP_INPUT_TOKEN_WARNING,
        }
        self.history.append(result)
        self.save()
        return {**result, "transition_samples": transition_samples}

    def _update_actor(self, records: Sequence[Mapping[str, Any]],
                      trajectory_advantages: torch.Tensor,
                      clip_epsilon: float = 0.2) -> float:
        """Apply the exact length-normalized GRPO loss in bounded decision batches."""
        self.actor_optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for trajectory_index, trajectory in enumerate(records):
            policy_records = list(trajectory["policy_records"])
            decision_count = len(policy_records)
            if decision_count == 0:
                raise ValueError("every direct actor trajectory requires at least one decision")
            advantage = trajectory_advantages[trajectory_index].detach()
            scale = 1.0 / (len(records) * decision_count)
            for start in range(0, decision_count, self.actor_microbatch):
                batch = policy_records[start:start + self.actor_microbatch]
                current = replay_joint_log_probabilities(
                    self.actor, self.encoder, batch, self.device
                )
                old = torch.tensor([
                    float(record.get("masked_old_log_probability",
                                     record.get("maskedOldLogProbability")))
                    for record in batch
                ], dtype=current.dtype, device=self.device)
                ratio = torch.exp(current - old.detach())
                clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                surrogate = torch.minimum(ratio * advantage, clipped * advantage)
                batch_loss = -surrogate.sum() * scale
                batch_loss.backward()
                total_loss += float(batch_loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        return total_loss

    def _precache_group_text(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Batch frozen text encoding for all node-actor replay inputs."""
        precache = getattr(self.encoder, "precache", None)
        if not callable(precache):
            return

        def texts() -> Any:
            for trajectory in records:
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
                if behavior_kind != "actor":
                    raise ValueError("LHTB GRPO accepts only direct actor behavior, never MCTS")
                policy_state = record.get("policy_state", record.get("policyState"))
                if not isinstance(policy_state, Mapping):
                    raise ValueError("actor record is missing its node-local policy state")
                context_node = str(record.get(
                    "context_node_id", record.get("contextNodeId", "")
                ))
                if not context_node or context_node != str(policy_state.get("context_node_id", "")):
                    raise ValueError("actor decision must identify its exact scheduler context node")
                local_context = policy_state.get("context_node")
                if not isinstance(local_context, Mapping) or str(
                    local_context.get("id", "")
                ) != context_node:
                    raise ValueError("actor decision must save the node's complete local context")
                if not isinstance(local_context.get("ancestry"), Sequence):
                    raise ValueError("actor decision must save the node's explicit ancestry")
                controller_candidates = policy_state.get("candidates")
                if not isinstance(controller_candidates, Sequence) or not controller_candidates:
                    raise ValueError("actor state requires a six-action Controller mask")
                kinds = [str(item.get("kind", "")) for item in controller_candidates
                         if isinstance(item, Mapping)]
                if len(kinds) != len(controller_candidates) or len(set(kinds)) != len(kinds):
                    raise ValueError("Controller state allows at most one candidate per action")
                if any(kind not in ORGANIZATION_ACTIONS for kind in kinds):
                    raise ValueError("actor state contains a Runtime or unknown action kind")
                forbidden_payload_fields = {
                    "conditional_payload", "command", "child_specification", "report",
                    "connection", "target_node_id", "final_output",
                }
                if any(forbidden_payload_fields.intersection(item)
                       for item in controller_candidates if isinstance(item, Mapping)):
                    raise ValueError("Controller action candidates must not contain Worker payloads")
                selected_action = str(record.get(
                    "selected_action", record.get("selectedAction", "")
                ))
                if selected_action not in kinds:
                    raise ValueError("recorded Controller action is absent from its legal mask")
                if record.get("mcts_search_samples", record.get("mctsSearchSamples")):
                    raise ValueError("direct actor trajectories cannot contain MCTS search samples")
                forbidden_search_fields = (
                    "mcts_search_trace", "mctsSearchTrace", "mcts_visit_counts",
                    "mctsVisitCounts", "mcts_behavior_probabilities",
                    "mctsBehaviorProbabilities", "root_target_value", "rootTargetValue",
                    "selected_child_target_value", "selectedChildTargetValue",
                )
                if any(record.get(key) is not None for key in forbidden_search_fields):
                    raise ValueError("direct actor trajectories cannot contain search/value fields")

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "learned_information_realization", "benchmark": "lhtb",
            "objective": "official_terminal_reward_group_relative_grpo",
            "actor_model_revision": LHTB_ACTOR_MODEL_REVISION,
            "sampling_protocol": "direct_node_actor_on_policy_no_search",
            "inference_protocol": "shared_actor_invoked_separately_for_each_scheduled_node",
            "controller_action_space": list(ORGANIZATION_ACTIONS),
            "worker_policy": "frozen_semantic_payload_generator",
            "payload_policy": "excluded_from_actor_input_and_grpo_probability",
            "reward_source": "official_lhtb_terminal_verifier_only",
            "actor_microbatch": self.actor_microbatch,
            "groups": self.groups, "actor_steps": self.actor_steps,
            "updated_group_ids": sorted(self.updated_group_ids), "history": self.history,
        }

    def save(self) -> None:
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
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
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        metadata = payload.get("metadata", {})
        self.groups = int(metadata.get("groups", 0))
        self.actor_steps = int(metadata.get("actor_steps", 0))
        self.updated_group_ids = set(metadata.get("updated_group_ids", []))
        self.history = list(metadata.get("history", []))
