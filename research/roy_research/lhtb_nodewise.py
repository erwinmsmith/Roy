from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .lhtb import require_training_task
from .lhtb_value_metrics import _state_graph
from .model import FrozenTextEncoder, graph_tensors
from .organization import LHTB_POLICY_INTERFACE_REVISION, ORGANIZATION_ACTIONS
from .organization_model import InformationRealizationPolicy, LHTB_ACTOR_MODEL_REVISION
from .organization_replay import replay_joint_log_probabilities
from .value_model import EpistemicValueModel, LHTB_VALUE_MODEL_REVISION


NODEWISE_ALGORITHM_REVISION = "forced-finalize-delta-v-nodewise-grpo-v1"
FINALIZE_LABEL_SCHEMA_VERSION = 1
MACRO_GROUP_SCHEMA_VERSION = 1
NODEWISE_GROUP_SIZE = 8
MACRO_BOUNDARIES = {
    "worker_phase_complete",
    "external_observation_recorded",
    "parent_report_integrated",
    "branch_pruned",
    "official_verifier_complete",
}


def build_forced_finalize_label(
    *,
    label_id: str,
    task_id: str,
    split: str,
    process_state: Mapping[str, Any],
    task_utilities: Sequence[float],
    finalizer_revision: str,
    task_checksum: str,
    environment_digest: str,
    checkpoint_id: str,
    clone_provenance: Mapping[str, Any],
    verifier_provenance: Sequence[Mapping[str, Any]],
    sample_seeds: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Build one auditable V(S) target from official forced-finalize probes.

    The fixed finalizer is allowed to read the supplied snapshot, but it may not
    perform another structural action.  Every score must come from the official
    LHTB verifier; no model score or epistemic feature is accepted as a label.
    These values are task utilities U_T, not the derived Controller reward R_t.
    """
    if not label_id or not task_id or not checkpoint_id:
        raise ValueError("forced-finalize labels require stable IDs")
    fingerprint = str(process_state.get("fingerprint") or "")
    if not fingerprint:
        raise ValueError("forced-finalize state requires an immutable fingerprint")
    if not task_utilities:
        raise ValueError("forced-finalize labels require at least one verifier score")
    normalized = [float(value) for value in task_utilities]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized):
        raise ValueError("official LHTB forced-finalize scores must be in [0,1]")
    seeds = list(sample_seeds if sample_seeds is not None else range(len(normalized)))
    if len(seeds) != len(normalized) or len(set(int(value) for value in seeds)) != len(seeds):
        raise ValueError("forced-finalize probes require one unique seed per score")
    if len(verifier_provenance) != len(normalized) or any(
        not str(value.get("harbor_result_path", ""))
        or not str(value.get("harbor_result_sha256", ""))
        for value in verifier_provenance
    ):
        raise ValueError("forced-finalize probes require official Harbor result provenance")
    _validate_clone_provenance(
        clone_provenance, fingerprint, environment_digest,
        context="forced-finalize label",
    )
    return {
        "schema_version": FINALIZE_LABEL_SCHEMA_VERSION,
        "label_id": label_id,
        "benchmark": "lhtb",
        "task_id": task_id,
        "split": split,
        "checkpoint_id": checkpoint_id,
        "state_fingerprint": fingerprint,
        "process_state": dict(process_state),
        "finalizer_revision": finalizer_revision,
        "finalizer_policy": "frozen_finalize_now_no_structural_actions",
        "task_utility_source": "official_lhtb_verifier",
        "task_checksum": task_checksum,
        "environment_digest": environment_digest,
        "clone_provenance": dict(clone_provenance),
        "samples": [
            {"seed": int(seed), "official_lhtb_task_utility": utility,
             **dict(provenance)}
            for seed, utility, provenance in zip(seeds, normalized, verifier_provenance)
        ],
        "k": len(normalized),
        "value_target": sum(normalized) / len(normalized),
    }


class LHTBNodeWiseDeltaVTrainer:
    """Forced-finalize value learning and one-state macro-action GRPO.

    Value fitting and actor updates are deliberately separate operations.  A
    macro-action group is scored by one frozen value revision, and updating the
    actor never mutates that value definition.
    """

    def __init__(
        self,
        checkpoint: Path,
        manifest: Sequence[Mapping[str, object]],
        *,
        learning_rate: float = 1e-4,
        value_learning_rate: float = 1e-4,
        device_name: str = "cpu",
        seed: int = 20260829,
        encoder: Any | None = None,
        resume: bool = False,
    ) -> None:
        torch.manual_seed(seed)
        self.checkpoint = checkpoint
        self.manifest = list(manifest)
        self.device = torch.device(device_name)
        self.encoder = encoder or FrozenTextEncoder(device=device_name, local_only=True)
        self.actor = InformationRealizationPolicy().to(self.device)
        self.value = EpistemicValueModel().to(self.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.AdamW(
            self.value.parameters(), lr=value_learning_rate
        )
        self.actor_revision = 0
        self.value_revision = 0
        self.actor_steps = 0
        self.value_steps = 0
        self.updated_macro_group_ids: set[str] = set()
        self.used_value_label_ids: set[str] = set()
        self.history: List[Mapping[str, Any]] = []
        if resume and checkpoint.exists():
            self._restore()

    def update_value(
        self,
        labels: Sequence[Mapping[str, Any]],
        *,
        epochs: int = 4,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("value epochs and batch size must be positive")
        fresh = self._validate_value_labels(labels)
        self._precache_states([value["process_state"] for value in fresh])
        targets = torch.tensor(
            [float(value["value_target"]) for value in fresh],
            dtype=torch.float32,
            device=self.device,
        )
        self.value.train()
        losses: List[float] = []
        for _ in range(epochs):
            for start in range(0, len(fresh), batch_size):
                batch = fresh[start:start + batch_size]
                predictions = self._value_predictions(
                    [value["process_state"] for value in batch]
                )
                target = targets[start:start + len(batch)]
                loss = F.huber_loss(predictions, target)
                self.value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
                self.value_optimizer.step()
                self.value_steps += 1
                losses.append(float(loss.detach().cpu()))
        self.value.eval()
        with torch.no_grad():
            prediction = self._value_predictions(
                [value["process_state"] for value in fresh]
            ).detach().cpu().tolist()
        target_values = [float(value["value_target"]) for value in fresh]
        metrics = _value_metrics(fresh, prediction, target_values)
        self.value_revision += 1
        self.used_value_label_ids.update(str(value["label_id"]) for value in fresh)
        result = {
            "operation": "value_update",
            "value_revision": self.value_revision,
            "labels": len(fresh),
            "optimizer_steps": len(losses),
            "mean_huber_loss": sum(losses) / max(1, len(losses)),
            **metrics,
        }
        self.history.append(result)
        self.save()
        return result

    def update_macro_group(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        clip_epsilon: float = 0.2,
    ) -> Dict[str, Any]:
        group_id = self._validate_macro_group(records)
        self.value.eval()
        base_state = records[0]["base_state"]
        successor_states = [value["successor_state"] for value in records]
        self._precache_states([base_state, *successor_states])
        with torch.no_grad():
            values = self._value_predictions([base_state, *successor_states])
        base_value = float(values[0].detach().cpu())
        successor_values = values[1:]
        delta = successor_values - values[0]
        deviation = delta.std(unbiased=False)
        has_signal = float(delta.max() - delta.min()) > 1e-12
        advantages = (
            torch.zeros_like(delta)
            if not has_signal
            else (delta - delta.mean()) / (deviation + 1e-8)
        ).detach()
        actor_updated = False
        actor_loss: float | None = None
        if has_signal:
            policy_records = [value["policy_record"] for value in records]
            current = replay_joint_log_probabilities(
                self.actor, self.encoder, policy_records, self.device
            )
            old = torch.tensor(
                [float(value["masked_old_log_probability"]) for value in policy_records],
                dtype=current.dtype,
                device=self.device,
            )
            ratio = torch.exp(current - old.detach())
            clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self.actor_steps += 1
            self.actor_revision += 1
            actor_updated = True
            actor_loss = float(loss.detach().cpu())
        self.updated_macro_group_ids.add(group_id)
        result = {
            "operation": "nodewise_macro_action_update",
            "group_id": group_id,
            "actor_updated": actor_updated,
            "actor_skip_reason": None if actor_updated else "zero_delta_v_variance",
            "actor_loss": actor_loss,
            "actor_revision": self.actor_revision,
            "value_revision": self.value_revision,
            "value_frozen_during_group": True,
            "base_value": base_value,
            "successor_values": [float(value.detach().cpu()) for value in successor_values],
            "derived_process_rewards": [float(value.detach().cpu()) for value in delta],
            "advantages": [float(value.detach().cpu()) for value in advantages],
            "actions": [str(value["selected_action"]) for value in records],
            "derived_reward_definition": "V_psi(S_next)-V_psi(S_base)",
            "derived_reward_source": "frozen_forced_finalize_state_value_increment",
            "task_utility_source": "official_lhtb_verifier_for_value_supervision_only",
        }
        self.history.append(result)
        self.save()
        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "learned_information_realization",
            "benchmark": "lhtb",
            "algorithm_revision": NODEWISE_ALGORITHM_REVISION,
            "actor_model_revision": LHTB_ACTOR_MODEL_REVISION,
            "value_model_revision": LHTB_VALUE_MODEL_REVISION,
            "actor_revision": self.actor_revision,
            "value_revision": self.value_revision,
            "actor_steps": self.actor_steps,
            "value_steps": self.value_steps,
            "training_protocol": "same_state_macro_action_delta_v_grpo_no_mcts",
            "value_label_protocol": "frozen_finalize_now_official_lhtb_task_utility",
            "derived_reward_definition": "R_t=V_psi(S_t+1)-V_psi(S_t)",
            "inference_protocol": "shared_actor_per_actual_node_without_value_or_search",
            "controller_action_space": list(ORGANIZATION_ACTIONS),
            "updated_macro_group_ids": sorted(self.updated_macro_group_ids),
            "used_value_label_ids": sorted(self.used_value_label_ids),
            "history": self.history,
        }

    def save(self) -> None:
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "metadata": self.metadata(),
        }, self.checkpoint)

    def _restore(self) -> None:
        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        metadata = payload.get("metadata", {})
        if metadata.get("algorithm_revision") != NODEWISE_ALGORITHM_REVISION:
            raise ValueError("checkpoint is not a compatible node-wise Delta-V model")
        if metadata.get("actor_model_revision") != LHTB_ACTOR_MODEL_REVISION:
            raise ValueError("node-wise checkpoint has an incompatible actor revision")
        if metadata.get("value_model_revision") != LHTB_VALUE_MODEL_REVISION:
            raise ValueError("node-wise checkpoint has an incompatible value revision")
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.value.load_state_dict(payload["value_state_dict"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(payload["value_optimizer_state_dict"])
        self.actor_revision = int(metadata.get("actor_revision", 0))
        self.value_revision = int(metadata.get("value_revision", 0))
        self.actor_steps = int(metadata.get("actor_steps", 0))
        self.value_steps = int(metadata.get("value_steps", 0))
        self.updated_macro_group_ids = set(metadata.get("updated_macro_group_ids", []))
        self.used_value_label_ids = set(metadata.get("used_value_label_ids", []))
        self.history = list(metadata.get("history", []))

    def _validate_value_labels(
        self, labels: Sequence[Mapping[str, Any]]
    ) -> List[Mapping[str, Any]]:
        fresh = [value for value in labels
                 if str(value.get("label_id", "")) not in self.used_value_label_ids]
        if not fresh:
            raise ValueError("value update contains no fresh forced-finalize labels")
        if len({str(value.get("label_id", "")) for value in fresh}) != len(fresh):
            raise ValueError("forced-finalize label IDs must be unique")
        for value in fresh:
            require_training_task(str(value.get("task_id", "")), self.manifest)
            if value.get("benchmark") != "lhtb" or value.get("split") != "train":
                raise ValueError("value learning accepts only LHTB train snapshots")
            if value.get("task_utility_source") != "official_lhtb_verifier":
                raise ValueError("V(S) labels must come from the official LHTB verifier")
            if value.get("finalizer_policy") != "frozen_finalize_now_no_structural_actions":
                raise ValueError("V(S) labels require the frozen finalize-now policy")
            state = value.get("process_state")
            if not isinstance(state, Mapping) or str(state.get("fingerprint", "")) != str(
                value.get("state_fingerprint", "")
            ):
                raise ValueError("forced-finalize label/state fingerprint mismatch")
            _validate_clone_provenance(
                value.get("clone_provenance"), str(value["state_fingerprint"]),
                str(value.get("environment_digest", "")),
                context="forced-finalize label",
            )
            samples = value.get("samples")
            if not isinstance(samples, Sequence) or len(samples) != int(value.get("k", 0)):
                raise ValueError("forced-finalize label is missing its K verifier probes")
            scores = [float(sample.get("official_lhtb_task_utility", -1))
                      for sample in samples if isinstance(sample, Mapping)]
            if len(scores) != len(samples) or any(not 0.0 <= score <= 1.0 for score in scores):
                raise ValueError("forced-finalize probes require legal official task utilities")
            if any(not str(sample.get("harbor_result_path", ""))
                   or not str(sample.get("harbor_result_sha256", ""))
                   for sample in samples if isinstance(sample, Mapping)):
                raise ValueError("forced-finalize probes require Harbor result provenance")
            if abs(sum(scores) / len(scores) - float(value.get("value_target", -1))) > 1e-9:
                raise ValueError("forced-finalize value target is not the MC score mean")
        return fresh

    def _validate_macro_group(self, records: Sequence[Mapping[str, Any]]) -> str:
        if len(records) != NODEWISE_GROUP_SIZE:
            raise ValueError("node-wise GRPO requires exactly G=8 macro-action outcomes")
        group_ids = {str(value.get("group_id", "")) for value in records}
        if len(group_ids) != 1 or "" in group_ids:
            raise ValueError("macro-action outcomes must share one group ID")
        group_id = next(iter(group_ids))
        if group_id in self.updated_macro_group_ids:
            raise ValueError(f"macro-action group was already optimized: {group_id}")
        task_ids = {str(value.get("task_id", "")) for value in records}
        if len(task_ids) != 1:
            raise ValueError("macro-action outcomes must share one LHTB task")
        require_training_task(next(iter(task_ids)), self.manifest)
        immutable = (
            "base_state_fingerprint", "checkpoint_id", "context_node_id",
            "task_checksum", "environment_digest", "runtime_config",
            "policy_revision", "value_revision",
        )
        for key in immutable:
            values = {json.dumps(value.get(key), sort_keys=True) for value in records}
            if len(values) != 1 or values == {"null"}:
                raise ValueError(f"macro-action outcomes must share {key}")
        if int(records[0]["policy_revision"]) != self.actor_revision:
            raise ValueError("macro-action group is stale relative to the current actor")
        if int(records[0]["value_revision"]) != self.value_revision:
            raise ValueError("macro-action group was not scored by the current frozen value model")
        if {int(value.get("sample_index", -1)) for value in records} != set(range(8)):
            raise ValueError("node-wise group sample indices must be zero through seven")
        if len({int(value.get("organization_seed", -1)) for value in records}) != 8:
            raise ValueError("node-wise group requires eight unique organization seeds")
        for value in records:
            if value.get("benchmark") != "lhtb" or value.get("split") != "train":
                raise ValueError("node-wise GRPO accepts only LHTB train records")
            if value.get("sampling_protocol") != "same_state_direct_macro_action_no_mcts":
                raise ValueError("node-wise GRPO rejects search or full-trajectory samples")
            if value.get("macro_boundary") not in MACRO_BOUNDARIES:
                raise ValueError("macro action did not reach a meaningful control boundary")
            base = value.get("base_state")
            successor = value.get("successor_state")
            if not isinstance(base, Mapping) or not isinstance(successor, Mapping):
                raise ValueError("macro-action outcome requires base and successor states")
            if str(base.get("fingerprint", "")) != str(value["base_state_fingerprint"]):
                raise ValueError("macro-action base fingerprint mismatch")
            if not str(successor.get("fingerprint", "")):
                raise ValueError("macro-action successor requires an immutable fingerprint")
            _validate_clone_provenance(
                value.get("clone_provenance"), str(value["base_state_fingerprint"]),
                str(value["environment_digest"]), context="macro-action outcome",
            )
            record = value.get("policy_record")
            if not isinstance(record, Mapping) or record.get("behavior_policy") != "actor":
                raise ValueError("macro action must be sampled directly by the node actor")
            if record.get("mcts_search_samples"):
                raise ValueError("MCTS samples are invalid for node-wise GRPO")
            if str(record.get("state_fingerprint", "")) != str(value["base_state_fingerprint"]):
                raise ValueError("policy record was not sampled from the shared base state")
            if str(record.get("context_node_id", "")) != str(value["context_node_id"]):
                raise ValueError("policy record was sampled for another node")
            action = str(value.get("selected_action", ""))
            if action not in ORGANIZATION_ACTIONS or action != str(record.get("selected_action", "")):
                raise ValueError("macro-action kind and policy record disagree")
            if record.get("policy_state", {}).get("interface_revision") \
                    != LHTB_POLICY_INTERFACE_REVISION:
                raise ValueError("macro-action group uses a stale actor interface")
        return group_id

    def _precache_states(self, states: Iterable[Mapping[str, Any]]) -> None:
        precache = getattr(self.encoder, "precache", None)
        if not callable(precache):
            return
        graphs = [_state_graph(state) for state in states]
        precache(
            str(node.get("text") or node.get("kind") or "")
            for graph in graphs for node in graph.get("nodes", [])
            if isinstance(node, Mapping)
        )

    def _value_predictions(self, states: Sequence[Mapping[str, Any]]) -> Tensor:
        graphs = [tuple(value.to(self.device) for value in graph_tensors(
            _state_graph(state), self.encoder
        )) for state in states]
        return self.value.forward_batch(graphs)


def _value_metrics(
    labels: Sequence[Mapping[str, Any]], predictions: Sequence[float], targets: Sequence[float]
) -> Dict[str, float]:
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    mae = sum(abs(value) for value in errors) / len(errors)
    by_task: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_task[str(label["task_id"])].append(index)
    correct = 0
    pairs = 0
    for indices in by_task.values():
        for left_index, left in enumerate(indices):
            for right in indices[left_index + 1:]:
                target_delta = targets[left] - targets[right]
                if abs(target_delta) <= 1e-12:
                    continue
                prediction_delta = predictions[left] - predictions[right]
                correct += int(prediction_delta * target_delta > 0)
                pairs += 1
    spearman = _spearman(predictions, targets)
    return {
        "value_rmse": rmse,
        "value_mae": mae,
        "value_spearman": spearman,
        "pairwise_ranking_accuracy": correct / pairs if pairs else 0.0,
        "ranking_pairs": float(pairs),
    }


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or max(left) == min(left) or max(right) == min(right):
        return 0.0

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        cursor = 0
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and values[order[end]] == values[order[cursor]]:
                end += 1
            rank = (cursor + end - 1) / 2.0
            for index in order[cursor:end]:
                result[index] = rank
            cursor = end
        return result

    left_rank = ranks(left)
    right_rank = ranks(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum((a - left_mean) * (b - right_mean)
                    for a, b in zip(left_rank, right_rank))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_rank)
        * sum((value - right_mean) ** 2 for value in right_rank)
    )
    return numerator / denominator if denominator else 0.0


def _validate_clone_provenance(
    value: Any,
    source_fingerprint: str,
    environment_digest: str,
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping) or value.get("complete") is not True:
        raise ValueError(f"{context} requires a complete checkpoint clone")
    if value.get("mode") not in ("full_clone", "deterministic_replay"):
        raise ValueError(f"{context} requires full clone or deterministic replay")
    if str(value.get("source_state_fingerprint", "")) != source_fingerprint:
        raise ValueError(f"{context} clone uses another source state")
    if str(value.get("source_environment_digest", "")) != environment_digest:
        raise ValueError(f"{context} clone uses another environment")
    if not str(value.get("clone_audit_id", "")):
        raise ValueError(f"{context} requires a clone audit ID")
