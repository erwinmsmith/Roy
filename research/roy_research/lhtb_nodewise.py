from __future__ import annotations

import json
import math
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .lhtb import require_training_task
from .lhtb_results import official_lhtb_task_utility
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
MIA_REWARD_DEFINITION = "R_t^MIA=V_psi(S_t+1)-V_psi(S_t)"
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
            {"seed": int(seed), "environment_utility": utility,
             "official_lhtb_task_utility": utility,
             **dict(provenance)}
            for seed, utility, provenance in zip(seeds, normalized, verifier_provenance)
        ],
        "k": len(normalized),
        "value_target": sum(normalized) / len(normalized),
    }


def import_nodewise_macro_group(
    *,
    base_run: Path,
    samples_root: Path,
    group_id: str,
    task_id: str,
    split: str,
    epoch: int,
    policy_revision: int,
    value_revision: int,
    environment_digest: str,
    expected: int = NODEWISE_GROUP_SIZE,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Import one exact-checkpoint node-wise group and its V(S) labels.

    Harbor's official verifier utility is retained only as a forced-finalize
    target for the base and successor states.  Macro records intentionally carry
    no precomputed reward: ``LHTBNodeWiseDeltaVTrainer`` must derive
    ``R_t^MIA`` from one frozen value revision during the actor update.
    """
    if expected != NODEWISE_GROUP_SIZE:
        raise ValueError("node-wise macro import requires exactly G=8 successors")
    base = _load_nodewise_run(base_run)
    base_state = base["state"]
    base_snapshot = base["snapshot"]
    base_fingerprint = str(base_state.get("fingerprint") or "")
    if not base_fingerprint:
        raise ValueError("node-wise base state has no fingerprint")
    _validate_nodewise_identity(base, task_id, environment_digest, require_restorable=True)

    labels = [_nodewise_value_label(
        loaded=base,
        label_id=f"{group_id}:base",
        task_id=task_id,
        split=split,
    )]
    records: List[Dict[str, Any]] = []
    sample_dirs = sorted(
        (path for path in samples_root.iterdir()
         if path.is_dir() and path.name.startswith("sample-")),
        key=lambda path: int(path.name.split("-", 1)[1]),
    )
    if len(sample_dirs) != expected:
        raise ValueError(
            f"expected {expected} node-wise successor runs, found {len(sample_dirs)}"
        )
    base_policy_count = len(base_snapshot.get("policyRecords") or [])
    seeds: set[int] = set()
    for index, sample_dir in enumerate(sample_dirs):
        loaded = _load_nodewise_run(sample_dir)
        _validate_nodewise_identity(
            loaded, task_id, environment_digest, require_restorable=False
        )
        snapshot = loaded["snapshot"]
        policy_records = list(snapshot.get("policyRecords") or [])
        if len(policy_records) != base_policy_count + 1:
            raise ValueError(
                f"sample {index} did not execute exactly one new Controller action"
            )
        policy_record = _normalize_policy_record(policy_records[-1])
        if policy_record["state_fingerprint"] != base_fingerprint:
            raise ValueError(f"sample {index} was not sampled from the shared base state")
        seed = int(snapshot.get("organizationSeed", -1))
        if seed < 0 or seed in seeds:
            raise ValueError("node-wise successors require unique organization seeds")
        seeds.add(seed)
        successor_state = loaded["state"]
        action = str(policy_record["selected_action"])
        records.append({
            "schema_version": MACRO_GROUP_SCHEMA_VERSION,
            "group_id": group_id,
            "benchmark": "lhtb",
            "task_id": task_id,
            "split": split,
            "epoch": int(epoch),
            "sample_index": index,
            "organization_seed": seed,
            "checkpoint_id": str(base["checkpoint"].get("payload_digest") or ""),
            "context_node_id": str(policy_record["context_node_id"]),
            "base_state_fingerprint": base_fingerprint,
            "base_state": base_state,
            "successor_state": successor_state,
            "selected_action": action,
            "policy_record": policy_record,
            "policy_revision": int(policy_revision),
            "value_revision": int(value_revision),
            "macro_boundary": _macro_boundary(action, base_state, successor_state),
            "sampling_protocol": "same_state_direct_macro_action_no_mcts",
            "task_checksum": str(base["result"].get("task_checksum") or ""),
            "environment_digest": environment_digest,
            "runtime_config": {
                "macro_steps": 1,
                "controller": "shared_node_actor",
                "mcts_enabled": False,
            },
            "clone_provenance": {
                "mode": str(base["checkpoint"].get("mode") or "full_clone"),
                "complete": bool(base["checkpoint"].get("complete")),
                "source_state_fingerprint": base_fingerprint,
                "source_environment_digest": environment_digest,
                "clone_audit_id": _sha256_json({
                    "base_checkpoint": base["checkpoint"],
                    "successor_checkpoint": loaded["checkpoint"],
                    "sample_index": index,
                }),
            },
            "environment_utility": float(loaded["utility"]),
            "environment_utility_role": "value_supervision_only",
            "mia_reward": None,
            "mia_reward_emitted": False,
        })
        labels.append(_nodewise_value_label(
            loaded=loaded,
            label_id=f"{group_id}:successor:{index}",
            task_id=task_id,
            split=split,
        ))
    return labels, records


def _load_nodewise_run(path: Path) -> Dict[str, Any]:
    artifacts = path / "artifacts"
    required = {
        "snapshot": artifacts / "session.json",
        "state": artifacts / "state.json",
        "checkpoint": artifacts / "environment-checkpoint" / "checkpoint.json",
    }
    missing = [str(value) for value in required.values() if not value.is_file()]
    if missing:
        raise ValueError(f"node-wise run is missing artifacts: {missing}")
    results = []
    for result_path in path.rglob("result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(result.get("verifier_result"), Mapping):
            results.append((result_path, result))
    if len(results) != 1:
        raise ValueError(f"node-wise run requires one official Harbor result, found {len(results)}")
    result_path, result = results[0]
    if result.get("exception_info"):
        raise ValueError("node-wise Harbor result contains an exception")
    return {
        key: json.loads(value.read_text(encoding="utf-8"))
        for key, value in required.items()
    } | {
        "result": result,
        "result_path": result_path.resolve(),
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "utility": official_lhtb_task_utility(result),
    }


def _validate_nodewise_identity(
    loaded: Mapping[str, Any], task_id: str, environment_digest: str,
    *, require_restorable: bool,
) -> None:
    checkpoint = loaded["checkpoint"]
    state = loaded["state"]
    snapshot = loaded["snapshot"]
    if str(checkpoint.get("task_id") or "") != task_id:
        raise ValueError("node-wise checkpoint task mismatch")
    if str(checkpoint.get("source_environment_digest") or "") != environment_digest:
        raise ValueError("node-wise checkpoint environment digest mismatch")
    if not checkpoint.get("complete"):
        raise ValueError("node-wise checkpoint is incomplete")
    restorable = checkpoint.get(
        "restorable", checkpoint.get("mode") == "full_clone"
    )
    if require_restorable and (
        restorable is not True
        or checkpoint.get("mode") not in ("full_clone", "deterministic_replay")
    ):
        raise ValueError("node-wise base checkpoint is not restorable")
    if checkpoint.get("mode") not in (
        "full_clone", "deterministic_replay", "isolated_instance_observation"
    ):
        raise ValueError("node-wise checkpoint has an unsupported capture mode")
    fingerprint = str(state.get("fingerprint") or "")
    if str(checkpoint.get("source_state_fingerprint") or "") != fingerprint:
        raise ValueError("node-wise checkpoint/state fingerprint mismatch")
    states = list(snapshot.get("processStates") or [])
    if not states or str(states[-1].get("fingerprint") or "") != fingerprint:
        raise ValueError("node-wise snapshot/state fingerprint mismatch")


def _nodewise_value_label(
    *, loaded: Mapping[str, Any], label_id: str, task_id: str, split: str
) -> Dict[str, Any]:
    checkpoint = loaded["checkpoint"]
    state = loaded["state"]
    environment_digest = str(checkpoint["source_environment_digest"])
    return build_forced_finalize_label(
        label_id=label_id,
        task_id=task_id,
        split=split,
        process_state=state,
        task_utilities=[float(loaded["utility"])],
        finalizer_revision="artifact-identity-a0-v1",
        task_checksum=str(loaded["result"].get("task_checksum") or ""),
        environment_digest=environment_digest,
        checkpoint_id=str(checkpoint.get("payload_digest") or ""),
        clone_provenance={
            "mode": (
                "isolated_instance"
                if checkpoint.get("mode") == "isolated_instance_observation"
                else str(checkpoint.get("mode") or "full_clone")
            ),
            "complete": bool(checkpoint.get("complete")),
            "source_state_fingerprint": str(state["fingerprint"]),
            "source_environment_digest": environment_digest,
            "clone_audit_id": _sha256_json(checkpoint),
        },
        verifier_provenance=[{
            "harbor_result_path": str(loaded["result_path"]),
            "harbor_result_sha256": str(loaded["result_sha256"]),
        }],
        sample_seeds=[int(loaded["snapshot"].get("organizationSeed", 0))],
    )


def _normalize_policy_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "behavior_policy": record.get("behaviorPolicy", record.get("behavior_policy")),
        "state_fingerprint": str(record.get(
            "stateFingerprint", record.get("state_fingerprint", "")
        )),
        "context_node_id": str(record.get(
            "contextNodeId", record.get("context_node_id", "")
        )),
        "candidate_id": str(record.get("candidateId", record.get("candidate_id", ""))),
        "masked_old_log_probability": float(record.get(
            "maskedOldLogProbability", record.get("masked_old_log_probability")
        )),
        "selected_action": str(record.get(
            "selectedAction", record.get("selected_action", "")
        )),
        "policy_state": dict(record.get("policyState", record.get("policy_state")) or {}),
        "mcts_search_samples": record.get(
            "mctsSearchSamples", record.get("mcts_search_samples")
        ),
    }


def _macro_boundary(
    action: str, base_state: Mapping[str, Any], successor_state: Mapping[str, Any]
) -> str:
    if action == "FINISH":
        return "official_verifier_complete"
    if action == "RETURN":
        return "parent_report_integrated"
    if action == "PRUNE":
        return "branch_pruned"
    if len(successor_state.get("externalObservations", [])) > len(
        base_state.get("externalObservations", [])
    ):
        return "external_observation_recorded"
    return "worker_phase_complete"


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


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
        mia_rewards = successor_values - values[0]
        deviation = mia_rewards.std(unbiased=False)
        has_signal = float(mia_rewards.max() - mia_rewards.min()) > 1e-12
        advantages = (
            torch.zeros_like(mia_rewards)
            if not has_signal
            else (mia_rewards - mia_rewards.mean()) / (deviation + 1e-8)
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
            "mia_rewards": [float(value.detach().cpu()) for value in mia_rewards],
            # Historical audit alias. New consumers must use ``mia_rewards``.
            "derived_process_rewards": [
                float(value.detach().cpu()) for value in mia_rewards
            ],
            "advantages": [float(value.detach().cpu()) for value in advantages],
            "actions": [str(value["selected_action"]) for value in records],
            "mia_reward_definition": MIA_REWARD_DEFINITION,
            "mia_reward_source": "frozen_forced_finalize_state_value_increment",
            "environment_utility_role": "value_supervision_and_evaluation_only",
            "environment_utility_used_by_actor": False,
            # Compatibility metadata for already-collected experiment readers.
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
            "value_supervision_available": bool(
                self.value_revision > 0 and self.used_value_label_ids
            ),
            "training_protocol": "same_state_macro_action_delta_v_grpo_no_mcts",
            "value_label_protocol": "frozen_finalize_now_official_lhtb_task_utility",
            "mia_reward_definition": MIA_REWARD_DEFINITION,
            "environment_utility_symbol": "u_env",
            "environment_utility_role": "value_supervision_and_evaluation_only",
            "environment_utility_used_by_actor": False,
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
            scores = [float(sample.get(
                "environment_utility",
                sample.get("official_lhtb_task_utility", -1),
            ))
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
        if self.value_revision <= 0 or not self.used_value_label_ids:
            raise ValueError(
                "node-wise GRPO requires a forced-finalize-trained V_psi revision"
            )
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
    if value.get("mode") not in (
        "full_clone", "deterministic_replay", "isolated_instance"
    ):
        raise ValueError(
            f"{context} requires full clone, deterministic replay or isolated instance"
        )
    if str(value.get("source_state_fingerprint", "")) != source_fingerprint:
        raise ValueError(f"{context} clone uses another source state")
    if str(value.get("source_environment_digest", "")) != environment_digest:
        raise ValueError(f"{context} clone uses another environment")
    if not str(value.get("clone_audit_id", "")):
        raise ValueError(f"{context} requires a clone audit ID")
