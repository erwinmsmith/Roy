from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from roy_research.lhtb import build_lhtb_split
from roy_research.cli import main as research_cli_main
from roy_research.lhtb_nodewise import (
    LHTBNodeWiseDeltaVTrainer,
    MIA_REWARD_DEFINITION,
    NODEWISE_ALGORITHM_REVISION,
    build_forced_finalize_label,
    import_nodewise_macro_group,
)
from roy_research.organization import LHTB_POLICY_INTERFACE_REVISION
from roy_research.organization_replay import sample_organization_decision


class FakeEncoder384:
    def encode(self, texts):
        rows = []
        for text in texts:
            value = float((sum(str(text).encode("utf-8")) % 17) + 1) / 17
            rows.append(torch.full((384,), value, dtype=torch.float32))
        return torch.stack(rows) if rows else torch.zeros((0, 384), dtype=torch.float32)

    def precache(self, _texts):
        return None


def process_state(fingerprint: str, child_count: int = 0):
    nodes = [{"id": "root", "kind": "agent", "text": "solve task", "status": "running"}]
    edges = []
    for index in range(child_count):
        child_id = f"child-{index}"
        nodes.append({"id": child_id, "kind": "agent", "text": f"evidence {index}",
                      "status": "completed"})
        edges.append({"kind": "derivation", "from": "root", "to": child_id})
    return {"sequence": child_count, "fingerprint": fingerprint,
            "event_graph": {"nodes": nodes, "edges": edges}}


def policy_state(fingerprint: str):
    return {
        "interface_revision": LHTB_POLICY_INTERFACE_REVISION,
        "state_fingerprint": fingerprint,
        "event_graph": process_state(fingerprint)["event_graph"],
        "context_node_id": "root",
        "context_node": {"id": "root", "depth": 0, "status": "running",
                         "local_objective": "solve task", "requirements": [],
                         "ancestry": [], "recent_runtime_events": []},
        "candidates": [
            {"id": "controller:CONTINUE", "kind": "CONTINUE",
             "actor_node_id": "root", "description": "continue", "legal": True,
             "scheduler_complexity": 0},
            {"id": "controller:DERIVE_INFO", "kind": "DERIVE_INFO",
             "actor_node_id": "root", "description": "derive info", "legal": True,
             "scheduler_complexity": 0},
            {"id": "controller:DERIVE_ORG", "kind": "DERIVE_ORG",
             "actor_node_id": "root", "description": "derive org", "legal": True,
             "scheduler_complexity": 0},
            {"id": "controller:FINISH", "kind": "FINISH",
             "actor_node_id": "root", "description": "finish", "legal": True,
             "scheduler_complexity": 0},
        ],
        "envelope": {"id": "open", "minimum_nodes": 0, "maximum_nodes": 1_000_000,
                     "minimum_depth": 0, "maximum_depth": 1_000_000,
                     "mode": "expansive"},
        "resources": {"llm_calls_remaining_fraction": 1.0,
                      "tool_calls_remaining_fraction": 1.0,
                      "nodes_remaining_fraction": 1.0,
                      "depth_remaining_fraction": 1.0,
                      "decisions_remaining_fraction": 1.0},
        "num_real_residual_gaps": 1,
        "num_child_proposals": 0,
        "stop_legal_reason": "root_can_submit",
    }


class NodeWiseDeltaVTests(unittest.TestCase):
    def setUp(self):
        manifest = [value.to_dict() for value in build_lhtb_split()]
        self.manifest = manifest
        self.task_id = next(str(value["task_id"]) for value in manifest
                            if value["split"] == "train")

    def test_forced_finalize_label_is_official_mc_value(self):
        state = process_state("m3", 2)
        label = build_forced_finalize_label(
            label_id="label-1", task_id=self.task_id, split="train",
            process_state=state, task_utilities=[0.2, 0.5, 0.8],
            finalizer_revision="frozen-a0", task_checksum="task-sha",
            environment_digest="sha256:environment", checkpoint_id="checkpoint-3",
            clone_provenance={"mode": "full_clone", "complete": True,
                "source_state_fingerprint": "m3",
                "source_environment_digest": "sha256:environment",
                "clone_audit_id": "clone-label-1"},
            verifier_provenance=[
                {"harbor_result_path": f"/audit/result-{index}.json",
                 "harbor_result_sha256": f"sha-{index}"}
                for index in range(3)
            ],
            sample_seeds=[11, 12, 13],
        )
        self.assertAlmostEqual(label["value_target"], 0.5)
        self.assertEqual(label["task_utility_source"], "official_lhtb_verifier")
        self.assertEqual(label["finalizer_policy"],
                         "frozen_finalize_now_no_structural_actions")
        self.assertEqual(label["state_fingerprint"], "m3")
        self.assertEqual(label["samples"][0]["environment_utility"], 0.2)

    def test_finalize_label_cli_hashes_official_harbor_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.json"
            result_path = root / "result.json"
            output = root / "labels.jsonl"
            state_path.write_text(json.dumps(process_state("m-cli")))
            result_path.write_text(json.dumps({
                "verifier_result": {"rewards": {"reward": 0.625}},
            }))
            research_cli_main([
                "lhtb-finalize-label", "--state", str(state_path),
                "--output", str(output), "--label-id", "cli-label",
                "--task-id", self.task_id, "--split", "train",
                "--checkpoint-id", "checkpoint-cli",
                "--finalizer-revision", "frozen-a0", "--task-checksum", "task-sha",
                "--environment-digest", "sha256:environment",
                "--clone-mode", "full_clone", "--clone-audit-id", "clone-cli",
                "--harbor-result", str(result_path), "--sample-seed", "7",
            ])
            label = json.loads(output.read_text())
            self.assertEqual(label["value_target"], 0.625)
            self.assertEqual(len(label["samples"][0]["harbor_result_sha256"]), 64)

    def test_value_update_rejects_non_official_or_non_frozen_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = LHTBNodeWiseDeltaVTrainer(
                Path(directory) / "model.pt", self.manifest, encoder=FakeEncoder384()
            )
            label = build_forced_finalize_label(
                label_id="label-1", task_id=self.task_id, split="train",
                process_state=process_state("m0"), task_utilities=[0.4],
                finalizer_revision="frozen-a0", task_checksum="task-sha",
                environment_digest="sha256:environment", checkpoint_id="checkpoint-0",
                clone_provenance={"mode": "deterministic_replay", "complete": True,
                    "source_state_fingerprint": "m0",
                    "source_environment_digest": "sha256:environment",
                    "clone_audit_id": "clone-label-1"},
                verifier_provenance=[{"harbor_result_path": "/audit/result.json",
                                      "harbor_result_sha256": "sha-result"}],
            )
            invalid = dict(label)
            invalid["task_utility_source"] = "learned_self_judge"
            with self.assertRaisesRegex(ValueError, "official LHTB verifier"):
                trainer.update_value([invalid], epochs=1)
            update = trainer.update_value([label], epochs=1)
            self.assertEqual(update["value_revision"], 1)
            self.assertEqual(update["labels"], 1)

            replay = label
            second = build_forced_finalize_label(
                label_id="label-2", task_id=self.task_id, split="train",
                process_state=process_state("m1", 1), task_utilities=[0.8],
                finalizer_revision="frozen-a0", task_checksum="task-sha",
                environment_digest="sha256:environment", checkpoint_id="checkpoint-1",
                clone_provenance={"mode": "deterministic_replay", "complete": True,
                    "source_state_fingerprint": "m1",
                    "source_environment_digest": "sha256:environment",
                    "clone_audit_id": "clone-label-2"},
                verifier_provenance=[{"harbor_result_path": "/audit/result-2.json",
                                      "harbor_result_sha256": "sha-result-2"}],
            )
            replay_update = trainer.update_value(
                [second], replay_labels=[replay], epochs=1, batch_size=2
            )
            self.assertEqual(replay_update["fresh_labels"], 1)
            self.assertEqual(replay_update["replay_labels"], 1)
            self.assertEqual(replay_update["training_labels"], 2)
            self.assertIn("fresh_value_spearman", replay_update)

    def test_nodewise_group_uses_same_state_frozen_delta_v_and_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            encoder = FakeEncoder384()
            trainer = LHTBNodeWiseDeltaVTrainer(
                checkpoint, self.manifest, encoder=encoder
            )
            trainer.value_revision = 1
            trainer.used_value_label_ids.add("bootstrap-value-label")
            state = policy_state("shared-m4")
            records = []
            for index in range(8):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(index + 1)
                _, actor_record = sample_organization_decision(
                    trainer.actor, encoder, copy.deepcopy(state), generator=generator,
                    device=torch.device("cpu"),
                )
                records.append({
                    "schema_version": 1,
                    "group_id": "node:task:checkpoint-4",
                    "benchmark": "lhtb",
                    "task_id": self.task_id,
                    "split": "train",
                    "sample_index": index,
                    "organization_seed": 100 + index,
                    "checkpoint_id": "checkpoint-4",
                    "context_node_id": "root",
                    "base_state_fingerprint": "shared-m4",
                    "base_state": process_state("shared-m4"),
                    "successor_state": process_state(f"successor-{index}", index % 4),
                    "selected_action": actor_record["selected_action"],
                    "policy_record": actor_record,
                    "policy_revision": 0,
                    "value_revision": 1,
                    "macro_boundary": "worker_phase_complete",
                    "sampling_protocol": "same_state_direct_macro_action_no_mcts",
                    "task_checksum": "task-sha",
                    "environment_digest": "sha256:environment",
                    "runtime_config": {"worker_phase": 1},
                    "clone_provenance": {"mode": "full_clone", "complete": True,
                        "source_state_fingerprint": "shared-m4",
                        "source_environment_digest": "sha256:environment",
                        "clone_audit_id": f"macro-clone-{index}"},
                })

            values = torch.tensor(
                [0.25, 0.10, 0.20, 0.30, 0.40, 0.15, 0.35, 0.50, 0.45],
                dtype=torch.float32,
            )
            trainer._value_predictions = lambda _states: values.to(trainer.device)
            update = trainer.update_macro_group(records)
            self.assertTrue(update["actor_updated"])
            self.assertEqual(update["value_revision"], 1)
            self.assertEqual(update["actor_revision"], 1)
            self.assertAlmostEqual(update["derived_process_rewards"][0], -0.15, places=6)
            self.assertEqual(update["mia_rewards"], update["derived_process_rewards"])
            self.assertEqual(update["mia_reward_definition"], MIA_REWARD_DEFINITION)
            self.assertFalse(update["environment_utility_used_by_actor"])
            self.assertAlmostEqual(sum(update["advantages"]), 0.0, places=5)
            self.assertEqual(update["derived_reward_source"],
                             "frozen_forced_finalize_state_value_increment")

            restored = LHTBNodeWiseDeltaVTrainer(
                checkpoint, self.manifest, encoder=encoder, resume=True
            )
            self.assertEqual(restored.actor_revision, 1)
            self.assertEqual(restored.value_revision, 1)
            self.assertEqual(restored.metadata()["algorithm_revision"],
                             NODEWISE_ALGORITHM_REVISION)
            with self.assertRaisesRegex(ValueError, "already optimized"):
                restored.update_macro_group(records)

    def test_nodewise_group_rejects_untrained_constant_value(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = LHTBNodeWiseDeltaVTrainer(
                Path(directory) / "model.pt", self.manifest, encoder=FakeEncoder384()
            )
            with self.assertRaisesRegex(ValueError, "forced-finalize-trained"):
                trainer.update_macro_group([{}] * 8)

    def test_nodewise_import_separates_environment_utility_from_mia_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = "sha256:environment"
            base = root / "base"
            self._write_nodewise_run(
                base, "base-state", digest, 0.0, [], 90
            )
            samples = root / "samples"
            for index in range(8):
                record = {
                    "behaviorPolicy": "actor",
                    "stateFingerprint": "base-state",
                    "contextNodeId": "root",
                    "candidateId": "controller:CONTINUE",
                    "maskedOldLogProbability": -0.5,
                    "selectedAction": "CONTINUE",
                    "policyState": policy_state("base-state"),
                }
                self._write_nodewise_run(
                    samples / f"sample-{index}", f"next-{index}", digest,
                    index / 7, [record], 100 + index,
                )
            labels, records = import_nodewise_macro_group(
                base_run=base, samples_root=samples, group_id="node:g8",
                task_id=self.task_id, split="train", epoch=0,
                decision_round=1,
                policy_revision=0, value_revision=0,
                environment_digest=digest,
            )
            self.assertEqual(len(labels), 9)
            self.assertEqual(len(records), 8)
            self.assertEqual({record["decision_round"] for record in records}, {1})
            self.assertEqual(records[-1]["environment_utility"], 1.0)
            self.assertIsNone(records[-1]["mia_reward"])
            self.assertFalse(records[-1]["mia_reward_emitted"])
            self.assertEqual(
                {record["base_state_fingerprint"] for record in records},
                {"base-state"},
            )
            self.assertEqual(
                {record["organization_seed"] for record in records},
                set(range(100, 108)),
            )

    def test_nodewise_import_accepts_isolated_nonrestorable_successor_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = "sha256:service-environment"
            base = root / "base"
            self._write_nodewise_run(
                base, "base-service", digest, 0.0, [], 10,
                mode="deterministic_replay", restorable=True,
            )
            samples = root / "samples"
            for index in range(8):
                record = {
                    "behaviorPolicy": "actor",
                    "stateFingerprint": "base-service",
                    "contextNodeId": "root",
                    "candidateId": "controller:CONTINUE",
                    "maskedOldLogProbability": -0.5,
                    "selectedAction": "CONTINUE",
                    "policyState": policy_state("base-service"),
                }
                self._write_nodewise_run(
                    samples / f"sample-{index}", f"service-next-{index}",
                    digest, 0.25, [record], 20 + index,
                    mode=("isolated_instance_observation" if index == 0
                          else "deterministic_replay"),
                    restorable=index != 0,
                )
            labels, records = import_nodewise_macro_group(
                base_run=base, samples_root=samples, group_id="service:g8",
                task_id=self.task_id, split="train", epoch=0,
                policy_revision=0, value_revision=0,
                environment_digest=digest,
            )
            self.assertEqual(labels[1]["clone_provenance"]["mode"],
                             "isolated_instance")
            self.assertTrue(labels[1]["clone_provenance"]["complete"])
            self.assertEqual(len(records), 8)

    def _write_nodewise_run(
        self, root: Path, fingerprint: str, digest: str, utility: float,
        policy_records, seed: int, *, mode: str = "full_clone",
        restorable=None,
    ) -> None:
        artifacts = root / "artifacts"
        checkpoint = artifacts / "environment-checkpoint"
        trial = root / "jobs" / "job" / "trial"
        checkpoint.mkdir(parents=True)
        trial.mkdir(parents=True)
        state = process_state(fingerprint)
        (artifacts / "state.json").write_text(json.dumps(state))
        (artifacts / "session.json").write_text(json.dumps({
            "organizationSeed": seed,
            "processStates": [state],
            "policyRecords": policy_records,
        }))
        checkpoint_payload = {
            "complete": True,
            "mode": mode,
            "payload_digest": f"payload-{fingerprint}",
            "source_environment_digest": digest,
            "source_state_fingerprint": fingerprint,
            "task_id": self.task_id,
        }
        if restorable is not None:
            checkpoint_payload["restorable"] = restorable
        (checkpoint / "checkpoint.json").write_text(json.dumps(checkpoint_payload))
        (trial / "result.json").write_text(json.dumps({
            "task_checksum": "task-sha",
            "exception_info": None,
            "agent_result": {"metadata": {
                "nodewise_protocol": "same_checkpoint_single_node_macro_action_v1",
                "finalizer_revision": "frozen-bounded-local-readout-a0-20260830",
            }},
            "verifier_result": {"rewards": {"reward": utility}},
        }))


if __name__ == "__main__":
    unittest.main()
