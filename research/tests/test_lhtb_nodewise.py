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
    NODEWISE_ALGORITHM_REVISION,
    build_forced_finalize_label,
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

    def test_nodewise_group_uses_same_state_frozen_delta_v_and_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            encoder = FakeEncoder384()
            trainer = LHTBNodeWiseDeltaVTrainer(
                checkpoint, self.manifest, encoder=encoder
            )
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
                    "value_revision": 0,
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
            self.assertEqual(update["value_revision"], 0)
            self.assertEqual(update["actor_revision"], 1)
            self.assertAlmostEqual(update["derived_process_rewards"][0], -0.15, places=6)
            self.assertAlmostEqual(sum(update["advantages"]), 0.0, places=5)
            self.assertEqual(update["derived_reward_source"],
                             "frozen_forced_finalize_state_value_increment")

            restored = LHTBNodeWiseDeltaVTrainer(
                checkpoint, self.manifest, encoder=encoder, resume=True
            )
            self.assertEqual(restored.actor_revision, 1)
            self.assertEqual(restored.value_revision, 0)
            self.assertEqual(restored.metadata()["algorithm_revision"],
                             NODEWISE_ALGORITHM_REVISION)
            with self.assertRaisesRegex(ValueError, "already optimized"):
                restored.update_macro_group(records)


if __name__ == "__main__":
    unittest.main()
