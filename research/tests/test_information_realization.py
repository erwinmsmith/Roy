from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from roy_research.organization import (
    DEFAULT_EXPLORATION_GROUP,
    envelope_legal_actions,
    require_single_terminal_utility,
    validate_exploration_group,
)
from roy_research.organization_model import (
    InformationRealizationPolicy,
    expected_resource_log_probs,
    mixed_behavior_log_probs,
    uniform_exploration_log_probs,
)
from roy_research.organization_replay import (
    replay_joint_log_probability,
    sample_organization_decision,
)
from roy_research.organization_training import (
    organization_group_advantages,
    single_objective_organization_grpo_loss,
)
from roy_research.tau3 import TAU3_COMMIT, build_tau3_manifest, manifest_summary


class InformationRealizationTests(unittest.TestCase):
    def test_eight_envelopes_control_exploration_without_reward(self) -> None:
        validate_exploration_group(DEFAULT_EXPLORATION_GROUP)
        candidates = [
            {"kind": "DERIVE", "legal": True},
            {"kind": "EXECUTE", "legal": True},
            {"kind": "STOP", "legal": True},
        ]
        mask = envelope_legal_actions(candidates, DEFAULT_EXPLORATION_GROUP[4], 2, 1)
        self.assertEqual(mask, [True, True, False])
        full = envelope_legal_actions(candidates, DEFAULT_EXPLORATION_GROUP[4], 9, 4)
        self.assertEqual(full, [False, True, True])

    def test_training_accepts_only_terminal_task_utility(self) -> None:
        self.assertEqual(require_single_terminal_utility({"terminal_utility": 0.75}), 0.75)
        with self.assertRaisesRegex(ValueError, "forbidden reward"):
            require_single_terminal_utility({
                "terminal_utility": 1.0,
                "communication_reward": 0.1,
            })

    def test_policy_supports_variable_nodes_and_open_candidates(self) -> None:
        model = InformationRealizationPolicy(hidden_dim=64, layers=2)
        node_embeddings = torch.randn(4, 384)
        node_types = torch.tensor([0, 2, 7, 1])
        scalar_features = torch.zeros(4, 3)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_types = torch.tensor([1, 2, 3])
        states, graph = model.encode_graph(
            node_embeddings, node_types, scalar_features, edge_index, edge_types
        )
        resources = torch.zeros(5)
        active = model.active_node_logits(states, graph, resources, torch.ones(4, dtype=torch.bool))
        candidates = model.candidate_logits(
            graph,
            states[0],
            torch.randn(11, 384),
            torch.tensor([0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 0]),
            torch.zeros(11, 4),
            resources,
            torch.ones(11, dtype=torch.bool),
        )
        self.assertEqual(tuple(active.shape), (4,))
        self.assertEqual(tuple(candidates.shape), (11,))

    def test_expected_resource_projection_is_not_a_per_trajectory_cap(self) -> None:
        logits = torch.tensor([0.0, 2.0])
        costs = torch.tensor([0.0, 10.0])
        projected = expected_resource_log_probs(
            logits, torch.tensor([True, True]), costs, expected_budget=4.0
        )
        probabilities = projected.exp()
        self.assertLessEqual(float((probabilities * costs).sum()), 4.00001)
        self.assertGreater(float(probabilities[1]), 0.0)
        self.assertGreater(float(probabilities[1] * costs[1]), 0.0)

    def test_exploration_mixture_records_exact_behavior_distribution(self) -> None:
        policy = torch.log(torch.tensor([0.8, 0.2]))
        exploration = uniform_exploration_log_probs(torch.tensor([True, True]))
        mixed = mixed_behavior_log_probs(policy, exploration, 0.25)
        self.assertAlmostEqual(float(mixed.exp().sum()), 1.0, places=6)
        self.assertTrue(torch.allclose(mixed.exp(), torch.tensor([0.575, 0.425])))

    def test_sample_and_replay_use_the_same_joint_conditional_policy(self) -> None:
        class FakeEncoder:
            def encode(self, texts):
                rows = []
                for text in texts:
                    value = float(sum(str(text).encode("utf-8")) % 97) / 97.0
                    rows.append(torch.full((384,), value))
                return torch.stack(rows) if rows else torch.zeros((0, 384))

        model = InformationRealizationPolicy(hidden_dim=64, layers=2)
        state = {
            "state_fingerprint": "state-1",
            "event_graph": {
                "nodes": [
                    {"id": "root", "kind": "agent", "text": "root objective"},
                    {"id": "child", "kind": "agent", "text": "child objective"},
                ],
                "edges": [{"from": "root", "to": "child", "kind": "derivation"}],
            },
            "active_node_ids": ["root", "child"],
            "active_node_legal": [True, True],
            "candidates": [
                {
                    "id": "execute",
                    "kind": "EXECUTE",
                    "description": "reason locally",
                    "legal": True,
                    "expected_resource_cost": 1.0,
                },
                {
                    "id": "derive",
                    "kind": "DERIVE",
                    "description": "resolve the remaining evidence gap",
                    "legal": True,
                    "expected_resource_cost": 3.0,
                    "resolves_gap": True,
                    "depth_delta": 1,
                },
            ],
            "resources": {},
            "node_count": 2,
            "maximum_depth_reached": 1,
            "envelope": DEFAULT_EXPLORATION_GROUP[3].to_dict(),
            "exploration_alpha": 0.4,
            "expected_resource_budget": 2.0,
        }
        generator = torch.Generator().manual_seed(17)
        _candidate, record = sample_organization_decision(
            model, FakeEncoder(), state, generator=generator, device=torch.device("cpu")
        )
        replayed = replay_joint_log_probability(
            model, FakeEncoder(), record, torch.device("cpu")
        )
        self.assertLessEqual(float(record["projected_expected_resource_cost"]), 2.00001)
        self.assertAlmostEqual(
            float(replayed.detach()), float(record["policy_log_probability"]), places=6
        )

    def test_single_objective_group_advantage_and_loss(self) -> None:
        records = [
            {"id": f"t-{index}", "group_id": "g", "terminal_utility": utility}
            for index, utility in enumerate((0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25, 0.75))
        ]
        advantages = organization_group_advantages(records)
        self.assertEqual(len(advantages), 8)
        self.assertAlmostEqual(sum(value.advantage for value in advantages), 0.0, places=6)
        current = [torch.tensor([-0.2, -0.3], requires_grad=True) for _ in records]
        behavior = [torch.tensor([-0.2, -0.3]) for _ in records]
        loss = single_objective_organization_grpo_loss(
            current, behavior, [value.advantage for value in advantages]
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_tau3_manifest_keeps_official_test_and_knowledge_held_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            domains = root / "data" / "tau2" / "domains"
            for domain in ("airline", "retail", "telecom"):
                path = domains / domain
                path.mkdir(parents=True)
                (path / "tasks.json").write_text(json.dumps([
                    {"id": f"{domain}-0"}, {"id": f"{domain}-1"}, {"id": f"{domain}-2"}
                ]))
                (path / "split_tasks.json").write_text(json.dumps({
                    "train": [f"{domain}-0", f"{domain}-1"],
                    "test": [f"{domain}-2"],
                    "base": [f"{domain}-{index}" for index in range(3)],
                }))
            knowledge = domains / "banking_knowledge"
            knowledge.mkdir(parents=True)
            (knowledge / "tasks.json").write_text(json.dumps([
                {"id": "knowledge-0"}, {"id": "knowledge-1"}
            ]))
            completed = Mock(stdout=f"{TAU3_COMMIT}\n")
            with patch("roy_research.tau3.subprocess.run", return_value=completed):
                manifest = build_tau3_manifest(root, validation_modulus=10_000)
            summary = manifest_summary(manifest)
            self.assertEqual(summary["by_split"]["test"], 3)
            self.assertEqual(summary["by_split"]["heldout"], 2)
            self.assertFalse(any(
                value.domain == "banking_knowledge" and value.split == "train"
                for value in manifest
            ))


if __name__ == "__main__":
    unittest.main()
