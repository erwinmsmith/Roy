from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from roy_research.organization import (
    ExplorationEnvelope,
    RuntimeBudget,
    envelope_legal_actions,
    require_single_terminal_utility,
    training_envelope,
    validate_exploration_group,
)
from roy_research.organization_model import InformationRealizationPolicy
from roy_research.organization_replay import (
    OrganizationGRPOTrainer,
    replay_joint_log_probability,
    sample_organization_decision,
)
from roy_research.organization_training import (
    organization_group_advantages,
    single_objective_organization_grpo_loss,
)
from roy_research.tau3 import TAU3_COMMIT, build_tau3_manifest, manifest_summary
from roy_research.tau3_agent import (
    _bound_tool_call_payload,
    _communication_candidate_allowed,
    _is_user_interaction_requirement,
    _is_failed_tool_observation,
    _matching_tool_names,
    _is_stop_message,
    _non_thinking_arguments,
    _normalize_report,
    _objective_fingerprint,
    _residual_child_specifications,
    _stop_content,
    _tool_acquisition_candidates,
    _topology_summary,
    _tool_access_residuals,
    _tool_argument_prompt,
    _user_acquisition_question,
)


class InformationRealizationTests(unittest.TestCase):
    def test_one_shared_envelope_controls_group_without_reward(self) -> None:
        envelope = training_envelope(0, 4)
        validate_exploration_group((envelope,) * 8)
        candidates = [
            {"kind": "DERIVE", "legal": True},
            {"kind": "EXECUTE", "legal": True},
            {"kind": "STOP", "legal": True},
        ]
        mask = envelope_legal_actions(candidates, envelope, 2, 1, True)
        self.assertEqual(mask, [True, False, False])
        no_gap = envelope_legal_actions(candidates, envelope, 2, 1, False)
        self.assertEqual(no_gap, [True, True, True])
        full = envelope_legal_actions(candidates, envelope, 24, 8, True)
        self.assertEqual(full, [False, True, True])
        with self.assertRaisesRegex(ValueError, "share one exploration envelope"):
            validate_exploration_group((envelope,) * 7 + (training_envelope(3, 4),))

    def test_training_envelope_anneals_without_synthetic_requirements(self) -> None:
        self.assertEqual(
            [(training_envelope(epoch, 4).minimum_nodes,
              training_envelope(epoch, 4).minimum_depth) for epoch in range(4)],
            [(6, 3), (4, 2), (2, 1), (0, 0)],
        )
        self.assertIsNone(RuntimeBudget().maximum_nodes)
        self.assertEqual(training_envelope(0, 4).maximum_nodes, 24)

    def test_depth_floor_prefers_only_depth_increasing_derivations(self) -> None:
        envelope = ExplorationEnvelope("depth", 3, 12, 3, 5, "expansive")
        candidates = [
            {"kind": "DERIVE", "legal": True, "resulting_depth": 2},
            {"kind": "DERIVE", "legal": True, "resulting_depth": 3},
            {"kind": "ACQUIRE", "legal": True},
        ]
        self.assertEqual(
            envelope_legal_actions(candidates, envelope, 3, 2, True),
            [False, True, False],
        )
        self.assertEqual(
            envelope_legal_actions(candidates, envelope, 3, 2, False),
            [True, True, True],
        )

    def test_floor_prefers_report_grounded_acquisition_when_no_child_exists(self) -> None:
        envelope = ExplorationEnvelope("acquire", 6, 12, 3, 5, "expansive")
        candidates = [
            {"kind": "ACQUIRE", "legal": True, "resolves_gap": False},
            {"kind": "ACQUIRE", "legal": True, "resolves_gap": True},
            {"kind": "EXECUTE", "legal": True},
            {"kind": "STOP", "legal": True},
        ]
        self.assertEqual(
            envelope_legal_actions(candidates, envelope, 2, 1, True),
            [False, True, False, False],
        )

    def test_training_accepts_only_terminal_task_utility(self) -> None:
        self.assertEqual(require_single_terminal_utility({"terminal_utility": 0.75}), 0.75)
        with self.assertRaisesRegex(ValueError, "forbidden reward"):
            require_single_terminal_utility({
                "terminal_utility": 1.0,
                "communication_reward": 0.1,
            })

    def test_node_report_normalizes_real_model_shape_variants(self) -> None:
        report = _normalize_report({
            "conclusion": "done",
            "uncertainty": "moderate",
            "coverage": ["refund policy"],
            "claims": "one claim",
            "residual_requirements": {
                "description": "look up booking",
                "possible_external_access": "reservation tool",
            },
        }, {"id": "root"})
        self.assertEqual(report["uncertainty"], {"summary": "moderate"})
        self.assertEqual(report["coverage"], {"summary": ["refund policy"]})
        self.assertEqual(report["claims"], ["one claim"])
        self.assertEqual(
            report["residual_requirements"][0]["possible_external_access"],
            ["reservation tool"],
        )

    def test_report_recovers_unlabeled_proposals_and_symbolic_dependencies(self) -> None:
        report = _normalize_report({
            "residual_requirements": [
                "Retrieve reservation details with get_reservation_details.",
                "Search and compare the cheapest business flights.",
            ],
            "proposed_children": [
                {
                    "id": "child-1",
                    "objective": "Retrieve and inspect reservation details",
                    "termination_condition": "Reservation fields are available",
                },
                {
                    "id": "child-2",
                    "objective": "Compare business flight options",
                    "termination_condition": "The cheapest round trip is identified",
                    "depends_on_node_ids": ["child-1"],
                },
            ],
        }, {"id": "root"})
        self.assertEqual(len(report["residual_requirements"]), 2)
        self.assertEqual(len(report["proposed_children"]), 2)
        self.assertEqual(
            report["proposed_children"][0]["triggering_gap_id"],
            report["residual_requirements"][0]["id"],
        )
        specifications = _residual_child_specifications(
            report, {"id": "root"}, ["root"]
        )
        self.assertEqual(specifications[1]["depends_on_node_ids"], ["child-1"])
        self.assertEqual(
            _objective_fingerprint("Retrieve reservation K1NW8N."),
            _objective_fingerprint("retrieve RESERVATION k1nw8n"),
        )

    def test_residual_requirement_becomes_open_child_specification(self) -> None:
        report = _normalize_report({
            "residual_requirements": [
                {
                    "description": "Independently verify the applicable booking policy",
                    "termination_condition": "Return the verified policy with evidence",
                },
                {
                    "id": "gap-policy",
                    "description": "Check the cancellation policy",
                },
            ],
            "proposed_children": [],
        }, {"id": "root"})
        specifications = _residual_child_specifications(
            report, {"id": "root"}, ["root", "node-1"]
        )
        self.assertEqual(len(specifications), 2)
        self.assertEqual(specifications[0]["origin"], "model_reported_residual")
        self.assertEqual(
            specifications[0]["termination_condition"],
            "Return the verified policy with evidence",
        )
        self.assertEqual(specifications[1]["triggering_gap_id"], "gap-policy")

    def test_explicit_child_wins_and_dependencies_are_validated(self) -> None:
        report = _normalize_report({
            "residual_requirements": [{"id": "gap-1", "description": "Cross-check fare"}],
            "proposed_children": [{
                "triggering_gap_id": "gap-1",
                "objective": "Independently verify the fare",
                "termination_condition": "Return the matching fare",
                "depends_on_node_ids": ["node-1", "missing"],
            }],
        }, {"id": "node-2"})
        specifications = _residual_child_specifications(
            report, {"id": "node-2"}, ["root", "node-1", "node-2"]
        )
        self.assertEqual(len(specifications), 1)
        self.assertEqual(specifications[0]["objective"], "Independently verify the fare")
        self.assertEqual(specifications[0]["depends_on_node_ids"], ["node-1"])

    def test_user_interaction_gap_is_acquired_instead_of_recursively_derived(self) -> None:
        report = _normalize_report({
            "residual_requirements": [{
                "id": "gap-user-id",
                "description": "Ask the user for their user_id and wait for the next user message",
                "possible_external_access": ["user"],
            }],
            "proposed_children": [{
                "triggering_gap_id": "gap-user-id",
                "objective": "Prompt the user for their user_id",
                "termination_condition": "Receive the next user message",
            }],
        }, {"id": "root"})
        self.assertTrue(_is_user_interaction_requirement(report["residual_requirements"][0]))
        self.assertEqual(
            _residual_child_specifications(report, {"id": "root"}, ["root"]), []
        )
        question = _user_acquisition_question(report["residual_requirements"][0])
        self.assertIn("user_id", question)
        self.assertTrue(_is_user_interaction_requirement({
            "description": "Obtain the user's user ID before account lookup",
            "possible_external_access": ["user_interaction"],
        }))

    def test_tool_access_gap_is_acquired_instead_of_recursively_derived(self) -> None:
        report = _normalize_report({
            "residual_requirements": [{
                "id": "gap-details",
                "description": "Look up the user's account details",
                "possible_external_access": ["get_user_details tool"],
            }],
        }, {"id": "root"})
        self.assertEqual(
            _residual_child_specifications(report, {"id": "root"}, ["root"]), []
        )
        self.assertEqual(
            _matching_tool_names(
                report["residual_requirements"],
                ["get_user_details", "cancel_reservation"],
            ),
            {"get_user_details"},
        )
        self.assertEqual(
            [value["id"] for value in _tool_access_residuals([
                report["residual_requirements"][0],
                {
                    "id": "gap-user",
                    "description": "Ask the user for their user_id",
                    "possible_external_access": ["user"],
                },
            ])],
            ["gap-details"],
        )

    def test_tool_candidates_require_a_gap_and_do_not_repeat_semantically(self) -> None:
        tools = [
            {"name": "get_user_details", "short_desc": "Get user details"},
            {"name": "cancel_reservation", "short_desc": "Cancel a reservation"},
        ]
        self.assertEqual(_tool_acquisition_candidates("root", [], tools, set()), [])
        residual = {
            "id": "gap-details",
            "description": "Look up account details",
            "possible_external_access": ["get_user_details tool"],
        }
        candidates = _tool_acquisition_candidates("root", [residual], tools, set())
        self.assertEqual([value["tool_name"] for value in candidates], ["get_user_details"])
        self.assertNotIn("properties", candidates[0]["description"])
        self.assertEqual(
            _tool_acquisition_candidates(
                "root", [residual], tools, {str(candidates[0]["id"])}
            ),
            [],
        )
        self.assertTrue(_is_failed_tool_observation("Error: missing user_id"))
        self.assertTrue(_is_failed_tool_observation('{"error": "invalid id"}'))
        self.assertFalse(_is_failed_tool_observation('{"user_id": "valid"}'))
        reworded = {
            **residual,
            "id": "new-gap-id",
            "description": "Retrieve the same account with slightly different wording",
        }
        self.assertEqual(
            _tool_acquisition_candidates(
                "root", [reworded], tools, {str(candidates[0]["id"])}
            ),
            [],
        )

    def test_optional_communication_candidates_are_sparse_and_acyclic(self) -> None:
        root = {"id": "root", "status": "reported", "report": {"conclusion": "x"}}
        first = {"id": "node-1", "status": "reported", "report": {"conclusion": "y"}}
        second = {"id": "node-2", "status": "ready", "report": None}
        self.assertTrue(_communication_candidate_allowed(
            first, second, ("node-1", "node-2"), set(), set(), set(),
            "connect:node-1:node-2",
        ))
        self.assertFalse(_communication_candidate_allowed(
            second, first, ("node-2", "node-1"), set(), set(), set(),
            "connect:node-2:node-1",
        ))
        self.assertFalse(_communication_candidate_allowed(
            root, second, ("root", "node-2"), set(), {"node-2"}, set(),
            "connect:root:node-2",
        ))

    def test_saved_topology_summary_distinguishes_tree_and_dag(self) -> None:
        nodes = [
            {"id": "root", "depth": 0},
            {"id": "node-1", "depth": 1},
            {"id": "node-2", "depth": 1},
        ]
        derivations = [
            {"from": "root", "to": "node-1"},
            {"from": "root", "to": "node-2"},
        ]
        tree = _topology_summary(nodes, derivations, [], [])
        self.assertEqual(tree["class"], "fan_out_tree")
        self.assertEqual(tree["derived_agent_count"], 2)
        dag = _topology_summary(
            nodes,
            derivations,
            [{"from": "node-1", "to": "node-2"}],
            [{"from": "root", "to": "node-2"}],
        )
        self.assertEqual(dag["class"], "hybrid_dag")

    def test_acquire_generates_arguments_without_delegating_tool_identity(self) -> None:
        arguments = _non_thinking_arguments({
            "max_tokens": 50000,
            "extra_body": {"provider_option": "preserved", "thinking": {"type": "enabled"}},
        })
        self.assertEqual(arguments["max_tokens"], 50000)
        self.assertEqual(arguments["extra_body"]["provider_option"], "preserved")
        self.assertEqual(arguments["extra_body"]["thinking"], {"type": "disabled"})
        prompt = _tool_argument_prompt(
            "get_user_details",
            {"type": "function", "function": {"name": "get_user_details"}},
            {"selected_tool": "get_user_details"},
        )
        self.assertIn("Selected tool: get_user_details", prompt)
        self.assertIn("Return only one JSON object", prompt)
        self.assertIn("Do not return a tool name", prompt)
        bound = _bound_tool_call_payload(
            "update_reservation_flights",
            {"reservation_id": "ABC123"},
            "call-test",
        )
        self.assertEqual(bound["name"], "update_reservation_flights")
        self.assertEqual(bound["arguments"], {"reservation_id": "ABC123"})
        self.assertEqual(bound["requestor"], "assistant")
        stop_token = "###ROY_ORGANIZATION_STOP###"
        marked = _stop_content("final answer", stop_token)
        self.assertTrue(_is_stop_message(Mock(content=marked), stop_token))
        self.assertEqual(_stop_content(marked, stop_token).count(stop_token), 1)

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
                    "scheduler_complexity": 1.0,
                },
                {
                    "id": "derive",
                    "kind": "DERIVE",
                    "description": "resolve the remaining evidence gap",
                    "legal": True,
                    "scheduler_complexity": 3.0,
                    "resolves_gap": True,
                    "depth_delta": 1,
                },
            ],
            "resources": {},
            "node_count": 2,
            "maximum_depth_reached": 1,
            "envelope": ExplorationEnvelope(
                "test", 0, 12, 0, 5, "expansive"
            ).to_dict(),
            "unresolved_gap_exists": True,
            "organization_temperature": 2.0,
        }
        generator = torch.Generator().manual_seed(17)
        _candidate, record = sample_organization_decision(
            model, FakeEncoder(), state, generator=generator, device=torch.device("cpu")
        )
        replayed = replay_joint_log_probability(
            model, FakeEncoder(), record, torch.device("cpu")
        )
        self.assertAlmostEqual(
            float(replayed.detach()), float(record["masked_old_log_probability"]), places=6
        )
        self.assertNotIn("behavior_log_probability", record)
        self.assertNotIn("exploration_alpha", record)

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

    def test_zero_variance_complete_group_does_not_step_optimizer(self) -> None:
        trainer = OrganizationGRPOTrainer.__new__(OrganizationGRPOTrainer)
        trainer.groups = 0
        trainer.trajectories = 0
        trainer.optimizer_steps = 0
        trainer.zero_variance_groups = 0
        trainer.losses = []
        trainer.loss_sum = 0.0
        trainer.updated_group_ids = set()
        trainer.seed = 17
        trainer.save = Mock()
        envelope = training_envelope(0, 1).to_dict()
        records = [{
            "id": f"trajectory-{index}",
            "group_id": "task:epoch",
            "benchmark": "tau3",
            "split": "train",
            "terminal": True,
            "truncated": False,
            "terminal_utility": 0.0,
            "envelope": envelope,
            "rollout_index": index,
            "environment_seed": 9,
            "initial_snapshot_fingerprint": "same",
            "runtime_budget": RuntimeBudget().to_dict(),
            "policy_records": [{"unused": True}],
        } for index in range(8)]
        result = trainer.update_group(records)
        self.assertFalse(result["update_applied"])
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertEqual(trainer.zero_variance_groups, 1)
        trainer.save.assert_called_once()

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
