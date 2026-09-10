from __future__ import annotations

import copy

import pytest

from roy_research.io import write_jsonl
from roy_research.training_free.observed_topology import (
    build_catalog, canonical_matrix, validate_template,
)


def test_canonicalization_preserves_root_direction_and_exact_weights():
    first = {"agent_ids": ["child", "A0", "parent"],
             "values": [[0, 0, 0.5], [0, 0, 0], [0, 1, 0]]}
    relabeled = {"agent_ids": ["A0", "other", "leaf"],
                 "values": [[0, 0, 0], [1, 0, 0], [0, 0.5, 0]]}
    assert canonical_matrix(first) == canonical_matrix(relabeled)
    assert canonical_matrix(first).positive_edge_count() == 2
    assert canonical_matrix(first).total_capacity() == 1.5


def test_catalog_deduplicates_repairs_and_never_selects_by_answer_score(tmp_path):
    chain = {"agent_ids": ["A0", "x", "y"], "values": [[0, 0, 0], [0, 0, 1], [1, 0, 0]]}
    dormant = {"agent_ids": ["A0", "x", "y"], "values": [[0, 0, 0], [1, 0, 0], [0, 0, 0]]}
    row = {
        "task_id": "t1", "worker_model": "model", "benchmark": "MATH", "split": "test",
        "method": "training_free_information_flow_search", "run_status": "completed",
        "search_architecture": {"matrix_objective": "precision_logdet"},
        "evaluation": {"score": 1}, "final_matrix": chain,
        "rounds": [{"round_index": 0, "committed": True, "selected_subgraph_id": "f",
                    "frontiers": {"f": {"matrix": chain}, "g": {"matrix": dormant}}}],
    }
    path = tmp_path / "traces.jsonl"
    write_jsonl(path, [row, row])
    first = build_catalog([path])
    assert first["source_completed_trajectories"] == 1
    assert len(first["selected"]) == 1
    selected = first["selected"][0]
    assert selected["root_reaching_agents"] == 3
    assert selected["task_counts"]["committed"] == 1
    assert selected["evaluation_protocol"] == "retrospective_test_topology_reuse"
    changed = copy.deepcopy(row)
    changed["evaluation"] = {"score": 0}
    write_jsonl(path, [changed])
    assert build_catalog([path]) == first


def test_template_rejects_dormant_nodes_and_count_mismatch():
    template = {"schema_version": 1, "template_id": "example", "matrix": {
        "agent_ids": ["A0", "A1", "A2"], "values": [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    }}
    with pytest.raises(ValueError, match="directed path"):
        validate_template(template, 3)
    with pytest.raises(ValueError, match="size"):
        validate_template(template, 2)
