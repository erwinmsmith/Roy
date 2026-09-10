"""Extract score-blind, root-preserving topology templates from Roy traces."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .matrix import InformationMatrix


@lru_cache(maxsize=32768)
def _canonical(values: tuple[tuple[float, ...], ...], root: int) -> tuple[float, ...]:
    others = [i for i in range(len(values)) if i != root]
    return min(
        tuple(values[i][j] for i in (root, *order) for j in (root, *order))
        for order in itertools.permutations(others)
    )


def canonical_matrix(raw: Mapping[str, Any]) -> InformationMatrix:
    matrix = InformationMatrix(list(raw["agent_ids"]), [list(row) for row in raw["values"]])
    matrix.validate()
    if "A0" not in matrix.agent_ids or not 2 <= len(matrix.agent_ids) <= 5:
        raise ValueError("observed template requires A0 and 2..5 Agents")
    size = len(matrix.agent_ids)
    key = _canonical(tuple(tuple(row) for row in matrix.values), matrix.agent_ids.index("A0"))
    return InformationMatrix(
        [f"A{i}" for i in range(size)],
        [list(key[i * size:(i + 1) * size]) for i in range(size)],
    )


def validate_template(template: Mapping[str, Any], agent_count: int) -> InformationMatrix:
    if template.get("schema_version") != 1 or not template.get("template_id"):
        raise ValueError("observed template requires schema_version=1 and template_id")
    matrix = canonical_matrix(template["matrix"])
    if len(matrix.agent_ids) != agent_count:
        raise ValueError("observed matrix size does not match fixed Agent count")
    if matrix.sources_reaching("A0") != set(matrix.agent_ids):
        raise ValueError("every template Agent must have a directed path to A0")
    return matrix


def build_catalog(paths: list[Path]) -> dict[str, Any]:
    # Later supplied files are repairs/continuations. Keep only the latest
    # completed row per model, benchmark, objective and task id.
    latest = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("run_status", "completed") != "completed":
                    continue
                if row.get("method") != "training_free_information_flow_search":
                    continue
                architecture = row.get("search_architecture", {})
                key = (row.get("worker_model"), row.get("benchmark"),
                       architecture.get("matrix_objective"), row["task_id"])
                # Discard bulky X snapshots and thousands of numerical search
                # intermediates immediately; extraction needs only frontier matrices.
                latest[key] = ({
                    "split": row.get("split", "unknown"),
                    "final_matrix": row.get("final_matrix"),
                    "rounds": [{
                        "round_index": s["round_index"],
                        "committed": s.get("committed", False),
                        "selected_subgraph_id": s.get("selected_subgraph_id"),
                        "baseline": {"matrix": s["baseline"]["matrix"]} if s.get("baseline") else None,
                        "frontiers": {k: {"matrix": v["matrix"]}
                                      for k, v in s.get("frontiers", {}).items()},
                    } for s in row.get("rounds", [])],
                }, str(path))

    entries: dict[tuple[str, str], dict[str, Any]] = {}
    allocated = defaultdict(int)
    for (model, benchmark, objective, task_id), (row, path) in latest.items():
        matrices = [("final", -1, "final", row.get("final_matrix"))]
        for record in row.get("rounds", []):
            index = record["round_index"]
            if record.get("baseline"):
                kind = ("committed" if record.get("committed")
                        and record.get("selected_subgraph_id") is None else "potential")
                matrices.append((kind, index, "baseline", record["baseline"]["matrix"]))
            for frontier_id, frontier in record.get("frontiers", {}).items():
                kind = ("committed" if record.get("committed")
                        and record.get("selected_subgraph_id") == frontier_id else "potential")
                matrices.append((kind, index, frontier_id, frontier["matrix"]))
        for kind, index, frontier_id, raw in matrices:
            if not raw or not 2 <= len(raw["agent_ids"]) <= 5:
                continue
            matrix = canonical_matrix(raw)
            size = len(matrix.agent_ids)
            reaching = len(matrix.sources_reaching("A0"))
            allocated[(benchmark, kind, size, reaching)] += 1
            fingerprint = hashlib.sha256(
                json.dumps(matrix.to_dict(), sort_keys=True).encode()
            ).hexdigest()[:12]
            template_id = f"observed-n{size}-{fingerprint}"
            entry = entries.setdefault((benchmark, template_id), {
                "schema_version": 1,
                "template_id": template_id,
                "benchmark": benchmark,
                "agent_count": size,
                "root_reaching_agents": reaching,
                "matrix": matrix.to_dict(),
                "selection_uses_evaluation_scores": False,
                "source_splits": set(),
                "source_models": set(),
                "source_objectives": set(),
                "task_sets": defaultdict(set),
                "examples": {},
            })
            entry["source_splits"].add(row.get("split", "unknown"))
            entry["source_models"].add(str(model))
            entry["source_objectives"].add(str(objective))
            entry["task_sets"][kind].add(task_id)
            entry["examples"].setdefault(kind, {
                "path": path, "task_id": task_id, "round_index": index,
                "frontier_id": frontier_id, "original_matrix": raw,
            })

    templates = []
    for entry in entries.values():
        entry["task_counts"] = {
            kind: len(entry["task_sets"].get(kind, set()))
            for kind in ("committed", "potential", "final")
        }
        del entry["task_sets"]
        for name in ("source_splits", "source_models", "source_objectives"):
            entry[name] = sorted(entry[name])
        entry["evaluation_protocol"] = (
            "retrospective_test_topology_reuse" if "test" in entry["source_splits"]
            else "optimization_topology_frozen_before_test"
        )
        templates.append(entry)
    selected = []
    for benchmark in sorted({item["benchmark"] for item in templates}):
        for size in (2, 3, 4):
            eligible = [item for item in templates if item["benchmark"] == benchmark
                        and item["agent_count"] == size and item["root_reaching_agents"] == size]
            if not eligible:
                continue
            eligible.sort(key=lambda item: (
                -item["task_counts"]["committed"], -item["task_counts"]["potential"],
                -item["task_counts"]["final"], item["template_id"],
            ))
            selected.append(eligible[0])
    return {
        "schema_version": 1,
        "scope": "optimized_frontier_winners_committed_and_final_matrices",
        "numerical_search_intermediate_matrices_included": False,
        "selection_rule": "root_connected_exact_N_then_task_frequency_committed_potential_final",
        "evaluation_scores_read": False,
        "source_completed_trajectories": len(latest),
        "dimension_counts": [
            {"benchmark": b, "kind": k, "allocated_agents": n,
             "root_reaching_agents": r, "occurrences": c}
            for (b, k, n, r), c in sorted(allocated.items())
        ],
        "selected": selected,
        "templates": sorted(templates, key=lambda item: (item["benchmark"], item["template_id"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    catalog = build_catalog(args.inputs)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    for template in catalog["selected"]:
        name = f"observed{template['agent_count']}-{template['benchmark'].lower()}.json"
        (args.output_directory / name).write_text(json.dumps(template, indent=2) + "\n")
    print(json.dumps({"trajectories": catalog["source_completed_trajectories"],
                      "unique_templates": len(catalog["templates"]),
                      "selected": catalog["selected"]}, indent=2))


if __name__ == "__main__":
    main()
