from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def paired_bootstrap_interval(
    left: Sequence[float],
    right: Sequence[float],
    samples: int = 2_000,
    seed: int = 20260815,
) -> Dict[str, float | str]:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must be non-empty and equal length")
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    randomizer = np.random.default_rng(seed)
    means = np.asarray([
        float(randomizer.choice(differences, size=len(differences), replace=True).mean())
        for _ in range(samples)
    ])
    low, high = np.quantile(means, [0.025, 0.975])
    conclusion = "positive" if low > 0 else "negative" if high < 0 else "inconclusive"
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "conclusion": conclusion,
    }


def summarize_groups(groups: Iterable[Dict[str, object]]) -> Dict[str, object]:
    values = list(groups)
    by_family: Dict[str, int] = {}
    reconstruction_errors: List[float] = []
    acquisitions: List[float] = []
    activations: List[float] = []
    utilities: List[float] = []
    work: List[int] = []
    spans: List[int] = []
    nodes: List[int] = []
    communication_edges: List[int] = []
    waits: List[int] = []
    redundant_branches: List[int] = []
    for group in values:
        task = dict(group.get("task", {}))
        family = str(task.get("family", "unknown"))
        by_family[family] = by_family.get(family, 0) + 1
        diagnostic = dict(group.get("mechanism_diagnostics", {}))
        if "reconstruction_error" in diagnostic:
            reconstruction_errors.append(float(diagnostic["reconstruction_error"]))
            acquisitions.append(float(diagnostic.get("acquisition", 0.0)))
            activations.append(float(diagnostic.get("activation", 0.0)))
        results = list(group.get("results", []))
        utilities.extend(float(result["utility"]) for result in results)
        work.append(len(results))
        spans.append(len({int(result.get("repeat", 0)) for result in results}))
        waits.append(sum(int(result.get("wait_ms", 0)) for result in results))
        branch_values = list(dict(group.get("branch_values", {})).values())
        redundant_branches.append(max(0, len(branch_values) - 1))
        checkpoint = dict(group.get("checkpoint", {}))
        event_graph = dict(checkpoint.get("event_graph", {}))
        nodes.append(len(event_graph.get("nodes", [])))
        communication_edges.append(sum(
            1 for edge in event_graph.get("edges", []) if edge.get("kind") == "communication"
        ))
    return {
        "groups": len(values),
        "by_family": by_family,
        "max_reconstruction_error": max(reconstruction_errors, default=0.0),
        "mean_g_acq": float(np.mean(acquisitions)) if acquisitions else 0.0,
        "mean_g_act": float(np.mean(activations)) if activations else 0.0,
        "mean_rollout_utility": float(np.mean(utilities)) if utilities else 0.0,
        "mean_work": float(np.mean(work)) if work else 0.0,
        "mean_span": float(np.mean(spans)) if spans else 0.0,
        "mean_nodes": float(np.mean(nodes)) if nodes else 0.0,
        "mean_wait_ms": float(np.mean(waits)) if waits else 0.0,
        "mean_redundant_branches": float(np.mean(redundant_branches)) if redundant_branches else 0.0,
        "mean_communication_edges": float(np.mean(communication_edges)) if communication_edges else 0.0,
    }
