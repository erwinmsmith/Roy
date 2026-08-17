from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List

from .analysis import paired_bootstrap_interval
from .controlled import TERMINAL_SUCCESS_THRESHOLD


def evaluate_controlled_arms(
    groups: Iterable[Dict[str, Any]], seed: int = 20260815, split: str | None = None
) -> Dict[str, Any]:
    """Evaluate deterministic structural baselines on already matched Q^mu groups.

    Learned Node-only/full scores are added by the training/evaluation command; this
    function provides the non-parametric controls and an oracle diagnostic.
    """
    values = [group for group in groups if split is None or group["task"]["split"] == split]
    randomizer = random.Random(seed)
    policies = {
        "no_derivation": lambda group: "CONTINUE",
        "random_derivation": lambda group: randomizer.choice(list(group["action_values"])),
        "fixed_branch_count": lambda group: "BRANCH",
        "roy_heuristic": _roy_heuristic,
        "rollout_policy_oracle": _joint_oracle,
    }
    rows: Dict[str, Dict[str, Any]] = {}
    baseline_utilities: List[float] = []
    direct_successes: List[float] = []
    for name, policy in policies.items():
        utilities: List[float] = []
        regrets: List[float] = []
        for group in values:
            action_values = {key: float(value) for key, value in group["action_values"].items()}
            selected = policy(group)
            if selected not in action_values:
                selected = "CONTINUE"
            joint_values = dict(action_values)
            branch_values = [float(value) for value in group.get("branch_values", {}).values()]
            if branch_values:
                joint_values["BRANCH"] = max(branch_values)
            selected_utility = joint_values[selected] if name == "rollout_policy_oracle" else action_values[selected]
            oracle_value = max(joint_values.values())
            utilities.append(selected_utility)
            regrets.append(oracle_value - selected_utility)
        if name == "no_derivation":
            baseline_utilities = utilities
            direct_successes = [float(value >= TERMINAL_SUCCESS_THRESHOLD) for value in utilities]
        successes = [float(value >= TERMINAL_SUCCESS_THRESHOLD) for value in utilities]
        rows[name] = {
            "episodes": len(utilities),
            "display_name": "direct (no_derivation)" if name == "no_derivation" else name,
            "mean_utility": sum(utilities) / max(1, len(utilities)),
            "mean_rollout_policy_regret": sum(regrets) / max(1, len(regrets)),
            "success_threshold": TERMINAL_SUCCESS_THRESHOLD,
            "successes": int(sum(successes)),
            "success_rate": sum(successes) / max(1, len(successes)),
            "paired_success_vs_direct": None if name == "no_derivation" else paired_bootstrap_interval(
                successes, direct_successes
            ),
            "paired_vs_no_derivation": None if name == "no_derivation" else paired_bootstrap_interval(
                utilities, baseline_utilities
            ),
        }
    return {"schema_version": 1, "arms": rows, "rollout_policy": "controlled-deterministic-mu-v1"}


def _roy_heuristic(group: Dict[str, Any]) -> str:
    nodes = group["checkpoint"]["event_graph"]["nodes"]
    unresolved = any(node.get("kind") == "dependency" and node.get("status") == "unresolved" for node in nodes)
    if unresolved and "BRANCH" in group["action_values"]:
        return "BRANCH"
    return "RETURN" if "RETURN" in group["action_values"] else "CONTINUE"


def _joint_oracle(group: Dict[str, Any]) -> str:
    values = {key: float(value) for key, value in group["action_values"].items()}
    branches = [float(value) for value in group.get("branch_values", {}).values()]
    if branches:
        values["BRANCH"] = max(branches)
    return max(values, key=values.get)
