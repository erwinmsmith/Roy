from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


TRAIN_EPOCHS = 4
GROUP_SIZE = 8
MAX_ROLLOUT_SECONDS = 6 * 60 * 60
ROLLOUT_FINALIZATION_MARGIN_SECONDS = 10 * 60
CONCURRENCY = 4
MAX_RESPONSE_TOKENS = 32768
SOLVE_THRESHOLD = 0.95


@dataclass(frozen=True)
class ScheduledGroup:
    epoch: int
    task_id: str
    group_id: str
    organization_seeds: Tuple[int, ...]


def build_training_schedule(
    manifest: Sequence[Mapping[str, object]], seed: int = 20260820
) -> List[ScheduledGroup]:
    train = sorted(str(value["task_id"]) for value in manifest if value.get("split") == "train")
    if len(train) != 30:
        raise ValueError("formal LHTB training requires exactly 30 train tasks")
    result: List[ScheduledGroup] = []
    for epoch in range(TRAIN_EPOCHS):
        for task_id in train:
            group_id = f"lhtb:{epoch}:{task_id}"
            seeds = tuple(int(hashlib.sha256(
                f"{seed}:{group_id}:{index}".encode("utf-8")
            ).hexdigest()[:8], 16) for index in range(GROUP_SIZE))
            result.append(ScheduledGroup(epoch, task_id, group_id, seeds))
    if len(result) * GROUP_SIZE != 960:
        raise AssertionError("formal LHTB schedule must contain 960 rollouts")
    return result


def disk_preflight(path: Path, minimum_total_gb: int = 200,
                   minimum_free_fraction: float = 0.15) -> Dict[str, Any]:
    usage = shutil.disk_usage(path)
    gib = 1024 ** 3
    free_fraction = usage.free / max(1, usage.total)
    memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    cpu_count = os.cpu_count() or 0
    architecture = platform.machine()
    result = {
        "path": str(path.resolve()), "architecture": architecture,
        "cpu_count": cpu_count, "memory_gb": memory_bytes / gib,
        "total_gb": usage.total / gib,
        "free_gb": usage.free / gib, "free_fraction": free_fraction,
        "minimum_total_gb": minimum_total_gb, "minimum_free_fraction": minimum_free_fraction,
        "recommended_total_gb": 300,
    }
    if architecture not in ("x86_64", "AMD64"):
        raise RuntimeError("formal LHTB VM must be x86_64")
    if cpu_count < 16:
        raise RuntimeError("formal LHTB VM requires at least 16 CPUs")
    if memory_bytes < 64 * gib:
        raise RuntimeError("formal LHTB VM requires at least 64 GB RAM")
    if usage.total < minimum_total_gb * gib:
        raise RuntimeError(f"LHTB volume is below {minimum_total_gb} GB")
    if free_fraction < minimum_free_fraction:
        raise RuntimeError("LHTB volume has less than 15% free; checkpoint and stop")
    return result


def select_dev_checkpoint(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    valid = [value for value in records if value.get("split") == "dev"]
    if not valid:
        raise ValueError("checkpoint selection requires dev metrics")
    by_epoch: Dict[int, List[Mapping[str, Any]]] = {}
    for value in valid:
        by_epoch.setdefault(int(value["epoch"]), []).append(value)
    candidates = []
    for epoch, values in by_epoch.items():
        task_ids = {str(value["task_id"]) for value in values}
        if len(task_ids) != 8:
            raise ValueError(f"epoch {epoch} does not cover all eight dev tasks")
        candidates.append({
            "epoch": epoch,
            "checkpoint": values[0]["checkpoint"],
            "mean_reward": float(np.mean([float(value["reward"]) for value in values])),
            "tokens": int(sum(int(value["tokens"]) for value in values)),
        })
    return sorted(candidates, key=lambda value: (
        -value["mean_reward"], value["tokens"], value["epoch"]
    ))[0]


def paired_bootstrap(
    learned: Sequence[float], direct: Sequence[float], repeats: int = 10000,
    seed: int = 20260820,
) -> Tuple[float, float]:
    if len(learned) != len(direct) or not learned:
        raise ValueError("paired bootstrap requires aligned non-empty values")
    differences = np.asarray(learned) - np.asarray(direct)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(repeats, len(differences)))
    samples = differences[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_test(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if any(value.get("split") != "test" for value in records):
        raise ValueError("final LHTB report accepts test records only")
    arms = ("single_agent_direct", "roy_runtime_heuristic", "learned_information_realization")
    summary: Dict[str, Any] = {"arms": {}}
    for arm in arms:
        values = [value for value in records if value.get("arm") == arm]
        if not values:
            raise ValueError(f"missing test arm {arm}")
        rewards = [float(value.get("reward", value.get("terminal_reward"))) for value in values]
        summary["arms"][arm] = {
            "episodes": len(values), "mean_reward": float(np.mean(rewards)),
            "success_rate": float(np.mean([reward >= SOLVE_THRESHOLD for reward in rewards])),
            "tokens": int(sum(int(value.get("tokens", 0)) for value in values)),
            "mean_wall_time_seconds": float(np.mean([
                float(value.get("wall_time_seconds", 0)) for value in values
            ])),
            "mean_nodes": float(np.mean([_terminal_graph_size(value)[0] for value in values])),
            "mean_dag_edges": float(np.mean([_terminal_graph_size(value)[1] for value in values])),
            "environment_failures": sum(bool(value.get("environment_failure")) for value in values),
            "mean_process_reward_magnitude": float(np.mean([
                np.mean(np.abs(value.get("process_rewards", [])))
                if value.get("process_rewards") else 0.0 for value in values
            ])),
            "failure_cases": [{"task_id": value.get("task_id"),
                "repeat": value.get("repeat", value.get("rollout_index")),
                "reward": value.get("reward", value.get("terminal_reward")),
                "termination_type": value.get("termination_type"),
                "harbor_result_path": value.get("harbor_result_path")}
                for value in values
                if float(value.get("reward", value.get("terminal_reward"))) < SOLVE_THRESHOLD],
        }
    keyed = lambda arm: {
        (str(value["task_id"]), int(value.get("repeat", value.get("rollout_index", 0)))):
        float(value.get("reward", value.get("terminal_reward")))
        for value in records if value.get("arm") == arm
    }
    direct = keyed("single_agent_direct")
    for arm in arms[1:]:
        compared = keyed(arm)
        keys = sorted(set(direct).intersection(compared))
        interval = paired_bootstrap([compared[key] for key in keys], [direct[key] for key in keys])
        summary["arms"][arm]["paired_difference_vs_direct"] = float(np.mean([
            compared[key] - direct[key] for key in keys
        ]))
        summary["arms"][arm]["paired_bootstrap_95_ci"] = interval
        summary["arms"][arm]["conclusion"] = (
            "inconclusive" if interval[0] <= 0 <= interval[1] else
            "higher" if interval[0] > 0 else "lower"
        )
    return summary


def _terminal_graph_size(record: Mapping[str, Any]) -> Tuple[int, int]:
    states = list(record.get("process_states", []))
    if not states:
        return 0, 0
    state = states[-1]
    return len(state.get("nodes", [])), len(state.get("dagEdges", state.get("dag_edges", [])))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_harbor_group_config(
    path: Path, task_id: str, jobs_dir: Path, arm: str,
    initial_fingerprint: str, organization_seed: int, attempts: int = GROUP_SIZE,
    official_timeout: bool = False, environment_backend: str = "docker",
    native_runtime_root: Path | None = None,
    native_template_root: Path | None = None,
    allow_network_degraded: bool = False,
    max_retries: int = 2,
    concurrency: int = CONCURRENCY,
    dataset_path: str = "./tasks",
    nodewise_macro_steps: int | None = None,
    nodewise_source_snapshot: Path | None = None,
    nodewise_source_checkpoint: Path | None = None,
    nodewise_output_snapshot: Path | None = None,
    nodewise_output_state: Path | None = None,
    nodewise_output_checkpoint: Path | None = None,
) -> None:
    if arm not in ("single_agent_direct", "roy_runtime_heuristic",
                   "learned_information_realization", "frozen_finalize_now",
                   "nodewise_checkpoint_finalize"):
        raise ValueError(f"unknown LHTB arm {arm}")
    if environment_backend == "docker":
        environment = {"type": "docker", "force_build": False, "delete": True}
    elif environment_backend == "native":
        if native_runtime_root is None or native_template_root is None:
            raise ValueError("native Harbor config requires runtime and template roots")
        environment = {
            "import_path": "roy_research.native_environment:NativeProcessEnvironment",
            "force_build": False,
            "delete": True,
            "kwargs": {
                "runtime_root": str(native_runtime_root),
                "template_root": str(native_template_root),
                "allow_network_degraded": allow_network_degraded,
            },
        }
    else:
        raise ValueError(f"unknown LHTB environment backend {environment_backend}")
    if max_retries < 0:
        raise ValueError("Harbor max_retries cannot be negative")
    if concurrency < 1:
        raise ValueError("Harbor concurrency must be positive")
    if arm == "nodewise_checkpoint_finalize":
        if attempts != 1:
            raise ValueError("node-wise checkpoint finalize requires exactly one Harbor attempt")
        if nodewise_macro_steps not in (0, 1):
            raise ValueError("node-wise checkpoint finalize requires zero or one macro step")
        if not all((nodewise_output_snapshot, nodewise_output_state,
                    nodewise_output_checkpoint)):
            raise ValueError("node-wise checkpoint finalize requires all output paths")
        if bool(nodewise_source_snapshot) != bool(nodewise_source_checkpoint):
            raise ValueError("node-wise source snapshot and checkpoint must be paired")
        if nodewise_macro_steps == 1 and not nodewise_source_snapshot:
            raise ValueError("one-step node-wise finalize requires a source checkpoint")
    def agent_config() -> Dict[str, Any]:
        if arm == "frozen_finalize_now":
            return {
                "import_path": "roy_research.harbor_agent:FrozenFinalizeNowAgent",
                "model_name": "frozen/artifact-identity-a0-v1",
                "kwargs": {},
                "env": {"ROY_LHTB_ENVIRONMENT_BACKEND": environment_backend},
            }
        if arm == "nodewise_checkpoint_finalize":
            return {
                "import_path": "roy_research.harbor_agent:NodewiseCheckpointFinalizeAgent",
                "model_name": "deepseek/deepseek-v4-flash",
                "kwargs": {
                    "macro_steps": nodewise_macro_steps,
                    "source_snapshot_path": (str(nodewise_source_snapshot)
                                             if nodewise_source_snapshot else None),
                    "source_checkpoint_path": (str(nodewise_source_checkpoint)
                                               if nodewise_source_checkpoint else None),
                    "output_snapshot_path": str(nodewise_output_snapshot),
                    "output_state_path": str(nodewise_output_state),
                    "output_checkpoint_path": str(nodewise_output_checkpoint),
                    "organization_seed": organization_seed,
                    "initial_snapshot_fingerprint": initial_fingerprint,
                    "environment_revision": "lhtb-pinned",
                    "rpc_timeout": 720,
                },
                "env": {
                    "ROY_LHTB_ENVIRONMENT_BACKEND": environment_backend,
                    "ROY_LHTB_ARM": "learned_information_realization",
                    "ROY_LHTB_INITIAL_FINGERPRINT": initial_fingerprint,
                    "ROY_LHTB_ORGANIZATION_SEED": str(organization_seed),
                },
            }
        kwargs: Dict[str, Any] = {"rpc_timeout": 720}
        agent_env = {"ROY_LHTB_ARM": arm,
                     "ROY_LHTB_ENVIRONMENT_BACKEND": environment_backend,
                     "ROY_LHTB_INITIAL_FINGERPRINT": initial_fingerprint,
                     "ROY_LHTB_ORGANIZATION_SEED": str(organization_seed),
                     "HB_CONTINUE_MODE": "same_conversation"}
        return {
            "import_path": "roy_research.harbor_agent:RoyHarborAgent",
            "model_name": "deepseek/deepseek-v4-flash",
            "kwargs": kwargs,
            "env": agent_env,
        }
    if arm == "learned_information_realization":
        # Every rollout samples the current node actor directly. Distinct
        # organization seeds produce different on-policy trajectories; topology
        # is observed after rollout and is never assigned as an input profile.
        agents = [agent_config() for _ in range(attempts)]
        harbor_attempts = 1
    else:
        agents = [agent_config()]
        harbor_attempts = attempts
    value = {
        "job_name": f"roy-{task_id}-{organization_seed}",
        "jobs_dir": str(jobs_dir), "n_attempts": harbor_attempts,
        "n_concurrent_trials": min(concurrency, attempts), "timeout_multiplier": 1.0,
        "retry": {"max_retries": max_retries},
        "environment": environment,
        "agents": agents,
        "datasets": [{"path": dataset_path, "task_names": [task_id]}],
    }
    if not official_timeout and arm != "frozen_finalize_now":
        for agent in value["agents"]:
            agent["override_timeout_sec"] = MAX_ROLLOUT_SECONDS
            agent["kwargs"]["rollout_timeout_sec"] = (
                MAX_ROLLOUT_SECONDS - ROLLOUT_FINALIZATION_MARGIN_SECONDS
            )
    write_json(path, value)


def write_lhtb_svg(path: Path, summary: Mapping[str, Any]) -> None:
    arms = list(summary.get("arms", {}).items())
    labels = {"single_agent_direct": "Direct", "roy_runtime_heuristic": "Roy heuristic",
              "learned_information_realization": "Learned Roy"}
    colors = ("#667085", "#A66B3D", "#245D75")
    rows = []
    for index, (arm, value) in enumerate(arms):
        reward = float(value["mean_reward"])
        y = 54 + index * 46
        rows.append(f'<text x="20" y="{y+17}" font-family="system-ui" font-size="13">'
                    f'{html.escape(labels.get(arm, arm))}</text>')
        rows.append(f'<rect x="170" y="{y}" width="{reward*500:.1f}" height="24" '
                    f'rx="4" fill="{colors[index % len(colors)]}"/>')
        rows.append(f'<text x="{180+reward*500:.1f}" y="{y+17}" '
                    f'font-family="ui-monospace" font-size="12">{reward:.3f}</text>')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="220">',
           '<rect width="100%" height="100%" fill="#fbfaf7"/>',
           '<text x="20" y="28" font-family="system-ui" font-size="18" font-weight="700">'
           'LHTB final test mean reward</text>', *rows, '</svg>']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
