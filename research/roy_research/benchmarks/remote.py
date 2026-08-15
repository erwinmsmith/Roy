from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RemoteBenchmark:
    id: str
    repository: str
    revision: str
    license: str
    setup: List[str]
    run_template: List[str]
    result_glob: str


REMOTE_BENCHMARKS = {
    "tau-knowledge": RemoteBenchmark(
        id="tau-knowledge",
        repository="https://github.com/sierra-research/tau2-bench.git",
        revision="c3398666e6559e3a063da3fc04b5acf7f941464e",
        license="repository-defined; verify before redistribution",
        setup=["uv sync --extra knowledge"],
        run_template=[
            "uv", "run", "tau2", "run", "--domain", "banking_knowledge",
            "--agent-llm", "deepseek/deepseek-v4-flash",
            "--user-llm", "deepseek/deepseek-v4-flash",
            "--num-trials", "3", "--num-tasks", "5",
        ],
        result_glob="data/simulations/**/*",
    ),
    "tua": RemoteBenchmark(
        id="tua",
        repository="https://github.com/facebookresearch/TUA-Bench.git",
        revision="3497fd320abcafaf4797424192c891a593fd7964",
        license="CC-BY-NC",
        setup=["uv run setup-env"],
        run_template=[
            "uv", "run", "harbor", "run", "-p", "tasks",
            "-a", "terminus-2", "-m", "deepseek/deepseek-v4-flash",
            "-o", "jobs/roy-cs-grpo-pilot", "--yes",
        ],
        result_glob="jobs/roy-cs-grpo-pilot/**/*",
    ),
}


def build_remote_manifest(token_limit: int = 10_000_000) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "token_limit": token_limit,
        "task_selection": {"strategy": "pinned_prefix", "count": 5, "seed": 20260815},
        "repeats": 3,
        "arms": ["no_derivation", "roy_heuristic", "node_only", "full_v0_v4"],
        "expected_episodes": 120,
        "benchmarks": [asdict(benchmark) for benchmark in REMOTE_BENCHMARKS.values()],
        "notes": [
            "Voice mode is intentionally excluded from the first Roy pilot.",
            "The runner must persist the token ledger and stop before the hard limit.",
            "Container image digests are captured during prepare and verified before run.",
            "ROY_DRY_RUN=1 prints the complete 120-episode matrix without API calls.",
            "Do not redistribute benchmark assets in Roy artifacts.",
        ],
    }
