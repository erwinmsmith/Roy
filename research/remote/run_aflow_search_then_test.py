#!/usr/bin/env python3
"""Run leakage-safe AFlow search, freeze the winner, then evaluate test once.

The pinned AFlow checkout is treated as read-only.  A private ``workspace``
package is staged below ``--run-root`` because upstream AFlow writes generated
Python workflows in place.  Every search round is scored only on AFlow's
``*_validate.jsonl`` optimization split.  The test split is loaded only after
the best round has been selected from validation scores.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from roy_research.io import atomic_json, read_jsonl, write_jsonl
from roy_research.providers import PersistentTokenLedger
from roy_research.training_free.aflow import AFlowDataset, AFlowEvaluator

from run_aflow_model_eval import AFlowModelAdapter, build_client


EXPERIMENTS = {
    "MATH": ("math", ["Custom", "ScEnsemble", "Programmer"]),
    "HumanEval": ("code", ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aflow-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--provider", choices=("deepseek", "openai-compatible"), default="deepseek")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--provider-max-retries", type=int, default=6)
    parser.add_argument("--provider-retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--token-limit", type=int, default=100_000_000)
    parser.add_argument("--max-task-attempts", type=int, default=5)
    parser.add_argument("--sample", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--validation-rounds", type=int, default=1)
    parser.add_argument("--optimization-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aflow-python", type=Path)
    parser.add_argument("--human-eval-sandbox-command")
    return parser.parse_args()


def _copy_init(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, destination)
    else:
        destination.write_text("", encoding="utf-8")


def stage_workspace(aflow_root: Path, run_root: Path, benchmark: str, resume: bool) -> None:
    workspace = run_root / "workspace"
    if run_root.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {run_root}; pass --resume")
    data_source = aflow_root / "data"
    if not data_source.is_dir():
        raise FileNotFoundError(f"missing AFlow data directory: {data_source}")
    run_root.mkdir(parents=True, exist_ok=True)
    data_link = run_root / "data"
    if not data_link.exists():
        data_link.symlink_to(data_source, target_is_directory=True)
    if workspace.exists():
        return
    source = aflow_root / "workspace" / benchmark / "workflows"
    if not source.exists():
        raise FileNotFoundError(f"missing AFlow seed workflows: {source}")
    target = workspace / benchmark / "workflows"
    target.mkdir(parents=True, exist_ok=True)
    _copy_init(aflow_root / "workspace" / "__init__.py", workspace / "__init__.py")
    _copy_init(
        aflow_root / "workspace" / benchmark / "__init__.py",
        workspace / benchmark / "__init__.py",
    )
    _copy_init(source / "__init__.py", target / "__init__.py")
    shutil.copytree(source / "template", target / "template")
    shutil.copytree(source / "round_1", target / "round_1")


class AuditedEvaluationUtils:
    """AFlow EvaluationUtils replacement with serial, resumable audit rows."""

    def __init__(
        self,
        *,
        benchmark: str,
        optimization_tasks: list[Any],
        test_tasks: list[Any] | None,
        evaluator: AFlowEvaluator,
        ledger: PersistentTokenLedger,
        max_task_attempts: int,
        validation_rounds: int,
    ) -> None:
        self.benchmark = benchmark
        self.optimization_tasks = optimization_tasks
        self.test_tasks = test_tasks
        self.evaluator = evaluator
        self.ledger = ledger
        self.max_task_attempts = max_task_attempts
        self.validation_rounds = validation_rounds

    async def _evaluate_tasks(
        self,
        graph_class: Any,
        tasks: list[Any],
        output: Path,
        *,
        split: str,
        workflow_round: int,
        resume: bool = False,
    ) -> float:
        workflow = graph_class(
            name=self.benchmark, llm_config=_EXEC_CONFIG, dataset=self.benchmark,
        )
        previous = list(read_jsonl(output)) if resume and output.exists() else []
        completed = {str(row["task_id"]): row for row in previous}
        scores = [float(row.get("evaluation", {}).get("score", 0.0)) for row in previous]
        mismatches: list[dict[str, Any]] = []
        for task in tasks:
            if task.task_id in completed:
                continue
            before = int(self.ledger.snapshot()["used"])
            failures: list[dict[str, Any]] = []
            prediction = ""
            evaluation: dict[str, Any] = {"score": 0.0, "failure": "task_execution_failed"}
            status = "failed"
            successful_attempt = None
            for attempt in range(1, self.max_task_attempts + 1):
                try:
                    if self.benchmark == "MATH":
                        prediction, _cost = await workflow(task.instruction)
                    else:
                        prediction, _cost = await workflow(
                            task.instruction, task.evaluator_payload["entry_point"]
                        )
                    evaluation = self.evaluator.score(task, str(prediction))
                    status = "completed"
                    successful_attempt = attempt
                    break
                except Exception as error:
                    failures.append({
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
            score = float(evaluation.get("score", 0.0))
            row = {
                "schema_version": 1,
                "method": "aflow_search" if split == "optimization" else "aflow_frozen_test",
                "task_id": task.task_id,
                "benchmark": self.benchmark,
                "split": split,
                "workflow_round": workflow_round,
                "run_status": status,
                "execution_attempt": successful_attempt,
                "failed_attempts": failures,
                "prediction": str(prediction),
                "evaluation": evaluation,
                "all_attempts_total_tokens": int(self.ledger.snapshot()["used"]) - before,
            }
            write_jsonl(output, [row], append=output.exists())
            scores.append(score)
            if score == 0.0:
                mismatches.append({
                    "question": task.instruction,
                    "right_answer": task.evaluator_payload.get("solution", "evaluator-only"),
                    "model_output": str(prediction),
                    "extracted_output": evaluation.get("extracted_prediction", evaluation.get("details", "")),
                })
            print(json.dumps({
                "phase": split,
                "round": workflow_round,
                "task_id": task.task_id,
                "run_status": status,
                "score": score,
                "completed": len(scores),
                "total": len(tasks),
            }), flush=True)
        if split == "optimization":
            # Upstream expects a list of mismatch dictionaries, not a wrapper.
            (output.parent / "log.json").write_text(
                json.dumps(mismatches, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return sum(scores) / len(tasks) if tasks else 0.0

    async def evaluate_graph(
        self, optimizer: Any, directory: str, validation_n: int, data: list[Any], initial: bool = False,
    ) -> float:
        round_number = optimizer.round if initial else optimizer.round + 1
        scores = []
        for repetition in range(validation_n):
            output = Path(directory) / f"optimization-{repetition}.jsonl"
            score = await self._evaluate_tasks(
                optimizer.graph,
                self.optimization_tasks,
                output,
                split="optimization",
                workflow_round=round_number,
            )
            scores.append(score)
            data.append(optimizer.data_utils.create_result_data(round_number, score, 0.0, 0.0))
            result_path = optimizer.data_utils.get_results_file_path(
                f"{optimizer.root_path}/workflows"
            )
            optimizer.data_utils.save_results(result_path, data)
        return sum(scores) / len(scores)

    async def evaluate_initial_round(
        self, optimizer: Any, graph_path: str, directory: str, validation_n: int, data: list[Any],
    ) -> list[Any]:
        await self.evaluate_graph(optimizer, directory, validation_n, data, initial=True)
        return data

    async def evaluate_frozen_test(self, graph_class: Any, round_number: int, output: Path) -> float:
        if self.test_tasks is None:
            raise RuntimeError("test tasks must be loaded only after workflow selection")
        return await self._evaluate_tasks(
            graph_class,
            self.test_tasks,
            output,
            split="test",
            workflow_round=round_number,
            resume=True,
        )


def select_round(results_path: Path) -> tuple[int, dict[str, float]]:
    values = json.loads(results_path.read_text(encoding="utf-8"))
    scores: dict[int, list[float]] = defaultdict(list)
    for value in values:
        scores[int(value["round"])].append(float(value["score"]))
    if not scores:
        raise RuntimeError(f"no successful validation scores in {results_path}")
    means = {round_number: sum(items) / len(items) for round_number, items in scores.items()}
    winner = min(means, key=lambda round_number: (-means[round_number], round_number))
    return winner, {str(key): value for key, value in sorted(means.items())}


def workflow_fingerprint(run_root: Path, benchmark: str, round_number: int) -> dict[str, str]:
    directory = run_root / "workspace" / benchmark / "workflows" / f"round_{round_number}"
    result = {}
    for name in ("graph.py", "prompt.py"):
        path = directory / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


_OPT_CONFIG = object()
_EXEC_CONFIG = object()


def main() -> None:
    args = parse_args()
    if min(
        args.max_output_tokens,
        args.max_task_attempts,
        args.sample,
        args.max_rounds,
        args.validation_rounds,
    ) < 1:
        raise ValueError("token, attempt, sample, and round limits must be positive")
    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass

    aflow_root = args.aflow_root.resolve()
    manifest = args.manifest.resolve()
    run_root = args.run_root.resolve()
    stage_workspace(aflow_root, run_root, args.benchmark, args.resume)

    sys.path.insert(0, str(aflow_root))
    sys.path.insert(0, str(run_root))
    os.chdir(run_root)

    ledger = PersistentTokenLedger(run_root / "tokens.json", args.token_limit)
    args.events = run_root / "events.jsonl"
    client = build_client(args, ledger)
    opt_adapter = AFlowModelAdapter(
        client,
        max_tokens=args.max_output_tokens,
        temperature=args.temperature,
        purpose="aflow_workflow_optimization",
    )
    exec_adapter = AFlowModelAdapter(
        client,
        max_tokens=args.max_output_tokens,
        temperature=args.temperature,
        purpose="aflow_workflow_execution",
    )

    def factory(config: Any) -> AFlowModelAdapter:
        return opt_adapter if config is _OPT_CONFIG else exec_adapter

    async_llm = importlib.import_module("scripts.async_llm")
    async_llm.create_llm_instance = factory
    optimizer_module = importlib.import_module("scripts.optimizer")
    optimizer_module.create_llm_instance = factory
    upstream_extract = optimizer_module.Optimizer._extract_fields_from_response

    def strict_extract(instance: Any, response: str) -> dict[str, str] | None:
        parsed = upstream_extract(instance, response)
        required = ("modification", "graph", "prompt")
        if not parsed or any(not str(parsed.get(field, "")).strip() for field in required):
            return None
        return parsed

    optimizer_module.Optimizer._extract_fields_from_response = strict_extract

    dataset = AFlowDataset(aflow_root, manifest)
    optimization_tasks = dataset.load(
        args.benchmark, "optimization", args.optimization_limit,
    )
    aflow_python = (args.aflow_python or (aflow_root / ".venv/bin/python")).expanduser()
    evaluator = AFlowEvaluator(
        aflow_root,
        aflow_python,
        human_eval_sandbox_prefix=(
            __import__("shlex").split(args.human_eval_sandbox_command)
            if args.human_eval_sandbox_command else None
        ),
    )
    evaluation_utils = AuditedEvaluationUtils(
        benchmark=args.benchmark,
        optimization_tasks=optimization_tasks,
        test_tasks=None,
        evaluator=evaluator,
        ledger=ledger,
        max_task_attempts=args.max_task_attempts,
        validation_rounds=args.validation_rounds,
    )

    question_type, operators = EXPERIMENTS[args.benchmark]
    optimizer = optimizer_module.Optimizer(
        dataset=args.benchmark,
        question_type=question_type,
        opt_llm_config=_OPT_CONFIG,
        exec_llm_config=_EXEC_CONFIG,
        check_convergence=True,
        operators=operators,
        optimized_path="workspace",
        sample=args.sample,
        initial_round=1,
        max_rounds=args.max_rounds,
        validation_rounds=args.validation_rounds,
    )
    optimizer.evaluation_utils = evaluation_utils
    optimizer.optimize("Graph")

    results_path = run_root / "workspace" / args.benchmark / "workflows" / "results.json"
    winner, validation_scores = select_round(results_path)
    frozen_files = workflow_fingerprint(run_root, args.benchmark, winner)
    selection = {
        "benchmark": args.benchmark,
        "selection_split": "optimization",
        "selected_round": winner,
        "selected_score": validation_scores[str(winner)],
        "validation_scores": validation_scores,
        "frozen_files_sha256": frozen_files,
        "test_accessed": False,
        "seed": args.seed,
    }
    atomic_json(run_root / "selection.json", selection)
    # This is the first test-split access.  The workflow round is already frozen
    # in selection.json and cannot be changed using test outcomes.
    test_tasks = dataset.load(args.benchmark, "test", args.test_limit)
    evaluation_utils.test_tasks = test_tasks
    importlib.invalidate_caches()
    graph_class = optimizer.graph_utils.load_graph(
        winner, f"workspace/{args.benchmark}/workflows"
    )
    test_score = asyncio.run(
        evaluation_utils.evaluate_frozen_test(graph_class, winner, run_root / "final-test.jsonl")
    )
    if workflow_fingerprint(run_root, args.benchmark, winner) != frozen_files:
        raise RuntimeError("frozen workflow changed during test evaluation")
    selection["test_score"] = test_score
    selection["test_records"] = len(test_tasks)
    selection["test_accessed"] = True
    atomic_json(run_root / "selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
