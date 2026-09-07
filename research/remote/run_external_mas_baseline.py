#!/usr/bin/env python3
"""Run one source-pinned external MAS baseline on AFlow benchmark records."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict

from roy_research.external_baselines import (
    AuditedCaller,
    build_baseline,
    source_fingerprint,
    verify_checkout,
)
from roy_research.io import read_jsonl, write_jsonl
from roy_research.providers import PersistentTokenLedger, ProviderCircuitOpenError
from roy_research.training_free.aflow import AFlowDataset, AFlowEvaluator

from run_aflow_model_eval import build_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dylan", "autoagents", "evoagent"), required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--aflow-root", type=Path, required=True)
    parser.add_argument("--aflow-python", type=Path)
    parser.add_argument("--aflow-manifest", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("MATH", "HumanEval"), required=True)
    parser.add_argument("--split", choices=("optimization", "test"), default="test")
    parser.add_argument("--provider", choices=("deepseek", "openai-compatible"), default="deepseek")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--provider-max-retries", type=int, default=6)
    parser.add_argument("--provider-retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--token-limit", type=int, default=100_000_000)
    parser.add_argument("--max-task-attempts", type=int, default=3)
    parser.add_argument("--evoagent-iterations", type=int, default=3)
    parser.add_argument("--autoagents-review-rounds", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--human-eval-sandbox-command")
    return parser.parse_args()


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["task_id"]) for row in read_jsonl(path)
        if row.get("run_status") == "completed"
    }


def run(args: argparse.Namespace) -> None:
    for value, label in (
        (args.max_output_tokens, "max-output-tokens"),
        (args.max_task_attempts, "max-task-attempts"),
        (args.evoagent_iterations, "evoagent-iterations"),
    ):
        if value < 1:
            raise ValueError(f"{label} must be positive")
    if args.autoagents_review_rounds < 0:
        raise ValueError("autoagents-review-rounds cannot be negative")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --resume")

    baseline_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    spec: Dict[str, Any] = baseline_manifest["baselines"][args.method]
    revision = verify_checkout(args.repository_root, str(spec["revision"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    ledger = PersistentTokenLedger(args.ledger, args.token_limit)
    client = build_client(args, ledger)
    caller = AuditedCaller(
        client,
        method=args.method,
        max_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )
    baseline = build_baseline(
        args.method,
        args.repository_root,
        caller,
        evoagent_iterations=args.evoagent_iterations,
        autoagents_review_rounds=args.autoagents_review_rounds,
    )
    fingerprint = source_fingerprint(baseline.sources)

    dataset = AFlowDataset(args.aflow_root, args.aflow_manifest)
    tasks = dataset.load(args.benchmark, args.split, args.limit)
    evaluator = AFlowEvaluator(
        args.aflow_root,
        args.aflow_python or args.aflow_root / ".venv/bin/python",
        human_eval_sandbox_prefix=(
            shlex.split(args.human_eval_sandbox_command)
            if args.human_eval_sandbox_command else None
        ),
    )
    completed = _completed_ids(args.output) if args.resume else set()

    for task in tasks:
        if task.task_id in completed:
            continue
        before = int(ledger.snapshot()["used"])
        failures = []
        row = None
        for attempt in range(1, args.max_task_attempts + 1):
            call_offset = len(caller.calls)
            try:
                result = baseline.run(task)
                evaluation = evaluator.score(task, result.prediction)
                row = {
                    "schema_version": 1,
                    "method": args.method,
                    "task_id": task.task_id,
                    "benchmark": args.benchmark,
                    "split": args.split,
                    "provider": args.provider,
                    "model": args.model,
                    "run_status": "completed",
                    "execution_attempt": attempt,
                    "failed_attempts": failures,
                    "prediction": result.prediction,
                    "evaluation": evaluation,
                    "algorithm_trace": result.trace,
                    "call_trace": caller.calls[call_offset:],
                }
                break
            except ProviderCircuitOpenError:
                raise
            except Exception as error:
                failures.append({
                    "attempt": attempt,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
        if row is None:
            row = {
                "schema_version": 1,
                "method": args.method,
                "task_id": task.task_id,
                "benchmark": args.benchmark,
                "split": args.split,
                "provider": args.provider,
                "model": args.model,
                "run_status": "failed",
                "failed_attempts": failures,
                "prediction": "",
                "evaluation": {"score": 0.0, "failure": "task_execution_failed"},
            }
        row.update({
            "upstream_repository": spec["repository"],
            "upstream_revision": revision,
            "upstream_integration": spec["integration"],
            "upstream_source_sha256": fingerprint,
            "all_attempts_total_tokens": int(ledger.snapshot()["used"]) - before,
        })
        write_jsonl(args.output, [row], append=args.output.exists())
        print(json.dumps({
            "method": args.method,
            "task_id": task.task_id,
            "run_status": row["run_status"],
            "score": row["evaluation"].get("score"),
            "tokens": row["all_attempts_total_tokens"],
        }), flush=True)


if __name__ == "__main__":
    run(parse_args())
