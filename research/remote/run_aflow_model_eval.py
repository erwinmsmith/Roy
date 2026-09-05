#!/usr/bin/env python3
"""Evaluate an upstream AFlow workflow with Roy's audited model clients.

The adapter leaves the pinned AFlow checkout unchanged, injects credentials only
through the selected environment variable, writes one resumable JSONL row per
task, and uses Roy's isolated HumanEval scorer.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from roy_research.io import read_jsonl, write_jsonl
from roy_research.providers import (
    DeepSeekClient,
    OpenAICompatibleClient,
    PersistentTokenLedger,
)
from roy_research.training_free.aflow import AFlowDataset, AFlowEvaluator


class AFlowModelAdapter:
    """Expose AFlow's small async LLM interface over a Roy client."""

    def __init__(self, client: Any, *, max_tokens: int, temperature: float) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def __call__(self, prompt: str) -> str:
        completion = await asyncio.to_thread(
            self.client.complete,
            [{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            metadata={"purpose": "aflow_workflow_execution"},
        )
        self.total_input_tokens += completion.prompt_tokens
        self.total_output_tokens += completion.completion_tokens
        return completion.content

    async def call_with_format(self, prompt: str, formatter: Any) -> Any:
        response = await self(formatter.prepare_prompt(prompt))
        valid, parsed = formatter.validate_response(response)
        if not valid:
            from scripts.formatter import FormatError

            raise FormatError(f"{formatter.format_error_message()}. Raw response: {response}")
        return parsed

    def get_usage_summary(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            # AFlow expects this field, but an unknown endpoint price must not
            # be silently replaced by a fabricated cost.
            "total_cost": 0.0,
            "call_count": None,
            "history": [],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aflow-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("MATH", "HumanEval"), required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--provider", choices=("deepseek", "openai-compatible"), default="deepseek")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--provider-max-retries", type=int, default=6)
    parser.add_argument("--provider-retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--token-limit", type=int, default=20_000_000)
    parser.add_argument("--max-task-attempts", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--human-eval-sandbox-command")
    return parser.parse_args()


def build_client(args: argparse.Namespace, ledger: PersistentTokenLedger) -> Any:
    common = {
        "model": args.model,
        "timeout": args.timeout,
        "event_log": args.events,
        "max_retries": args.provider_max_retries,
        "retry_base_seconds": args.provider_retry_base_seconds,
    }
    if args.provider == "deepseek":
        if args.base_url:
            common["base_url"] = args.base_url
        return DeepSeekClient(ledger, **common)
    if not args.base_url:
        raise ValueError("--base-url is required for openai-compatible providers")
    return OpenAICompatibleClient(
        ledger,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_output_tokens=args.max_output_tokens,
        **common,
    )


async def run(args: argparse.Namespace) -> None:
    if args.round < 1 or args.max_output_tokens < 1 or args.max_task_attempts < 1:
        raise ValueError("round, max-output-tokens, and max-task-attempts must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --resume")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.events.parent.mkdir(parents=True, exist_ok=True)

    aflow_root = args.aflow_root.resolve()
    sys.path.insert(0, str(aflow_root))
    os.chdir(aflow_root)

    ledger = PersistentTokenLedger(args.ledger, args.token_limit)
    client = build_client(args, ledger)
    adapter = AFlowModelAdapter(
        client, max_tokens=args.max_output_tokens, temperature=args.temperature,
    )

    # Patch only the factory imported by the selected upstream graph. AFlow's
    # workflow and operators remain the source of execution semantics.
    async_llm = importlib.import_module("scripts.async_llm")
    async_llm.create_llm_instance = lambda _config: adapter
    graph_module = importlib.import_module(
        f"workspace.{args.benchmark}.workflows.round_{args.round}.graph"
    )
    workflow = graph_module.Workflow(
        name=args.benchmark, llm_config=None, dataset=args.benchmark,
    )

    dataset = AFlowDataset(aflow_root, args.manifest)
    tasks = dataset.load(args.benchmark, "test", args.limit)
    completed_ids = {
        str(row.get("task_id")) for row in read_jsonl(args.output)
    } if args.resume and args.output.exists() else set()
    evaluator = AFlowEvaluator(
        aflow_root,
        aflow_root / ".venv/bin/python",
        human_eval_sandbox_prefix=(
            shlex.split(args.human_eval_sandbox_command)
            if args.human_eval_sandbox_command else None
        ),
    )

    for task in tasks:
        if task.task_id in completed_ids:
            continue
        before = int(ledger.snapshot()["used"])
        failures = []
        row = None
        for attempt in range(1, args.max_task_attempts + 1):
            try:
                if args.benchmark == "MATH":
                    prediction, _cost = await workflow(task.instruction)
                else:
                    prediction, _cost = await workflow(
                        task.instruction, task.evaluator_payload["entry_point"]
                    )
                evaluation = evaluator.score(task, str(prediction))
                row = {
                    "schema_version": 1,
                    "method": f"aflow_round_{args.round}",
                    "task_id": task.task_id,
                    "benchmark": args.benchmark,
                    "split": "test",
                    "provider": args.provider,
                    "model": args.model,
                    "workflow_round": args.round,
                    "run_status": "completed",
                    "execution_attempt": attempt,
                    "failed_attempts": failures,
                    "prediction": str(prediction),
                    "evaluation": evaluation,
                }
                break
            except Exception as error:  # keep AFlow's per-sample total behavior
                failures.append({
                    "attempt": attempt,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
        if row is None:
            row = {
                "schema_version": 1,
                "method": f"aflow_round_{args.round}",
                "task_id": task.task_id,
                "benchmark": args.benchmark,
                "split": "test",
                "provider": args.provider,
                "model": args.model,
                "workflow_round": args.round,
                "run_status": "failed",
                "failed_attempts": failures,
                "prediction": "",
                "evaluation": {"score": 0.0, "failure": "task_execution_failed"},
            }
        row["all_attempts_total_tokens"] = int(ledger.snapshot()["used"]) - before
        write_jsonl(args.output, [row], append=args.output.exists())
        print(json.dumps({
            "task_id": task.task_id,
            "run_status": row["run_status"],
            "score": row["evaluation"].get("score"),
            "tokens": row["all_attempts_total_tokens"],
        }), flush=True)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
