#!/usr/bin/env python3
"""Summarize comparable MAS JSONL outputs without loading model event logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from roy_research.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args()


def _key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    method = str(row.get("method") or row.get("arm") or "unknown")
    if method == "fixed_mas_star":
        protocol = row.get("fixed_mas_protocol", {})
        count = protocol.get("agent_count") if isinstance(protocol, dict) else None
        method = f"fixed_mas_star_n{count}" if count is not None else method
    benchmark = str(row.get("benchmark") or "unknown")
    model = str(row.get("model") or row.get("worker_model") or "unknown")
    return model, benchmark, method


def summarize(paths: Iterable[Path]) -> list[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        for row in read_jsonl(path):
            task_id = str(row.get("task_id", ""))
            if task_id:
                grouped[_key(row)][task_id] = row
    summaries = []
    for (model, benchmark, method), rows_by_task in sorted(grouped.items()):
        rows = list(rows_by_task.values())
        completed = [row for row in rows if row.get("run_status", "completed") == "completed"]
        scores = [float(row.get("evaluation", {}).get("score", 0.0)) for row in completed]
        summaries.append({
            "model": model,
            "benchmark": benchmark,
            "method": method,
            "observed_tasks": len(rows),
            "completed_tasks": len(completed),
            "correct": sum(score >= 1.0 for score in scores),
            "accuracy": sum(scores) / len(scores) if scores else None,
            "total_tokens": sum(int(row.get("all_attempts_total_tokens", 0)) for row in rows),
            "mean_tokens_per_completed_task": (
                sum(int(row.get("all_attempts_total_tokens", 0)) for row in completed) / len(completed)
                if completed else None
            ),
        })
    return summaries


if __name__ == "__main__":
    print(json.dumps(summarize(parse_args().inputs), ensure_ascii=False, indent=2))
