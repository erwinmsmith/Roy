from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "remote"))
from summarize_mas_results import summarize

from roy_research.io import write_jsonl


def test_summary_deduplicates_resumed_tasks_and_counts_tokens(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    write_jsonl(path, [
        {"method": "dylan", "model": "m", "benchmark": "MATH", "task_id": "1",
         "run_status": "failed", "evaluation": {"score": 0}, "all_attempts_total_tokens": 30},
        {"method": "dylan", "model": "m", "benchmark": "MATH", "task_id": "1",
         "run_status": "completed", "evaluation": {"score": 1}, "all_attempts_total_tokens": 20},
        {"method": "dylan", "model": "m", "benchmark": "MATH", "task_id": "2",
         "run_status": "completed", "evaluation": {"score": 0}, "all_attempts_total_tokens": 10},
    ])
    result = summarize([path])[0]
    assert result["observed_tasks"] == 2
    assert result["accuracy"] == 0.5
    assert result["total_tokens"] == 30
