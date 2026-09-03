from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .types import BenchmarkTask


class AFlowDataset:
    """Leakage-safe loader for Roy's pinned AFlow MATH and HumanEval records."""

    def __init__(self, root: Path, manifest_path: Path) -> None:
        self.root = root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def verify(self, benchmark: str, split: str) -> None:
        if benchmark not in ("MATH", "HumanEval"):
            raise ValueError("training-free V1 supports only MATH and HumanEval")
        if split not in ("optimization", "test"):
            raise ValueError("split must be optimization or test")
        expected_revision = str(self.manifest["source"]["revision"])
        revision = subprocess.check_output(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True,
        ).strip()
        if revision != expected_revision:
            raise ValueError(f"AFlow revision mismatch: {revision} != {expected_revision}")
        spec = self.manifest["benchmarks"][benchmark][split]
        path = self.root / spec["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"AFlow dataset hash mismatch: {path}")

    def load(self, benchmark: str, split: str, limit: int | None = None) -> List[BenchmarkTask]:
        self.verify(benchmark, split)
        spec = self.manifest["benchmarks"][benchmark][split]
        records = _read_jsonl(self.root / spec["path"])
        if limit is not None:
            records = records[:limit]
        public_tests = self._public_tests() if benchmark == "HumanEval" else {}
        tasks: List[BenchmarkTask] = []
        for index, record in enumerate(records):
            if benchmark == "MATH":
                task_id = f"MATH/{split}/{index}"
                tasks.append(BenchmarkTask(
                    task_id=task_id,
                    benchmark="MATH",
                    instruction=str(record["problem"]),
                    public_tests=[],
                    evaluator_payload={"solution": record["solution"]},
                ))
            else:
                task_id = str(record["task_id"])
                tasks.append(BenchmarkTask(
                    task_id=task_id,
                    benchmark="HumanEval",
                    instruction=str(record["prompt"]),
                    public_tests=list(public_tests.get(task_id, [])),
                    evaluator_payload={
                        "entry_point": record["entry_point"],
                        "canonical_solution": record["canonical_solution"],
                        "test": record["test"],
                    },
                ))
        return tasks

    def _public_tests(self) -> Dict[str, List[str]]:
        spec = self.manifest["benchmarks"]["HumanEval"]["public_tests"]
        path = self.root / spec["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"AFlow public-test hash mismatch: {path}")
        return {
            str(record["problem_id"]): [str(test) for test in record.get("test", [])]
            for record in _read_jsonl(path)
        }


class AFlowEvaluator:
    """Invoke the pinned upstream scorers; HumanEval must have an isolation prefix."""

    def __init__(
        self,
        root: Path,
        python: Path,
        *,
        human_eval_sandbox_prefix: Sequence[str] | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.root = root.resolve()
        # Keep the virtual-environment launcher path intact. Resolving its symlink
        # silently switches to the base interpreter and loses the venv packages.
        self.python = python.expanduser().absolute()
        self.sandbox_prefix = list(human_eval_sandbox_prefix or [])
        self.timeout_seconds = timeout_seconds

    def score(self, task: BenchmarkTask, prediction: str) -> Dict[str, Any]:
        if task.benchmark == "MATH":
            script = """
import json, sys
from benchmarks.math import MATHBenchmark
value = json.load(sys.stdin)
benchmark = MATHBenchmark('MATH', '/dev/null', '/tmp')
score, extracted = benchmark.calculate_score(value['expected'], value['prediction'])
print(json.dumps({'score': float(score), 'extracted_prediction': extracted}))
"""
            return self._invoke(script, {
                "expected": task.evaluator_payload["solution"], "prediction": prediction,
            })
        if not self.sandbox_prefix:
            raise RuntimeError(
                "HumanEval executes generated code; configure a reviewed process-isolation prefix"
            )
        script = """
import json, sys
from benchmarks.humaneval import HumanEvalBenchmark
value = json.load(sys.stdin)
benchmark = HumanEvalBenchmark('HumanEval', '/dev/null', '/tmp')
status, details = benchmark.check_solution(
    value['prediction'], value['test'], value['entry_point']
)
print(json.dumps({'score': 1.0 if status == benchmark.PASS else 0.0, 'details': details}))
"""
        return self._invoke(script, {
            "prediction": prediction,
            "test": task.evaluator_payload["test"],
            "entry_point": task.evaluator_payload["entry_point"],
        }, prefix=self.sandbox_prefix)

    def _invoke(
        self,
        script: str,
        payload: Dict[str, Any],
        *,
        prefix: Sequence[str] = (),
    ) -> Dict[str, Any]:
        command = [*prefix, str(self.python), "-c", script]
        completed = subprocess.run(
            command,
            cwd=self.root,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"AFlow scorer failed ({completed.returncode}): {completed.stderr.strip()}"
            )
        return json.loads(completed.stdout)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object AFlow record in {path}")
                records.append(value)
    return records
