from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


TAU3_REPOSITORY = "https://github.com/sierra-research/tau2-bench.git"
TAU3_TAG = "v1.0.1"
TAU3_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU3_TRAINING_DOMAINS = ("airline", "retail", "telecom")
TAU3_KNOWLEDGE_DOMAIN = "banking_knowledge"


@dataclass(frozen=True)
class Tau3TaskReference:
    benchmark: str
    revision: str
    domain: str
    task_id: str
    split: str
    official_split: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verify_tau3_root(root: Path) -> Dict[str, Any]:
    if not (root / "data" / "tau2" / "domains").is_dir():
        raise ValueError(f"not a tau3 checkout: {root}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != TAU3_COMMIT:
        raise ValueError(f"tau3 revision mismatch: expected {TAU3_COMMIT}, got {commit}")
    return {"repository": TAU3_REPOSITORY, "tag": TAU3_TAG, "commit": commit}


def build_tau3_manifest(root: Path, validation_modulus: int = 10) -> List[Tau3TaskReference]:
    """Build a leakage-safe text/knowledge manifest from official tau3 data.

    Airline, retail and telecom retain their official train/test boundary. A
    deterministic subset of official train is reserved for validation. The
    banking_knowledge base has no official training split in tau3 v1.0.1, so
    all 97 tasks remain held out and are never emitted as training examples.
    """

    verify_tau3_root(root)
    if validation_modulus < 2:
        raise ValueError("validation_modulus must be at least two")
    domains_root = root / "data" / "tau2" / "domains"
    records: List[Tau3TaskReference] = []
    for domain in TAU3_TRAINING_DOMAINS:
        tasks = _task_ids(domains_root / domain / _tasks_filename(domain))
        splits = json.loads((domains_root / domain / "split_tasks.json").read_text())
        for official_split in ("train", "test"):
            for raw_identifier in splits[official_split]:
                task_id = _resolve_task_id(raw_identifier, tasks)
                split = official_split
                if official_split == "train" and _validation_bucket(domain, task_id, validation_modulus):
                    split = "validation"
                records.append(
                    Tau3TaskReference("tau3", TAU3_COMMIT, domain, task_id, split, official_split)
                )

    knowledge_tasks = _task_ids(domains_root / TAU3_KNOWLEDGE_DOMAIN / "tasks.json")
    records.extend(
        Tau3TaskReference(
            "tau3", TAU3_COMMIT, TAU3_KNOWLEDGE_DOMAIN, task_id, "heldout", "base"
        )
        for task_id in knowledge_tasks
    )
    return sorted(records, key=lambda value: (value.split, value.domain, value.task_id))


def manifest_summary(records: Iterable[Tau3TaskReference]) -> Dict[str, Any]:
    values = list(records)
    by_split: Dict[str, int] = {}
    by_domain: Dict[str, int] = {}
    for value in values:
        by_split[value.split] = by_split.get(value.split, 0) + 1
        by_domain[value.domain] = by_domain.get(value.domain, 0) + 1
    return {"tasks": len(values), "by_split": by_split, "by_domain": by_domain}


def _tasks_filename(domain: str) -> str:
    return "tasks.json"


def _task_ids(path: Path) -> List[str]:
    tasks = json.loads(path.read_text())
    if not isinstance(tasks, list):
        raise ValueError(f"unexpected tau3 task format: {path}")
    return [str(task.get("id", index)) for index, task in enumerate(tasks)]


def _resolve_task_id(value: object, task_ids: List[str]) -> str:
    candidate = str(value)
    if candidate in task_ids:
        return candidate
    if isinstance(value, int) and 0 <= value < len(task_ids):
        return task_ids[value]
    raise ValueError(f"tau3 split references unknown task: {value}")


def _validation_bucket(domain: str, task_id: str, modulus: int) -> bool:
    digest = hashlib.sha256(f"{domain}:{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus == 0

