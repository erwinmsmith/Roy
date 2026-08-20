from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


LHTB_REPOSITORY = "https://github.com/zli12321/LHTB.git"
LHTB_COMMIT = "84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54"
LHTB_SPLIT_SEED = "roy-lhtb-20260820"

# The pinned README defines these eight taxonomy rows. Keeping membership explicit
# makes a changed upstream task list fail closed instead of silently changing splits.
LHTB_TAXONOMY: Dict[str, Sequence[str]] = {
    "interactive_games_and_puzzles": (
        "2048", "chess-mate", "generals-bot-arena", "rush_hour_campaign",
        "snake_maze_campaign", "sokoban", "sudoku-recovery", "super-mario",
    ),
    "multimodal_and_imaging_analysis": (
        "audio-visual-event-alignment", "dicom-radiology-audit",
        "document-table-layout-reconstruction", "microscopy-cell-count-qc-audit",
        "satellite-flood-change-detection-audit", "scientific-figure-data-reconstruction",
    ),
    "software_and_reverse_engineering": (
        "commit0-multilib-tdd", "great-expectations-audit",
        "langchain-version-migration", "robotics-slam-benchmark-repair",
        "riscv-core-debug", "unknown-config-semantics",
    ),
    "scientific_computing_and_simulation": (
        "epidemic-inverse-control-audit", "materials-phase-diagram-audit",
        "nbody-accel-iterative", "opensees-seismic-structural-regression-audit",
        "spice-ephemeris-regression", "su2-airfoil-regression",
    ),
    "earth_climate_and_energy": (
        "climate-netcdf-extreme-event-audit", "epa-swmm-stormwater-regression-audit",
        "gdal-proj-raster-regression", "matpower-opf-regression",
        "modflow6-groundwater-regression-audit", "nrel-pysam-hybrid-renewables-audit",
    ),
    "systems_performance_and_security": (
        "duckdb-optimizer-closure", "grammar-fuzz-coverage-hunt", "poc-exploit-craft",
        "spot-scheduler-traces", "vector-db-iterative-build",
    ),
    "research_reproduction_and_ml": (
        "alp-paper-reproduction", "apex-openroad-ibex-signoff", "foldseek-paper-reproduction",
        "tabular-data-feature-covshift", "unison-paper-reproduction",
    ),
    "apex_professional_workflows": (
        "apex-ib244-matter", "apex-investment-banking-matter", "apex-law433-matter",
        "apex-management-consulting-matter",
    ),
}

_EXPECTED_TASK_COUNT = 46


@dataclass(frozen=True)
class LHTBTask:
    task_id: str
    category: str
    split: str
    order_hash: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "split": self.split,
            "order_hash": self.order_hash,
        }


def _ordered(task_ids: Iterable[str], category: str, seed: str) -> List[str]:
    return sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{seed}:{category}:{task_id}".encode("utf-8")
        ).hexdigest(),
    )


def build_lhtb_split(
    taxonomy: Mapping[str, Sequence[str]] = LHTB_TAXONOMY,
    seed: str = LHTB_SPLIT_SEED,
) -> List[LHTBTask]:
    seen: set[str] = set()
    records: List[LHTBTask] = []
    for category, task_ids in taxonomy.items():
        if len(task_ids) < 3:
            raise ValueError(f"category {category} needs at least three tasks")
        if seen.intersection(task_ids):
            raise ValueError(f"duplicate LHTB task in category {category}")
        seen.update(task_ids)
        ordered = _ordered(task_ids, category, seed)
        for index, task_id in enumerate(ordered):
            split = "dev" if index == 0 else "test" if index == 1 else "train"
            digest = hashlib.sha256(
                f"{seed}:{category}:{task_id}".encode("utf-8")
            ).hexdigest()
            records.append(LHTBTask(task_id, category, split, digest))
    if len(records) != _EXPECTED_TASK_COUNT:
        raise ValueError(f"pinned LHTB taxonomy must contain {_EXPECTED_TASK_COUNT} tasks")
    counts = {split: sum(value.split == split for value in records)
              for split in ("train", "dev", "test")}
    if counts != {"train": 30, "dev": 8, "test": 8}:
        raise ValueError(f"invalid LHTB split counts: {counts}")
    return sorted(records, key=lambda value: (value.split, value.category, value.order_hash))


def verify_lhtb_checkout(root: Path, records: Sequence[LHTBTask]) -> Dict[str, object]:
    task_root = root / "tasks"
    actual = {path.name for path in task_root.iterdir() if path.is_dir()}
    expected = {value.task_id for value in records}
    if actual != expected:
        raise ValueError(
            f"pinned LHTB task mismatch; missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    return {"repository": LHTB_REPOSITORY, "commit": LHTB_COMMIT, "tasks": len(actual)}


def write_lhtb_manifest(path: Path, records: Sequence[LHTBTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "repository": LHTB_REPOSITORY,
        "commit": LHTB_COMMIT,
        "split_seed": LHTB_SPLIT_SEED,
        "tasks": [value.to_dict() for value in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_lhtb_manifest(path: Path) -> List[Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("commit") != LHTB_COMMIT:
        raise ValueError("LHTB manifest commit does not match the pinned revision")
    tasks = list(payload.get("tasks", []))
    if len(tasks) != _EXPECTED_TASK_COUNT:
        raise ValueError("LHTB manifest must contain exactly 46 tasks")
    return tasks


def require_training_task(task_id: str, manifest: Sequence[Mapping[str, object]]) -> None:
    matches = [value for value in manifest if value.get("task_id") == task_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicated LHTB task: {task_id}")
    if matches[0].get("split") != "train":
        raise ValueError(f"trainer rejects non-train LHTB task {task_id}")
