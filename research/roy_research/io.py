from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    handle_context = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" \
        else path.open("r", encoding="utf-8")
    with handle_context as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        count = 0
        handle_context = gzip.open(path, "at", encoding="utf-8", compresslevel=1) \
            if path.suffix == ".gz" else path.open("a", encoding="utf-8")
        with handle_context as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        return count
    records = list(records)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        handle_context = gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) \
            if path.suffix == ".gz" else temporary.open("w", encoding="utf-8")
        with handle_context as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return len(records)


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
