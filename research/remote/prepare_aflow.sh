#!/usr/bin/env bash
set -euo pipefail

action="${1:-check}"
roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
aflow_root="${AFLOW_ROOT:-${HOME}/rivermind-data/benchmarks/AFlow}"
manifest="${AFLOW_MANIFEST:-${roy_root}/research/config/aflow_benchmarks.json}"
revision="3f457218fc716093fe53f6df8a5d5e6379d66346"
venv="${AFLOW_VENV:-${aflow_root}/.venv}"
python_spec="${AFLOW_PYTHON_SPEC:-3.9}"

clone_checkout() {
  mkdir -p "$(dirname -- "${aflow_root}")"
  if [[ ! -d "${aflow_root}/.git" ]]; then
    git clone https://github.com/FoundationAgents/AFlow.git "${aflow_root}"
  fi
  if [[ -n "$(git -C "${aflow_root}" status --short)" ]]; then
    echo "AFlow checkout is dirty; refusing to change its revision" >&2
    exit 4
  fi
  git -C "${aflow_root}" fetch origin "${revision}"
  git -C "${aflow_root}" checkout --detach "${revision}"
}

check_checkout() {
  [[ -f "${manifest}" && -d "${aflow_root}/.git" ]] || {
    echo "Roy AFlow manifest and external AFlow checkout are required" >&2
    exit 4
  }
  python3 - "${manifest}" "${aflow_root}" "${revision}" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

manifest_path, root_path, revision = sys.argv[1:]
root = pathlib.Path(root_path)
manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
head = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
if head != revision or manifest["source"]["revision"] != revision:
    raise SystemExit(f"AFlow revision mismatch: checkout={head}, expected={revision}")

files = []
for dataset, spec in manifest["benchmarks"].items():
    for split in ("optimization", "test", "public_tests"):
        if split not in spec:
            continue
        expected = spec[split]
        path = root / expected["path"]
        if not path.is_file():
            raise SystemExit(f"missing AFlow file: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected["sha256"]:
            raise SystemExit(f"AFlow hash mismatch: {path}")
        records = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SystemExit(f"non-object record: {path}:{line_number}")
                records += 1
        if records != expected["records"]:
            raise SystemExit(
                f"AFlow record mismatch: {path}: {records} != {expected['records']}"
            )
        files.append({
            "dataset": dataset,
            "split": split,
            "records": records,
            "sha256": digest,
        })
print(json.dumps({
    "repository": manifest["source"]["repository"],
    "revision": head,
    "datasets": len(manifest["benchmarks"]),
    "files": files,
}, sort_keys=True))
PY
}

download_data() {
  local bootstrap_python="${AFLOW_BOOTSTRAP_PYTHON:-python3}"
  (
    cd "${aflow_root}"
    "${bootstrap_python}" - <<'PY'
from data.download_data import download
download(["datasets"], force_download=False)
PY
  )
}

prepare_env() {
  command -v uv >/dev/null || { echo "uv is required" >&2; exit 4; }
  uv venv --python "${python_spec}" "${venv}"
  uv pip install --python "${venv}/bin/python" -r "${aflow_root}/requirements.txt"
}

smoke() {
  [[ -x "${venv}/bin/python" ]] || {
    echo "missing AFlow environment: ${venv}" >&2
    exit 4
  }
  (
    cd "${aflow_root}"
    "${venv}/bin/python" - <<'PY'
import asyncio
import json
from pathlib import Path

from benchmarks.drop import DROPBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.hotpotqa import HotpotQABenchmark
from benchmarks.math import MATHBenchmark

def first(name):
    return json.loads((Path("data/datasets") / name).read_text().splitlines()[0])

async def main():
    checks = []
    row = first("gsm8k_validate.jsonl")
    benchmark = GSM8KBenchmark(
        "GSM8K", "data/datasets/gsm8k_validate.jsonl", "/tmp"
    )
    checks.append((
        "GSM8K", len(await benchmark.load_data()),
        benchmark.calculate_score(
            float(row["answer"]), benchmark.extract_number(row["answer"])
        )[0],
    ))
    for name, benchmark_type, filename, answer_key in (
        ("HotpotQA", HotpotQABenchmark, "hotpotqa_validate.jsonl", "answer"),
        ("DROP", DROPBenchmark, "drop_validate.jsonl", "ref_text"),
        ("MATH", MATHBenchmark, "math_validate.jsonl", "solution"),
    ):
        row = first(filename)
        benchmark = benchmark_type(name, f"data/datasets/{filename}", "/tmp")
        checks.append((
            name, len(await benchmark.load_data()),
            benchmark.calculate_score(row[answer_key], row[answer_key])[0],
        ))
    if not all(score == 1 for _, _, score in checks):
        raise SystemExit(f"AFlow scorer smoke failed: {checks}")
    print(json.dumps({"safe_scorer_smoke": checks}))

asyncio.run(main())
PY
  )
  echo "HumanEval and MBPP smoke intentionally require a reviewed process sandbox."
}

case "${action}" in
  clone)
    clone_checkout
    ;;
  download)
    clone_checkout
    download_data
    check_checkout
    ;;
  env)
    prepare_env
    ;;
  check)
    check_checkout
    ;;
  smoke)
    check_checkout
    smoke
    ;;
  prepare)
    clone_checkout
    download_data
    check_checkout
    prepare_env
    smoke
    ;;
  *)
    echo "usage: $0 {clone|download|env|check|smoke|prepare}" >&2
    exit 2
    ;;
esac
