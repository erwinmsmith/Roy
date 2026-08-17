#!/usr/bin/env bash
set -euo pipefail

mode="${1:-dry-run}"
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest_path="${ROY_REMOTE_MANIFEST:-${bundle_dir}/manifest.json}"
work_dir="${ROY_REMOTE_WORK_DIR:-${bundle_dir}/work}"
results_dir="${ROY_REMOTE_RESULTS_DIR:-${bundle_dir}/results}"
ledger_path="${ROY_TOKEN_LEDGER:-${results_dir}/token-ledger.json}"
token_limit="${ROY_TOKEN_LIMIT:-10000000}"
proxy_port="${ROY_PROXY_PORT:-18080}"

for command_name in git uv python3; do
  command -v "${command_name}" >/dev/null || { echo "missing required command: ${command_name}" >&2; exit 2; }
done
test -f "${manifest_path}" || { echo "manifest not found: ${manifest_path}" >&2; exit 2; }
mkdir -p "${work_dir}" "${results_dir}"

read_manifest() {
  python3 - "${manifest_path}" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
path = sys.argv[2].split(".")
for key in path:
    value = value[int(key)] if isinstance(value, list) else value[key]
print(value)
PY
}

clone_pinned() {
  local index="$1"
  local name="$2"
  local repository revision destination actual
  repository="$(read_manifest "benchmarks.${index}.repository")"
  revision="$(read_manifest "benchmarks.${index}.revision")"
  destination="${work_dir}/${name}"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${repository}" "${destination}"
  fi
  git -C "${destination}" fetch --depth=1 origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
  actual="$(git -C "${destination}" rev-parse HEAD)"
  [[ "${actual}" == "${revision}" ]] || { echo "revision mismatch for ${name}" >&2; exit 3; }
}

prepare() {
  audit_tua_images
  prepare_tau
  clone_pinned 1 TUA-Bench
  (cd "${work_dir}/TUA-Bench" && uv run setup-env)
  if command -v docker >/dev/null; then
    docker image inspect $(docker image ls --format '{{.Repository}}:{{.Tag}}' | sort) \
      --format '{{index .RepoDigests 0}}' 2>/dev/null | sort -u > "${results_dir}/container-digests.lock" || true
  fi
  python3 - "${manifest_path}" "${results_dir}/prepared.json" <<'PY'
import json, pathlib, sys, time
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "schema_version": 1,
    "prepared_at": int(time.time()),
    "revisions": {b["id"]: b["revision"] for b in manifest["benchmarks"]},
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

prepare_tau() {
  clone_pinned 0 tau2-bench
  (cd "${work_dir}/tau2-bench" && uv sync --extra knowledge)
  python3 - "${manifest_path}" "${results_dir}/prepared-tau.json" <<'PY'
import json, pathlib, sys, time
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
benchmark = manifest["benchmarks"][0]
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "schema_version": 1,
    "prepared_at": int(time.time()),
    "revisions": {benchmark["id"]: benchmark["revision"]},
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "tau Knowledge host environment prepared; stamp: ${results_dir}/prepared-tau.json"
}

audit_tua_images() {
  command -v docker >/dev/null || { echo "missing required command: docker" >&2; exit 2; }
  command -v curl >/dev/null || { echo "missing required command: curl" >&2; exit 2; }
  if ! command -v docker-credential-desktop >/dev/null && \
     [[ -x /Applications/Docker.app/Contents/Resources/bin/docker-credential-desktop ]]; then
    export PATH="/Applications/Docker.app/Contents/Resources/bin:${PATH}"
  fi
  docker buildx version >/dev/null || { echo "docker buildx is required for manifest audit" >&2; exit 2; }
  local lock_path="${results_dir}/tua-image-audit.lock"
  local image digest status name url
  : > "${lock_path}"
  for image in \
    debian:bookworm \
    ubuntu:24.04 \
    ubuntu:20.04 \
    docker.io/cellprofiler/cellprofiler:4.2.8 \
    docker.io/openfoam/openfoam11-paraview510; do
    digest="$(docker buildx imagetools inspect --raw "${image}" | python3 -c 'import hashlib, sys; print("sha256:" + hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')"
    [[ "${digest}" == sha256:* ]] || { echo "missing TUA base image: ${image}" >&2; exit 5; }
    echo "image ${image} ${digest}" >> "${lock_path}"
  done
  while read -r name url; do
    status="$(curl -sSIL --max-time 30 -o /dev/null -w '%{http_code}' "${url}")"
    [[ "${status}" == "200" ]] || { echo "missing TUA setup asset: ${name} status=${status}" >&2; exit 5; }
    echo "asset ${name} ${status}" >> "${lock_path}"
  done <<'URLS'
drawio https://raw.githubusercontent.com/jgraph/drawio-diagrams/dev/blog/waypoint-shape.drawio
mri https://drive.google.com/uc?export=download&id=1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a
video https://huggingface.co/datasets/facebook/PE-Video/resolve/main/test/000000.tar?download=true
floorplan https://huggingface.co/buckets/bebeberr/floor-plans/resolve/floorplan.png?download=true
comstock https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/comstock_amy2018_release_3/timeseries_individual_buildings/by_state/upgrade=0/state=WA/100094-0.parquet
cellprofiler https://huggingface.co/buckets/bebeberr/cell-profiler/resolve/1-162hrh2ax2.tif?download=true
URLS
  echo "TUA image and setup-asset audit passed; lock: ${lock_path}"
}

print_matrix() {
  for benchmark in tau-knowledge tua; do
    for arm in no_derivation roy_heuristic node_only full_v0_v4; do
      for repeat in 0 1 2; do
        echo "${benchmark} arm=${arm} repeat=${repeat} tasks=5"
      done
    done
  done
  echo "planned episodes: 120; hard token limit: ${token_limit}"
}

run_pilot() {
  test -f "${results_dir}/prepared.json" || { echo "run prepare first" >&2; exit 4; }
  test -n "${DEEPSEEK_API_KEY:-}" || { echo "DEEPSEEK_API_KEY is required" >&2; exit 4; }
  [[ "${ROY_EXTERNAL_ADAPTER_READY:-0}" == "1" ]] || {
    echo "refusing invalid arm comparison: set ROY_EXTERNAL_ADAPTER_READY=1 only after the host's tau2 and Harbor agents consume ROY_STRUCTURAL_ARM and ROY_STRUCTURAL_POLICY_COMMAND" >&2
    exit 4
  }
  export PYTHONPATH="${bundle_dir}/..${PYTHONPATH:+:${PYTHONPATH}}"
  python3 "${bundle_dir}/token_proxy.py" --ledger "${ledger_path}" --limit "${token_limit}" --port "${proxy_port}" &
  proxy_pid=$!
  trap 'kill "${proxy_pid}" 2>/dev/null || true' EXIT
  export DEEPSEEK_API_BASE="http://127.0.0.1:${proxy_port}/v1"
  export OPENAI_API_BASE="http://127.0.0.1:${proxy_port}/v1"
  export ROY_STRUCTURAL_POLICY_COMMAND="${ROY_STRUCTURAL_POLICY_COMMAND:-python3 -m roy_research.policy_server}"

  for arm in no_derivation roy_heuristic node_only full_v0_v4; do
    export ROY_STRUCTURAL_ARM="${arm}"
    (cd "${work_dir}/tau2-bench" && uv run tau2 run \
      --domain banking_knowledge \
      --retrieval-config bm25 \
      --agent-llm deepseek/deepseek-v4-flash \
      --user-llm deepseek/deepseek-v4-flash \
      --num-trials 3 --num-tasks 5 --seed 20260815 --auto-resume \
      --save-to "roy-${arm}")
    (cd "${work_dir}/TUA-Bench" && uv run harbor run \
      -p tasks -a terminus-2 -m deepseek/deepseek-v4-flash \
      --n-tasks 5 --n-attempts 3 \
      --agent-env "ROY_STRUCTURAL_ARM=${arm}" \
      --agent-env "ROY_STRUCTURAL_POLICY_COMMAND=${ROY_STRUCTURAL_POLICY_COMMAND}" \
      -o "jobs/roy-${arm}" --yes)
  done
}

case "${mode}" in
  dry-run) print_matrix ;;
  audit-images) audit_tua_images ;;
  prepare-tau) prepare_tau ;;
  prepare) prepare ;;
  run) run_pilot ;;
  *) echo "usage: $0 [dry-run|audit-images|prepare-tau|prepare|run]" >&2; exit 2 ;;
esac
