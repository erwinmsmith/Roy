#!/usr/bin/env bash
set -euo pipefail

roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
benchmark_root="${LHTB_ROOT:-${HOME}/rivermind-data/benchmarks/LHTB}"
lhtb_revision="84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54"
mode="${1:-check}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_roy_env.sh
source "${script_dir}/load_roy_env.sh"
load_deepseek_api_key "${roy_root}"

export ROY_LHTB_TORCH_THREADS="${ROY_LHTB_TORCH_THREADS:-4}"
export ROY_LHTB_TORCH_INTEROP_THREADS="${ROY_LHTB_TORCH_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${ROY_LHTB_TORCH_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${ROY_LHTB_TORCH_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${ROY_LHTB_TORCH_THREADS}}"

for command_name in docker git git-lfs node python3 uv; do
  command -v "${command_name}" >/dev/null || {
    echo "missing required command: ${command_name}" >&2
    exit 2
  }
done

[[ "$(uname -m)" == "x86_64" ]] || {
  echo "formal LHTB host must be x86_64" >&2
  exit 2
}
docker info >/dev/null
docker buildx version >/dev/null
node_major="$(node -p 'process.versions.node.split(`.`)[0]')"
[[ "${node_major}" -ge 22 ]] || { echo "Node 22 or newer is required" >&2; exit 2; }

PYTHONPATH="${roy_root}/research" python3 -m roy_research lhtb-preflight \
  --path "$(dirname "${benchmark_root}")" \
  --output "${roy_root}/research/output/lhtb/preflight.json"

prepare_checkout() {
  if [[ ! -d "${benchmark_root}/.git" ]]; then
    mkdir -p "$(dirname "${benchmark_root}")"
    git clone https://github.com/zli12321/LHTB.git "${benchmark_root}"
  fi
  [[ -z "$(git -C "${benchmark_root}" status --porcelain)" ]] || {
    echo "LHTB checkout has local changes; refusing to switch revisions" >&2
    exit 3
  }
  git -C "${benchmark_root}" fetch origin "${lhtb_revision}"
  git -C "${benchmark_root}" checkout --detach "${lhtb_revision}"
  [[ "$(git -C "${benchmark_root}" rev-parse HEAD)" == "${lhtb_revision}" ]]
  git -C "${benchmark_root}" lfs pull
  if [[ ! -x "${roy_root}/research/.venv/bin/python" ]]; then
    uv venv --python 3.12 "${roy_root}/research/.venv"
  fi
  "${roy_root}/research/.venv/bin/python" -c \
    'import sys; assert sys.version_info[:2] == (3, 12), "formal LHTB runner requires Python 3.12"'
  uv pip install --python "${roy_root}/research/.venv/bin/python" -e "${roy_root}/research"
  uv pip install --python "${roy_root}/research/.venv/bin/python" -e "${benchmark_root}/harbor"
  "${roy_root}/research/.venv/bin/python" - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
)
PY
  cd "${roy_root}"
  npm ci
  npm run build
  PYTHONPATH="${roy_root}/research" "${roy_root}/research/.venv/bin/python" \
    -m roy_research lhtb-manifest --lhtb-root "${benchmark_root}" \
    --output "${roy_root}/research/output/lhtb/manifest.json"
  PYTHONPATH="${roy_root}/research" "${roy_root}/research/.venv/bin/python" \
    -m roy_research lhtb-schedule \
    --manifest "${roy_root}/research/output/lhtb/manifest.json" \
    --output "${roy_root}/research/output/lhtb/schedule.json"
}

oracle_smoke() {
  cd "${benchmark_root}"
  "${roy_root}/research/.venv/bin/harbor" run -c configs/examples/oracle_smoke.yaml
  docker image ls --digests --format '{{.Repository}} {{.Tag}} {{.Digest}}' \
    | sort -u > "${roy_root}/research/output/lhtb/docker-digests.lock"
}

roy_smoke() {
  [[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY is required" >&2; exit 4; }
  export PYTHONPATH="${roy_root}/research${PYTHONPATH:+:${PYTHONPATH}}"
  export ROY_LHTB_NODE_COMMAND="node ${roy_root}/dist/cli/LhtbAgent.js"
  export HB_CONTINUE_MODE=same_conversation
  export ROY_LHTB_SEMANTIC_COMMAND="${roy_root}/research/.venv/bin/python -m roy_research.semantic_server"
  export ROY_LHTB_SEMANTIC_ROOT="${roy_root}/research/output/lhtb/semantic-smoke"
  export ROY_LHTB_POLICY_COMMAND="${roy_root}/research/.venv/bin/python -m roy_research.lhtb_policy_server"
  export ROY_LHTB_MODEL="${roy_root}/research/output/lhtb/scheduler-structural-initial.pt"
  if [[ ! -f "${ROY_LHTB_MODEL}" ]]; then
    "${roy_root}/research/.venv/bin/python" -m roy_research lhtb-init \
      --manifest "${roy_root}/research/output/lhtb/manifest.json" --model "${ROY_LHTB_MODEL}"
  fi
  cd "${benchmark_root}"
  "${roy_root}/research/.venv/bin/harbor" run \
    -c "${roy_root}/research/config/lhtb_roy_smoke.yaml"
  "${roy_root}/research/.venv/bin/python" -m roy_research lhtb-smoke-validate \
    --jobs-dir "${benchmark_root}/jobs" \
    --output "${roy_root}/research/output/lhtb/smoke-validation.json"
}

case "${mode}" in
  check) ;;
  prepare) prepare_checkout ;;
  oracle-smoke) prepare_checkout; oracle_smoke ;;
  roy-smoke) prepare_checkout; roy_smoke ;;
  *) echo "usage: $0 [check|prepare|oracle-smoke|roy-smoke]" >&2; exit 2 ;;
esac

echo "LHTB ${mode} completed without destructive Docker pruning"
