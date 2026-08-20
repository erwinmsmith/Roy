#!/usr/bin/env bash
set -euo pipefail

roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
lhtb_root="${LHTB_ROOT:-${HOME}/rivermind-data/benchmarks/LHTB}"
execution_base="${ROY_LHTB_NATIVE_EXECUTION_BASE:-/tmp/Dev_4/roy-lhtb-native}"
native_root="${ROY_LHTB_NATIVE_ROOT:-${execution_base}/runtime}"
template_root="${ROY_LHTB_NATIVE_TEMPLATE_ROOT:-${execution_base}/templates}"
audit_path="${ROY_LHTB_NATIVE_AUDIT:-${roy_root}/research/output/lhtb/native/audit.json}"
node_runtime="${roy_root}/research/.runtime/node-v22"
python_bin="${roy_root}/research/.venv/bin/python"
harbor_bin="${roy_root}/research/.venv/bin/harbor"
revision="84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54"
mode="${1:-check}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_roy_env.sh
source "${script_dir}/load_roy_env.sh"
load_deepseek_api_key "${roy_root}"

install_system_dependencies() {
  [[ "$(id -u)" == "0" ]] || { echo "native prepare requires root for apt packages" >&2; exit 2; }
  # GPUHome images can retain sources from an older Ubuntu release. Install only
  # missing packages and pin them to the running OS suite to prevent cross-release
  # upgrades. ROY_LHTB_APT_UPDATE=false reuses a previously refreshed package index.
  # shellcheck source=/dev/null
  source /etc/os-release
  local apt_suite="${VERSION_CODENAME:?missing VERSION_CODENAME}" package_name
  local -a packages=()
  command -v asciinema >/dev/null || packages+=(asciinema)
  command -v cc >/dev/null || packages+=(build-essential)
  command -v cp >/dev/null || packages+=(coreutils)
  command -v curl >/dev/null || packages+=(ca-certificates curl)
  command -v git >/dev/null || packages+=(git)
  command -v git-lfs >/dev/null || packages+=(git-lfs)
  command -v proot >/dev/null || packages+=(proot)
  command -v setpriv >/dev/null || packages+=(util-linux)
  command -v tmux >/dev/null || packages+=(tmux)
  command -v xz >/dev/null || packages+=(xz-utils)
  ldconfig -p | grep 'libblas\.so' >/dev/null || packages+=(libblas3)
  ldconfig -p | grep 'libgfortran\.so' >/dev/null || packages+=(libgfortran5)
  ldconfig -p | grep 'liblapack\.so' >/dev/null || packages+=(liblapack3)
  ldconfig -p | grep 'libquadmath\.so' >/dev/null || packages+=(libquadmath0)
  if (( ${#packages[@]} )); then
    if [[ "${ROY_LHTB_APT_UPDATE:-true}" == "true" ]]; then
      apt-get update
    fi
    for package_name in "${!packages[@]}"; do
      packages[package_name]="${packages[package_name]}/${apt_suite}"
    done
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      "${packages[@]}"
  fi
  if ! command -v uv >/dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv
    python3 -m pip install uv || python3 -m pip install --break-system-packages uv
  fi
  uv python install 3.12
}

install_node_22() {
  if [[ -x "${node_runtime}/bin/node" ]]; then
    export PATH="${node_runtime}/bin:${PATH}"
    return
  fi
  local download_dir checksums archive
  download_dir="${roy_root}/research/.runtime/downloads"
  mkdir -p "${download_dir}" "${node_runtime}"
  checksums="${download_dir}/SHASUMS256-v22.txt"
  curl --fail --location --retry 3 \
    https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt -o "${checksums}"
  archive="$(awk '$2 ~ /linux-x64.tar.xz$/ {print $2; exit}' "${checksums}")"
  [[ -n "${archive}" ]] || { echo "cannot resolve latest verified Node 22 archive" >&2; exit 3; }
  curl --fail --location --retry 3 \
    "https://nodejs.org/dist/latest-v22.x/${archive}" -o "${download_dir}/${archive}"
  (cd "${download_dir}" && grep "  ${archive}$" "${checksums}" | sha256sum -c -)
  tar -xJf "${download_dir}/${archive}" -C "${node_runtime}" --strip-components=1
  export PATH="${node_runtime}/bin:${PATH}"
}

check_environment() {
  for command_name in git git-lfs node npm proot python3 setpriv tmux timeout uv; do
    command -v "${command_name}" >/dev/null || {
      echo "missing required command: ${command_name}" >&2
      exit 2
    }
  done
  [[ "$(uname -m)" == "x86_64" ]] || { echo "native LHTB host must be x86_64" >&2; exit 2; }
  node_major="$(node -p 'process.versions.node.split(`.`)[0]')"
  [[ "${node_major}" -ge 22 ]] || { echo "Node 22 or newer is required" >&2; exit 2; }
}

prepare_checkout() {
  if [[ ! -d "${lhtb_root}/.git" ]]; then
    mkdir -p "$(dirname "${lhtb_root}")"
    git clone https://github.com/zli12321/LHTB.git "${lhtb_root}"
  fi
  [[ -z "$(git -C "${lhtb_root}" status --porcelain)" ]] || {
    echo "LHTB checkout has local changes; refusing to switch revisions" >&2
    exit 3
  }
  git -C "${lhtb_root}" fetch origin "${revision}"
  git -C "${lhtb_root}" checkout --detach "${revision}"
  [[ "$(git -C "${lhtb_root}" rev-parse HEAD)" == "${revision}" ]]
  git -C "${lhtb_root}" lfs pull
  if [[ ! -x "${python_bin}" ]]; then
    uv venv --python 3.12 "${roy_root}/research/.venv"
  fi
  uv pip install --python "${python_bin}" -e "${roy_root}/research"
  uv pip install --python "${python_bin}" -e "${lhtb_root}/harbor"
  "${python_bin}" - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
)
PY
  cd "${roy_root}"
  npm ci
  npm run build
  PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-manifest \
    --lhtb-root "${lhtb_root}" --output "${roy_root}/research/output/lhtb/manifest.json"
  PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-native-preflight \
    --runtime-root "${native_root}" \
    --output "${roy_root}/research/output/lhtb/native/preflight.json"
}

provision_smoke_tasks() {
  local task_id
  for task_id in great-expectations-audit poc-exploit-craft \
    opensees-seismic-structural-regression-audit; do
    PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-native-provision \
      --lhtb-root "${lhtb_root}" --template-root "${template_root}" \
      --specs "${roy_root}/research/config/lhtb_native_provisioning.json" \
      --task-id "${task_id}" --python 3.11
  done
  PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-native-audit \
    --lhtb-root "${lhtb_root}" --manifest "${roy_root}/research/config/lhtb_split.json" \
    --template-root "${template_root}" --output "${audit_path}" \
    --allow-network-degraded
}

oracle_smoke() {
  local config="${roy_root}/research/output/lhtb/native/oracle-smoke.json"
  PYTHONPATH="${roy_root}/research" "${python_bin}" - "${config}" "${template_root}" \
    "${native_root}" "${roy_root}/research/output/lhtb/native/oracle-jobs" <<'PY'
import json, sys
value = {
    "job_name": "roy-native-oracle-smoke",
    "jobs_dir": sys.argv[4],
    "n_attempts": 1,
    "n_concurrent_trials": 1,
    "environment": {
        "import_path": "roy_research.native_environment:NativeProcessEnvironment",
        "force_build": False,
        "delete": True,
        "kwargs": {"template_root": sys.argv[2], "runtime_root": sys.argv[3]},
    },
    "agents": [{"name": "oracle"}],
    "datasets": [{"path": "./tasks", "task_names": ["great-expectations-audit"]}],
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(value, indent=2) + "\n")
PY
  export PYTHONPATH="${roy_root}/research${PYTHONPATH:+:${PYTHONPATH}}"
  export ROY_LHTB_NATIVE_ROOT="${native_root}"
  cd "${lhtb_root}"
  "${harbor_bin}" run -c "${config}" --yes
}

roy_smoke() {
  [[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY is required" >&2; exit 4; }
  local task_id category seed fingerprint job_dir config
  export PYTHONPATH="${roy_root}/research${PYTHONPATH:+:${PYTHONPATH}}"
  export ROY_LHTB_NODE_COMMAND="node ${roy_root}/dist/cli/LhtbAgent.js"
  export ROY_LHTB_SEMANTIC_COMMAND="${python_bin} -m roy_research.semantic_server"
  export ROY_LHTB_SEMANTIC_ROOT="${roy_root}/research/output/lhtb/native/semantic-smoke"
  export ROY_LHTB_POLICY_COMMAND="${python_bin} -m roy_research.lhtb_policy_server"
  export ROY_LHTB_MODEL="${roy_root}/research/output/lhtb/native/smoke-initial.pt"
  export ROY_LHTB_NATIVE_ROOT="${native_root}"
  if [[ ! -f "${ROY_LHTB_MODEL}" ]]; then
    "${python_bin}" -m roy_research lhtb-init \
      --manifest "${roy_root}/research/config/lhtb_split.json" --model "${ROY_LHTB_MODEL}"
  fi
  seed=20260820
  for task_id in great-expectations-audit poc-exploit-craft \
    opensees-seismic-structural-regression-audit; do
    category="$("${python_bin}" - "${roy_root}/research/config/lhtb_split.json" "${task_id}" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]:
    if value["task_id"] == sys.argv[2]: print(value["category"]); break
PY
)"
    fingerprint="$(printf '%s' "native-smoke:${task_id}:${revision}" | sha256sum | awk '{print $1}')"
    job_dir="${roy_root}/research/output/lhtb/native/smoke-jobs/${task_id}"
    config="${roy_root}/research/output/lhtb/native/configs/${task_id}.json"
    mkdir -p "${job_dir}" "$(dirname "${config}")"
    "${python_bin}" -m roy_research lhtb-group-config --output "${config}" \
      --jobs-dir "${job_dir}" --task-id "${task_id}" \
      --arm learned_information_realization --initial-fingerprint "${fingerprint}" \
      --organization-seed "${seed}" --attempts 8 --environment-backend native \
      --native-runtime-root "${native_root}" --native-template-root "${template_root}" \
      --allow-network-degraded
    cd "${lhtb_root}"
    "${harbor_bin}" run -c "${config}" --yes
    cd "${roy_root}"
    seed=$((seed + 1))
  done
  "${python_bin}" -m roy_research lhtb-smoke-validate \
    --jobs-dir "${roy_root}/research/output/lhtb/native/smoke-jobs" \
    --output "${roy_root}/research/output/lhtb/native/smoke-validation.json"
}

case "${mode}" in
  check) export PATH="${node_runtime}/bin:${PATH}"; check_environment ;;
  prepare) install_system_dependencies; install_node_22; check_environment; prepare_checkout ;;
  provision) install_node_22; check_environment; prepare_checkout; provision_smoke_tasks ;;
  oracle-smoke) install_node_22; check_environment; prepare_checkout; provision_smoke_tasks; oracle_smoke ;;
  roy-smoke) install_node_22; check_environment; prepare_checkout; provision_smoke_tasks; roy_smoke ;;
  *) echo "usage: $0 [check|prepare|provision|oracle-smoke|roy-smoke]" >&2; exit 2 ;;
esac

echo "LHTB-native ${mode} completed; results are not official leaderboard comparable"
