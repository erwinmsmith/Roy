#!/usr/bin/env bash
set -euo pipefail

roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
lhtb_root="${LHTB_ROOT:-${HOME}/rivermind-data/benchmarks/LHTB}"
execution_base="${ROY_LHTB_NATIVE_EXECUTION_BASE:-${HOME}/rivermind-data/lhtb-native}"
native_root="${ROY_LHTB_NATIVE_ROOT:-${execution_base}/runtime}"
template_root="${ROY_LHTB_NATIVE_TEMPLATE_ROOT:-${execution_base}/templates}"
audit_path="${ROY_LHTB_NATIVE_AUDIT:-${roy_root}/research/output/lhtb/native/audit.json}"
node_runtime="${roy_root}/research/.runtime/node-v22"
proot_runtime="${execution_base}/tools/proot-v5.3.1-99a84175"
python_bin="${roy_root}/research/.venv/bin/python"
harbor_bin="${roy_root}/research/.venv/bin/harbor"
native_task_python="${ROY_LHTB_NATIVE_TASK_PYTHON:-3.12}"
export ROY_LHTB_OCI_MIRROR="${ROY_LHTB_OCI_MIRROR:-dockerproxy.net}"
native_task_gid=210000
revision="84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54"
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
  command -v setfacl >/dev/null || packages+=(acl)
  command -v cc >/dev/null || packages+=(build-essential)
  command -v cp >/dev/null || packages+=(coreutils)
  command -v curl >/dev/null || packages+=(ca-certificates curl)
  command -v git >/dev/null || packages+=(git)
  command -v git-lfs >/dev/null || packages+=(git-lfs)
  command -v proot >/dev/null || packages+=(proot)
  command -v setpriv >/dev/null || packages+=(util-linux)
  command -v skopeo >/dev/null || packages+=(skopeo)
  command -v tmux >/dev/null || packages+=(tmux)
  command -v umoci >/dev/null || packages+=(umoci)
  command -v xz >/dev/null || packages+=(xz-utils)
  ldconfig -p | grep 'libblas\.so' >/dev/null || packages+=(libblas3)
  ldconfig -p | grep 'libgfortran\.so' >/dev/null || packages+=(libgfortran5)
  ldconfig -p | grep 'liblapack\.so' >/dev/null || packages+=(liblapack3)
  ldconfig -p | grep 'libquadmath\.so' >/dev/null || packages+=(libquadmath0)
  [[ -x /usr/games/stockfish ]] || packages+=(stockfish)
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
  if [[ -x /usr/games/stockfish && ! -e /usr/local/bin/stockfish ]]; then
    ln -s /usr/games/stockfish /usr/local/bin/stockfish
  fi
  if ! command -v uv >/dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv
    python3 -m pip install uv || python3 -m pip install --break-system-packages uv
  fi
  setfacl -m "g:${native_task_gid}:--x" /
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

install_proot_runtime() {
  local binary expected actual traversal
  mkdir -p "${proot_runtime}"
  exec 9>"${proot_runtime}/.install.lock"
  flock 9
  binary="${proot_runtime}/bin/proot"
  expected="b7f2adf5a225000a164f4905aabefeebe11c4c1d5bedff5e1fe8866c48dd70d2"
  if [[ ! -x "${binary}" ]]; then
    mkdir -p "$(dirname "${binary}")"
    curl --fail --location --retry 3 \
      https://proot.gitlab.io/proot/bin/proot -o "${binary}.download"
    actual="$(sha256sum "${binary}.download" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || {
      echo "downloaded PRoot checksum mismatch: ${actual}" >&2
      return 3
    }
    chmod 755 "${binary}.download"
    mv "${binary}.download" "${binary}"
  fi
  actual="$(sha256sum "${binary}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "installed PRoot checksum mismatch: ${actual}" >&2
    return 3
  }
  export PATH="${proot_runtime}/bin:${PATH}"
  "${binary}" --version | grep -q 'v5.3.1-99a84175' || {
    echo "unexpected PRoot runtime version" >&2
    return 3
  }
  # Native trials run with a dedicated unprivileged UID/GID. Grant that group
  # execute-only traversal through every parent of the data-backed runtime;
  # directory listing and writes remain forbidden by the ACL.
  traversal="$(realpath -m "${execution_base}")"
  while [[ "${traversal}" != "/" ]]; do
    [[ -d "${traversal}" ]] && setfacl -m "g:${native_task_gid}:--x" "${traversal}"
    traversal="$(dirname "${traversal}")"
  done
  setfacl -m "g:${native_task_gid}:--x" /
  flock -u 9
}

check_environment() {
  for command_name in flock git git-lfs node npm proot python3 setpriv skopeo tmux timeout umoci uv; do
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
  mkdir -p "${roy_root}/research/.runtime"
  exec 8>"${roy_root}/research/.runtime/.prepare-checkout.lock"
  flock 8
  if [[ ! -d "${lhtb_root}/.git" ]]; then
    mkdir -p "$(dirname "${lhtb_root}")"
    git clone https://github.com/zli12321/LHTB.git "${lhtb_root}"
  fi
  [[ -z "$(git -C "${lhtb_root}" status --porcelain)" ]] || {
    echo "LHTB checkout has local changes; refusing to switch revisions" >&2
    exit 3
  }
  if [[ "$(git -C "${lhtb_root}" rev-parse HEAD 2>/dev/null || true)" != "${revision}" ]]; then
    git -C "${lhtb_root}" fetch origin "${revision}"
  fi
  git -C "${lhtb_root}" checkout --detach "${revision}"
  [[ "$(git -C "${lhtb_root}" rev-parse HEAD)" == "${revision}" ]]
  if ! git -C "${lhtb_root}" lfs fsck; then
    git -C "${lhtb_root}" lfs pull
    git -C "${lhtb_root}" lfs fsck
  fi
  if [[ ! -x "${python_bin}" ]]; then
    uv venv --python 3.12 "${roy_root}/research/.venv"
  fi
  uv pip install --python "${python_bin}" -e "${roy_root}/research"
  uv pip install --python "${python_bin}" -e "${lhtb_root}/harbor"
  "${python_bin}" - <<'PY'
from sentence_transformers import SentenceTransformer
model_id = "sentence-transformers/all-MiniLM-L6-v2"
revision = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
try:
    SentenceTransformer(model_id, revision=revision, local_files_only=True)
except OSError:
    SentenceTransformer(model_id, revision=revision)
PY
  cd "${roy_root}"
  npm ci
  npm run build
  PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-manifest \
    --lhtb-root "${lhtb_root}" --output "${roy_root}/research/output/lhtb/manifest.json"
  PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-native-preflight \
    --runtime-root "${native_root}" \
    --output "${roy_root}/research/output/lhtb/native/preflight.json"
  flock -u 8
}

provision_smoke_tasks() {
  local task_id
  local -a smoke_tasks
  IFS=',' read -r -a smoke_tasks <<< "${ROY_LHTB_SMOKE_TASKS:-great-expectations-audit,poc-exploit-craft,opensees-seismic-structural-regression-audit}"
  for task_id in "${smoke_tasks[@]}"; do
    PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-native-provision \
      --lhtb-root "${lhtb_root}" --template-root "${template_root}" \
      --specs "${roy_root}/research/config/lhtb_native_provisioning.json" \
      --task-id "${task_id}" --python "${native_task_python}"
  done
  PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-native-audit \
    --lhtb-root "${lhtb_root}" --manifest "${roy_root}/research/config/lhtb_split.json" \
    --template-root "${template_root}" --output "${audit_path}" \
    --allow-network-degraded
}

provision_reviewed_tasks() {
  local task_id
  while IFS= read -r task_id; do
    PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-native-provision \
      --lhtb-root "${lhtb_root}" --template-root "${template_root}" \
      --specs "${roy_root}/research/config/lhtb_native_provisioning.json" \
      --task-id "${task_id}" --python "${native_task_python}"
  done < <("${python_bin}" - "${roy_root}/research/config/lhtb_native_provisioning.json" <<'PY'
import json, sys
for task_id in sorted(json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]):
    print(task_id)
PY
)
  PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-native-audit \
    --lhtb-root "${lhtb_root}" --manifest "${roy_root}/research/config/lhtb_split.json" \
    --template-root "${template_root}" --output "${audit_path}" \
    --allow-network-degraded
}

oracle_smoke() {
  local config="${roy_root}/research/output/lhtb/native/oracle-smoke.json"
  local jobs_root="${roy_root}/research/output/lhtb/native/oracle-jobs"
  local job_name="roy-native-oracle-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  local overlay_root="${roy_root}/research/output/lhtb/native/oracle-overlays/${job_name}/tasks"
  PYTHONPATH="${roy_root}/research" "${python_bin}" - "${config}" "${template_root}" \
    "${native_root}" "${jobs_root}" "${job_name}" "${overlay_root}" "${lhtb_root}" <<'PY'
import json, pathlib, sys
task_id = "great-expectations-audit"
source = pathlib.Path(sys.argv[7]) / "tasks" / task_id
target = pathlib.Path(sys.argv[6]) / task_id
target.mkdir(parents=True)
for entry in source.iterdir():
    if entry.name != "task.toml":
        (target / entry.name).symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
text = (source / "task.toml").read_text(encoding="utf-8")
needle = "continue_until_timeout = true"
if text.count(needle) != 1:
    raise SystemExit(f"oracle overlay expected one {needle!r} in {task_id}")
(target / "task.toml").write_text(text.replace(needle, "continue_until_timeout = false"),
                                  encoding="utf-8")
(target / ".roy-native-source-task").write_text(str(source.resolve()) + "\n",
                                                 encoding="utf-8")
value = {
    "job_name": sys.argv[5],
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
    "datasets": [{"path": sys.argv[6], "task_names": [task_id]}],
}

open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(value, indent=2) + "\n")
PY
  export PYTHONPATH="${roy_root}/research${PYTHONPATH:+:${PYTHONPATH}}"
  export ROY_LHTB_NATIVE_ROOT="${native_root}"
  unset HB_CONTINUE_MODE
  cd "${lhtb_root}"
  "${harbor_bin}" run -c "${config}" --yes
  "${python_bin}" - "${jobs_root}/${job_name}/result.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
stats = result["stats"]
evaluation = next(iter(stats["evals"].values()))
mean_reward = float(evaluation["metrics"][0]["mean"])
if (stats["n_completed_trials"] != 1 or stats["n_errored_trials"] != 0
        or mean_reward < 0.95):
    raise SystemExit(
        f"native oracle smoke failed: completed={stats['n_completed_trials']} "
        f"errored={stats['n_errored_trials']} mean={mean_reward}"
    )
print(json.dumps({"oracle_smoke": "passed", "mean": mean_reward,
                  "result": sys.argv[1]}))
PY
}

oracle_suite() {
  local run_id config jobs_root job_name suite_audit overlay_root
  local -a audit_network_args=()
  if [[ "${ROY_LHTB_ALLOW_NETWORK_DEGRADED:-false}" == "true" ]]; then
    audit_network_args+=(--allow-network-degraded)
  fi
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  config="${roy_root}/research/output/lhtb/native/configs/oracle-suite-${run_id}.json"
  jobs_root="${roy_root}/research/output/lhtb/native/oracle-jobs"
  job_name="roy-native-oracle-suite-${run_id}"
  suite_audit="${roy_root}/research/output/lhtb/native/oracle-suite-audit-${run_id}.json"
  overlay_root="${roy_root}/research/output/lhtb/native/oracle-overlays/${job_name}/tasks"
  mkdir -p "$(dirname "${config}")" "${jobs_root}"
  PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research lhtb-native-audit \
    --lhtb-root "${lhtb_root}" --manifest "${roy_root}/research/config/lhtb_split.json" \
    --template-root "${template_root}" --output "${suite_audit}" \
    "${audit_network_args[@]}"
  PYTHONPATH="${roy_root}/research" "${python_bin}" - \
    "${config}" "${suite_audit}" "${template_root}" "${native_root}" \
    "${jobs_root}" "${job_name}" "${overlay_root}" "${lhtb_root}" <<'PY'
import json, os, pathlib, sys
audit = json.load(open(sys.argv[2], encoding="utf-8"))
allow_degraded = os.environ.get("ROY_LHTB_ALLOW_NETWORK_DEGRADED", "false") == "true"
accepted_statuses = {"compatible", "degraded"} if allow_degraded else {"compatible"}
tasks = [value["task_id"] for value in audit["tasks"]
         if value["status"] in accepted_statuses]
requested = [value.strip() for value in
             os.environ.get("ROY_LHTB_ORACLE_TASKS", "").split(",") if value.strip()]
if requested:
    unknown = sorted(set(requested) - set(tasks))
    if unknown:
        raise SystemExit(f"requested oracle tasks are not compatible: {unknown}")
    tasks = requested
if not tasks:
    raise SystemExit("oracle suite has no compatible native tasks")
source_tasks = pathlib.Path(sys.argv[8]) / "tasks"
overlay_tasks = pathlib.Path(sys.argv[7])
needle = "continue_until_timeout = true"
for task_id in tasks:
    source = source_tasks / task_id
    target = overlay_tasks / task_id
    target.mkdir(parents=True)
    for entry in source.iterdir():
        if entry.name != "task.toml":
            (target / entry.name).symlink_to(
                entry.resolve(), target_is_directory=entry.is_dir()
            )
    text = (source / "task.toml").read_text(encoding="utf-8")
    if text.count(needle) > 1:
        raise SystemExit(f"oracle overlay found repeated {needle!r} in {task_id}")
    (target / "task.toml").write_text(
        text.replace(needle, "continue_until_timeout = false"), encoding="utf-8"
    )
    (target / ".roy-native-source-task").write_text(
        str(source.resolve()) + "\n", encoding="utf-8"
    )
value = {
    "job_name": sys.argv[6], "jobs_dir": sys.argv[5],
    "n_attempts": 1, "n_concurrent_trials": min(4, len(tasks)),
    "environment": {
        "import_path": "roy_research.native_environment:NativeProcessEnvironment",
        "force_build": False, "delete": True,
        "kwargs": {"template_root": sys.argv[3], "runtime_root": sys.argv[4],
                   "allow_network_degraded": allow_degraded},
    },
    "agents": [{"name": "oracle"}],
    "datasets": [{"path": sys.argv[7], "task_names": tasks}],
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(value, indent=2) + "\n")
print(json.dumps({"oracle_tasks": len(tasks), "task_ids": tasks}))
PY
  export PYTHONPATH="${roy_root}/research${PYTHONPATH:+:${PYTHONPATH}}"
  export ROY_LHTB_NATIVE_ROOT="${native_root}"
  unset HB_CONTINUE_MODE
  cd "${lhtb_root}"
  "${harbor_bin}" run -c "${config}" --yes
  cd "${roy_root}"
  PYTHONPATH="${roy_root}/research" "${python_bin}" - \
    "${jobs_root}/${job_name}" "${suite_audit}" "${config}" <<'PY'
import json, pathlib, sys
from roy_research.lhtb_results import official_lhtb_reward
from roy_research.lhtb_native import normalize_native_task_id
root = pathlib.Path(sys.argv[1])
audit = json.load(open(sys.argv[2], encoding="utf-8"))
config = json.load(open(sys.argv[3], encoding="utf-8"))
expected = set(config["datasets"][0]["task_names"])
results = {}
for path in root.rglob("result.json"):
    value = json.load(open(path, encoding="utf-8"))
    if "task_checksum" not in value:
        continue
    task_id = normalize_native_task_id(str(value.get("task_name")))
    try:
        reward = official_lhtb_reward(value)
    except ValueError:
        reward = None
    results[task_id] = {
        "reward": reward,
        "exception": (value.get("exception_info") or {}).get("exception_type"),
        "result": str(path),
    }
missing = sorted(expected - set(results))
failed = {task: value for task, value in results.items()
          if value["exception"] is not None or value["reward"] is None
          or value["reward"] < 0.95}
summary = {"expected": len(expected), "completed": len(results),
           "missing": missing, "failed": failed, "results": results}
(root / "oracle-suite-validation.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({"expected": len(expected), "completed": len(results),
                  "failed": sorted(failed), "missing": missing}))
if missing or failed:
    raise SystemExit("native oracle suite failed validation")
PY
  echo "Native oracle suite saved at ${jobs_root}/${job_name}"
}

roy_smoke() {
  [[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY is required" >&2; exit 4; }
  local task_id category seed fingerprint job_dir config result_path run_id smoke_root
  local -a smoke_tasks validation_args
  run_id="smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  smoke_root="${roy_root}/research/output/lhtb/native/smoke-jobs/${run_id}"
  export PYTHONPATH="${roy_root}/research${PYTHONPATH:+:${PYTHONPATH}}"
  export ROY_LHTB_NODE_COMMAND="node ${roy_root}/dist/cli/LhtbAgent.js"
  export HB_CONTINUE_MODE=same_conversation
  export ROY_LHTB_SEMANTIC_COMMAND="${python_bin} -m roy_research.semantic_server"
  export ROY_LHTB_SEMANTIC_ROOT="${smoke_root}/semantic"
  export ROY_LHTB_POLICY_COMMAND="${python_bin} -m roy_research.lhtb_policy_server"
  export ROY_LHTB_MODEL="${roy_root}/research/output/lhtb/native/scheduler-structural-initial.pt"
  export ROY_LHTB_NATIVE_ROOT="${native_root}"
  export ROY_LHTB_MCTS_ENABLED="${ROY_LHTB_MCTS_ENABLED:-true}"
  export ROY_LHTB_MCTS_SIMULATIONS="${ROY_LHTB_MCTS_SIMULATIONS:-16}"
  export ROY_LHTB_MCTS_MAX_DEPTH="${ROY_LHTB_MCTS_MAX_DEPTH:-3}"
  export ROY_LHTB_MCTS_CPUCT="${ROY_LHTB_MCTS_CPUCT:-1.5}"
  export ROY_LHTB_MCTS_TEMPERATURE="${ROY_LHTB_MCTS_TEMPERATURE:-1}"
  export ROY_LHTB_MCTS_AGENT_EXPANSIONS="${ROY_LHTB_MCTS_AGENT_EXPANSIONS:-4}"
  export ROY_LHTB_MCTS_PROPOSAL_ATTEMPTS="${ROY_LHTB_MCTS_PROPOSAL_ATTEMPTS:-2}"
  if [[ "${ROY_LHTB_MCTS_ENABLED}" == "true" ]]; then
    export ROY_LHTB_ORGANIZATION_INTERVAL=1
  else
    export ROY_LHTB_ORGANIZATION_INTERVAL="${ROY_LHTB_ORGANIZATION_INTERVAL:-5}"
  fi
  if [[ ! -f "${ROY_LHTB_MODEL}" ]]; then
    "${python_bin}" -m roy_research lhtb-init \
      --manifest "${roy_root}/research/config/lhtb_split.json" --model "${ROY_LHTB_MODEL}"
  fi
  seed=20260820
  IFS=',' read -r -a smoke_tasks <<< \
    "${ROY_LHTB_SMOKE_TASKS:-great-expectations-audit}"
  for task_id in "${smoke_tasks[@]}"; do
    category="$("${python_bin}" - "${roy_root}/research/config/lhtb_split.json" "${task_id}" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]:
    if value["task_id"] == sys.argv[2]: print(value["category"]); break
PY
)"
    fingerprint="$(printf '%s' "native-smoke:${task_id}:${revision}:${ROY_LHTB_MCTS_ENABLED}:${ROY_LHTB_MCTS_SIMULATIONS}:${ROY_LHTB_MCTS_MAX_DEPTH}:${ROY_LHTB_MCTS_AGENT_EXPANSIONS}:${ROY_LHTB_MCTS_PROPOSAL_ATTEMPTS}" | sha256sum | awk '{print $1}')"
    job_dir="${smoke_root}/${task_id}"
    config="${roy_root}/research/output/lhtb/native/configs/${run_id}-${task_id}.json"
    mkdir -p "${job_dir}" "$(dirname "${config}")"
    "${python_bin}" -m roy_research lhtb-group-config --output "${config}" \
      --jobs-dir "${job_dir}" --task-id "${task_id}" \
      --arm learned_information_realization --initial-fingerprint "${fingerprint}" \
      --organization-seed "${seed}" --attempts 8 --environment-backend native \
      --native-runtime-root "${native_root}" --native-template-root "${template_root}" \
      --allow-network-degraded --max-retries "${ROY_LHTB_MAX_ENV_RETRIES:-8}" \
      --concurrency "${ROY_LHTB_CONCURRENCY:-4}"
    cd "${lhtb_root}"
    "${harbor_bin}" run -c "${config}" --yes
    cd "${roy_root}"
    result_path="${job_dir}/roy-${task_id}-${seed}/result.json"
    "${python_bin}" - "${job_dir}" "${task_id}" <<'PY'
import json, pathlib, sys
from roy_research.lhtb_results import official_lhtb_reward
accepted = 0
failures = []
for path in pathlib.Path(sys.argv[1]).rglob("result.json"):
    result = json.load(open(path, encoding="utf-8"))
    if "task_checksum" not in result:
        continue
    exception = result.get("exception_info") or {}
    exception_type = str(exception.get("exception_type", ""))
    try:
        official_lhtb_reward(result)
        reward_available = True
    except ValueError:
        reward_available = False
    if not exception or (reward_available and exception_type in {
        "TimeoutError", "AgentTimeoutError", "AgentTimeout"
    }):
        accepted += 1
    else:
        failures.append(exception_type or "MissingVerifierReward")
if accepted != 8:
    raise SystemExit(
        f"Roy smoke group failed for {sys.argv[2]}: "
        f"accepted={accepted} failures={failures}"
    )
print(json.dumps({"task_id": sys.argv[2], "accepted": accepted,
                  "job_dir": sys.argv[1]}))
PY
    seed=$((seed + 1))
  done
  validation_args=(--jobs-dir "${smoke_root}"
    --output "${smoke_root}/smoke-validation.json" --max-input-tokens 15000000)
  for task_id in "${smoke_tasks[@]}"; do validation_args+=(--task-id "${task_id}"); done
  "${python_bin}" -m roy_research lhtb-smoke-validate "${validation_args[@]}"
  echo "Roy native smoke saved at ${smoke_root}"
}

case "${mode}" in
  check) export PATH="${node_runtime}/bin:${proot_runtime}/bin:${PATH}"; check_environment ;;
  prepare) install_system_dependencies; install_node_22; install_proot_runtime; check_environment; prepare_checkout ;;
  provision) install_node_22; install_proot_runtime; check_environment; prepare_checkout; provision_reviewed_tasks ;;
  oracle-smoke) install_node_22; install_proot_runtime; check_environment; prepare_checkout; provision_smoke_tasks; oracle_smoke ;;
  oracle-suite) install_node_22; install_proot_runtime; check_environment; prepare_checkout; oracle_suite ;;
  roy-smoke) install_node_22; install_proot_runtime; check_environment; prepare_checkout; provision_smoke_tasks; roy_smoke ;;
  *) echo "usage: $0 [check|prepare|provision|oracle-smoke|oracle-suite|roy-smoke]" >&2; exit 2 ;;
esac

echo "LHTB-native ${mode} completed; results are not official leaderboard comparable"
