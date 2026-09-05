#!/usr/bin/env bash
set -euo pipefail

roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
aflow_root="${AFLOW_ROOT:-${HOME}/rivermind-data/benchmarks/AFlow}"
python_bin="${ROY_TF_PYTHON:-${roy_root}/research/.venv/bin/python}"
aflow_python="${AFLOW_PYTHON:-${aflow_root}/.venv/bin/python}"
run_root="${1:?usage: run_training_free_matrix.sh RUN_ROOT MODEL [LIMIT]}"
model="${2:?usage: run_training_free_matrix.sh RUN_ROOT MODEL [LIMIT]}"
limit="${3:-8}"
provider="${ROY_TF_PROVIDER:-deepseek}"
api_key_env="${ROY_TF_API_KEY_ENV:-OPENAI_API_KEY}"
base_url="${ROY_TF_BASE_URL:-}"

[[ "${limit}" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT must be positive" >&2; exit 2; }
[[ -x "${python_bin}" && -x "${aflow_python}" ]] || {
  echo "Roy and AFlow Python environments are required" >&2
  exit 2
}
if [[ "${provider}" == "openai-compatible" ]]; then
  [[ -n "${base_url}" ]] || { echo "ROY_TF_BASE_URL is required" >&2; exit 2; }
  [[ -n "${!api_key_env:-}" ]] || { echo "${api_key_env} is required" >&2; exit 2; }
fi

mkdir -p "${run_root}"
if find "${run_root}" -maxdepth 1 -name '*.jsonl' -print -quit | grep -q .; then
  echo "refusing to overwrite an existing experiment in ${run_root}" >&2
  exit 2
fi

provider_args=(--provider "${provider}")
if [[ "${provider}" == "openai-compatible" ]]; then
  provider_args+=(--base-url "${base_url}" --api-key-env "${api_key_env}")
  if [[ -n "${ROY_TF_PROVIDER_MAX_OUTPUT_TOKENS:-}" ]]; then
    provider_args+=(--provider-max-output-tokens "${ROY_TF_PROVIDER_MAX_OUTPUT_TOKENS}")
  fi
fi
common=(
  --aflow-root "${aflow_root}"
  --split test
  --limit "${limit}"
  "${provider_args[@]}"
  --worker-model "${model}"
  --candidate-model "${model}"
  --max-task-attempts "${ROY_TF_MAX_TASK_ATTEMPTS:-3}"
  --provider-max-retries "${ROY_TF_PROVIDER_MAX_RETRIES:-6}"
  --provider-retry-base-seconds "${ROY_TF_PROVIDER_RETRY_BASE_SECONDS:-5}"
  --timeout "${ROY_TF_TIMEOUT:-300}"
  --score
  --aflow-python "${aflow_python}"
)
math_sandbox="${ROY_TF_MATH_SANDBOX:-env -i PATH=/usr/bin:/bin setpriv --reuid=210234 --regid=210000 --clear-groups --no-new-privs}"
he_sandbox="${ROY_TF_HE_SANDBOX:-env -i PATH=/usr/bin PYTHONPATH=${aflow_root}/.venv/lib/python3.12/site-packages:${aflow_root} setpriv --reuid=210232 --regid=210000 --clear-groups --no-new-privs}"
pids_tmp="${run_root}/pids.tsv.tmp"
: > "${pids_tmp}"

launch() {
  local name="$1" config="$2" benchmark="$3" arm="$4" token_limit="$5"
  local sandbox="${math_sandbox}"
  local -a benchmark_args=()
  if [[ "${benchmark}" == "HumanEval" ]]; then
    sandbox="${he_sandbox}"
    benchmark_args+=(--human-eval-sandbox-command "${he_sandbox}")
  fi
  nohup env PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research \
    training-free-run "${common[@]}" \
    --config "${roy_root}/${config}" \
    --benchmark "${benchmark}" \
    --arm "${arm}" \
    --token-limit "${token_limit}" \
    --tool-sandbox-command "${sandbox}" \
    "${benchmark_args[@]}" \
    --output "${run_root}/${name}.jsonl" \
    --ledger "${run_root}/${name}.ledger.json" \
    --events "${run_root}/${name}.events.jsonl" \
    > "${run_root}/${name}.log" 2>&1 &
  printf '%s\t%s\n' "${name}" "$!" | tee -a "${pids_tmp}"
}

launch scalar-math research/config/training_free_v1.json MATH roy \
  "${ROY_TF_ROY_TOKEN_LIMIT:-10000000}"
launch logdet-math research/config/training_free_logdet_v1.json MATH roy \
  "${ROY_TF_ROY_TOKEN_LIMIT:-10000000}"
launch direct-math research/config/training_free_v1.json MATH single_agent_direct \
  "${ROY_TF_DIRECT_TOKEN_LIMIT:-3000000}"
launch scalar-humaneval research/config/training_free_v1.json HumanEval roy \
  "${ROY_TF_ROY_TOKEN_LIMIT:-10000000}"
launch logdet-humaneval research/config/training_free_logdet_v1.json HumanEval roy \
  "${ROY_TF_ROY_TOKEN_LIMIT:-10000000}"
launch direct-humaneval research/config/training_free_v1.json HumanEval single_agent_direct \
  "${ROY_TF_DIRECT_TOKEN_LIMIT:-3000000}"

mv "${pids_tmp}" "${run_root}/pids.tsv"
