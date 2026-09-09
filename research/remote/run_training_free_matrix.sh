#!/usr/bin/env bash
set -euo pipefail

roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
aflow_root="${AFLOW_ROOT:-${HOME}/rivermind-data/benchmarks/AFlow}"
python_bin="${ROY_TF_PYTHON:-${roy_root}/research/.venv/bin/python}"
aflow_python="${AFLOW_PYTHON:-${aflow_root}/.venv/bin/python}"
run_root="${1:?usage: run_training_free_matrix.sh RUN_ROOT MODEL [LIMIT|all]}"
model="${2:?usage: run_training_free_matrix.sh RUN_ROOT MODEL [LIMIT|all]}"
limit="${3:-8}"
provider="${ROY_TF_PROVIDER:-deepseek}"
api_key_env="${ROY_TF_API_KEY_ENV:-OPENAI_API_KEY}"
base_url="${ROY_TF_BASE_URL:-}"
wait_for_pid="${ROY_TF_WAIT_FOR_PID:-}"
wait_seconds="${ROY_TF_WAIT_SECONDS:-30}"
resume="${ROY_TF_RESUME:-false}"
include_continual="${ROY_TF_INCLUDE_CONTINUAL:-false}"
auto_resume="${ROY_TF_AUTO_RESUME:-false}"
auto_resume_delay="${ROY_TF_AUTO_RESUME_DELAY_SECONDS:-60}"
auto_resume_max_delay="${ROY_TF_AUTO_RESUME_MAX_DELAY_SECONDS:-900}"
deferred_exit_code="${ROY_TF_PROVIDER_DEFERRED_EXIT_CODE:-75}"
resume_runner="${roy_root}/research/remote/run_with_deferred_resume.sh"

[[ "${limit}" == "all" || "${limit}" =~ ^[1-9][0-9]*$ ]] || {
  echo "LIMIT must be a positive integer or 'all'" >&2
  exit 2
}
[[ "${wait_seconds}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ROY_TF_WAIT_SECONDS must be positive" >&2
  exit 2
}
if [[ -n "${wait_for_pid}" && ! "${wait_for_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROY_TF_WAIT_FOR_PID must be a positive process id" >&2
  exit 2
fi
[[ "${resume}" == "true" || "${resume}" == "false" ]] || {
  echo "ROY_TF_RESUME must be true or false" >&2
  exit 2
}
[[ "${include_continual}" == "true" || "${include_continual}" == "false" ]] || {
  echo "ROY_TF_INCLUDE_CONTINUAL must be true or false" >&2
  exit 2
}
[[ "${auto_resume}" == "true" || "${auto_resume}" == "false" ]] || {
  echo "ROY_TF_AUTO_RESUME must be true or false" >&2
  exit 2
}
if [[ "${auto_resume}" == "true" ]]; then
  [[ "${resume}" == "true" ]] || {
    echo "ROY_TF_AUTO_RESUME requires ROY_TF_RESUME=true" >&2
    exit 2
  }
  [[ -x "${resume_runner}" ]] || {
    echo "resumable provider runner is unavailable: ${resume_runner}" >&2
    exit 2
  }
  [[ "${auto_resume_delay}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ROY_TF_AUTO_RESUME_DELAY_SECONDS must be positive" >&2
    exit 2
  }
  [[ "${auto_resume_max_delay}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ROY_TF_AUTO_RESUME_MAX_DELAY_SECONDS must be positive" >&2
    exit 2
  }
  [[ "${deferred_exit_code}" =~ ^[1-9][0-9]*$ ]] \
    && (( deferred_exit_code <= 255 )) || {
      echo "ROY_TF_PROVIDER_DEFERRED_EXIT_CODE must be between 1 and 255" >&2
      exit 2
    }
fi
[[ -x "${python_bin}" && -x "${aflow_python}" ]] || {
  echo "Roy and AFlow Python environments are required" >&2
  exit 2
}
if [[ "${provider}" == "openai-compatible" ]]; then
  [[ -n "${base_url}" ]] || { echo "ROY_TF_BASE_URL is required" >&2; exit 2; }
  [[ -n "${!api_key_env:-}" ]] || { echo "${api_key_env} is required" >&2; exit 2; }
fi

if [[ -n "${wait_for_pid}" ]]; then
  echo "waiting for process ${wait_for_pid} before starting ${run_root}"
  while kill -0 "${wait_for_pid}" 2>/dev/null; do
    sleep "${wait_seconds}"
  done
fi

mkdir -p "${run_root}"
if [[ "${resume}" != "true" ]] \
  && find "${run_root}" -maxdepth 1 -name '*.jsonl' -print -quit | grep -q .; then
  echo "refusing to overwrite an existing experiment in ${run_root}" >&2
  exit 2
fi

provider_args=(--provider "${provider}")
if [[ "${provider}" == "openai-compatible" ]]; then
  provider_args+=(--base-url "${base_url}" --api-key-env "${api_key_env}")
  if [[ -n "${ROY_TF_PROVIDER_MAX_OUTPUT_TOKENS:-}" ]]; then
    provider_args+=(--provider-max-output-tokens "${ROY_TF_PROVIDER_MAX_OUTPUT_TOKENS}")
  fi
  if [[ -n "${ROY_TF_PROVIDER_CONTEXT_WINDOW_TOKENS:-}" ]]; then
    provider_args+=(
      --provider-context-window-tokens "${ROY_TF_PROVIDER_CONTEXT_WINDOW_TOKENS}"
      --provider-context-safety-tokens "${ROY_TF_PROVIDER_CONTEXT_SAFETY_TOKENS:-1024}"
    )
  fi
fi
common=(
  --aflow-root "${aflow_root}"
  --split test
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
if [[ "${limit}" != "all" ]]; then
  common+=(--limit "${limit}")
fi
if [[ "${resume}" == "true" ]]; then
  common+=(--resume)
fi
if [[ "${auto_resume}" == "true" ]]; then
  common+=(--provider-deferred-exit-code "${deferred_exit_code}")
fi
math_sandbox="${ROY_TF_MATH_SANDBOX:-env -i PATH=/usr/bin:/bin setpriv --reuid=210234 --regid=210000 --clear-groups --no-new-privs}"
he_sandbox="${ROY_TF_HE_SANDBOX:-env -i PATH=/usr/bin PYTHONPATH=${aflow_root}/.venv/lib/python3.12/site-packages:${aflow_root} setpriv --reuid=210232 --regid=210000 --clear-groups --no-new-privs}"
pids_tmp="${run_root}/pids.tsv.tmp"
: > "${pids_tmp}"
serial="${ROY_TF_SERIAL:-false}"
concurrency="${ROY_TF_CONCURRENCY:-0}"
[[ "${concurrency}" =~ ^[0-9]+$ ]] || {
  echo "ROY_TF_CONCURRENCY must be zero (unlimited) or a positive integer" >&2
  exit 2
}

throttle() {
  local oldest_pid
  if [[ "${serial}" == "true" || "${concurrency}" == "0" ]]; then
    return
  fi
  while (( $(jobs -pr | wc -l) >= concurrency )); do
    oldest_pid="$(jobs -pr | head -n 1)"
    [[ -n "${oldest_pid}" ]] || break
    wait "${oldest_pid}" || true
  done
}

launch() {
  local name="$1" config="$2" benchmark="$3" arm="$4" token_limit="$5"
  local sandbox="${math_sandbox}"
  local -a benchmark_args=()
  if [[ "${benchmark}" == "HumanEval" ]]; then
    sandbox="${he_sandbox}"
    benchmark_args+=(--human-eval-sandbox-command "${he_sandbox}")
  fi
  local -a command=(env PYTHONPATH="${roy_root}/research" "${python_bin}" -m roy_research \
    training-free-run "${common[@]}" \
    --config "${roy_root}/${config}" \
    --benchmark "${benchmark}" \
    --arm "${arm}" \
    --token-limit "${token_limit}" \
    --tool-sandbox-command "${sandbox}" \
    "${benchmark_args[@]}" \
    --output "${run_root}/${name}.jsonl" \
    --ledger "${run_root}/${name}.ledger.json" \
    --events "${run_root}/${name}.events.jsonl")
  local -a supervised_command=("${command[@]}")
  if [[ "${auto_resume}" == "true" ]]; then
    supervised_command=("${resume_runner}" \
      "${run_root}/${name}.log" \
      "${run_root}/${name}.jsonl" \
      "${auto_resume_delay}" \
      "${auto_resume_max_delay}" \
      "${deferred_exit_code}" \
      -- "${command[@]}")
  fi
  if [[ "${serial}" == "true" ]]; then
    printf '%s\t%s\n' "${name}" "running" | tee -a "${pids_tmp}"
    local status=0
    if [[ "${resume}" == "true" ]]; then
      if [[ "${auto_resume}" == "true" ]]; then
        "${supervised_command[@]}" || status=$?
      else
        "${command[@]}" >> "${run_root}/${name}.log" 2>&1 || status=$?
      fi
    else
      "${command[@]}" > "${run_root}/${name}.log" 2>&1 || status=$?
    fi
    printf '%s\t%s\n' "${name}" "exit=${status}" | tee -a "${pids_tmp}"
  else
    throttle
    if [[ "${resume}" == "true" ]]; then
      if [[ "${auto_resume}" == "true" ]]; then
        nohup "${supervised_command[@]}" \
          >> "${run_root}/${name}.launcher.log" 2>&1 &
      else
        nohup "${command[@]}" >> "${run_root}/${name}.log" 2>&1 &
      fi
    else
      nohup "${command[@]}" > "${run_root}/${name}.log" 2>&1 &
    fi
    printf '%s\t%s\n' "${name}" "$!" | tee -a "${pids_tmp}"
  fi
}

launch direct-math research/config/training_free_v1.json MATH single_agent_direct \
  "${ROY_TF_DIRECT_TOKEN_LIMIT:-3000000}"
launch direct-humaneval research/config/training_free_v1.json HumanEval single_agent_direct \
  "${ROY_TF_DIRECT_TOKEN_LIMIT:-3000000}"
launch scalar-math research/config/training_free_v1.json MATH roy \
  "${ROY_TF_ROY_TOKEN_LIMIT:-10000000}"
launch scalar-humaneval research/config/training_free_v1.json HumanEval roy \
  "${ROY_TF_ROY_TOKEN_LIMIT:-10000000}"
launch logdet-math research/config/training_free_logdet_v1.json MATH roy \
  "${ROY_TF_ROY_TOKEN_LIMIT:-10000000}"
launch logdet-humaneval research/config/training_free_logdet_v1.json HumanEval roy \
  "${ROY_TF_ROY_TOKEN_LIMIT:-10000000}"
if [[ "${include_continual}" == "true" ]]; then
  launch continual-logdet-math research/config/training_free_continual_v1.json MATH roy_continual \
    "${ROY_TF_CONTINUAL_TOKEN_LIMIT:-100000000}"
  launch continual-logdet-humaneval research/config/training_free_continual_v1.json HumanEval roy_continual \
    "${ROY_TF_CONTINUAL_TOKEN_LIMIT:-100000000}"
fi

mv "${pids_tmp}" "${run_root}/pids.tsv"
