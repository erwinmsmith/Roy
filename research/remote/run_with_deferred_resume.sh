#!/usr/bin/env bash
set -euo pipefail

log_path="${1:?usage: run_with_deferred_resume.sh LOG OUTPUT BASE_DELAY MAX_DELAY EXIT_CODE -- COMMAND...}"
output_path="${2:?usage: run_with_deferred_resume.sh LOG OUTPUT BASE_DELAY MAX_DELAY EXIT_CODE -- COMMAND...}"
base_delay="${3:?usage: run_with_deferred_resume.sh LOG OUTPUT BASE_DELAY MAX_DELAY EXIT_CODE -- COMMAND...}"
max_delay="${4:?usage: run_with_deferred_resume.sh LOG OUTPUT BASE_DELAY MAX_DELAY EXIT_CODE -- COMMAND...}"
deferred_exit_code="${5:?usage: run_with_deferred_resume.sh LOG OUTPUT BASE_DELAY MAX_DELAY EXIT_CODE -- COMMAND...}"
shift 5
[[ "${1:-}" == "--" ]] || {
  echo "run_with_deferred_resume.sh requires -- before COMMAND" >&2
  exit 2
}
shift
(( $# > 0 )) || { echo "run_with_deferred_resume.sh requires COMMAND" >&2; exit 2; }
[[ "${base_delay}" =~ ^[1-9][0-9]*$ ]] || {
  echo "BASE_DELAY must be a positive integer" >&2
  exit 2
}
[[ "${max_delay}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_DELAY must be a positive integer" >&2
  exit 2
}
[[ "${deferred_exit_code}" =~ ^[1-9][0-9]*$ ]] \
  && (( deferred_exit_code <= 255 )) || {
    echo "EXIT_CODE must be between 1 and 255" >&2
    exit 2
  }
(( base_delay <= max_delay )) || {
  echo "BASE_DELAY cannot exceed MAX_DELAY" >&2
  exit 2
}

mkdir -p "$(dirname "${log_path}")"
delay="${base_delay}"
child_pid=""

completed_rows() {
  if [[ ! -f "${output_path}" ]]; then
    echo 0
    return
  fi
  grep -c '"run_status": "completed"' "${output_path}" || true
}

forward_signal() {
  local signal="$1"
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "-${signal}" "${child_pid}" 2>/dev/null || kill "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  exit 128
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

while true; do
  before="$(completed_rows)"
  printf '%s supervisor=start completed_rows=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${before}" \
    >> "${log_path}.supervisor"
  "$@" >> "${log_path}" 2>&1 &
  child_pid="$!"
  status=0
  wait "${child_pid}" || status=$?
  child_pid=""
  after="$(completed_rows)"
  printf '%s supervisor=exit status=%s completed_rows=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${status}" "${after}" >> "${log_path}.supervisor"
  if (( status == 0 )); then
    exit 0
  fi
  if (( status != deferred_exit_code )); then
    exit "${status}"
  fi
  if (( after > before )); then
    delay="${base_delay}"
  else
    delay=$(( delay * 2 ))
    (( delay > max_delay )) && delay="${max_delay}"
  fi
  printf '%s supervisor=deferred cooldown_seconds=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${delay}" >> "${log_path}.supervisor"
  sleep "${delay}"
done
