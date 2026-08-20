#!/usr/bin/env bash
set -euo pipefail

roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
lhtb_root="${LHTB_ROOT:-${HOME}/rivermind-data/benchmarks/LHTB}"
python_bin="${roy_root}/research/.venv/bin/python"
harbor_bin="${roy_root}/research/.venv/bin/harbor"
run_root="${ROY_LHTB_RUN_ROOT:-${roy_root}/research/output/lhtb/formal}"
manifest="${ROY_LHTB_MANIFEST:-${roy_root}/research/config/lhtb_split.json}"
selection="${run_root}/selected-checkpoint.json"
results="${run_root}/test-results.jsonl"

[[ -n "${DEEPSEEK_API_KEY:-}" && -f "${selection}" ]] || {
  echo "DeepSeek key and selected-checkpoint.json are required" >&2
  exit 4
}
selected_model="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${selection}")"
selected_epoch="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["epoch"])' "${selection}")"
[[ -f "${selected_model}" ]] || { echo "selected checkpoint is missing" >&2; exit 4; }

mkdir -p "${run_root}/jobs" "${run_root}/configs"
export PYTHONPATH="${roy_root}/research${PYTHONPATH:+:${PYTHONPATH}}"
export ROY_LHTB_NODE_COMMAND="node ${roy_root}/dist/cli/LhtbAgent.js"
export ROY_LHTB_POLICY_COMMAND="${python_bin} -m roy_research.lhtb_policy_server"
export ROY_LHTB_SEMANTIC_COMMAND="${python_bin} -m roy_research.semantic_server"
export ROY_LHTB_SEMANTIC_ROOT="${run_root}/semantic-test"
export ROY_LHTB_MODEL="${selected_model}"
export DEEPSEEK_MODEL_REVISION="${DEEPSEEK_MODEL_REVISION:-deepseek-v4-flash-api-alias}"

while IFS=$'\t' read -r task_id category; do
  task_tree="$(git -C "${lhtb_root}" rev-parse "HEAD:tasks/${task_id}")"
  initial_fingerprint="$(printf '%s' "test:${task_tree}:official-timeout" | shasum -a 256 | awk '{print $1}')"
  seed="$((16#$(printf '%s' "test:${task_id}" | shasum -a 256 | cut -c1-8)))"
  for arm in single_agent_direct roy_runtime_heuristic learned_information_realization; do
    if [[ -f "${results}" ]] && "${python_bin}" - "${results}" "${task_id}" "${arm}" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
raise SystemExit(0 if len([x for x in rows if x["task_id"] == sys.argv[2] and x["arm"] == sys.argv[3]]) == 3 else 1)
PY
    then
      continue
    fi
    job_dir="${run_root}/jobs/test-${arm}-${task_id}"
    config_path="${run_root}/configs/test-${arm}-${task_id}.json"
    mkdir -p "${job_dir}"
    "${python_bin}" -m roy_research lhtb-group-config --output "${config_path}" \
      --jobs-dir "${job_dir}" --task-id "${task_id}" --arm "${arm}" \
      --initial-fingerprint "${initial_fingerprint}" --organization-seed "${seed}" \
      --attempts 3 --official-timeout
    cd "${lhtb_root}"
    "${harbor_bin}" run -c "${config_path}" --yes
    cd "${roy_root}"
    digest="$(docker image ls --no-trunc --format '{{.Repository}} {{.ID}}' \
      | awk -v task="${task_id}" 'index($1, task) {print $2}' | sort -u | head -n 1)"
    [[ "${digest}" == sha256:* ]] || { echo "missing image digest for ${task_id}" >&2; exit 5; }
    "${python_bin}" -m roy_research lhtb-import-group --job-dir "${job_dir}" \
      --output "${results}" --group-id "test:${arm}:${task_id}" --task-id "${task_id}" \
      --category "${category}" --split test --epoch "${selected_epoch}" \
      --policy-revision 0 --docker-digest "${digest}" --expected 3 --arm "${arm}"
  done
done < <("${python_bin}" - "${manifest}" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]:
    if value["split"] == "test": print(value["task_id"], value["category"], sep="\t")
PY
)

"${python_bin}" -m roy_research lhtb-report --results "${results}" \
  --checkpoint "${selected_model}" --output "${run_root}/test-summary.json"
echo "frozen LHTB test complete: ${run_root}/test-summary.json"
