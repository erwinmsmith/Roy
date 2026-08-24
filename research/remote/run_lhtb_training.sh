#!/usr/bin/env bash
set -euo pipefail

roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
lhtb_root="${LHTB_ROOT:-${HOME}/rivermind-data/benchmarks/LHTB}"
python_bin="${roy_root}/research/.venv/bin/python"
harbor_bin="${roy_root}/research/.venv/bin/harbor"
run_root="${ROY_LHTB_RUN_ROOT:-${roy_root}/research/output/lhtb/formal}"
manifest="${ROY_LHTB_MANIFEST:-${roy_root}/research/config/lhtb_split.json}"
schedule="${run_root}/schedule.json"
model="${run_root}/checkpoints/current.pt"
trajectories="${run_root}/train-trajectories.jsonl"
updates="${run_root}/update-audit.jsonl"
dev_trajectories="${run_root}/dev-trajectories.jsonl"
dev_metrics="${run_root}/dev-metrics.jsonl"
environment_backend="${ROY_LHTB_ENVIRONMENT_BACKEND:-docker}"
native_runtime_root="${ROY_LHTB_NATIVE_ROOT:-${roy_root}/research/output/lhtb/native/runtime}"
native_template_root="${ROY_LHTB_NATIVE_TEMPLATE_ROOT:-${roy_root}/research/output/lhtb/native/templates}"
native_audit="${ROY_LHTB_NATIVE_AUDIT:-${roy_root}/research/output/lhtb/native/audit.json}"
allow_network_degraded="${ROY_LHTB_ALLOW_NETWORK_DEGRADED:-false}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_roy_env.sh
source "${script_dir}/load_roy_env.sh"
load_deepseek_api_key "${roy_root}"

[[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY is required" >&2; exit 4; }
[[ -x "${python_bin}" && -x "${harbor_bin}" ]] || {
  if [[ "${environment_backend}" == "native" ]]; then
    echo "run prepare_lhtb_native.sh prepare first" >&2
  else
    echo "run prepare_lhtb.sh prepare first" >&2
  fi
  exit 4
}
[[ "$(git -C "${lhtb_root}" rev-parse HEAD)" == \
  "84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54" ]] || {
  echo "LHTB checkout is not at the pinned commit" >&2
  exit 4
}

mkdir -p "${run_root}/checkpoints" "${run_root}/jobs" "${run_root}/configs"
export PYTHONPATH="${roy_root}/research${PYTHONPATH:+:${PYTHONPATH}}"
export ROY_LHTB_NODE_COMMAND="node ${roy_root}/dist/cli/LhtbAgent.js"
export HB_CONTINUE_MODE=same_conversation
export ROY_LHTB_POLICY_COMMAND="${python_bin} -m roy_research.lhtb_policy_server"
export ROY_LHTB_SEMANTIC_COMMAND="${python_bin} -m roy_research.semantic_server"
export ROY_LHTB_SEMANTIC_ROOT="${run_root}/semantic"
export ROY_LHTB_MODEL="${model}"
export DEEPSEEK_MODEL_REVISION="${DEEPSEEK_MODEL_REVISION:-deepseek-v4-flash-api-alias}"
export ROY_LHTB_ORGANIZATION_INTERVAL="${ROY_LHTB_ORGANIZATION_INTERVAL:-5}"
export ROY_LHTB_EXPLORATION_MIN_NODES="${ROY_LHTB_EXPLORATION_MIN_NODES:-3}"
export ROY_LHTB_EXPLORATION_MIN_DEPTH="${ROY_LHTB_EXPLORATION_MIN_DEPTH:-2}"

group_environment_args=(--environment-backend "${environment_backend}"
  --max-retries "${ROY_LHTB_MAX_ENV_RETRIES:-8}")
if [[ "${environment_backend}" == "native" ]]; then
  [[ -f "${native_audit}" ]] || { echo "native audit is missing" >&2; exit 4; }
  group_environment_args+=(--native-runtime-root "${native_runtime_root}"
    --native-template-root "${native_template_root}")
  if [[ "${allow_network_degraded}" == "true" ]]; then
    group_environment_args+=(--allow-network-degraded)
  fi
fi

cd "${roy_root}"
npm run build
if [[ ! -f "${model}" ]]; then
  "${python_bin}" -m roy_research lhtb-init --manifest "${manifest}" --model "${model}"
fi
"${python_bin}" -m roy_research lhtb-schedule --manifest "${manifest}" --output "${schedule}"

resolve_digest() {
  local task_id="$1" digest
  if [[ "${environment_backend}" == "native" ]]; then
    "${python_bin}" -m roy_research lhtb-native-digest \
      --audit "${native_audit}" --task-id "${task_id}"
    return
  fi
  digest="$(docker image ls --no-trunc --format '{{.Repository}} {{.ID}}' \
    | awk -v task="${task_id}" 'index($1, task) {print $2}' | sort -u | head -n 1)"
  [[ "${digest}" == sha256:* ]] || {
    echo "cannot resolve immutable Docker image ID for ${task_id}" >&2
    return 5
  }
  printf '%s' "${digest}"
}

run_dev_epoch() {
  local dev_epoch="$1" checkpoint="${run_root}/checkpoints/epoch-${dev_epoch}.pt"
  if [[ -f "${dev_metrics}" ]] && "${python_bin}" - "${dev_metrics}" "${dev_epoch}" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
raise SystemExit(0 if len([x for x in rows if x["epoch"] == int(sys.argv[2])]) == 8 else 1)
PY
  then
    echo "skip completed dev epoch ${dev_epoch}"
    return
  fi
  while IFS=$'\t' read -r task_id category; do
    if [[ -f "${dev_trajectories}" ]] && "${python_bin}" - "${dev_trajectories}" "${dev_epoch}" "${task_id}" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
raise SystemExit(0 if any(x["epoch"] == int(sys.argv[2]) and x["task_id"] == sys.argv[3] for x in rows) else 1)
PY
    then
      continue
    fi
    local group_id="dev:${dev_epoch}:${task_id}"
    local task_tree initial_fingerprint job_dir config_path digest revision seed
    task_tree="$(git -C "${lhtb_root}" rev-parse "HEAD:tasks/${task_id}")"
    initial_fingerprint="$(printf '%s' "${group_id}:${task_tree}:3600:32768" | shasum -a 256 | awk '{print $1}')"
    seed="$((16#$(printf '%s' "${group_id}" | shasum -a 256 | cut -c1-8)))"
    revision="$("${python_bin}" - "${model}" <<'PY'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["metadata"]["groups"])
PY
)"
    job_dir="${run_root}/jobs/dev-${dev_epoch}-${task_id}"
    config_path="${run_root}/configs/dev-${dev_epoch}-${task_id}.json"
    mkdir -p "${job_dir}"
    "${python_bin}" -m roy_research lhtb-group-config --output "${config_path}" \
      --jobs-dir "${job_dir}" --task-id "${task_id}" --arm learned_information_realization \
      --initial-fingerprint "${initial_fingerprint}" --organization-seed "${seed}" --attempts 1 \
      "${group_environment_args[@]}"
    cd "${lhtb_root}"
    "${harbor_bin}" run -c "${config_path}" --yes
    cd "${roy_root}"
    digest="$(resolve_digest "${task_id}")"
    "${python_bin}" -m roy_research lhtb-import-group --job-dir "${job_dir}" \
      --output "${dev_trajectories}" --group-id "${group_id}" --task-id "${task_id}" \
      --category "${category}" --split dev --epoch "${dev_epoch}" \
      --policy-revision "${revision}" --environment-digest "${digest}" \
      --environment-backend "${environment_backend}" --expected 1
  done < <("${python_bin}" - "${manifest}" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]:
    if value["split"] == "dev": print(value["task_id"], value["category"], sep="\t")
PY
)
  cp "${model}" "${checkpoint}"
  "${python_bin}" -m roy_research lhtb-dev-metrics --trajectories "${dev_trajectories}" \
    --checkpoint "${checkpoint}" --epoch "${dev_epoch}" --output "${dev_metrics}"
}

"${python_bin}" - "${schedule}" "${manifest}" <<'PY' | {
import json, sys
schedule = json.load(open(sys.argv[1], encoding="utf-8"))["groups"]
manifest = {x["task_id"]: x for x in json.load(open(sys.argv[2], encoding="utf-8"))["tasks"]}
for group in schedule:
    print(group["epoch"], group["task_id"], group["group_id"],
          group["organization_seeds"][0], manifest[group["task_id"]]["category"], sep="\t")
PY
current_epoch=-1
while IFS=$'\t' read -r epoch task_id group_id base_seed category; do
  if [[ "${current_epoch}" -ge 0 && "${epoch}" != "${current_epoch}" ]]; then
    run_dev_epoch "${current_epoch}"
  fi
  current_epoch="${epoch}"
  if "${python_bin}" - "${model}" "${group_id}" <<'PY'
import sys, torch
value = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(0 if sys.argv[2] in value.get("metadata", {}).get("updated_group_ids", []) else 1)
PY
  then
    echo "skip completed ${group_id}"
    continue
  fi

  if [[ "${environment_backend}" == "native" ]]; then
    "${python_bin}" -m roy_research lhtb-native-preflight --runtime-root "${native_runtime_root}" \
      --output "${run_root}/preflight-latest.json"
  else
    "${python_bin}" -m roy_research lhtb-preflight --path "$(dirname "${lhtb_root}")" \
      --output "${run_root}/preflight-latest.json"
  fi
  policy_revision="$("${python_bin}" - "${model}" <<'PY'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["metadata"]["groups"])
PY
)"
  task_tree="$(git -C "${lhtb_root}" rev-parse "HEAD:tasks/${task_id}")"
  initial_fingerprint="$(printf '%s' "${group_id}:${task_tree}:3600:32768" | shasum -a 256 | awk '{print $1}')"
  job_dir="${run_root}/jobs/${epoch}-${task_id}"
  config_path="${run_root}/configs/${epoch}-${task_id}.json"
  mkdir -p "${job_dir}"
  "${python_bin}" -m roy_research lhtb-group-config \
    --output "${config_path}" --jobs-dir "${job_dir}" --task-id "${task_id}" \
    --arm learned_information_realization --initial-fingerprint "${initial_fingerprint}" \
    --organization-seed "${base_seed}" "${group_environment_args[@]}"

  cd "${lhtb_root}"
  "${harbor_bin}" run -c "${config_path}" --yes
  cd "${roy_root}"

  environment_digest="$(resolve_digest "${task_id}")"
  "${python_bin}" -m roy_research lhtb-import-group \
    --job-dir "${job_dir}" --output "${trajectories}" --group-id "${group_id}" \
    --task-id "${task_id}" --category "${category}" --split train --epoch "${epoch}" \
    --policy-revision "${policy_revision}" --environment-digest "${environment_digest}" \
    --environment-backend "${environment_backend}"
  "${python_bin}" -m roy_research lhtb-update \
    --manifest "${manifest}" --trajectories "${trajectories}" --model "${model}" \
    --updates "${updates}" --resume
done
if [[ "${current_epoch}" -ge 0 ]]; then run_dev_epoch "${current_epoch}"; fi
}

"${python_bin}" -m roy_research lhtb-select --metrics "${dev_metrics}" \
  --output "${run_root}/selected-checkpoint.json"

echo "formal LHTB training schedule complete: ${model}"
