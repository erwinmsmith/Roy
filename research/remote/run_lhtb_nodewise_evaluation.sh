#!/usr/bin/env bash
set -euo pipefail

# Evaluate all four frozen node-wise checkpoints on dev, select once without
# test leakage, then run the locked three-arm LHTB test protocol.

roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
lhtb_root="${LHTB_ROOT:-${HOME}/rivermind-data/benchmarks/LHTB}"
python_bin="${roy_root}/research/.venv/bin/python"
harbor_bin="${roy_root}/research/.venv/bin/harbor"
run_root="${ROY_LHTB_RUN_ROOT:-${roy_root}/research/output/lhtb/native/formal-nodewise}"
manifest="${ROY_LHTB_MANIFEST:-${roy_root}/research/config/lhtb_split.json}"
native_runtime_root="${ROY_LHTB_NATIVE_ROOT:-${HOME}/rivermind-data/lhtb-native/runtime}"
native_template_root="${ROY_LHTB_NATIVE_TEMPLATE_ROOT:-${HOME}/rivermind-data/lhtb-native/templates}"
native_audit="${ROY_LHTB_NATIVE_AUDIT:-${roy_root}/research/output/lhtb/native/audit.json}"
trajectories="${run_root}/dev-trajectories.jsonl"
metrics="${run_root}/dev-metrics.jsonl"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=load_roy_env.sh
source "${script_dir}/load_roy_env.sh"
load_deepseek_api_key "${roy_root}"

[[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY is required" >&2; exit 4; }
[[ -x "${python_bin}" && -x "${harbor_bin}" && -f "${native_audit}" ]] || {
  echo "prepared Roy research and audited native LHTB environments are required" >&2
  exit 4
}
[[ "$(git -C "${lhtb_root}" rev-parse HEAD)" == \
  "84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54" ]] || {
  echo "LHTB checkout is not at the pinned commit" >&2
  exit 4
}

mkdir -p "${run_root}/dev-jobs" "${run_root}/dev-configs"
export PYTHONPATH="${roy_root}/research${PYTHONPATH:+:${PYTHONPATH}}"
export ROY_LHTB_ENVIRONMENT_BACKEND=native
export ROY_LHTB_NODE_COMMAND="node ${roy_root}/dist/cli/LhtbAgent.js"
export HB_CONTINUE_MODE=same_conversation
export ROY_LHTB_POLICY_COMMAND="${python_bin} -m roy_research.lhtb_policy_server"
export ROY_LHTB_SEMANTIC_COMMAND="${python_bin} -m roy_research.semantic_server"
export ROY_LHTB_SEMANTIC_ROOT="${run_root}/semantic-dev"
export ROY_LHTB_MCTS_ENABLED=false
export ROY_LHTB_ORGANIZATION_INTERVAL=1
export DEEPSEEK_MODEL_REVISION="${DEEPSEEK_MODEL_REVISION:-deepseek-v4-flash-api-alias}"

has_dev_record() {
  local epoch="$1" task_id="$2"
  [[ -f "${trajectories}" ]] && "${python_bin}" - "${trajectories}" "${epoch}" "${task_id}" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
matches = [row for row in rows if int(row.get("epoch", -1)) == int(sys.argv[2])
           and row.get("task_id") == sys.argv[3]]
raise SystemExit(0 if len(matches) == 1 and matches[0].get("complete") is True else 1)
PY
}

for epoch in 0 1 2 3; do
  checkpoint="${run_root}/checkpoints/epoch-${epoch}-latest.pt"
  [[ -f "${checkpoint}" ]] || { echo "missing frozen epoch checkpoint ${checkpoint}" >&2; exit 5; }
  export ROY_LHTB_MODEL="${checkpoint}"
  revision="$("${python_bin}" - "${checkpoint}" <<'PY'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["metadata"]["actor_revision"])
PY
)"
  while IFS=$'\t' read -r task_id category; do
    if has_dev_record "${epoch}" "${task_id}"; then
      continue
    fi
    group_id="dev:${epoch}:${task_id}"
    task_tree="$(git -C "${lhtb_root}" rev-parse "HEAD:tasks/${task_id}")"
    initial_fingerprint="$(printf '%s' "${group_id}:${task_tree}:21600:32768:nodewise-actor" \
      | shasum -a 256 | awk '{print $1}')"
    seed="$((16#$(printf '%s' "${group_id}" | shasum -a 256 | cut -c1-8)))"
    job_dir="${run_root}/dev-jobs/${epoch}-${task_id}"
    config="${run_root}/dev-configs/${epoch}-${task_id}.json"
    imported="${job_dir}/imported.jsonl"
    mkdir -p "${job_dir}"
    "${python_bin}" -m roy_research lhtb-group-config \
      --output "${config}" --jobs-dir "${job_dir}" --task-id "${task_id}" \
      --arm learned_information_realization --initial-fingerprint "${initial_fingerprint}" \
      --organization-seed "${seed}" --attempts 1 --official-timeout \
      --environment-backend native --native-runtime-root "${native_runtime_root}" \
      --native-template-root "${native_template_root}" --allow-network-degraded \
      --max-retries "${ROY_LHTB_MAX_ENV_RETRIES:-2}" --concurrency 1
    (
      cd "${lhtb_root}"
      "${harbor_bin}" run -c "${config}" --yes
    ) >"${job_dir}/harbor.log" 2>&1
    digest="$("${python_bin}" -m roy_research lhtb-native-digest \
      --audit "${native_audit}" --task-id "${task_id}")"
    if [[ ! -f "${imported}" ]]; then
      "${python_bin}" -m roy_research lhtb-import-group --job-dir "${job_dir}" \
        --output "${imported}" --group-id "${group_id}" --task-id "${task_id}" \
        --category "${category}" --split dev --epoch "${epoch}" \
        --policy-revision "${revision}" --environment-digest "${digest}" \
        --environment-backend native --expected 1 --arm learned_information_realization
    fi
    "${python_bin}" - "${imported}" <<'PY' || {
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
raise SystemExit(0 if len(rows) == 1 and rows[0].get("complete") is True else 1)
PY
      echo "dev environment failure for ${group_id}; refusing checkpoint selection" >&2
      exit 5
    }
    "${python_bin}" - "${imported}" "${trajectories}" "${group_id}" <<'PY'
import json, pathlib, sys
source, destination, group_id = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
existing = set()
if destination.exists():
    existing = {str(json.loads(line).get("group_id"))
                for line in destination.read_text(encoding="utf-8").splitlines()
                if line.strip()}
if group_id not in existing:
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(source.read_text(encoding="utf-8"))
PY
  done < <("${python_bin}" - "${manifest}" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]:
    if row["split"] == "dev":
        print(row["task_id"], row["category"], sep="\t")
PY
  )
  if ! [[ -f "${metrics}" ]] || ! "${python_bin}" - "${metrics}" "${epoch}" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
raise SystemExit(0 if len([row for row in rows if int(row["epoch"]) == int(sys.argv[2])]) == 8 else 1)
PY
  then
    "${python_bin}" -m roy_research lhtb-dev-metrics \
      --trajectories "${trajectories}" --checkpoint "${checkpoint}" \
      --epoch "${epoch}" --output "${metrics}"
  fi
done

"${python_bin}" -m roy_research lhtb-select --metrics "${metrics}" \
  --output "${run_root}/selected-checkpoint.json"

export ROY_LHTB_ALLOW_NETWORK_DEGRADED=true
exec "${script_dir}/run_lhtb_test.sh"
