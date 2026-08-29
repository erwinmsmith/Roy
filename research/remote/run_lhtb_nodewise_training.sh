#!/usr/bin/env bash
set -euo pipefail

# Formal Roy x LHTB node-wise Delta-V training.
#
# One scheduled group is eight independent one-macro-action successors cloned
# from the exact same node checkpoint.  The successor used to continue a task
# into the next epoch is committed by hash before outcomes are observed; task
# utility, Delta-V and topology are never used to pick it.

roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
lhtb_root="${LHTB_ROOT:-${HOME}/rivermind-data/benchmarks/LHTB}"
python_bin="${roy_root}/research/.venv/bin/python"
harbor_bin="${roy_root}/research/.venv/bin/harbor"
run_root="${ROY_LHTB_RUN_ROOT:-${roy_root}/research/output/lhtb/native/formal-nodewise}"
manifest="${ROY_LHTB_MANIFEST:-${roy_root}/research/config/lhtb_split.json}"
schedule="${run_root}/schedule.json"
model="${run_root}/checkpoints/current.pt"
native_runtime_root="${ROY_LHTB_NATIVE_ROOT:-${HOME}/rivermind-data/lhtb-native/runtime}"
native_template_root="${ROY_LHTB_NATIVE_TEMPLATE_ROOT:-${HOME}/rivermind-data/lhtb-native/templates}"
native_audit="${ROY_LHTB_NATIVE_AUDIT:-${roy_root}/research/output/lhtb/native/audit.json}"
dataset_path="${ROY_LHTB_DATASET_PATH:-${roy_root}/research/output/lhtb/native/oracle-overlays/roy-native-oracle-suite-20260824T045107Z/tasks}"
parallelism="${ROY_LHTB_NODEWISE_CONCURRENCY:-4}"
max_retries="${ROY_LHTB_MAX_ENV_RETRIES:-0}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=load_roy_env.sh
source "${script_dir}/load_roy_env.sh"
load_deepseek_api_key "${roy_root}"

[[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY is required" >&2; exit 4; }
[[ -x "${python_bin}" && -x "${harbor_bin}" ]] || {
  echo "the prepared Roy research virtual environment is required" >&2
  exit 4
}
[[ -f "${native_audit}" && -d "${dataset_path}" ]] || {
  echo "the reviewed native audit and frozen-finalize task overlay are required" >&2
  exit 4
}
[[ "$(git -C "${lhtb_root}" rev-parse HEAD)" == \
  "84d7ba5ee34fae6c11f0d7cb8ed5faa73a9ece54" ]] || {
  echo "LHTB checkout is not at the pinned commit" >&2
  exit 4
}
[[ "${parallelism}" =~ ^[1-9][0-9]*$ && "${parallelism}" -le 8 ]] || {
  echo "ROY_LHTB_NODEWISE_CONCURRENCY must be between 1 and 8" >&2
  exit 4
}

mkdir -p "${run_root}/checkpoints" "${run_root}/groups" "${run_root}/task-state"
export PYTHONPATH="${roy_root}/research${PYTHONPATH:+:${PYTHONPATH}}"
export ROY_LHTB_ENVIRONMENT_BACKEND=native
export ROY_LHTB_NODE_COMMAND="node ${roy_root}/dist/cli/LhtbAgent.js"
export ROY_LHTB_POLICY_COMMAND="${python_bin} -m roy_research.lhtb_policy_server"
export ROY_LHTB_SEMANTIC_COMMAND="${python_bin} -m roy_research.semantic_server"
export ROY_LHTB_SEMANTIC_ROOT="${run_root}/semantic"
export ROY_LHTB_MODEL="${model}"
export ROY_LHTB_MCTS_ENABLED=false
export ROY_LHTB_ORGANIZATION_INTERVAL=1
export DEEPSEEK_MODEL_REVISION="${DEEPSEEK_MODEL_REVISION:-deepseek-v4-flash-api-alias}"
export ROY_LHTB_TORCH_THREADS="${ROY_LHTB_TORCH_THREADS:-4}"
export ROY_LHTB_TORCH_INTEROP_THREADS="${ROY_LHTB_TORCH_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${ROY_LHTB_TORCH_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${ROY_LHTB_TORCH_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${ROY_LHTB_TORCH_THREADS}}"

cd "${roy_root}"
npm run build
"${python_bin}" -m roy_research lhtb-native-preflight \
  --runtime-root "${native_runtime_root}" --output "${run_root}/preflight.json"
if [[ ! -f "${model}" ]]; then
  "${python_bin}" -m roy_research lhtb-nodewise-init \
    --manifest "${manifest}" --model "${model}"
fi
if [[ ! -f "${schedule}" ]]; then
  "${python_bin}" -m roy_research lhtb-schedule \
    --manifest "${manifest}" --output "${schedule}"
fi

model_revisions() {
  "${python_bin}" - "${model}" <<'PY'
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["metadata"]
print(int(m["actor_revision"]), int(m["value_revision"]))
PY
}

environment_digest() {
  "${python_bin}" -m roy_research lhtb-native-digest \
    --audit "${native_audit}" --task-id "$1"
}

state_fingerprint() {
  "${python_bin}" - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["fingerprint"])
PY
}

write_config() {
  local output_dir="$1" task_id="$2" seed="$3" macro_steps="$4"
  local initial_fingerprint="$5" source_run="${6:-}"
  local source_args=()
  mkdir -p "${output_dir}/artifacts" "${output_dir}/jobs"
  if [[ -n "${source_run}" ]]; then
    source_args=(
      --nodewise-source-snapshot "${source_run}/artifacts/session.json"
      --nodewise-source-checkpoint "${source_run}/artifacts/environment-checkpoint"
    )
  fi
  "${python_bin}" -m roy_research lhtb-group-config \
    --output "${output_dir}/config.json" --jobs-dir "${output_dir}/jobs" \
    --task-id "${task_id}" --arm nodewise_checkpoint_finalize \
    --initial-fingerprint "${initial_fingerprint}" --organization-seed "${seed}" \
    --attempts 1 --environment-backend native \
    --native-runtime-root "${native_runtime_root}" \
    --native-template-root "${native_template_root}" --allow-network-degraded \
    --max-retries "${max_retries}" --concurrency 1 --dataset-path "${dataset_path}" \
    --nodewise-macro-steps "${macro_steps}" "${source_args[@]}" \
    --nodewise-output-snapshot "${output_dir}/artifacts/session.json" \
    --nodewise-output-state "${output_dir}/artifacts/state.json" \
    --nodewise-output-checkpoint "${output_dir}/artifacts/environment-checkpoint"
}

run_config() {
  local output_dir="$1"
  if [[ -f "${output_dir}/artifacts/state.json" && \
        -f "${output_dir}/artifacts/session.json" && \
        -f "${output_dir}/artifacts/environment-checkpoint/checkpoint.json" && \
        -n "$(find "${output_dir}" -type f -name result.json -print -quit)" ]]; then
    return
  fi
  (
    cd "${lhtb_root}"
    "${harbor_bin}" run -c "${output_dir}/config.json" --yes
  ) >"${output_dir}/harbor.log" 2>&1
}

run_successors() {
  local samples_root="$1"
  local running=0 failures=0 pid
  local -a pids=()
  shift
  for output_dir in "$@"; do
    run_config "${output_dir}" &
    pids+=("$!")
    running=$((running + 1))
    if [[ "${running}" -ge "${parallelism}" ]]; then
      for pid in "${pids[@]}"; do wait "${pid}" || failures=$((failures + 1)); done
      pids=()
      running=0
    fi
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failures=$((failures + 1)); done
  [[ "${failures}" -eq 0 ]] || {
    echo "${failures} node-wise Harbor successor processes failed under ${samples_root}" >&2
    return 5
  }
}

filter_fresh_labels() {
  local source="$1" output="$2"
  "${python_bin}" - "${source}" "${output}" "${model}" \
    "${run_root}/value-labels.jsonl" <<'PY'
import json, pathlib, sys, torch
source, output, model, ledger = map(pathlib.Path, sys.argv[1:])
used_ids = set(torch.load(model, map_location="cpu", weights_only=False)
               .get("metadata", {}).get("used_value_label_ids", []))
seen_fingerprints = set()
if ledger.exists():
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seen_fingerprints.add(str(json.loads(line)["state_fingerprint"]))
fresh = []
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    fingerprint = str(row["state_fingerprint"])
    if str(row["label_id"]) in used_ids or fingerprint in seen_fingerprints:
        continue
    fresh.append(row)
    seen_fingerprints.add(fingerprint)
output.write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in fresh), encoding="utf-8")
print(len(fresh))
PY
}

append_unique_jsonl() {
  local source="$1" destination="$2" key="$3"
  "${python_bin}" - "${source}" "${destination}" "${key}" <<'PY'
import json, pathlib, sys
source, destination, key = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
existing = set()
if destination.exists():
    existing = {str(json.loads(x).get(key)) for x in destination.read_text(encoding="utf-8").splitlines() if x.strip()}
rows = [json.loads(x) for x in source.read_text(encoding="utf-8").splitlines() if x.strip()]
with destination.open("a", encoding="utf-8") as stream:
    for row in rows:
        if str(row.get(key)) not in existing:
            stream.write(json.dumps(row, sort_keys=True)+"\n")
PY
}

select_continuation() {
  local group_id="$1" samples_root="$2" output="$3"
  "${python_bin}" - "${group_id}" "${samples_root}" "${output}" <<'PY'
import hashlib, json, pathlib, sys
group_id, root, output = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
committed = int(hashlib.sha256((group_id+":continuation").encode()).hexdigest()[:8], 16) % 8
chosen = None
for offset in range(8):
    index = (committed + offset) % 8
    state = json.load(open(root/f"sample-{index}"/"artifacts"/"state.json", encoding="utf-8"))
    active = list(state.get("activeSubtree") or [])
    statuses = {str(x.get("id")): str(x.get("status")) for x in state.get("nodes", [])}
    actionable = [node for node in active if statuses.get(str(node)) not in {"returned", "pruned", "failed"}]
    if actionable:
        chosen = (index, state, actionable, offset)
        break
if chosen is None:
    payload = {"group_id": group_id, "committed_index": committed,
               "selected_index": None, "reset_to_root": True,
               "selection_uses_outcomes": False}
else:
    index, state, actionable, offset = chosen
    payload = {"group_id": group_id, "committed_index": committed,
               "selected_index": index, "selected_run": str((root/f"sample-{index}").resolve()),
               "selected_state_fingerprint": state["fingerprint"],
               "actionable_nodes": actionable, "feasibility_offset": offset,
               "reset_to_root": False, "selection_uses_outcomes": False,
               "selection_uses_utility_or_value": False}
output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
}

write_group_summary() {
  local group_dir="$1" actor_before="$2" value_before="$3"
  "${python_bin}" - "${group_dir}" "${model}" "${actor_before}" \
    "${value_before}" <<'PY'
import json, pathlib, statistics, sys, torch
root, model = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
records = [json.loads(x) for x in (root/"groups.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
meta = torch.load(model, map_location="cpu", weights_only=False)["metadata"]
group_id = records[0]["group_id"]
actor_update = next((x for x in reversed(meta.get("history", [])) if x.get("group_id") == group_id), None)
def topology(state):
    nodes = list(state.get("nodes") or [])
    edges = list(state.get("dagEdges") or [])
    deps = list(state.get("dependencies") or [])
    comm = [x for x in edges if str(x.get("kind", x.get("type", ""))).lower() in {"communication", "communicates", "shortcut"}]
    return {"nodes": len(nodes), "max_depth": max([int(x.get("depth", 0)) for x in nodes] or [0]),
            "dag_edges": len(edges), "dependencies": len(deps), "communication_edges": len(comm)}
utilities = [float(x["environment_utility"]) for x in records]
payload = {
  "group_id": group_id, "task_id": records[0]["task_id"], "epoch": records[0]["epoch"],
  "base_state_fingerprint": records[0]["base_state_fingerprint"],
  "context_node_id": records[0]["context_node_id"],
  "actions": [x["selected_action"] for x in records],
  "environment_utilities": utilities,
  "environment_utility_mean": statistics.fmean(utilities),
  "environment_utility_std": statistics.pstdev(utilities),
  "successor_topologies": [topology(x["successor_state"]) for x in records],
  "actor_revision_before": int(sys.argv[3]), "value_revision_before": int(sys.argv[4]),
  "actor_revision_after": int(meta["actor_revision"]), "value_revision_after": int(meta["value_revision"]),
  "actor_update": actor_update,
  "sampling_protocol": "same_state_direct_macro_action_no_mcts",
  "environment_utility_used_by_actor": False,
}
(root/"group-summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
print(json.dumps({k: payload[k] for k in ("group_id", "actions", "environment_utilities", "actor_revision_after", "value_revision_after")}, sort_keys=True))
PY
}

# Expand the checked 120-group schedule to a simple tab-separated stream.
"${python_bin}" - "${schedule}" "${manifest}" <<'PY' >"${run_root}/schedule.tsv"
import json, sys
schedule = json.load(open(sys.argv[1], encoding="utf-8"))["groups"]
manifest = {x["task_id"]: x for x in json.load(open(sys.argv[2], encoding="utf-8"))["tasks"]}
for group in schedule:
    print(group["epoch"], group["task_id"], manifest[group["task_id"]]["category"],
          group["group_id"], ",".join(map(str, group["organization_seeds"])), sep="\t")
PY

while IFS=$'\t' read -r epoch task_id category group_id seed_csv; do
  group_dir="${run_root}/groups/${epoch}-${task_id}"
  if [[ -f "${group_dir}/complete.json" ]]; then
    echo "skip completed ${group_id}"
    continue
  fi
  mkdir -p "${group_dir}/samples"
  read -r actor_revision value_revision < <(model_revisions)
  digest="$(environment_digest "${task_id}")"
  continuation_file="${run_root}/task-state/${task_id}.json"
  source_run=""
  if [[ -f "${continuation_file}" ]]; then
    source_run="$("${python_bin}" - "${continuation_file}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("selected_run", ""))
PY
)"
  fi

  if [[ -n "${source_run}" ]]; then
    base_run="${source_run}"
  else
    base_run="${group_dir}/base"
    if [[ ! -f "${base_run}/config.json" ]]; then
      root_seed="$((16#$(printf '%s' "${group_id}:base" | shasum -a 256 | cut -c1-8)))"
      root_fingerprint="$(printf '%s' "${task_id}:formal-nodewise-root" | shasum -a 256 | awk '{print $1}')"
      write_config "${base_run}" "${task_id}" "${root_seed}" 0 "${root_fingerprint}"
    fi
    run_config "${base_run}"
  fi
  base_fingerprint="$(state_fingerprint "${base_run}/artifacts/state.json")"

  IFS=',' read -r -a seeds <<<"${seed_csv}"
  sample_dirs=()
  for index in "${!seeds[@]}"; do
    sample_dir="${group_dir}/samples/sample-${index}"
    sample_dirs+=("${sample_dir}")
    if [[ ! -f "${sample_dir}/config.json" ]]; then
      write_config "${sample_dir}" "${task_id}" "${seeds[$index]}" 1 \
        "${base_fingerprint}" "${base_run}"
    fi
  done
  run_successors "${group_dir}/samples" "${sample_dirs[@]}"

  if [[ ! -f "${group_dir}/groups.jsonl" ]]; then
    "${python_bin}" -m roy_research lhtb-nodewise-import \
      --base-run "${base_run}" --samples-root "${group_dir}/samples" \
      --labels-output "${group_dir}/labels.jsonl" \
      --groups-output "${group_dir}/groups.jsonl" --group-id "${group_id}" \
      --task-id "${task_id}" --split train --epoch "${epoch}" \
      --policy-revision "${actor_revision}" --value-revision "${value_revision}" \
      --environment-digest "${digest}" --expected 8
  fi

  # Revision zero is a deliberate value-only bootstrap.  Every later group
  # updates the actor before fitting V on its own newly observed labels.
  if [[ "${value_revision}" -gt 0 ]]; then
    "${python_bin}" -m roy_research lhtb-node-update \
      --manifest "${manifest}" --groups "${group_dir}/groups.jsonl" \
      --model "${model}" --updates "${run_root}/actor-updates.jsonl" --resume
  fi

  fresh_count="$(filter_fresh_labels "${group_dir}/labels.jsonl" "${group_dir}/fresh-labels.jsonl")"
  if [[ "${fresh_count}" -gt 0 ]]; then
    "${python_bin}" -m roy_research lhtb-value-update \
      --manifest "${manifest}" --labels "${group_dir}/fresh-labels.jsonl" \
      --model "${model}" --updates "${run_root}/value-updates.jsonl" \
      --epochs "${ROY_LHTB_VALUE_EPOCHS_PER_GROUP:-4}" \
      --batch-size "${ROY_LHTB_VALUE_BATCH_SIZE:-32}" --resume
    append_unique_jsonl "${group_dir}/fresh-labels.jsonl" \
      "${run_root}/value-labels.jsonl" label_id
  fi
  append_unique_jsonl "${group_dir}/groups.jsonl" \
    "${run_root}/macro-groups.jsonl" group_id
  select_continuation "${group_id}" "${group_dir}/samples" \
    "${group_dir}/continuation.json" >"${group_dir}/continuation.log"
  cp "${group_dir}/continuation.json" "${continuation_file}"
  write_group_summary "${group_dir}" "${actor_revision}" "${value_revision}"
  "${python_bin}" - "${group_dir}/complete.json" "${group_id}" "${model}" <<'PY'
import datetime, json, pathlib, sys, torch
meta = torch.load(sys.argv[3], map_location="cpu", weights_only=False)["metadata"]
payload = {"group_id": sys.argv[2], "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "actor_revision": meta["actor_revision"], "value_revision": meta["value_revision"],
           "algorithm_revision": meta["algorithm_revision"]}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
PY
  cp "${model}" "${run_root}/checkpoints/epoch-${epoch}-latest.pt"
done <"${run_root}/schedule.tsv"

cp "${model}" "${run_root}/checkpoints/final.pt"
echo "formal node-wise LHTB training complete: ${model}"
