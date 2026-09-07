#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="${ROY_BENCHMARK_ROOT:-$HOME/rivermind-data/benchmarks}"
ACTION="${1:-prepare}"

sync_checkout() {
  local name="$1"
  local repository="$2"
  local revision="$3"
  local destination="$BENCHMARK_ROOT/$name"

  if [[ ! -d "$destination/.git" ]]; then
    if [[ "$ACTION" == "check" ]]; then
      echo "$name missing: $destination"
      return 1
    fi
    git clone "$repository" "$destination"
  fi
  if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
    echo "$name has tracked modifications; refusing to change its revision" >&2
    return 1
  fi
  if [[ "$ACTION" == "prepare" ]]; then
    git -C "$destination" fetch origin "$revision"
    git -C "$destination" checkout --detach "$revision"
  fi
  local actual
  actual="$(git -C "$destination" rev-parse HEAD)"
  if [[ "$actual" != "$revision" ]]; then
    echo "$name revision mismatch: $actual != $revision" >&2
    return 1
  fi
  echo "$name $actual"
}

mkdir -p "$BENCHMARK_ROOT"
sync_checkout DyLAN https://github.com/SALT-NLP/DyLAN.git 006e440a519f7cf21e2826f3b8033d84ae9bf07c
sync_checkout AutoAgents https://github.com/Link-AGI/AutoAgents.git 223ad991988d752c446d25a0381e647f6e71c92c
sync_checkout EvoAgent https://github.com/siyuyuan/evoagent.git fc6d087b119df69466c2372cfcaf588c040aaba8
