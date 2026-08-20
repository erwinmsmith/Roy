#!/usr/bin/env bash

# Load only the DeepSeek credential needed by the remote runners. This avoids
# executing arbitrary shell content from .env while still supporting the
# repository's normal local configuration file.
load_deepseek_api_key() {
  local roy_root="$1"
  local env_file="${ROY_ENV_FILE:-${roy_root}/.env}"
  local value

  [[ -z "${DEEPSEEK_API_KEY:-}" && -f "${env_file}" ]] || return 0
  value="$(python3 - "${env_file}" <<'PY'
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, raw_value = line.split("=", 1)
    if key.strip() != "DEEPSEEK_API_KEY":
        continue
    candidate = raw_value.strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
        try:
            candidate = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            pass
    print(candidate, end="")
    break
PY
)"
  if [[ -n "${value}" ]]; then
    export DEEPSEEK_API_KEY="${value}"
  fi
}
