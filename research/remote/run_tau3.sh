#!/usr/bin/env bash
set -euo pipefail

roy_repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tau3_checkout="${TAU3_ROOT:-$(cd "${roy_repository_root}/.." && pwd)/benchmarks/tau3-bench-v1.0.1}"
roy_python="${roy_repository_root}/research/.venv/bin/python"
tau3_python="${tau3_checkout}/.venv/bin/python"

if [[ ! -x "${roy_python}" ]]; then
  echo "Roy research Python is missing: ${roy_python}" >&2
  exit 1
fi
if [[ ! -x "${tau3_python}" || ! -d "${tau3_checkout}/src/tau2" ]]; then
  echo "Pinned tau3 checkout is missing or incomplete: ${tau3_checkout}" >&2
  exit 1
fi

roy_site_packages="$(${roy_python} -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="${roy_repository_root}/research:${roy_site_packages}:${tau3_checkout}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${tau3_python}" -m roy_research "$@"
