#!/usr/bin/env bash
set -euo pipefail

roy_root="${ROY_ROOT:-${HOME}/rivermind-data/roy}"
execution_base="${ROY_LHTB_NATIVE_EXECUTION_BASE:-/tmp/Dev_4/roy-lhtb-native}"
export ROY_LHTB_ENVIRONMENT_BACKEND=native
export ROY_LHTB_NATIVE_ROOT="${ROY_LHTB_NATIVE_ROOT:-${execution_base}/runtime}"
export ROY_LHTB_NATIVE_TEMPLATE_ROOT="${ROY_LHTB_NATIVE_TEMPLATE_ROOT:-${execution_base}/templates}"
export ROY_LHTB_NATIVE_AUDIT="${ROY_LHTB_NATIVE_AUDIT:-${roy_root}/research/output/lhtb/native/audit.json}"
export ROY_LHTB_RUN_ROOT="${ROY_LHTB_RUN_ROOT:-${roy_root}/research/output/lhtb/native/formal}"

exec "${roy_root}/research/remote/run_lhtb_training.sh"
