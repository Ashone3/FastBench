#!/usr/bin/env bash
# Optional native frame gates; keep the existing benchmark clock and scoring.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export EXTRA_ARGS="${EXTRA_ARGS:-} --native-moss-framewise"
export VIDEO_ROOT="${VIDEO_ROOT:-${PUBLIC_ROOT}}"
export RUN_ID="${RUN_ID:-moss-framewise-$(date +%Y%m%d_%H%M%S)}"
exec bash "${SCRIPT_DIR}/eval_model.sh" moss_vl_realtime "$@"
