#!/usr/bin/env bash
# Original AURA prompt with isolated AURAv2 acknowledgement placeholders.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-${PUBLIC_ROOT}/configs/aurav2.yaml}"
export VIDEO_ROOT="${VIDEO_ROOT:-${PUBLIC_ROOT}}"
exec bash "${SCRIPT_DIR}/eval_model.sh" aurav2 "$@"
