#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PUBLIC_ROOT"
export MODEL_FAMILY=aurav2
exec bash "${SCRIPT_DIR}/serve_model.sh" "$@"
