#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_FAMILY=aura
exec bash "${SCRIPT_DIR}/serve_model.sh" "$@"
