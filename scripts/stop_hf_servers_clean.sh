#!/usr/bin/env bash
# Stop every listener and loading HF server in the selected port range.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/hf_server_control.py" stop \
    --base-port "${BASE_PORT:-8000}" --num-gpus "${NUM_GPUS:-1}" \
    --timeout "${STOP_TIMEOUT_SECONDS:-20}" "$@"
