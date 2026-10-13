#!/usr/bin/env bash
# Check port ownership and the protocol of every newly launched MOSS process.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PUBLIC_ROOT"
export MODEL_FAMILY=moss_vl_realtime
export MODEL_TYPE=moss_vl
export NUM_GPUS="${NUM_GPUS:-1}"
export BASE_PORT="${BASE_PORT:-8000}"
export HOST="${HOST:-127.0.0.1}"
export LOG_DIR="${LOG_DIR:-${PUBLIC_ROOT}/logs/servers/moss_framewise}"
if [[ "${1:-}" == stop ]]; then
    exec bash "${SCRIPT_DIR}/serve_model.sh" stop
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash ${BASH_SOURCE[0]} [stop]" >&2
    exit 2
fi
bash "${SCRIPT_DIR}/serve_model.sh" stop
python "${SCRIPT_DIR}/moss_framewise_preflight.py" ports \
    --host "$HOST" --base-port "$BASE_PORT" --num-gpus "$NUM_GPUS" --wait-seconds 10
if ! bash "${SCRIPT_DIR}/serve_model.sh"; then
    bash "${SCRIPT_DIR}/serve_model.sh" stop
    exit 1
fi
if ! python "${SCRIPT_DIR}/moss_framewise_preflight.py" servers \
    --host "$HOST" --base-port "$BASE_PORT" --num-gpus "$NUM_GPUS" \
    --pid-file "${LOG_DIR}/server_pids_moss_vl_realtime.txt"; then
    bash "${SCRIPT_DIR}/serve_model.sh" stop
    exit 1
fi
echo "[OK] All MOSS framewise endpoints match newly launched processes."
