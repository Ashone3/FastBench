#!/usr/bin/env bash
# Release selected ports, start a fresh deployment, verify its loaded weights.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PUBLIC_ROOT"
case "${1:-}" in
    moss) DEPLOY_SCRIPT=deploy_moss_vl_realtime_framewise.sh; MODEL_FAMILY=moss_vl_realtime ;;
    aura) DEPLOY_SCRIPT=deploy_aura.sh; MODEL_FAMILY=aura ;;
    aurav2) DEPLOY_SCRIPT=deploy_aurav2.sh; MODEL_FAMILY=aurav2 ;;
    joyai) DEPLOY_SCRIPT=deploy_joyai_vl_interaction.sh; MODEL_FAMILY=joyai_vl_interaction ;;
    *) echo "Usage: bash ${BASH_SOURCE[0]} {moss|aura|aurav2|joyai}" >&2; exit 2 ;;
esac
[[ $# -eq 1 ]] || { echo 'Expected one model name' >&2; exit 2; }
export NUM_GPUS="${NUM_GPUS:-1}"
export BASE_PORT="${BASE_PORT:-8000}"
export HOST="${HOST:-127.0.0.1}"
export HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-600}"
bash "${SCRIPT_DIR}/stop_hf_servers_clean.sh"
mkdir -p "${PUBLIC_ROOT}/logs/servers"
export LOG_DIR
LOG_DIR="$(mktemp -d "${PUBLIC_ROOT}/logs/servers/clean_${1}_XXXXXXXX")"
echo "[INFO] Deployment logs: ${LOG_DIR}"
cleanup_failed_launch() {
    echo "[ERROR] Deployment/identity verification failed. Cleaning selected ports." >&2
    bash "${SCRIPT_DIR}/stop_hf_servers_clean.sh" || true
}
trap cleanup_failed_launch ERR
bash "${SCRIPT_DIR}/${DEPLOY_SCRIPT}"
python "${SCRIPT_DIR}/hf_server_control.py" verify \
    --host "$HOST" --base-port "$BASE_PORT" --num-gpus "$NUM_GPUS" \
    --pid-file "${LOG_DIR}/server_pids_${MODEL_FAMILY}.txt" --timeout "$HEALTH_TIMEOUT_SECONDS"
trap - ERR
echo "[OK] All servers are newly launched and checkpoint paths verified."
