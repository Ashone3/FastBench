#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=model_defaults.sh
source "${SCRIPT_DIR}/model_defaults.sh"

NUM_GPUS="${NUM_GPUS:-1}"
BASE_PORT="${BASE_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-300}"
LOG_DIR="${LOG_DIR:-${PUBLIC_ROOT}/logs/servers}"
PID_FILE="${LOG_DIR}/server_pids_${MODEL_FAMILY}.txt"

usage() {
    cat <<'USAGE'
Usage: bash scripts/serve_model.sh [stop]

Environment variables:
  MODEL_FAMILY          Model family (default: qwen3_vl)
  MODEL_PATH            Hugging Face model id or local checkpoint path
  NUM_GPUS              Number of one-process-per-GPU servers (default: 1)
  BASE_PORT             First OpenAI-compatible server port (default: 8000)
  TORCH_DTYPE           auto, bf16, fp16, or fp32 (default: bf16)
  ATTN_IMPL             sdpa or flash_attention_2 (default: sdpa)
USAGE
}

stop_servers() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "[INFO] No PID file found at $PID_FILE."
        return
    fi
    while IFS= read -r pid; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" || true
            echo "[INFO] Stopped PID $pid."
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "${1:-}" == "stop" ]]; then
    stop_servers
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo "[ERROR] Unknown argument: $1" >&2
    usage >&2
    exit 2
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_GPUS must be a positive integer." >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
stop_servers 2>/dev/null || true
: > "$PID_FILE"

echo "[INFO] Model family: $MODEL_FAMILY"
echo "[INFO] Model path:   $MODEL_PATH"
echo "[INFO] GPUs:         $NUM_GPUS"
echo "[INFO] Ports:        $BASE_PORT-$((BASE_PORT + NUM_GPUS - 1))"

for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    port=$((BASE_PORT + gpu))
    log_file="${LOG_DIR}/${MODEL_FAMILY}_gpu${gpu}_port${port}.log"
    CUDA_VISIBLE_DEVICES="$gpu" python "${PUBLIC_ROOT}/hf_openai_server.py" \
        --model-path "$MODEL_PATH" \
        --model-type "$MODEL_TYPE" \
        --host "$HOST" \
        --port "$port" \
        --device-map auto \
        --torch-dtype "$TORCH_DTYPE" \
        --attn-implementation "$ATTN_IMPL" \
        > "$log_file" 2>&1 &
    echo $! >> "$PID_FILE"
done

echo "[INFO] Waiting for health checks ..."
mapfile -t pids < "$PID_FILE"
all_healthy=1
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    port=$((BASE_PORT + gpu))
    healthy=0
    for ((attempt = 1; attempt <= HEALTH_TIMEOUT_SECONDS; attempt++)); do
        pid="${pids[$gpu]:-}"
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[ERROR] GPU $gpu server exited. See ${LOG_DIR}/${MODEL_FAMILY}_gpu${gpu}_port${port}.log" >&2
            break
        fi
        if curl -fsS "http://${HOST}:${port}/health" 2>/dev/null | grep -q 'ok'; then
            echo "[INFO] GPU $gpu is healthy on port $port."
            healthy=1
            break
        fi
        sleep 1
    done
    if [[ "$healthy" -ne 1 ]]; then
        all_healthy=0
    fi
done

if [[ "$all_healthy" -ne 1 ]]; then
    echo "[ERROR] One or more servers failed health checks." >&2
    exit 1
fi

echo "[INFO] API bases:"
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    echo "  http://${HOST}:$((BASE_PORT + gpu))/v1"
done
