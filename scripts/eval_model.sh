#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -gt 0 && "$1" != -* ]]; then
    export MODEL_FAMILY="$1"
    shift
fi
# shellcheck source=model_defaults.sh
source "${SCRIPT_DIR}/model_defaults.sh"

NUM_GPUS="${NUM_GPUS:-1}"
BASE_PORT="${BASE_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
NUM_WORKERS="${NUM_WORKERS:-$NUM_GPUS}"
NUM_SCORERS="${NUM_SCORERS:-$NUM_WORKERS}"
VIDEO_ROOT="${VIDEO_ROOT:-${PUBLIC_ROOT}/videos}"
OUTPUT_DIR="${OUTPUT_DIR:-${PUBLIC_ROOT}/runs}"
CONFIG_PATH="${CONFIG_PATH:-${PUBLIC_ROOT}/config.yaml}"
RUN_ID="${RUN_ID:-${MODEL_FAMILY}-$(date +%Y%m%d_%H%M%S)}"
MAX_SAMPLES=0
TIME_WINDOW=4.0
MODEL_VIDEO_FPS=2
TRIM_FPS=2
RESUME=0
SKIP_SCORING=0
DISABLE_LLM_JUDGE=0
FORCE_FOCUS=0
FOCUS_WINDOW_SECONDS=0
PROACTIVE_FOCUS=0
FULL_VIDEO_HIGH_FPS=0
ALLOW_EARLY_CORRECT=0

usage() {
    cat <<USAGE
Usage: bash scripts/eval_model.sh [model_family] [options]

Model families:
  qwen3_vl, aura, joyai_vl_interaction, moss_vl_realtime, videochat3_4b

Options:
  --model-path PATH              Model id or local checkpoint path
  --video-root PATH              Root directory containing benchmark videos
  --config PATH                  Evaluation YAML configuration
  --num-samples N                Run at most N samples (0 means all)
  --num-workers N                Concurrent inference workers
  --num-scorers N                Concurrent scoring workers
  --num-gpus N                   Number of local model servers
  --base-port N                  First local server port
  --run-id ID                    Stable run identifier for resuming
  --output-dir PATH              Directory for evaluation outputs
  --time-window SECONDS          Scoring response window (default: 4.0)
  --model-video-fps FPS          Normal model sampling FPS (default: 2)
  --trim-fps FPS                 Input FPS cap (default: 2)
  --force-focus                  Preserve source FPS near timestamp_focus
  --focus-window-seconds T       Focus window length
  --proactive-focus              Let model focus tokens control sampling
  --full-video-high-fps          Use high FPS for every video chunk
  --allow-early-correct          Score early correct responses normally
  --resume                       Resume an existing run
  --skip-scoring                 Run inference without a judge API
  --disable-llm-judge            Disable LLM judging in the evaluator
  -h, --help                     Show this help

The judge endpoint and key are supplied with STREAMEVAL_JUDGER_* environment variables.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="${2:?--model-path requires a value}"; shift 2 ;;
        --video-root) VIDEO_ROOT="${2:?--video-root requires a value}"; shift 2 ;;
        --config) CONFIG_PATH="${2:?--config requires a value}"; shift 2 ;;
        --num-samples) MAX_SAMPLES="${2:?--num-samples requires a value}"; shift 2 ;;
        --num-workers) NUM_WORKERS="${2:?--num-workers requires a value}"; shift 2 ;;
        --num-scorers) NUM_SCORERS="${2:?--num-scorers requires a value}"; shift 2 ;;
        --num-gpus) NUM_GPUS="${2:?--num-gpus requires a value}"; shift 2 ;;
        --base-port) BASE_PORT="${2:?--base-port requires a value}"; shift 2 ;;
        --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
        --time-window) TIME_WINDOW="${2:?--time-window requires a value}"; shift 2 ;;
        --model-video-fps) MODEL_VIDEO_FPS="${2:?--model-video-fps requires a value}"; shift 2 ;;
        --trim-fps) TRIM_FPS="${2:?--trim-fps requires a value}"; shift 2 ;;
        --force-focus) FORCE_FOCUS=1; shift ;;
        --focus-window-seconds) FOCUS_WINDOW_SECONDS="${2:?--focus-window-seconds requires a value}"; shift 2 ;;
        --proactive-focus) PROACTIVE_FOCUS=1; shift ;;
        --full-video-high-fps) FULL_VIDEO_HIGH_FPS=1; shift ;;
        --allow-early-correct) ALLOW_EARLY_CORRECT=1; shift ;;
        --resume) RESUME=1; shift ;;
        --skip-scoring) SKIP_SCORING=1; shift ;;
        --disable-llm-judge) DISABLE_LLM_JUDGE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value_name in NUM_GPUS NUM_WORKERS NUM_SCORERS MAX_SAMPLES; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 && "$value_name" != MAX_SAMPLES ]]; then
        echo "[ERROR] $value_name must be a non-negative integer (positive for worker/GPU counts)." >&2
        exit 2
    fi
done

if ! [[ "$BASE_PORT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] BASE_PORT must be a non-negative integer." >&2
    exit 2
fi

api_bases=""
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    [[ -n "$api_bases" ]] && api_bases+=",";
    api_bases+="http://${HOST}:$((BASE_PORT + gpu))/v1"
done

# Set the model-specific config variable without embedding a checkpoint path in source.
export "STREAMEVAL_${MODEL_ENV_PREFIX}_MODEL=${MODEL_PATH}"

args=(
    "${PUBLIC_ROOT}/run_stream_eval_parallel.py"
    --config "$CONFIG_PATH"
    --model-name "$MODEL_FAMILY"
    --video-root "$VIDEO_ROOT"
    --output-dir "$OUTPUT_DIR"
    --run-id "$RUN_ID"
    --num-workers "$NUM_WORKERS"
    --num-scorers "$NUM_SCORERS"
    --api-bases "$api_bases"
    --model-video-fps "$MODEL_VIDEO_FPS"
    --trim-fps "$TRIM_FPS"
    --time-window "$TIME_WINDOW"
)

if [[ "$MAX_SAMPLES" -gt 0 ]]; then args+=(--max-samples "$MAX_SAMPLES"); fi
if [[ "$RESUME" -eq 1 ]]; then args+=(--resume); fi
if [[ "$SKIP_SCORING" -eq 1 ]]; then args+=(--skip-scoring); fi
if [[ "$DISABLE_LLM_JUDGE" -eq 1 ]]; then args+=(--disable-llm-judge); fi
if [[ "$FORCE_FOCUS" -eq 1 ]]; then args+=(--force-focus --focus-window-seconds "$FOCUS_WINDOW_SECONDS"); fi
if [[ "$PROACTIVE_FOCUS" -eq 1 ]]; then args+=(--proactive-focus); fi
if [[ "$FULL_VIDEO_HIGH_FPS" -eq 1 ]]; then args+=(--full-video-high-fps 1); fi
if [[ "$ALLOW_EARLY_CORRECT" -eq 1 ]]; then args+=(--allow-early-correct); fi
if [[ -n "$NATIVE_FLAG" ]]; then args+=("$NATIVE_FLAG"); fi

echo "[INFO] Evaluating $MODEL_FAMILY with $NUM_WORKERS worker(s)."
python "${args[@]}"
