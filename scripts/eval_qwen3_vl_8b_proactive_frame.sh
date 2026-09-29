#!/usr/bin/env bash
# Run the Qwen3-VL-8B ProactiveFrame preset with its dedicated system prompt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Preserve the research preset; concurrency and paths remain configurable.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_SCORERS="${NUM_SCORERS:-20}"
BASE_PORT="${BASE_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
VIDEO_ROOT="${VIDEO_ROOT:-${VIDEO_ROOT_OVERRIDE:-${PUBLIC_ROOT}}}"
CONFIG_PATH="${CONFIG_PATH:-${PUBLIC_ROOT}/configs/proactive_frame.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PUBLIC_ROOT}/runs}"
RUN_ID="${RUN_ID:-qwen3-vl-8b-proactive-frame-$(date +%Y%m%d_%H%M%S)}"
MAX_SAMPLES=0
TIME_WINDOW_SECONDS=4.0
ACTIVE_WINDOW=4
FOCUS_WINDOW_SECONDS=3
RESUME=0
FORCE_FOCUS=0
ALLOW_EARLY_CORRECT=0
SKIP_SCORING=0
DISABLE_LLM_JUDGE=0
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: bash scripts/eval_qwen3_vl_8b_proactive_frame.sh [options]

Run ProactiveFrame using Qwen3-VL-8B servers started separately with
scripts/deploy_qwen3_vl_8b.sh. The dedicated configs/proactive_frame.yaml
contains the focus-control prompt and multilingual placeholder responses.

Preset: 2 FPS base sampling, model-controlled focus, 90 focus frames,
248832000 source pixels, time compression, and low-FPS history degradation.
The original --proactive-focus flag and Focus_Start/Focus_End tokens are kept.

Options:
  --num-samples N             Maximum video samples (0 means all)
  --model-path PATH           Model id or local checkpoint path
  --video-root PATH           Root to prepend to annotation video_path values
  --config PATH               Override the dedicated ProactiveFrame YAML
  --output-dir PATH           Output directory (default: runs/)
  --run-id ID                 Run identifier; reuse the same ID with --resume
  --resume                    Resume inference under the selected run ID
  --num-gpus N                Local server count (default: 8)
  --num-workers N             Inference worker count (default: 8)
  --num-scorers N             Scoring worker count (default: 20)
  --base-port N               First local server port (default: 8000)
  --time-window-seconds T     Scoring response window (default: 4.0)
  --time-window T             Alias for --time-window-seconds
  --active-window N           Sparse generation window (default: 4)
  --force-focus               Also enable annotation-guided focus (ablation)
  --focus-window-seconds T    Annotation-guided focus window (default: 3)
  --allow-early-correct       Disable the original early-response penalty
  --skip-scoring              Run inference without calling a judge
  --disable-llm-judge         Disable LLM judging; not for paper score comparisons
  --dry-run                   Print the command without running inference/scoring
  -h, --help                  Show this help

Configure credentials with STREAMEVAL_JUDGER_API_KEY in the environment.
Defaults: OpenRouter, https://openrouter.ai/api/v1, qwen/qwen3-235b-a22b-2507.
Overrides such as --force-focus change the experimental setting.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-samples) MAX_SAMPLES="${2:?--num-samples requires a value}"; shift 2 ;;
        --model-path) MODEL_PATH="${2:?--model-path requires a value}"; shift 2 ;;
        --video-root) VIDEO_ROOT="${2:?--video-root requires a value}"; shift 2 ;;
        --config) CONFIG_PATH="${2:?--config requires a value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
        --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
        --num-gpus) NUM_GPUS="${2:?--num-gpus requires a value}"; shift 2 ;;
        --num-workers) NUM_WORKERS="${2:?--num-workers requires a value}"; shift 2 ;;
        --num-scorers) NUM_SCORERS="${2:?--num-scorers requires a value}"; shift 2 ;;
        --base-port) BASE_PORT="${2:?--base-port requires a value}"; shift 2 ;;
        --time-window|--time-window-seconds) TIME_WINDOW_SECONDS="${2:?$1 requires a value}"; shift 2 ;;
        --active-window) ACTIVE_WINDOW="${2:?--active-window requires a value}"; shift 2 ;;
        --focus-window-seconds) FOCUS_WINDOW_SECONDS="${2:?--focus-window-seconds requires a value}"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --force-focus) FORCE_FOCUS=1; shift ;;
        --allow-early-correct) ALLOW_EARLY_CORRECT=1; shift ;;
        --skip-scoring) SKIP_SCORING=1; shift ;;
        --disable-llm-judge) DISABLE_LLM_JUDGE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for name in NUM_GPUS NUM_WORKERS NUM_SCORERS; do
    if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] $name must be a positive integer." >&2
        exit 2
    fi
done
for name in MAX_SAMPLES ACTIVE_WINDOW; do
    if ! [[ "${!name}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] $name must be a non-negative integer." >&2
        exit 2
    fi
done
for name in TIME_WINDOW_SECONDS FOCUS_WINDOW_SECONDS; do
    if ! [[ "${!name}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "[ERROR] $name must be a non-negative number." >&2
        exit 2
    fi
done
if ! [[ "$BASE_PORT" =~ ^[1-9][0-9]*$ ]] || ((BASE_PORT + NUM_GPUS - 1 > 65535)); then
    echo "[ERROR] The server port range must be within 1..65535." >&2
    exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Configuration not found: $CONFIG_PATH" >&2
    exit 2
fi

export STREAMEVAL_QWEN3_VL_MODEL="$MODEL_PATH"
export STREAMEVAL_JUDGER_BACKEND="${STREAMEVAL_JUDGER_BACKEND:-openrouter}"
export STREAMEVAL_JUDGER_API_BASE="${STREAMEVAL_JUDGER_API_BASE:-https://openrouter.ai/api/v1}"
export STREAMEVAL_JUDGER_MODEL="${STREAMEVAL_JUDGER_MODEL:-qwen/qwen3-235b-a22b-2507}"
# The caller supplies the judge key; no credential fallback is embedded here.

api_bases=""
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    if [[ -n "$api_bases" ]]; then api_bases+=","; fi
    api_bases+="http://${HOST}:$((BASE_PORT + gpu))/v1"
done

args=(
    "${PUBLIC_ROOT}/run_stream_eval_parallel.py"
    --model-name qwen3_vl
    --config "$CONFIG_PATH"
    --video-root "$VIDEO_ROOT"
    --output-dir "$OUTPUT_DIR"
    --run-id "$RUN_ID"
    --num-workers "$NUM_WORKERS"
    --num-scorers "$NUM_SCORERS"
    --api-bases "$api_bases"
    --trim-fps 2
    --model-video-fps 2
    --proactive-focus
    --max-focus-context-frames 90
    --time-compress
    --pixel-budget 248832000
    --low-fps-degeneration
    --time-window "$TIME_WINDOW_SECONDS"
    --active-window "$ACTIVE_WINDOW"
)
if [[ "$MAX_SAMPLES" -gt 0 ]]; then args+=(--max-samples "$MAX_SAMPLES"); fi
if [[ "$RESUME" -eq 1 ]]; then args+=(--resume); fi
if [[ "$FORCE_FOCUS" -eq 1 ]]; then args+=(--force-focus --focus-window-seconds "$FOCUS_WINDOW_SECONDS"); fi
if [[ "$ALLOW_EARLY_CORRECT" -eq 1 ]]; then args+=(--allow-early-correct); fi
if [[ "$SKIP_SCORING" -eq 1 ]]; then args+=(--skip-scoring); fi
if [[ "$DISABLE_LLM_JUDGE" -eq 1 ]]; then args+=(--disable-llm-judge); fi

# Preserve optional cache/root overrides supported by the research run_eval.sh.
if [[ -n "${STREAM_ADDR_ROOT:-}" ]]; then args+=(--stream-addr-root "$STREAM_ADDR_ROOT"); fi
if [[ -n "${CHUNK_CACHE_ROOT:-}" ]]; then args+=(--chunk-cache-root "$CHUNK_CACHE_ROOT"); fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' python "${args[@]}"
    printf '\n'
    exit 0
fi
exec python "${args[@]}"
