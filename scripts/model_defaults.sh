#!/usr/bin/env bash
# Shared model-family defaults for the public launchers.

set -euo pipefail

PUBLIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_FAMILY="${MODEL_FAMILY:-qwen3_vl}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_TYPE="${MODEL_TYPE:-}"
NATIVE_FLAG=""
MODEL_ENV_PREFIX=""

case "$MODEL_FAMILY" in
    qwen3_vl)
        MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
        MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
        MODEL_ENV_PREFIX="QWEN3_VL"
        ;;
    aura)
        MODEL_PATH="${MODEL_PATH:-Aura}"
        MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
        MODEL_ENV_PREFIX="AURA"
        NATIVE_FLAG="--native-aura"
        ;;
    joyai_vl_interaction)
        MODEL_PATH="${MODEL_PATH:-JoyAI-VL-Interaction}"
        MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
        MODEL_ENV_PREFIX="JOYAI_VL_INTERACTION"
        NATIVE_FLAG="--native-joyai"
        ;;
    moss_vl_realtime)
        MODEL_PATH="${MODEL_PATH:-MOSS-VL-Realtime}"
        MODEL_TYPE="${MODEL_TYPE:-moss_vl}"
        MODEL_ENV_PREFIX="MOSS_VL_REALTIME"
        NATIVE_FLAG="--native-moss"
        ;;
    videochat3_4b)
        MODEL_PATH="${MODEL_PATH:-MCG-NJU/VideoChat3-4B}"
        MODEL_TYPE="${MODEL_TYPE:-videochat3}"
        MODEL_ENV_PREFIX="VIDEOCHAT3_4B"
        NATIVE_FLAG="--native-videochat3"
        ;;
    *)
        echo "[ERROR] Unsupported MODEL_FAMILY: $MODEL_FAMILY" >&2
        echo "        Use qwen3_vl, aura, joyai_vl_interaction, moss_vl_realtime, or videochat3_4b." >&2
        exit 2
        ;;
esac

export PUBLIC_ROOT MODEL_FAMILY MODEL_PATH MODEL_TYPE MODEL_ENV_PREFIX NATIVE_FLAG
