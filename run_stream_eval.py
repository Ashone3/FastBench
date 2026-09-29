#!/usr/bin/env python3
"""
Self-contained StreamEval runner.

Design goals:
- Keep old inference/scoring semantics.
- Run from a standalone package (no imports from repo root modules).
- Support both old and new benchmark formats.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from tqdm import tqdm

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover
    repair_json = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"
PLACEHOLDERS_LOWER: set[str] = set()
FORWARD_TASK_TYPES: set[str] = {"forward", "future", "proactive"}

# qwen-vl-utils first resizes video frames on a 28px grid, then the Qwen3-VL
# processor uses a 32px grid (16px patches followed by 2x2 spatial merging).
QWEN_VL_UTILS_SPATIAL_FACTOR = 28
QWEN_VL_UTILS_VIDEO_MIN_PIXELS = 128 * QWEN_VL_UTILS_SPATIAL_FACTOR**2
QWEN_VL_UTILS_VIDEO_MAX_PIXELS = 768 * QWEN_VL_UTILS_SPATIAL_FACTOR**2
QWEN_VL_UTILS_IMAGE_MIN_PIXELS = 4 * QWEN_VL_UTILS_SPATIAL_FACTOR**2
QWEN_VL_UTILS_IMAGE_MAX_PIXELS = 16384 * QWEN_VL_UTILS_SPATIAL_FACTOR**2
QWEN3_VL_SPATIAL_TOKEN_FACTOR = 32
QWEN3_VL_TEMPORAL_PATCH_SIZE = 2
QWEN3_VL_VIDEO_MIN_TOTAL_PIXELS = 4096
QWEN3_VL_VIDEO_MAX_TOTAL_PIXELS = 25165824
QWEN3_VL_BASELINE_FRAMES = 120
QWEN3_VL_BASELINE_HEIGHT = 1080
QWEN3_VL_BASELINE_WIDTH = 1920
QWEN3_VL_DEFAULT_PIXEL_BUDGET = (
    QWEN3_VL_BASELINE_FRAMES
    * QWEN3_VL_BASELINE_HEIGHT
    * QWEN3_VL_BASELINE_WIDTH
)


def qwen3_vl_effective_frame_count(num_frames: int) -> int:
    """Match qwen-vl-utils, which floors sampled video frames to pairs."""
    frames = max(0, int(num_frames))
    return (frames // QWEN3_VL_TEMPORAL_PATCH_SIZE) * QWEN3_VL_TEMPORAL_PATCH_SIZE


def _smart_resize_dimensions(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int]:
    height = max(1, int(height))
    width = max(1, int(width))
    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = max(factor, math.ceil(height * beta / factor) * factor)
        resized_width = max(factor, math.ceil(width * beta / factor) * factor)
    return int(resized_height), int(resized_width)


def qwen3_vl_preprocess_dimensions(
    num_frames: int,
    height: int,
    width: int,
    scale: float = 1.0,
    scale_before_native_resize: bool = False,
) -> Tuple[int, int, int, int]:
    """Simulate qwen-vl-utils and Qwen3-VL spatial preprocessing."""
    safe_scale = max(0.0, min(1.0, float(scale)))
    if scale_before_native_resize:
        height = max(1, round(max(1, int(height)) * safe_scale))
        width = max(1, round(max(1, int(width)) * safe_scale))
    native_height, native_width = _smart_resize_dimensions(
        height,
        width,
        QWEN_VL_UTILS_SPATIAL_FACTOR,
        QWEN_VL_UTILS_VIDEO_MIN_PIXELS,
        QWEN_VL_UTILS_VIDEO_MAX_PIXELS,
    )
    utility_scale = 1.0 if scale_before_native_resize else safe_scale
    utility_height, utility_width = _smart_resize_dimensions(
        max(1, round(native_height * utility_scale)),
        max(1, round(native_width * utility_scale)),
        QWEN_VL_UTILS_SPATIAL_FACTOR,
        QWEN_VL_UTILS_IMAGE_MIN_PIXELS,
        QWEN_VL_UTILS_IMAGE_MAX_PIXELS,
    )

    factor = QWEN3_VL_SPATIAL_TOKEN_FACTOR
    processed_height = max(factor, round(utility_height / factor) * factor)
    processed_width = max(factor, round(utility_width / factor) * factor)
    frames = max(
        QWEN3_VL_TEMPORAL_PATCH_SIZE,
        qwen3_vl_effective_frame_count(num_frames),
    )
    temporal_frames = frames
    total_pixels = temporal_frames * processed_height * processed_width
    if total_pixels > QWEN3_VL_VIDEO_MAX_TOTAL_PIXELS:
        beta = math.sqrt(
            (frames * utility_height * utility_width)
            / QWEN3_VL_VIDEO_MAX_TOTAL_PIXELS
        )
        processed_height = max(
            factor,
            math.floor(utility_height / beta / factor) * factor,
        )
        processed_width = max(
            factor,
            math.floor(utility_width / beta / factor) * factor,
        )
    elif total_pixels < QWEN3_VL_VIDEO_MIN_TOTAL_PIXELS:
        beta = math.sqrt(
            QWEN3_VL_VIDEO_MIN_TOTAL_PIXELS
            / (frames * utility_height * utility_width)
        )
        processed_height = max(
            factor,
            math.ceil(utility_height * beta / factor) * factor,
        )
        processed_width = max(
            factor,
            math.ceil(utility_width * beta / factor) * factor,
        )
    return utility_height, utility_width, processed_height, processed_width


def qwen3_vl_visual_token_count(
    num_frames: int,
    height: int,
    width: int,
    scale: float = 1.0,
) -> int:
    """Estimate tokens after the native Qwen video preprocessing pipeline."""
    _, _, processed_height, processed_width = qwen3_vl_preprocess_dimensions(
        num_frames,
        height,
        width,
        scale,
    )
    temporal_tokens = (
        qwen3_vl_effective_frame_count(num_frames)
        // QWEN3_VL_TEMPORAL_PATCH_SIZE
    )
    return (
        temporal_tokens
        * (processed_height // QWEN3_VL_SPATIAL_TOKEN_FACTOR)
        * (processed_width // QWEN3_VL_SPATIAL_TOKEN_FACTOR)
    )


QWEN3_VL_DEFAULT_VISUAL_TOKEN_BUDGET = (
    QWEN3_VL_BASELINE_FRAMES // QWEN3_VL_TEMPORAL_PATCH_SIZE
) * qwen3_vl_visual_token_count(
    QWEN3_VL_TEMPORAL_PATCH_SIZE,
    QWEN3_VL_BASELINE_HEIGHT,
    QWEN3_VL_BASELINE_WIDTH,
)


ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::(.*))?\}$")


def resolve_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: resolve_env_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_refs(v) for v in value]
    if isinstance(value, str):
        match = ENV_PATTERN.match(value)
        if match:
            env_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.getenv(env_name, default)
    return value


def normalize_api_base(url: str, default_scheme: str = "https") -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        return text.rstrip("/")
    return f"{default_scheme}://{text}".rstrip("/")


def init_runtime_constants(config: Dict[str, Any]) -> None:
    global PLACEHOLDERS_LOWER, FORWARD_TASK_TYPES
    scoring_cfg = config.get("scoring", {}) if isinstance(config, dict) else {}

    placeholders = scoring_cfg.get("placeholder_responses", [])
    if not isinstance(placeholders, list):
        raise ValueError("config.scoring.placeholder_responses must be a list")
    PLACEHOLDERS_LOWER = {str(x).strip().lower() for x in placeholders}

    forward_types = scoring_cfg.get("forward_task_types", ["forward", "future", "proactive"])
    if not isinstance(forward_types, list):
        raise ValueError("config.scoring.forward_task_types must be a list")
    normalized = {str(x).strip().lower() for x in forward_types if str(x).strip()}
    FORWARD_TASK_TYPES = normalized or {"forward", "future", "proactive"}


def load_release_config(config_path: Optional[Path]) -> Tuple[Dict[str, Any], Path]:
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    cfg = resolve_env_refs(cfg)
    required_keys = ["default_benchmarks", "default_prompt", "benchmarks", "prompts", "models", "judger", "scoring"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return cfg, config_path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_placeholder(text: str, extra_placeholders: Optional[Iterable[str]] = None) -> bool:
    key = (text or "").strip().lower()
    if key in PLACEHOLDERS_LOWER:
        return True
    if not extra_placeholders:
        return False
    return key in {str(item).strip().lower() for item in extra_placeholders}


def time_to_seconds(t: str) -> int:
    parts = str(t).strip().split(":")
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        m, s = map(int, parts)
        return m * 60 + s
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    raise ValueError(f"Invalid time format: {t}")


def seconds_to_time(sec: int) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h <= 0:
        return f"{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def probe_video(video_path: str) -> Optional[Dict[str, Any]]:
    path = Path(video_path)
    if not path.exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_name,codec_type,profile,level,bit_rate,width,height,r_frame_rate,nb_frames,sample_rate,channels,channel_layout",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(out.stdout)
    except Exception:
        return None
    info = {
        "duration": float(data.get("format", {}).get("duration", 0.0)),
        "file_size": int(data.get("format", {}).get("size", path.stat().st_size)),
        "total_bitrate": int(data.get("format", {}).get("bit_rate", 0)),
        "video": None,
        "audio": None,
    }
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and info["video"] is None:
            fps = 0.0
            fps_raw = stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_raw.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 0.0
            except Exception:
                fps = 0.0
            info["video"] = {
                "index": stream.get("index", 0),
                "codec": stream.get("codec_name", ""),
                "profile": stream.get("profile", ""),
                "level": stream.get("level", 0),
                "width": int(stream.get("width", 0) or 0),
                "height": int(stream.get("height", 0) or 0),
                "fps": fps,
                "total_frames": int(stream.get("nb_frames", 0) or 0),
                "bitrate": int(stream.get("bit_rate", 0) or 0),
            }
        elif stream.get("codec_type") == "audio" and info["audio"] is None:
            info["audio"] = {
                "index": stream.get("index", 1),
                "codec": stream.get("codec_name", ""),
                "sample_rate": int(stream.get("sample_rate", 0) or 0),
                "channels": int(stream.get("channels", 0) or 0),
                "channel_layout": stream.get("channel_layout", ""),
                "bitrate": int(stream.get("bit_rate", 0) or 0),
            }
    return info


def build_time_interval_select_filter(target_fps: float) -> str:
    """Return the low-FPS selector used by the proactive-video builder."""
    if float(target_fps) <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    interval = 1.0 / float(target_fps)
    return (
        "select='if(isnan(prev_selected_t)\\,1\\,"
        f"gte(t-prev_selected_t\\,{interval:.12f}))'"
    )


def trim_video(
    video_path: str,
    trim_path: str,
    start_time: str,
    end_time: str,
    fps: Optional[float] = None,
) -> None:
    def _is_usable_chunk(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            if path.stat().st_size <= 0:
                return False
        except Exception:
            return False
        info = probe_video(str(path))
        if not isinstance(info, dict):
            return False
        return isinstance(info.get("video"), dict)

    out_path = Path(trim_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and _is_usable_chunk(out_path):
        return
    if out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass
    start_sec = max(0.0, float(time_to_seconds(start_time)))
    end_sec = max(start_sec + 1.0, float(time_to_seconds(end_time)))
    video_filters: List[str] = []
    if fps is not None:
        if float(fps) <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        # Run select on the full clip timeline before trimming. This preserves
        # the same global sampling phase as build_proactive_perception_videos.py
        # instead of restarting the sampler at every one-second chunk.
        video_filters.append(build_time_interval_select_filter(float(fps)))
    video_filters.extend(
        [
            f"trim=start={start_sec:.6f}:end={end_sec:.6f}",
            "setpts=PTS-STARTPTS",
        ]
    )
    audio_filter = (
        f"atrim=start={start_sec:.6f}:end={end_sec:.6f},"
        "asetpts=PTS-STARTPTS"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        ",".join(video_filters),
        "-af",
        audio_filter,
        "-vsync",
        "vfr",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-loglevel",
        "error",
    ]
    if fps is not None:
        # Selection has already happened above, so this only declares the
        # selected frames' cadence (including the final frame duration). Unlike
        # the old implementation it cannot create extra sampled frames.
        cmd.extend(["-r", str(float(fps))])
    cmd.append(str(out_path))
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        # Prevent reusing partial/corrupted outputs after ffmpeg failures.
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        raise


def merge_videos(clip_list: List[str], output_path: str) -> str:
    """Merge multiple video clips using ffmpeg concat demuxer (stream copy, no re-encoding)."""
    if not clip_list:
        raise ValueError("clip_list is empty")
    if len(clip_list) == 1:
        return clip_list[0]
    out = Path(output_path)
    if out.exists():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for clip in clip_list:
            f.write(f"file '{os.path.abspath(clip)}'\n")
        list_file = f.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                "-movflags", "+faststart",
                str(out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        os.unlink(list_file)
    return str(out)


def normalize_time_type_for_new_format(time_type: Any) -> str:
    value = str(time_type or "backward").strip().lower()
    if value in FORWARD_TASK_TYPES:
        return "forward"
    if value == "instant":
        return "instant"
    return "backward"


def normalize_task_type_preserving_legacy(task_type: Any, default: str = "DefaultType") -> str:
    if task_type is None:
        return default
    text = str(task_type).strip()
    if not text:
        return default
    if text.lower() == "forward":
        return "forward"
    return text


def is_forward_task(task_type: Any) -> bool:
    return str(task_type or "").strip().lower() in FORWARD_TASK_TYPES


def ensure_uuid(seed: str) -> str:
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]


def resolve_path(path_value: str, base_dirs: List[Path]) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    for base in base_dirs:
        maybe = base / candidate
        if maybe.exists():
            return maybe.resolve()
    return (base_dirs[0] / candidate).resolve()


def resolve_video_path(
    raw_video_path: str,
    benchmark_path: Path,
    video_root: Optional[Path],
) -> str:
    candidate = Path(raw_video_path)
    if candidate.is_absolute():
        return str(candidate)
    base_dirs = []
    if video_root is not None:
        base_dirs.append(video_root)
    base_dirs.extend([benchmark_path.parent, SCRIPT_DIR, Path.cwd()])
    return str(resolve_path(raw_video_path, base_dirs))


def convert_verified_responses_to_sqa(verified_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sqa: List[Dict[str, Any]] = []
    event_id = 1
    for qa in verified_responses:
        task_type = normalize_time_type_for_new_format(qa.get("time_type"))
        timestamp_question = qa.get("timestamp_question")
        question = qa.get("user_query", "")
        response = qa.get("response", "")
        capability = qa.get("capability")
        options = qa.get("options")

        if task_type == "forward":
            sqa.append(
                {
                    "event_id": event_id,
                    "timestamp": timestamp_question,
                    "type": task_type,
                    "question": question,
                    **({"capability": capability} if capability else {}),
                    **({"options": options} if options is not None else {}),
                }
            )
            event_id += 1
            sqa.append(
                {
                    "event_id": event_id,
                    "timestamp": qa.get("timestamp_proactive", timestamp_question),
                    "response": response,
                    **({"capability": capability} if capability else {}),
                }
            )
            event_id += 1
        else:
            sqa.append(
                {
                    "event_id": event_id,
                    "timestamp": timestamp_question,
                    "type": task_type,
                    "question": question,
                    "response": response,
                    **({"capability": capability} if capability else {}),
                    **({"options": options} if options is not None else {}),
                }
            )
            event_id += 1
    return sqa


def infer_duration(video_info: Optional[Dict[str, Any]], sqa: List[Dict[str, Any]]) -> float:
    if isinstance(video_info, dict) and video_info.get("duration"):
        return float(video_info["duration"])
    if sqa:
        timestamps = [time_to_seconds(item.get("timestamp", "00:00")) for item in sqa if item.get("timestamp")]
        if timestamps:
            return float(max(timestamps) + 10)
    return 0.0


def normalize_sample(
    sample: Dict[str, Any],
    idx: int,
    bench_name: str,
    benchmark_path: Path,
    video_root: Optional[Path],
    stream_addr_root: Optional[Path],
    need_video_info: bool,
) -> Dict[str, Any]:
    record = dict(sample)
    raw_video_path = record.get("video") or record.get("video_path", "")
    record["video"] = resolve_video_path(raw_video_path, benchmark_path, video_root) if raw_video_path else ""

    if "sqa" not in record and "verified_responses" in record:
        record["sqa"] = convert_verified_responses_to_sqa(record.get("verified_responses", []))
    else:
        record["sqa"] = record.get("sqa", [])

    verified = record.get("verified_responses")
    focus_timestamps = record.get("timestamp_focuses")
    if not isinstance(focus_timestamps, list):
        focus_timestamps = []
    else:
        focus_timestamps = list(focus_timestamps)
    if isinstance(verified, list):
        for item in verified:
            if not isinstance(item, dict):
                continue
            raw_focus = item.get("timestamp_focus")
            text = str(raw_focus).strip() if raw_focus is not None else ""
            if text and text not in focus_timestamps:
                focus_timestamps.append(text)
    if focus_timestamps:
        record["timestamp_focuses"] = focus_timestamps
        # Retain the legacy scalar field for callers that still consume it.
        if not str(record.get("timestamp_focus", "")).strip():
            record["timestamp_focus"] = focus_timestamps[0]

    if "id" not in record:
        record["id"] = idx + 1
    if "uuid" not in record:
        record["uuid"] = ensure_uuid(f"{bench_name}:{record['id']}:{record.get('video','')}")
    if "stream_addr" not in record:
        base = stream_addr_root if stream_addr_root is not None else (SCRIPT_DIR / "cache" / "stream_addr" / bench_name)
        record["stream_addr"] = str(base / record["uuid"])

    if need_video_info:
        current_video_info = record.get("video_info") if isinstance(record.get("video_info"), dict) else None
        if not current_video_info:
            probed = probe_video(record["video"]) if record.get("video") else None
            record["video_info"] = probed or {"duration": infer_duration(None, record["sqa"])}
        else:
            if not current_video_info.get("duration"):
                current_video_info["duration"] = infer_duration(current_video_info, record["sqa"])
            record["video_info"] = current_video_info
    else:
        record.setdefault("video_info", {"duration": infer_duration(None, record["sqa"])})

    record.setdefault("source", record.get("video_path", record.get("video", "")))
    return record


def load_samples_any_format(
    benchmark_path: Path,
    bench_name: str,
    video_root: Optional[Path],
    stream_addr_root: Optional[Path],
    need_video_info: bool,
) -> List[Dict[str, Any]]:
    if benchmark_path.suffix.lower() == ".jsonl":
        rows = load_jsonl(benchmark_path)
    else:
        raw = load_json(benchmark_path)
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict) and isinstance(raw.get("samples"), list):
            rows = raw["samples"]
        else:
            raise ValueError(f"Unsupported benchmark format in {benchmark_path}")

    normalized: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        normalized.append(
            normalize_sample(
                sample=row,
                idx=idx,
                bench_name=bench_name,
                benchmark_path=benchmark_path,
                video_root=video_root,
                stream_addr_root=stream_addr_root,
                need_video_info=need_video_info,
            )
        )
    return normalized


class OpenAICompatibleBackend:
    def __init__(self, name: str, cfg: Dict[str, Any]):
        self.name = name
        self.backend_type = str(cfg.get("backend", "openai_compatible")).strip().lower()
        self.api_base = normalize_api_base(str(cfg.get("api_base", "")))
        self.api_key = str(cfg.get("api_key", ""))
        self.model = str(cfg.get("model", ""))
        self.timeout = float(cfg.get("timeout", 120))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.temperature = float(cfg.get("temperature", 0.0))
        transport_raw = str(cfg.get("video_transport", "auto")).strip().lower()
        if transport_raw == "auto":
            if self.backend_type in {"openrouter", "zhizengzeng"}:
                self.video_transport = "image_frames_data_url"
            else:
                self.video_transport = "path"
        else:
            self.video_transport = transport_raw
        self.max_video_size_mb = float(cfg.get("max_video_size_mb", 30))
        self.video_frame_count = max(1, int(cfg.get("video_frame_count", 2)))
        self.video_frame_count_cap = max(1, int(cfg.get("video_frame_count_cap", 120)))
        self.video_frame_extract_fps = float(cfg.get("video_frame_extract_fps", 2.0))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1024))
        self.max_tokens_param = str(cfg.get("max_tokens_param", "auto")).strip().lower()
        self._video_data_url_cache: Dict[str, str] = {}
        self._video_frame_data_urls_cache: Dict[str, List[str]] = {}
        self._video_duration_cache: Dict[str, float] = {}
        self.dialog_dump_root = str(cfg.get("dialog_dump_root", "")).strip()
        self._dump_lock = threading.Lock()

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks)
        return ""

    def _resolve_max_tokens_param(self) -> str:
        if self.max_tokens_param in {"max_tokens", "max_completion_tokens"}:
            return self.max_tokens_param
        model_l = self.model.lower()
        # GPT-5 family requires max_completion_tokens on OpenAI-compatible APIs.
        if "gpt-5" in model_l:
            return "max_completion_tokens"
        return "max_tokens"

    def _video_to_data_url(self, video_path: str) -> str:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        stat = path.stat()
        max_bytes = int(self.max_video_size_mb * 1024 * 1024)
        if max_bytes > 0 and stat.st_size > max_bytes:
            raise ValueError(
                f"Video too large for base64 transport: {video_path} "
                f"({stat.st_size} bytes > {max_bytes})"
            )

        cache_key = f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
        cached = self._video_data_url_cache.get(cache_key)
        if cached is not None:
            return cached

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("video/"):
            mime_type = "video/mp4"

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"
        self._video_data_url_cache = {cache_key: data_url}
        return data_url

    def _image_to_data_url(self, image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _video_to_frame_data_urls(
        self,
        video_path: str,
        fps: Optional[float],
        frame_count: Optional[int] = None,
    ) -> List[str]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        stat = path.stat()
        eff_fps = float(fps if fps is not None else self.video_frame_extract_fps)
        if eff_fps <= 0:
            eff_fps = self.video_frame_extract_fps
        eff_frame_count = int(frame_count or self.video_frame_count)
        eff_frame_count = max(1, min(eff_frame_count, self.video_frame_count_cap))
        cache_key = (
            f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
            f"::{eff_frame_count}::{eff_fps:.6f}"
        )
        cached = self._video_frame_data_urls_cache.get(cache_key)
        if cached is not None:
            return cached

        with tempfile.TemporaryDirectory(prefix="stream_eval_frames_") as tmpdir:
            frame_pattern = str(Path(tmpdir) / "frame_%06d.jpg")
            extract_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vf",
                build_time_interval_select_filter(eff_fps),
                "-vsync",
                "vfr",
                "-q:v",
                "2",
                "-loglevel",
                "error",
                frame_pattern,
            ]
            proc = subprocess.run(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            sampled_frames = sorted(Path(tmpdir).glob("frame_*.jpg"))
            if proc.returncode != 0 or not sampled_frames:
                raise RuntimeError(
                    f"Frame extraction failed for {video_path}: "
                    f"exit_code={proc.returncode}; stderr_tail={(proc.stderr or '')[-1200:]}"
                )

            # The chunk and image transports now use the same timestamp-based
            # selector. Keep the selected frames in temporal order, using
            # frame_count only as a safety cap; never center-sample or repeat.
            selected_frames = sampled_frames[:eff_frame_count]
            data_urls = [self._image_to_data_url(frame_file) for frame_file in selected_frames]

        # Keep cache bounded to prevent unbounded memory growth.
        if len(self._video_frame_data_urls_cache) >= 256:
            self._video_frame_data_urls_cache.clear()
        self._video_frame_data_urls_cache[cache_key] = data_urls
        return data_urls

    def _probe_video_duration_seconds(self, video_path: str) -> float:
        cached = self._video_duration_cache.get(video_path)
        if cached is not None:
            return cached
        duration = 1.0
        try:
            info = probe_video(video_path)
            if isinstance(info, dict):
                val = float(info.get("duration", 0.0) or 0.0)
                if val > 0:
                    duration = val
        except Exception:
            duration = 1.0
        self._video_duration_cache[video_path] = duration
        return duration

    def _sanitize_path_segment(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return "unknown"
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        return text[:120] or "unknown"

    def _extract_sample_tag_from_video_path(self, video_path: str) -> str:
        path = Path(video_path)
        parent = path.parent.name.strip()
        if parent:
            return self._sanitize_path_segment(parent)
        stem = path.stem
        # Typical chunk name: video_<uuid>_<start>_<end>_trim*.mp4
        m = re.match(r"^video_([^_]+)_", stem)
        if m:
            return self._sanitize_path_segment(m.group(1))
        return self._sanitize_path_segment(stem or "unknown")

    def _decode_data_url(self, data_url: str) -> Tuple[bytes, str]:
        # data:<mime>;base64,<payload>
        if not data_url.startswith("data:"):
            raise ValueError("Unsupported non-data URL payload")
        header, encoded = data_url.split(",", 1)
        mime = "application/octet-stream"
        if ";" in header:
            mime = header[5:].split(";", 1)[0].strip() or mime
        ext = mimetypes.guess_extension(mime) or ".bin"
        return base64.b64decode(encoded), ext

    def _dump_dialog_artifacts(
        self,
        payload: Dict[str, Any],
        response_text: str,
        status_code: int,
        elapsed_ms: float,
        sample_tag: str = "",
        frame_durations: Optional[List[float]] = None,
    ) -> None:
        root = self.dialog_dump_root.strip()
        if not root:
            return
        root_dir = Path(root)
        base_dir = root_dir
        safe_sample_tag = self._sanitize_path_segment(sample_tag) if str(sample_tag).strip() else ""
        if safe_sample_tag:
            base_dir = base_dir / safe_sample_tag
        base_dir.mkdir(parents=True, exist_ok=True)
        folder_name = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
        req_dir = base_dir / folder_name
        req_dir.mkdir(parents=True, exist_ok=True)

        messages = payload.get("messages", [])
        dump_messages: List[Any] = []
        for msg in messages:
            if not isinstance(msg, dict):
                dump_messages.append(msg)
                continue
            dump_msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, list):
                dump_content: List[Any] = []
                for part in content:
                    if not isinstance(part, dict):
                        dump_content.append(part)
                        continue
                    dump_part = dict(part)
                    image_url = part.get("image_url")
                    if dump_part.get("type") == "image_url" and isinstance(image_url, dict):
                        dump_part["image_url"] = dict(image_url)
                        if "url" in image_url:
                            dump_part["image_url"]["url"] = "[data-url omitted]"
                    dump_content.append(dump_part)
                dump_msg["content"] = dump_content
            dump_messages.append(dump_msg)
        with open(req_dir / "messages.json", "w", encoding="utf-8") as f:
            json.dump(dump_messages, f, ensure_ascii=False, indent=2)

        text_lines: List[str] = []
        image_idx = 0
        dumped_images_in_order: List[Path] = []
        dumped_image_durations: List[float] = []
        duration_cursor = 0
        for msg_idx, msg in enumerate(messages):
            role = str(msg.get("role", "user"))
            text_lines.append(f"[{msg_idx}] role={role}")
            content_items = msg.get("content", [])
            if isinstance(content_items, str):
                text_lines.append("  (text content as string)")
                text_lines.append(content_items)
                continue
            if not isinstance(content_items, list):
                text_lines.append(f"  (unsupported content type: {type(content_items).__name__})")
                continue
            for part_idx, part in enumerate(content_items):
                part_type = str(part.get("type", ""))
                if part_type == "text":
                    text_lines.append(f"  ({part_idx}) text:")
                    text_lines.append(str(part.get("text", "")))
                elif part_type == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if isinstance(url, str) and url.startswith("data:"):
                        try:
                            raw, ext = self._decode_data_url(url)
                            image_idx += 1
                            img_name = f"image_{msg_idx:03d}_{part_idx:03d}_{image_idx:03d}{ext}"
                            with open(req_dir / img_name, "wb") as img_f:
                                img_f.write(raw)
                            dumped_images_in_order.append(req_dir / img_name)
                            if frame_durations is not None and duration_cursor < len(frame_durations):
                                dur = float(frame_durations[duration_cursor])
                            else:
                                dur = 0.5
                            dumped_image_durations.append(max(0.02, dur))
                            duration_cursor += 1
                            text_lines.append(f"  ({part_idx}) image: {img_name}")
                        except Exception as exc:
                            text_lines.append(f"  ({part_idx}) image decode failed: {exc}")
                    else:
                        text_lines.append(f"  ({part_idx}) image_url(non-data): {url}")
                elif part_type == "image":
                    text_lines.append(f"  ({part_idx}) image path: {part.get('image', '')}")
                elif part_type == "video":
                    text_lines.append(f"  ({part_idx}) video path: {part.get('video', '')}")
                elif part_type == "moss_realtime":
                    frames = part.get("frames") or []
                    prompts = part.get("prompts") or []
                    text_lines.append(
                        f"  ({part_idx}) moss_realtime frames={len(frames)} prompts={len(prompts)} "
                        f"reset={bool(part.get('reset', False))}"
                    )
                    for frame in frames:
                        if isinstance(frame, dict):
                            text_lines.append(
                                f"    frame t={float(frame.get('timestamp', 0.0)):.3f}s {frame.get('image', '')}"
                            )
                    for prompt in prompts:
                        text_lines.append(f"    prompt: {prompt}")

        with open(req_dir / "prompt.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines).strip() + "\n")

        meta = {
            "backend": self.backend_type,
            "model": self.model,
            "status_code": int(status_code),
            "elapsed_ms": round(float(elapsed_ms), 2),
            "saved_at": datetime.now().isoformat(),
        }
        with open(req_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(req_dir / "response.txt", "w", encoding="utf-8") as f:
            f.write((response_text or "").strip() + "\n")

        if dumped_images_in_order:
            list_file = req_dir / "_images_for_video.txt"
            try:
                with open(list_file, "w", encoding="utf-8") as f:
                    for idx, img in enumerate(dumped_images_in_order):
                        safe_path = img.resolve().as_posix().replace("'", r"'\''")
                        f.write(f"file '{safe_path}'\n")
                        dur = dumped_image_durations[idx] if idx < len(dumped_image_durations) else 0.5
                        f.write(f"duration {dur:.6f}\n")
                    safe_last = dumped_images_in_order[-1].resolve().as_posix().replace("'", r"'\''")
                    f.write(f"file '{safe_last}'\n")
                preview_cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_file.resolve()),
                    "-vsync",
                    "vfr",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-loglevel",
                    "error",
                    str((req_dir / "images_preview.mp4").resolve()),
                ]
                proc = subprocess.run(
                    preview_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if proc.returncode != 0:
                    with open(req_dir / "images_preview_error.txt", "w", encoding="utf-8") as f:
                        f.write((proc.stderr or "")[-2000:] + "\n")
            finally:
                if list_file.exists():
                    try:
                        list_file.unlink()
                    except Exception:
                        pass

        index_record = {
            "saved_at": meta["saved_at"],
            "sample_tag": safe_sample_tag or "unknown",
            "request_dir": str(req_dir),
            "request_dir_relative": str(req_dir.relative_to(root_dir)),
            "image_count": int(image_idx),
            "status_code": int(status_code),
            "elapsed_ms": round(float(elapsed_ms), 2),
            "model": self.model,
            "backend": self.backend_type,
        }
        index_path = root_dir / "index.jsonl"
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(index_record, ensure_ascii=False) + "\n")

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str, List[float]]:
        converted = []
        sample_tag = ""
        frame_durations: List[float] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue
            out_content = []
            for item in content:
                t = item.get("type")
                if t == "text":
                    if item.get("text"):
                        out_content.append({"type": "text", "text": item["text"]})
                elif t == "image":
                    image_path = str(item.get("image", ""))
                    if image_path and not sample_tag:
                        raw_sample_id = str(item.get("sample_id", "")).strip()
                        if raw_sample_id:
                            sample_tag = self._sanitize_path_segment(raw_sample_id)
                    if image_path:
                        image_entry: Dict[str, Any] = {"type": "image", "image": image_path}
                        for key in ("min_pixels", "max_pixels", "resized_height", "resized_width"):
                            if key in item:
                                image_entry[key] = int(item[key])
                        out_content.append(image_entry)
                elif t == "moss_realtime":
                    raw_sample_id = str(item.get("sample_id", "")).strip()
                    if raw_sample_id and not sample_tag:
                        sample_tag = self._sanitize_path_segment(raw_sample_id)
                    out_content.append(dict(item))
                elif t == "video":
                    video_path = str(item.get("video", ""))
                    if video_path and not sample_tag:
                        raw_sample_id = str(item.get("sample_id", "")).strip()
                        if raw_sample_id:
                            sample_tag = self._sanitize_path_segment(raw_sample_id)
                        else:
                            sample_tag = self._extract_sample_tag_from_video_path(video_path)
                    if self.video_transport == "image_frames_data_url":
                        # Default to fixed frame count (e.g. 2 fps => 2 frames per 1s chunk).
                        # Only force-focus chunks should expand to chunk-level max_frames.
                        if bool(item.get("use_max_frames", False)):
                            requested_count = item.get("max_frames", self.video_frame_count)
                            try:
                                requested_count = int(requested_count)
                            except Exception:
                                requested_count = self.video_frame_count
                        else:
                            requested_count = self.video_frame_count
                        frame_urls = self._video_to_frame_data_urls(
                            video_path=video_path,
                            fps=float(item.get("fps", self.video_frame_extract_fps)),
                            frame_count=requested_count,
                        )
                        logical_chunk_duration = float(item.get("chunk_duration_seconds", 0.0) or 0.0)
                        if logical_chunk_duration > 0:
                            chunk_duration = logical_chunk_duration
                        else:
                            chunk_duration = self._probe_video_duration_seconds(video_path)
                        per_frame_duration = max(0.02, float(chunk_duration) / max(1, len(frame_urls)))
                        if "chunk_start_seconds" in item:
                            chunk_start_seconds = float(item.get("chunk_start_seconds", 0.0) or 0.0)
                            out_content.append(
                                {
                                    "type": "text",
                                    "text": f"timestamp: {chunk_start_seconds:.2f}s",
                                }
                            )
                        for frame_url in frame_urls:
                            out_content.append({"type": "image_url", "image_url": {"url": frame_url}})
                            frame_durations.append(per_frame_duration)
                    elif self.video_transport == "base64_data_url":
                        data_url = self._video_to_data_url(video_path)
                        if self.backend_type in {"openrouter", "zhizengzeng"}:
                            out_content.append({"type": "video_url", "video_url": {"url": data_url}})
                        else:
                            out_content.append({"type": "input_video", "video_url": data_url})
                    else:
                        entry: Dict[str, Any] = {
                            "type": "video",
                            "video": video_path,
                            "fps": float(item.get("fps", 2.0)),
                        }
                        if "max_frames" in item:
                            entry["max_frames"] = item["max_frames"]
                        if "max_pixels" in item:
                            entry["max_pixels"] = item["max_pixels"]
                        if bool(item.get("low_fps_degenerated", False)):
                            entry["low_fps_degenerated"] = True
                        if bool(item.get("time_compress", False)):
                            entry["time_compress"] = True
                            entry["estimated_visual_tokens"] = int(
                                item.get("native_visual_tokens", 0) or 0
                            )
                            entry["estimated_source_pixels"] = int(
                                item.get("source_video_pixels", 0) or 0
                            )
                        if bool(item.get("resolution_compress", False)):
                            entry["resolution_compress"] = True
                            if "resized_height" in item and "resized_width" in item:
                                entry["resized_height"] = int(item["resized_height"])
                                entry["resized_width"] = int(item["resized_width"])
                            entry["processed_height"] = int(item["processed_height"])
                            entry["processed_width"] = int(item["processed_width"])
                            entry["estimated_visual_tokens"] = int(item["estimated_visual_tokens"])
                            entry["estimated_source_pixels"] = int(
                                item.get("estimated_source_pixels", 0) or 0
                            )
                        out_content.append(entry)
            if out_content:
                converted.append({"role": role, "content": out_content})
        return converted, sample_tag, frame_durations

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        skip_special_tokens: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not self.api_base:
            return {"response": "[ERROR] Missing api_base", "raw_response": "", "status_code": 500}
        if not self.model:
            return {"response": "[ERROR] Missing model name", "raw_response": "", "status_code": 500}

        norm_messages, sample_tag, frame_durations = self._normalize_messages(messages)
        payload = {
            "model": self.model,
            "messages": norm_messages,
            "stream": False,
            "temperature": self.temperature if temperature is None else float(temperature),
        }
        payload[self._resolve_max_tokens_param()] = int(max_new_tokens or self.max_new_tokens)
        if skip_special_tokens is not None:
            payload["skip_special_tokens"] = bool(skip_special_tokens)
        url = f"{self.api_base}/chat/completions"
        headers = self._build_headers()
        last_err = ""
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                body = resp.text
                if resp.status_code == 200:
                    parsed = resp.json()
                    choices = parsed.get("choices", [])
                    if not choices:
                        return {"response": "[ERROR] Empty choices", "raw_response": body, "status_code": 502}
                    msg = choices[0].get("message", {})
                    text = self._extract_text(msg.get("content", ""))
                    with self._dump_lock:
                        self._dump_dialog_artifacts(
                            payload=payload,
                            response_text=text,
                            status_code=200,
                            elapsed_ms=elapsed_ms,
                            sample_tag=sample_tag,
                            frame_durations=frame_durations,
                        )
                    return {"response": text.strip(), "raw_response": body, "status_code": 200}
                last_err = f"HTTP {resp.status_code}: {body[:400]}"
                with self._dump_lock:
                    self._dump_dialog_artifacts(
                        payload=payload,
                        response_text=last_err,
                        status_code=resp.status_code,
                        elapsed_ms=elapsed_ms,
                        sample_tag=sample_tag,
                        frame_durations=frame_durations,
                    )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                is_timeout = isinstance(exc, requests.exceptions.Timeout)
                error_kind = "Timeout" if is_timeout else type(exc).__name__
                last_err = f"{error_kind}: {exc}"
                with self._dump_lock:
                    self._dump_dialog_artifacts(
                        payload=payload,
                        response_text=f"[ERROR] {last_err}",
                        status_code=408 if is_timeout else 502,
                        elapsed_ms=elapsed_ms,
                        sample_tag=sample_tag,
                        frame_durations=frame_durations,
                    )
            time.sleep(0.5 * (2**attempt))
        return {"response": f"[ERROR] {last_err}", "raw_response": last_err, "status_code": 502}


class GeminiNativeBackend:
    """
    Gemini native format backend:
    POST /v1beta/models/{model}:generateContent?key=...
    """

    def __init__(self, name: str, cfg: Dict[str, Any]):
        self.name = name
        self.api_base = normalize_api_base(
            str(cfg.get("api_base", "https://generativelanguage.googleapis.com"))
        )
        self.api_key = str(cfg.get("api_key", ""))
        self.model = str(cfg.get("model", ""))
        self.timeout = float(cfg.get("timeout", 120))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1024))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_video_size_mb = float(cfg.get("max_video_size_mb", 30))
        self.video_metadata_fps = float(cfg.get("video_metadata_fps", 1.0))

    def _is_google_native(self) -> bool:
        return "googleapis.com" in self.api_base

    def _build_url(self) -> str:
        if not self.api_key:
            return ""
        path = f"/models/{self.model}:generateContent"
        base = self.api_base if "/v1beta" in self.api_base else f"{self.api_base}/v1beta"
        if self._is_google_native():
            return f"{base}{path}?key={self.api_key}"
        return f"{base}{path}"

    def _video_to_base64(self, video_path: str) -> str:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        max_bytes = int(self.max_video_size_mb * 1024 * 1024)
        if path.stat().st_size > max_bytes:
            raise ValueError(f"Video too large: {video_path} ({path.stat().st_size} bytes > {max_bytes})")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _merge_context_videos(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge consecutive video chunks across messages to reduce video count.

        Mirrors the MergeChunk._build_context() pattern from the original library:
        accumulate video-only messages, then merge them into one file when a text
        message is encountered.
        """
        merged: List[Dict[str, Any]] = []
        pending_videos: List[Dict[str, Any]] = []
        pending_role: Optional[str] = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])

            if role == "system":
                merged.append(msg)
                continue

            videos = [item for item in content if item.get("type") == "video"]
            non_videos = [item for item in content if item.get("type") != "video"]

            if non_videos:
                new_content: List[Dict[str, Any]] = []
                pending_videos.extend(videos)
                if pending_videos:
                    merged_path = self._merge_video_paths(
                        [v["video"] for v in pending_videos]
                    )
                    fps = pending_videos[0].get("fps", self.video_metadata_fps)
                    new_content.append({"type": "video", "video": merged_path, "fps": fps})
                    pending_videos.clear()
                new_content.extend(non_videos)
                merged.append({"role": role, "content": new_content})
            else:
                pending_videos.extend(videos)
                pending_role = role

        if pending_videos:
            merged_path = self._merge_video_paths(
                [v["video"] for v in pending_videos]
            )
            fps = pending_videos[0].get("fps", self.video_metadata_fps)
            merged.append({
                "role": pending_role or "user",
                "content": [{"type": "video", "video": merged_path, "fps": fps}],
            })

        return merged

    def _merge_video_paths(self, video_paths: List[str]) -> str:
        if len(video_paths) == 1:
            return video_paths[0]
        p0, p1 = Path(video_paths[0]), Path(video_paths[-1])
        s0, s1 = p0.stem.split("_"), p1.stem.split("_")
        prefix = "_".join(s0[:-2]) or s0[0]
        start_t = s0[-2] if len(s0) >= 2 else "0"
        end_t = s1[-1] if len(s1) >= 1 else "end"
        output_path = str(p0.parent / f"{prefix}_{start_t}_{end_t}.mp4")
        return merge_videos(video_paths, output_path)

    def _to_gemini_payload(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        messages = self._merge_context_videos(messages)

        system_text = ""
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            parts: List[Dict[str, Any]] = []
            for item in msg.get("content", []):
                item_type = item.get("type")
                if item_type == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        if role == "system":
                            system_text = f"{system_text}\n{text}".strip()
                        else:
                            parts.append({"text": text})
                elif item_type == "video":
                    video_path = str(item.get("video", ""))
                    b64 = self._video_to_base64(video_path)
                    parts.append({
                        "inline_data": {"mime_type": "video/mp4", "data": b64},
                        "video_metadata": {"fps": float(item.get("fps", self.video_metadata_fps))},
                    })

            if role != "system" and parts:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": parts})

        payload: Dict[str, Any] = {"contents": contents}
        if system_text:
            payload["system_instruction"] = {"parts": [{"text": system_text}]}
        payload["generationConfig"] = {
            "maxOutputTokens": int(max_new_tokens),
            "temperature": float(self.temperature),
        }
        return payload

    def _extract_text(self, response_json: Dict[str, Any]) -> str:
        candidates = response_json.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        text_chunks = [str(p.get("text", "")) for p in parts if isinstance(p, dict) and p.get("text")]
        return "\n".join(text_chunks).strip()

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        if not self.model:
            return {"response": "[ERROR] Missing model name", "raw_response": "", "status_code": 500}
        url = self._build_url()
        if not url:
            return {"response": "[ERROR] Missing Gemini api_key", "raw_response": "", "status_code": 500}

        try:
            payload = self._to_gemini_payload(messages, int(max_new_tokens or self.max_new_tokens))
        except Exception as exc:
            return {"response": f"[ERROR] payload build failed: {exc}", "raw_response": "", "status_code": 500}

        if not payload.get("contents"):
            return {"response": "[ERROR] Empty contents for Gemini request", "raw_response": "", "status_code": 500}

        headers = {"Content-Type": "application/json"}
        if not self._is_google_native():
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err = ""
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                body = resp.text
                if resp.status_code == 200:
                    parsed = resp.json()
                    text = self._extract_text(parsed)
                    return {"response": text, "raw_response": text, "status_code": 200}
                last_err = f"HTTP {resp.status_code}: {body[:400]}"
            except Exception as exc:
                last_err = str(exc)
            time.sleep(0.5 * (2**attempt))
        return {"response": f"[ERROR] {last_err}", "raw_response": last_err, "status_code": 502}


def build_backend(name: str, cfg: Dict[str, Any]) -> Any:
    backend_type = str(cfg.get("backend", "openai_compatible")).strip().lower()
    if backend_type in {"openai_compatible", "hf", "zhizengzeng", "openrouter"}:
        return OpenAICompatibleBackend(name, cfg)
    if backend_type in {"gemini_native", "gemini"}:
        return GeminiNativeBackend(name, cfg)
    raise ValueError(
        f"Unsupported backend '{backend_type}' for '{name}'. "
        "Supported: openai_compatible, hf, zhizengzeng, openrouter, gemini_native"
    )


class SessionModel:
    def __init__(
        self,
        backend: Any,
        max_video_length_in_seconds: float,
        max_context_frames: int,
        default_video_fps: float,
        max_focus_context_frames: int = 0,
        resolution_compress: bool = False,
        time_compress: bool = False,
        also_compress_focus: bool = False,
        low_fps_degeneration: bool = False,
        visual_token_budget: int = QWEN3_VL_DEFAULT_VISUAL_TOKEN_BUDGET,
        pixel_budget: Optional[int] = None,
    ):
        self.backend = backend
        self.max_video_length_in_seconds = float(max_video_length_in_seconds)
        self.max_context_frames = max(0, int(max_context_frames))
        self.max_focus_context_frames = max(0, int(max_focus_context_frames))
        self.default_video_fps = float(default_video_fps)
        self.resolution_compress = bool(resolution_compress)
        self.time_compress = bool(time_compress)
        self.also_compress_focus = bool(also_compress_focus)
        self.low_fps_degeneration = bool(low_fps_degeneration)
        self.visual_token_budget = max(1, int(visual_token_budget))
        self.pixel_budget = max(1, int(pixel_budget)) if pixel_budget is not None else None
        self.new_session()

    def new_session(self, chunk: Optional[Dict[str, Any]] = None) -> str:
        self.session_id = str(uuid.uuid4())
        self.context: List[Dict[str, Any]] = []
        self.video_chunk_info: Dict[str, Dict[str, Any]] = {}
        self.current_context_video_info: List[Dict[str, Any]] = []
        self.cum_video_length_in_seconds = 0.0
        self.cum_video_frames = 0
        self.cum_focus_video_frames = 0
        self.cum_native_visual_tokens = 0
        self.cum_source_video_pixels = 0
        if chunk:
            self.add_chunk(chunk)
        return self.session_id

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        new_content = []
        for ele in chunk.get("content", []):
            if ele.get("type") == "video":
                video_path = str(ele.get("video", ""))
                info = self.video_chunk_info.get(video_path)
                if info is None:
                    info = probe_video(video_path) or {"duration": 1.0, "video": {"fps": self.default_video_fps}}
                    self.video_chunk_info[video_path] = info
                duration = float(info.get("duration", 1.0) or 1.0)
                fps = float(ele.get("fps", self.default_video_fps))
                video_info = info.get("video") or {}
                num_frames = int(video_info.get("total_frames", 0) or 0)
                width = int(video_info.get("width", 0) or 0)
                height = int(video_info.get("height", 0) or 0)
                is_high_fps = bool(ele.get("use_max_frames", False))
                is_focus_chunk = bool(ele.get("is_focus_chunk", is_high_fps))
                native_visual_tokens = qwen3_vl_visual_token_count(
                    num_frames,
                    height,
                    width,
                )
                source_video_pixels = num_frames * height * width
                self.cum_video_length_in_seconds += duration
                self.cum_video_frames += num_frames
                self.cum_native_visual_tokens += native_visual_tokens
                self.cum_source_video_pixels += source_video_pixels
                if is_focus_chunk:
                    self.cum_focus_video_frames += num_frames
                video_item = {
                    "type": "video",
                    "video": video_path,
                    "max_frames": num_frames,
                    "context_frames": num_frames,
                    "max_pixels": 384 * 28 * 28,
                    "fps": fps,
                    "use_max_frames": is_high_fps,
                    "is_high_fps": is_high_fps,
                    "is_focus_chunk": is_focus_chunk,
                    "source_width": width,
                    "source_height": height,
                    "native_visual_tokens": native_visual_tokens,
                    "source_video_pixels": source_video_pixels,
                    "sample_id": str(ele.get("sample_id", "")).strip(),
                    "chunk_start_seconds": float(ele.get("chunk_start_seconds", 0.0) or 0.0),
                    "chunk_duration_seconds": float(ele.get("chunk_duration_seconds", 0.0) or 0.0),
                }
                if self.resolution_compress:
                    # Explicit dimensions make qwen-vl-utils use the size selected
                    # by the context allocator rather than its independent cap.
                    video_item.update(
                        {
                            "resolution_compress": True,
                        }
                    )
                if self.time_compress:
                    video_item["time_compress"] = True
                new_content.append(video_item)
            else:
                new_content.append(ele)

        self.context.append({"role": chunk.get("role", "user"), "content": new_content})
        self._truncate_context_videos()
        if self.resolution_compress:
            self._compress_context_video_resolutions()
        self._rebuild_video_info()

    def _remove_oldest_video(self, focus_only: bool = False) -> bool:
        for msg_idx, msg in enumerate(self.context):
            for item_idx, item in enumerate(msg.get("content", [])):
                if item.get("type") != "video":
                    continue
                if focus_only and not bool(item.get("is_focus_chunk", False)):
                    continue
                path = item.get("video", "")
                info = self.video_chunk_info.get(path, {})
                dur = float(info.get("duration", 1.0) or 1.0)
                video_info = info.get("video") or {}
                num_frames = int(
                    item.get("context_frames", video_info.get("total_frames", 0)) or 0
                )
                self.cum_video_length_in_seconds -= dur
                self.cum_video_frames -= num_frames
                self.cum_native_visual_tokens -= int(
                    item.get("native_visual_tokens", 0) or 0
                )
                self.cum_source_video_pixels -= int(
                    item.get("source_video_pixels", 0) or 0
                )
                if bool(item.get("is_focus_chunk", False)):
                    self.cum_focus_video_frames -= num_frames
                msg["content"] = msg["content"][:item_idx] + msg["content"][item_idx + 1 :]
                if not msg["content"]:
                    self.context = self.context[:msg_idx] + self.context[msg_idx + 1 :]
                return True
        return False

    def _degrade_oldest_focus_video(self) -> bool:
        for msg in self.context:
            for item in msg.get("content", []):
                if item.get("type") != "video" or not bool(item.get("is_focus_chunk", False)):
                    continue
                old_frames = int(item.get("context_frames", item.get("max_frames", 0)) or 0)
                duration = float(item.get("chunk_duration_seconds", 0.0) or 0.0)
                if duration <= 0:
                    info = self.video_chunk_info.get(str(item.get("video", "")), {})
                    duration = float(info.get("duration", 1.0) or 1.0)
                low_frames = max(2, int(duration * self.default_video_fps))
                low_frames = max(2, (low_frames // QWEN3_VL_TEMPORAL_PATCH_SIZE) * QWEN3_VL_TEMPORAL_PATCH_SIZE)
                low_frames = min(old_frames, low_frames) if old_frames > 0 else low_frames
                old_native_tokens = int(item.get("native_visual_tokens", 0) or 0)
                new_native_tokens = qwen3_vl_visual_token_count(
                    low_frames,
                    int(item.get("source_height", 0) or 0),
                    int(item.get("source_width", 0) or 0),
                )
                old_source_pixels = int(item.get("source_video_pixels", 0) or 0)
                new_source_pixels = (
                    low_frames
                    * int(item.get("source_height", 0) or 0)
                    * int(item.get("source_width", 0) or 0)
                )

                self.cum_focus_video_frames -= old_frames
                self.cum_video_frames -= max(0, old_frames - low_frames)
                self.cum_native_visual_tokens += new_native_tokens - old_native_tokens
                self.cum_source_video_pixels += new_source_pixels - old_source_pixels
                item["fps"] = self.default_video_fps
                item["max_frames"] = low_frames
                item["context_frames"] = low_frames
                item["use_max_frames"] = False
                item["is_high_fps"] = False
                item["is_focus_chunk"] = False
                item["native_visual_tokens"] = new_native_tokens
                item["source_video_pixels"] = new_source_pixels
                item["low_fps_degenerated"] = True
                return True
        return False

    def _truncate_context_videos(self) -> None:
        while True:
            general_window_exceeded = (
                self.cum_video_length_in_seconds > self.max_video_length_in_seconds
                or (
                    self.time_compress
                    and (
                        (
                            self.pixel_budget is not None
                            and self.cum_source_video_pixels > self.pixel_budget
                        )
                        or (
                            self.pixel_budget is None
                            and self.cum_native_visual_tokens > self.visual_token_budget
                        )
                    )
                )
                or (
                    not self.resolution_compress
                    and not self.time_compress
                    and self.max_context_frames > 0
                    and self.cum_video_frames > self.max_context_frames
                )
            )
            focus_window_exceeded = (
                self.max_focus_context_frames > 0
                and self.cum_focus_video_frames > self.max_focus_context_frames
            )
            if not general_window_exceeded and not focus_window_exceeded:
                break

            if focus_window_exceeded and self.low_fps_degeneration:
                if self._degrade_oldest_focus_video():
                    continue
                break

            # Preserve normal-FPS chunks when only the dedicated focus window
            # overflows; general time/frame overflow still evicts oldest-first.
            removed = self._remove_oldest_video(
                focus_only=focus_window_exceeded and not general_window_exceeded
            )
            if not removed:
                break

    def _compress_context_video_resolutions(self) -> None:
        video_items = [
            item
            for msg in self.context
            for item in msg.get("content", [])
            if item.get("type") == "video"
        ]
        if not video_items:
            return

        def item_plan(item: Dict[str, Any], scale: float) -> Tuple[int, int, int, int]:
            height = max(1, int(item.get("source_height", 0) or 0))
            width = max(1, int(item.get("source_width", 0) or 0))
            return qwen3_vl_preprocess_dimensions(
                item_frames(item),
                height,
                width,
                scale,
                scale_before_native_resize=self.pixel_budget is not None,
            )

        def item_frames(item: Dict[str, Any]) -> int:
            frames = int(item.get("max_frames", 0) or 0)
            if frames > 0:
                return frames
            duration = float(item.get("chunk_duration_seconds", 0.0) or 0.0)
            fps = float(item.get("fps", self.default_video_fps) or self.default_video_fps)
            return max(1, round(duration * fps))

        def item_tokens(item: Dict[str, Any], scale: float) -> int:
            _, _, processed_height, processed_width = item_plan(item, scale)
            temporal_tokens = (
                qwen3_vl_effective_frame_count(item_frames(item))
                // QWEN3_VL_TEMPORAL_PATCH_SIZE
            )
            return (
                temporal_tokens
                * (processed_height // QWEN3_VL_SPATIAL_TOKEN_FACTOR)
                * (processed_width // QWEN3_VL_SPATIAL_TOKEN_FACTOR)
            )

        def item_pixels(item: Dict[str, Any], scale: float) -> int:
            height = max(1, int(item.get("source_height", 0) or 0))
            width = max(1, int(item.get("source_width", 0) or 0))
            safe_scale = max(0.0, min(1.0, float(scale)))
            target_height = max(1, round(height * safe_scale))
            target_width = max(1, round(width * safe_scale))
            return item_frames(item) * target_height * target_width

        def item_budget_cost(item: Dict[str, Any], scale: float) -> int:
            if self.pixel_budget is not None:
                return item_pixels(item, scale)
            return item_tokens(item, scale)

        budget_limit = (
            self.pixel_budget
            if self.pixel_budget is not None
            else self.visual_token_budget
        )
        budget_unit = "pixels" if self.pixel_budget is not None else "visual tokens"
        fixed_focus_items = [
            item
            for item in video_items
            if bool(item.get("is_focus_chunk", False)) and not self.also_compress_focus
        ]
        compressible_items = [
            item
            for item in video_items
            if self.also_compress_focus or not bool(item.get("is_focus_chunk", False))
        ]
        fixed_focus_cost = sum(item_budget_cost(item, 1.0) for item in fixed_focus_items)
        if fixed_focus_cost > budget_limit:
            raise RuntimeError(
                "Resolution compression cannot fit the unscaled focus chunks: "
                f"{fixed_focus_cost} {budget_unit} exceed budget {budget_limit}"
            )

        compressible_scale = 1.0
        if (
            fixed_focus_cost
            + sum(item_budget_cost(item, 1.0) for item in compressible_items)
            > budget_limit
        ):
            low, high = 0.0, 1.0
            for _ in range(32):
                middle = (low + high) / 2.0
                total = fixed_focus_cost + sum(
                    item_budget_cost(item, middle) for item in compressible_items
                )
                if total <= budget_limit:
                    low = middle
                else:
                    high = middle
            compressible_scale = low

        total_cost = fixed_focus_cost
        for item in fixed_focus_items:
            _, _, processed_height, processed_width = item_plan(item, 1.0)
            tokens = item_tokens(item, 1.0)
            # Focus keeps Qwen's native smart-resize; it is only exempt from
            # the additional context-driven spatial compression.
            item.pop("resized_height", None)
            item.pop("resized_width", None)
            item["processed_height"] = processed_height
            item["processed_width"] = processed_width
            item["estimated_visual_tokens"] = tokens
            item["estimated_source_pixels"] = item_pixels(item, 1.0)
        for item in compressible_items:
            resized_height, resized_width, processed_height, processed_width = item_plan(
                item,
                compressible_scale,
            )
            tokens = item_tokens(item, compressible_scale)
            item["resized_height"] = resized_height
            item["resized_width"] = resized_width
            item["processed_height"] = processed_height
            item["processed_width"] = processed_width
            item["estimated_visual_tokens"] = tokens
            item["estimated_source_pixels"] = item_pixels(item, compressible_scale)
            total_cost += item_budget_cost(item, compressible_scale)

        if total_cost > budget_limit:
            raise RuntimeError(
                f"Resolution compression produced {total_cost} {budget_unit}, "
                f"above budget {budget_limit}"
            )

    def _rebuild_video_info(self) -> None:
        out: List[Dict[str, Any]] = []
        for msg in self.context:
            for item in msg.get("content", []):
                if item.get("type") == "video":
                    raw = self.video_chunk_info.get(item.get("video", ""), {})
                    if isinstance(raw, dict):
                        out.append(raw | {"fps": float(item.get("fps", self.default_video_fps))})
        self.current_context_video_info = out

    def generate(self, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        return self.backend.generate(self.context, max_new_tokens=max_new_tokens)


NATIVE_VIDEOCHAT3_SYSTEM = """
You are a helpful assistant specializing in streaming video analysis.
You will receive input frame by frame, each labeled with absolute time intervals
in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:

1. Use </Silence> when:
   - No relevant event has started, OR
   - The current input is irrelevant to the given question.

2. Use </Standby> when:
   - An event is in progress but has not yet completed, OR
   - The current input is relevant but the question cannot yet be answered.

3. Use </Response> only when:
   - An event has fully concluded, OR
   - The available information is sufficient to fully answer the question.
   Provide a complete description at this point.

Do not provide partial answers or speculate beyond the given information.
Whenever you deliver an answer, begin with </Response>.
""".strip()
NATIVE_VIDEOCHAT3_FPS = 4.0
NATIVE_VIDEOCHAT3_MAX_PIXELS = 224 * 224
NATIVE_VIDEOCHAT3_MAX_USER_ROUNDS = 32
NATIVE_VIDEOCHAT3_MAX_NEW_TOKENS = 128
NATIVE_VIDEOCHAT3_TEMPERATURE = 0.7
NATIVE_VIDEOCHAT3_SILENCE = "</Silence>"
NATIVE_VIDEOCHAT3_STANDBY = "</Standby>"
NATIVE_VIDEOCHAT3_RESPONSE_PREFIX = "</Response>"
NATIVE_VIDEOCHAT3_SCORE_PLACEHOLDERS = (
    NATIVE_VIDEOCHAT3_SILENCE,
    NATIVE_VIDEOCHAT3_STANDBY,
)


def native_videochat3_smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 28 * 28,
    max_pixels: int = NATIVE_VIDEOCHAT3_MAX_PIXELS,
    force_resize: bool = False,
) -> Tuple[int, int]:
    """Match VideoChat3-4B/demo_vc3_proactive.py smart_resize."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    if force_resize:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
        return h_bar, w_bar
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def native_videochat3_resize_image(src: Path, dst: Path, max_pixels: int) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(src) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        resized_height, resized_width = native_videochat3_smart_resize(
            height,
            width,
            min_pixels=28 * 28,
            max_pixels=max_pixels,
            force_resize=True,
        )
        resized = rgb.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst, quality=95)
    return resized_height, resized_width


def native_videochat3_extract_frames(video_path: str, out_dir: Path, target_fps: float) -> List[Path]:
    """Sample source frames the same way as VideoChat3's VideoFrameExtractor."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(video_path) or {}
    video_info = info.get("video") if isinstance(info.get("video"), dict) else {}
    src_fps = float((video_info or {}).get("fps") or 0.0)
    duration = float(info.get("duration") or 0.0)
    n_extract = int(duration * target_fps) if duration > 0 else 0
    if n_extract <= 0:
        raise RuntimeError(f"Cannot determine native VideoChat3 frame count for {video_path}")
    existing = sorted(out_dir.glob("src_*.jpg"))
    if len(existing) >= n_extract:
        return existing[:n_extract]
    for stale in existing:
        try:
            stale.unlink()
        except Exception:
            pass
    if src_fps > 0:
        interval = max(int(round(src_fps / target_fps)), 1)
        video_filter = f"select=not(mod(n\\,{interval}))"
    else:
        video_filter = f"fps={target_fps}"
    pattern = str(out_dir / "src_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        video_filter,
        "-vsync",
        "vfr",
        "-frames:v",
        str(n_extract),
        "-q:v",
        "2",
        pattern,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    frames = sorted(out_dir.glob("src_*.jpg"))
    if proc.returncode != 0 or not frames:
        tail = (proc.stderr or "")[-1200:]
        raise RuntimeError(f"Native VideoChat3 frame extraction failed for {video_path}: {tail}")
    return frames[:n_extract]


def format_native_videochat3_question(event: Dict[str, Any]) -> str:
    parts: List[str] = []
    question = str(event.get("question") or "").strip()
    if question:
        parts.append(question)
    if event.get("options") is not None:
        parts.append(f"Options: {event.get('options')}")
    return "\n".join(parts)


def prepare_native_videochat3_scored_text(text: str) -> str:
    stripped = (text or "").strip()
    prefix = NATIVE_VIDEOCHAT3_RESPONSE_PREFIX
    if stripped.lower().startswith(prefix.lower()):
        return stripped[len(prefix):].strip()
    return stripped


NATIVE_MOSS_SILENCE = "<|silence|>"
NATIVE_MOSS_CONTROL_TOKENS = (
    "<|round_start|>",
    "<|round_end|>",
    "<|response|>",
    "<|assistant|>",
)
NATIVE_MOSS_SCORE_PLACEHOLDERS = (NATIVE_MOSS_SILENCE,)


def prepare_native_moss_scored_text(text: str) -> str:
    """Map MOSS-VL-Realtime control tokens onto the benchmark answer text."""
    cleaned = text or ""
    for token in NATIVE_MOSS_CONTROL_TOKENS:
        cleaned = cleaned.replace(token, "")
    answer = cleaned.replace(NATIVE_MOSS_SILENCE, "").strip()
    if not answer:
        return NATIVE_MOSS_SILENCE
    return answer


def native_moss_chunk_timestamps(start: float, end: float, sample_fps: float) -> List[float]:
    """Match MOSS-VL ``iter_video_file``: ``ceil(duration * fps)`` stamps at ``start + i / fps``."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    duration = float(end) - float(start)
    if duration <= 1e-6:
        return []
    frame_count = max(1, int(math.ceil(duration * sample_fps - 1e-6)))
    begin = float(start)
    safe_end = max(begin, begin + duration - 1e-6)
    return [min(begin + index / float(sample_fps), safe_end) for index in range(frame_count)]


def _native_moss_tensor_to_pil(frame: Any) -> Any:
    import numpy as np
    from PIL import Image

    if hasattr(frame, "detach"):
        frame = frame.detach().cpu()
    if hasattr(frame, "ndim") and frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = frame.permute(1, 2, 0)
    array = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
    if array.dtype != np.uint8:
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] >= 4:
        array = array[..., :3]
    return Image.fromarray(array).convert("RGB")


def native_moss_extract_chunk_frames(
    video_path: str,
    start: float,
    end: float,
    sample_fps: float,
    out_dir: Path,
) -> List[Tuple[str, float]]:
    """Decode one benchmark chunk with MOSS-VL's timestamped frame sampler.

    ``sample_fps`` is the chunk policy FPS (the same value the other models receive
    as ``fps``). Focus windows therefore switch rate by changing this value, while
    every frame keeps an absolute timestamp so the realtime session sees the switch.
    """
    timestamps = native_moss_chunk_timestamps(start, end, sample_fps)
    if not timestamps:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    fps_tag = str(sample_fps).replace(".", "p")
    frame_dir = out_dir / f"{start:.3f}_{end:.3f}_{fps_tag}".replace("-", "m")
    manifest_path = frame_dir / "frames.json"
    if manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        if (
            isinstance(cached, list)
            and len(cached) == len(timestamps)
            and all(Path(str(item.get("image", ""))).exists() for item in cached if isinstance(item, dict))
        ):
            return [(str(item["image"]), float(item["timestamp"])) for item in cached]

    try:
        from torchcodec.decoders import VideoDecoder
    except Exception as exc:
        raise RuntimeError(
            "Native MOSS-VL frame sampling requires torchcodec, the decoder used by MOSS-VL-Realtime."
        ) from exc

    frame_dir.mkdir(parents=True, exist_ok=True)
    decoder = VideoDecoder(str(video_path), num_ffmpeg_threads=0)
    try:
        metadata = decoder.metadata
        begin_value = getattr(metadata, "begin_stream_seconds_from_content", None)
        safe_begin = max(0.0, float(begin_value)) if begin_value is not None else 0.0
        end_value = getattr(metadata, "end_stream_seconds_from_content", None)
        if end_value is not None:
            safe_stream_end = max(safe_begin, float(end_value) - 1e-6)
            clamped = [max(safe_begin, min(float(ts), safe_stream_end)) for ts in timestamps]
        else:
            clamped = [max(safe_begin, float(ts)) for ts in timestamps]
        batch = decoder.get_frames_played_at(clamped)
        pts_value = getattr(batch, "pts_seconds", None)
        if hasattr(pts_value, "detach"):
            pts_value = pts_value.detach().cpu()
        decoded_timestamps = pts_value.tolist() if hasattr(pts_value, "tolist") else []
        saved: List[Dict[str, Any]] = []
        last_timestamp: Optional[float] = None
        for offset, frame in enumerate(batch.data):
            timestamp = clamped[offset]
            if offset < len(decoded_timestamps):
                timestamp = float(decoded_timestamps[offset])
            if last_timestamp is not None and timestamp < last_timestamp:
                timestamp = last_timestamp
            last_timestamp = timestamp
            image = _native_moss_tensor_to_pil(frame)
            image_path = frame_dir / f"f{offset:04d}.jpg"
            image.save(image_path, quality=95)
            saved.append({"image": str(image_path), "timestamp": timestamp})
    finally:
        close = getattr(decoder, "close", None)
        if callable(close):
            close()
    if len(saved) != len(timestamps):
        raise RuntimeError(
            f"Native MOSS-VL decoded {len(saved)} frames for {video_path} "
            f"[{start:.3f}, {end:.3f}) at {sample_fps} fps, expected {len(timestamps)}"
        )
    manifest_path.write_text(json.dumps(saved), encoding="utf-8")
    return [(str(item["image"]), float(item["timestamp"])) for item in saved]


NATIVE_JOYAI_SILENCE = "</silence>"
NATIVE_JOYAI_RESPONSE = "</response>"
NATIVE_JOYAI_SCORE_PLACEHOLDERS = (
    NATIVE_JOYAI_SILENCE,
    "Focus_Start",
    "Focus_End",
)
NATIVE_JOYAI_MAX_PIXELS = 262144
NATIVE_JOYAI_CHUNK_TURNS = 100
NATIVE_JOYAI_QUERY_HEADER = "[User Query (IMPORTANT — follow this instruction)]"
NATIVE_JOYAI_SYSTEM = """You are a real-time video streaming assistant observing a continuous camera feed frame by frame. The last frame represents the current moment.
## Action Format
At every inference step you MUST choose exactly one of the following three actions:
**Stay silent** — output ONLY:
</silence>
Choose this when nothing noteworthy has changed in the scene, no user query is pending, or there is nothing useful to say.
**Speak** — output the token followed by a concise reply:
</response> Your reply here.
Choose this when you observe something worth reporting or a significant state change, or when you can answer a user question based on available evidence.

**Delegate** — when a question is too hard or error-prone to answer reliably yourself, speak a brief note that you're delegating, then hand the question to the background solver:
</response> Brief note that you're delegating. </delegation> <the question>"""
NATIVE_JOYAI_FOCUS_RULES = """## Focus control
The number of frames in a newly appended second can vary. You can proactively request a temporary higher-frame-rate view when the current sampling rate may miss fine-grained temporal details.

- Output the exact token Focus_Start whenever the current scene appears likely to contain an upcoming highlight, rapid motion, a brief event, a transition, an interaction, object manipulation, a subtle gesture, or any detail that could benefit from fine-grained temporal perception. Also output Focus_Start when an active question requires closer temporal observation. High-frame-rate sampling starts with the next second. Use a low threshold: when uncertain, prefer Focus_Start.
- While high-frame-rate sampling is active, output the exact token Focus_End once that interval has clearly ended and the scene has settled. Normal sampling resumes with the next second. Until then, keep focus active by answering the active question if possible, or outputting </silence>.
- Focus_Start and Focus_End must each be the entire response. Do not combine either token with </response>, </silence>, an answer, punctuation, or explanation.
- Do not output Focus_Start if high-frame-rate sampling is already active. Do not output Focus_End unless it is active.
- </silence> remains the stay-silent action when neither a focus change nor a spoken reply is needed."""


def normalize_native_joyai_output(text: str) -> str:
    """Match JoyAI live_adapter.normalize_model_output."""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    for token in ("<|im_end|>", "<|endoftext|>"):
        raw = raw.replace(token, "")
    raw = raw.strip()
    if not raw:
        return NATIVE_JOYAI_SILENCE
    marker_positions = []
    for marker in (NATIVE_JOYAI_RESPONSE, NATIVE_JOYAI_SILENCE):
        idx = raw.find(marker)
        if idx != -1:
            marker_positions.append((idx, marker))
    if marker_positions:
        _, marker = min(marker_positions, key=lambda item: item[0])
        if marker == NATIVE_JOYAI_SILENCE:
            return NATIVE_JOYAI_SILENCE
        response_text = raw.split(marker, 1)[1].strip()
        if not response_text:
            return NATIVE_JOYAI_RESPONSE
        first_line = " ".join(response_text.splitlines()[0].split())
        return f"{NATIVE_JOYAI_RESPONSE} {first_line}" if first_line else NATIVE_JOYAI_RESPONSE
    first_line = " ".join(raw.splitlines()[0].split())
    return f"{NATIVE_JOYAI_RESPONSE} {first_line}" if first_line else NATIVE_JOYAI_SILENCE


def prepare_native_joyai_scored_text(text: str) -> str:
    """Drop JoyAI control tokens and keep the spoken first line for scoring."""
    normalized = normalize_native_joyai_output(text)
    if not normalized.startswith(NATIVE_JOYAI_RESPONSE):
        return NATIVE_JOYAI_SILENCE
    payload = normalized[len(NATIVE_JOYAI_RESPONSE):].strip()
    return payload or NATIVE_JOYAI_SILENCE


def native_joyai_extract_chunk_frames(
    video_path: str,
    start: float,
    end: float,
    sample_fps: float,
    out_dir: Path,
) -> List[Tuple[str, float]]:
    """Sample one benchmark chunk as JoyAI image frames.

    The 1-second clock stays on StreamingState. ``sample_fps`` is that chunk's
    policy FPS, so a focus window can place more than one frame inside the second.
    JoyAI still receives them as one turn: a single ``<T seconds>`` tag plus images.
    """
    timestamps = native_moss_chunk_timestamps(start, end, sample_fps)
    if not timestamps:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    fps_tag = str(sample_fps).replace(".", "p")
    # ss6: seek with 6 decimals. 3-decimal formatting rounded 19.966667 up to
    # 19.967 and skipped past the last frame.
    frame_dir = out_dir / f"{start:.3f}_{end:.3f}_{fps_tag}_ss6".replace("-", "m")
    manifest_path = frame_dir / "frames.json"
    if manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        if (
            isinstance(cached, list)
            and len(cached) == len(timestamps)
            and all(Path(str(item.get("image", ""))).exists() for item in cached if isinstance(item, dict))
        ):
            return [(str(item["image"]), float(item["timestamp"])) for item in cached]

    frame_dir.mkdir(parents=True, exist_ok=True)
    for stale in frame_dir.glob("f*.jpg"):
        try:
            stale.unlink()
        except Exception:
            pass

    def _grab_frame(seek_seconds: float, image_path: Path) -> bool:
        if image_path.exists():
            image_path.unlink()
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{float(seek_seconds):.6f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(image_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return proc.returncode == 0 and image_path.is_file() and image_path.stat().st_size > 0

    saved = []
    past_last_frame = False
    frame_step = 1.0 / float(sample_fps)
    for offset, timestamp in enumerate(timestamps):
        image_path = frame_dir / f"f{offset:04d}.jpg"
        if past_last_frame and saved:
            image_path.write_bytes(Path(saved[-1]["image"]).read_bytes())
            saved.append({"image": str(image_path), "timestamp": float(timestamp)})
            continue
        seek = float(timestamp)
        if not _grab_frame(seek, image_path):
            # Container duration can sit one frame past the last decodable PTS.
            # Input seeking to that instant writes an empty file.
            for _ in range(4):
                seek = max(float(start), seek - frame_step)
                if _grab_frame(seek, image_path):
                    break
            else:
                if saved:
                    image_path.write_bytes(Path(saved[-1]["image"]).read_bytes())
                else:
                    raise RuntimeError(
                        f"Native JoyAI frame extraction failed for {video_path} "
                        f"at {float(timestamp):.6f}s"
                    )
            past_last_frame = True
        saved.append({"image": str(image_path), "timestamp": float(timestamp)})
    manifest_path.write_text(json.dumps(saved), encoding="utf-8")
    return [(str(item["image"]), float(item["timestamp"])) for item in saved]


def _native_joyai_frame_pixels(frames: List[Tuple[str, float]]) -> int:
    from PIL import Image

    total = 0
    for image_path, _timestamp in frames:
        try:
            with Image.open(image_path) as image:
                total += int(image.size[0]) * int(image.size[1])
        except Exception:
            total += NATIVE_JOYAI_MAX_PIXELS
    return total


def _native_joyai_user_message(messages: List[Dict[str, Any]], user_index: int) -> Optional[Dict[str, Any]]:
    seen = 0
    for message in messages:
        if message.get("role") != "user":
            continue
        if seen == user_index:
            return message
        seen += 1
    return None


def _replace_native_joyai_images(
    message: Dict[str, Any],
    frames: List[Tuple[str, float]],
    sample_id: str,
) -> None:
    content = [item for item in message.get("content", []) if item.get("type") != "image"]
    for offset, (image_path, _timestamp) in enumerate(frames):
        image_item: Dict[str, Any] = {
            "type": "image",
            "image": image_path,
            "max_pixels": NATIVE_JOYAI_MAX_PIXELS,
        }
        if offset == 0 and sample_id:
            image_item["sample_id"] = sample_id
        content.append(image_item)
    message["content"] = content


def _drop_native_joyai_turn(
    messages: List[Dict[str, Any]],
    turn_meta: List[Dict[str, Any]],
    user_index: int,
) -> bool:
    seen = 0
    user_pos = None
    for pos, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        if seen == user_index:
            user_pos = pos
            break
        seen += 1
    if user_pos is None or user_index >= len(turn_meta):
        return False
    del messages[user_pos]
    if user_pos < len(messages) and messages[user_pos].get("role") == "assistant":
        del messages[user_pos]
    del turn_meta[user_index]
    return True


def _native_joyai_message_query_text(message: Optional[Dict[str, Any]]) -> str:
    if not message:
        return ""
    for item in message.get("content", []):
        if item.get("type") != "text":
            continue
        text = str(item.get("text", ""))
        if text.startswith(NATIVE_JOYAI_QUERY_HEADER):
            return text
    return ""


def _native_joyai_history_has_query(messages: List[Dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "user" and _native_joyai_message_query_text(message)
        for message in messages
    )


def _inject_native_joyai_query(messages: List[Dict[str, Any]], query_text: str) -> None:
    """Keep a dropped question on the oldest remaining user turn."""
    if not query_text:
        return
    for message in messages:
        if message.get("role") != "user":
            continue
        content = list(message.get("content", []))
        if _native_joyai_message_query_text({"content": content}):
            return
        content.insert(0, {"type": "text", "text": query_text})
        message["content"] = content
        return


def _evict_native_joyai_turns_over_pixel_budget(
    messages: List[Dict[str, Any]],
    turn_meta: List[Dict[str, Any]],
    pixel_budget: int,
) -> None:
    """Drop oldest JoyAI turns until source pixels fit --time-compress --pixel-budget."""

    def total_pixels() -> int:
        return sum(int(meta.get("pixels", 0) or 0) for meta in turn_meta)

    while turn_meta and total_pixels() > int(pixel_budget):
        carried_query = _native_joyai_message_query_text(_native_joyai_user_message(messages, 0))
        if not _drop_native_joyai_turn(messages, turn_meta, 0):
            return
        if carried_query and not _native_joyai_history_has_query(messages):
            _inject_native_joyai_query(messages, carried_query)


NATIVE_AURA_SILENCE = "<|silent|>"
NATIVE_AURA_SCORE_PLACEHOLDERS = (NATIVE_AURA_SILENCE, "Focus_Start", "Focus_End")
NATIVE_AURA_MAX_ROUNDS = 45
NATIVE_AURA_ROUNDS_KEEP = 30
NATIVE_AURA_MAX_CONTEXT_QAS = 10
NATIVE_AURA_SYSTEM = (
    "You are receiving a live video stream where the final frame is the present moment. "
    "Respond only when a response is needed based on the user's message or the visual context. "
    "Otherwise, output `<|silent|>` to signify silence."
)
NATIVE_AURA_FOCUS_SYSTEM = """You are receiving a live video stream where the final frame is the present moment. At any moment, there is at most one active question.

Answer the active question as soon as the video seen so far and the prior dialogue contain enough information for a complete, factually grounded answer.

Stream input:
- Each new user turn appends one video chunk of at most 1 second. The chunk is a video, not a list of images.
- The number of frames in a chunk can vary. Do not assume a fixed frame count.
- You can request a temporary higher-frame-rate view when the current sampling rate may miss fine-grained temporal details.

Follow these output rules exactly:

- Focus control:
  1) Proactively monitor the video even when there is no active question. Output the exact token Focus_Start whenever the current scene appears likely to contain an upcoming highlight, rapid motion, a brief event, a transition, an interaction, object manipulation, a subtle gesture, or any detail that could benefit from fine-grained temporal perception. This preserves richer evidence for possible future backward-looking or instant questions.
  2) Also output Focus_Start when an active question requires closer temporal observation before it can be answered reliably. High-frame-rate sampling starts with the next video chunk.
  3) Use a low threshold for starting focus. When uncertain whether higher frame rate may help, prefer Focus_Start; unnecessary focus is acceptable.
  4) While high-frame-rate sampling is active, inspect each new chunk and output the exact token Focus_End once the interesting/high-motion/fine-grained interval has clearly ended and the scene has settled. Normal sampling resumes with the next video chunk. Until then, keep focus active by answering the active question if possible, or outputting <|silent|>.
  5) Focus_Start and Focus_End must each be the entire response. Do not combine either token with an answer, <|silent|>, punctuation, or explanation.
  6) Do not output Focus_Start if high-frame-rate sampling is already active. Do not output Focus_End unless it is active.

- For multiple-choice questions (those that explicitly list options such as A, B, C, D):
  Respond only with the correct option letter (e.g., "A"). Do not add punctuation, explanation, or extra text.

- For all other questions:
  Provide a complete answer using only information available in the stream up to this point.
  If the answer can be inferred with reasonable confidence, answer directly.

- Output the exact token <|silent|> only in one of these cases:
  1) There is no active question and neither Focus_Start nor Focus_End is appropriate.
  2) User explicitly asks to delay answering.
  3) Question unambiguously requires future content and no part can be determined yet."""


def prepare_native_aura_scored_text(text: str) -> str:
    """Map AURA's <|silent|> token onto the benchmark answer text."""
    raw = (text or "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if not raw or raw.startswith(NATIVE_AURA_SILENCE):
        return NATIVE_AURA_SILENCE
    cleaned = raw.replace(NATIVE_AURA_SILENCE, "").strip()
    return cleaned or NATIVE_AURA_SILENCE


def native_aura_ensure_min_frames(video_path: str, min_frames: int = 2) -> str:
    """Give Qwen3-VL the two frames its temporal patch expects.

    A trimmed one-second chunk can contain a single frame. qwen_vl_utils then
    raises ``nframes should in interval [2, 1], but got 0``. AURA duplicates
    that frame before generation; the padded file is only used by this path.
    """
    info = probe_video(video_path)
    video = info.get("video") if isinstance(info, dict) else None
    if not isinstance(video, dict):
        return video_path
    total = int(video.get("total_frames") or 0)
    if total >= min_frames or total <= 0:
        return video_path
    out_path = Path(video_path).with_name(f"{Path(video_path).stem}_aura{min_frames}f.mp4")
    existing = probe_video(str(out_path))
    existing_video = existing.get("video") if isinstance(existing, dict) else None
    if isinstance(existing_video, dict) and int(existing_video.get("total_frames") or 0) >= min_frames:
        return str(out_path)
    duration = float(info.get("duration") or 0.0)
    fps = float(video.get("fps") or 0.0)
    frame_dt = duration / total if duration > 0 else (1.0 / fps if fps > 0 else 0.5)
    pad = frame_dt * (min_frames - total)
    tmp_path = out_path.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"tpad=stop_mode=clone:stop_duration={pad:.6f}",
        "-frames:v",
        str(min_frames),
        "-an",
        "-c:v",
        "libx264",
        "-movflags",
        "+faststart",
        "-loglevel",
        "error",
    ]
    if fps > 0:
        cmd.extend(["-r", str(fps)])
    cmd.append(str(tmp_path))
    try:
        subprocess.run(cmd, check=True)
        tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        return video_path
    repaired = probe_video(str(out_path))
    repaired_video = repaired.get("video") if isinstance(repaired, dict) else None
    if not isinstance(repaired_video, dict) or int(repaired_video.get("total_frames") or 0) < min_frames:
        return video_path
    return str(out_path)


def _native_aura_prune_rounds(
    rounds: List[Dict[str, Any]],
    context_qas: List[List[Dict[str, str]]],
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
    """Match AURA's launch-script window: above 45 rounds, keep the latest 30.

    Moved silent turns are dropped. Spoken turns keep their text only, and at most
    10 of those question-answer groups stay in the text history.
    """
    if len(rounds) <= NATIVE_AURA_MAX_ROUNDS:
        return rounds, context_qas
    moved = rounds[:-NATIVE_AURA_ROUNDS_KEEP]
    remaining = rounds[-NATIVE_AURA_ROUNDS_KEEP:]
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for rnd in moved:
        if rnd.get("has_text") and current:
            groups.append(current)
            current = [rnd]
        else:
            current.append(rnd)
    if current:
        groups.append(current)
    history = list(context_qas)
    for group in groups:
        rewritten: List[Dict[str, str]] = []
        for rnd in group:
            if rnd.get("assistant") == NATIVE_AURA_SILENCE:
                continue
            rewritten.append({"role": "user", "content": str(rnd.get("text") or "")})
            rewritten.append({"role": "assistant", "content": str(rnd.get("assistant") or "")})
        if not rewritten:
            continue
        if group[0].get("has_text"):
            history.append(rewritten)
        elif history:
            history[-1].extend(rewritten)
        else:
            history.append(rewritten)
    if len(history) > NATIVE_AURA_MAX_CONTEXT_QAS:
        history = history[-NATIVE_AURA_MAX_CONTEXT_QAS:]
    return remaining, history


def _native_aura_video_item(content: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in content:
        if item.get("type") == "video":
            return item
    return None


def _native_aura_describe_round(
    content: List[Dict[str, Any]],
    assistant: str,
    question_text: str,
    is_focus: bool,
) -> Dict[str, Any]:
    """Record the frames and pixels of one AURA video turn.

    A high-frame-rate chunk also carries max_frames so Qwen3-VL keeps the frames
    stored in that file. Ordinary 2 fps chunks stay fps-only.
    """
    frames = 0
    height = 0
    width = 0
    video_item = _native_aura_video_item(content)
    if video_item is not None:
        info = probe_video(str(video_item.get("video", ""))) or {}
        video = info.get("video") if isinstance(info, dict) else None
        if isinstance(video, dict):
            frames = int(video.get("total_frames") or 0)
            height = int(video.get("height") or 0)
            width = int(video.get("width") or 0)
        if is_focus and frames > 0:
            video_item["max_frames"] = frames
            video_item["use_max_frames"] = True
    return {
        "content": content,
        "assistant": assistant,
        "text": question_text,
        "has_text": bool(question_text),
        "is_focus": bool(is_focus) and frames > 0,
        "focus_frames": frames if is_focus else 0,
        "video_frames": frames,
        "video_pixels": frames * height * width,
        "video_height": height,
        "video_width": width,
    }


def _native_aura_drop_oldest_video(rounds: List[Dict[str, Any]], focus_only: bool) -> bool:
    for index, rnd in enumerate(rounds):
        if focus_only and not rnd.get("is_focus"):
            continue
        content = list(rnd.get("content") or [])
        video_index = next(
            (item_index for item_index, item in enumerate(content) if item.get("type") == "video"),
            None,
        )
        if video_index is None:
            continue
        del content[video_index]
        rnd["content"] = content
        rnd["is_focus"] = False
        rnd["focus_frames"] = 0
        rnd["video_frames"] = 0
        rnd["video_pixels"] = 0
        if not content:
            del rounds[index]
        return True
    return False


def _native_aura_degrade_oldest_focus(rounds: List[Dict[str, Any]], low_fps: float) -> bool:
    """Sample an old high-fps chunk at the normal fps without rewriting the file."""
    for rnd in rounds:
        if not rnd.get("is_focus"):
            continue
        video_item = _native_aura_video_item(rnd.get("content") or [])
        if video_item is None:
            rnd["is_focus"] = False
            rnd["focus_frames"] = 0
            continue
        duration = float(video_item.get("chunk_duration_seconds") or 0.0)
        if duration <= 0:
            duration = 1.0
        old_frames = int(rnd.get("focus_frames") or rnd.get("video_frames") or 0)
        low_frames = max(2, int(duration * float(low_fps)))
        low_frames = max(2, (low_frames // QWEN3_VL_TEMPORAL_PATCH_SIZE) * QWEN3_VL_TEMPORAL_PATCH_SIZE)
        if old_frames > 0:
            low_frames = min(old_frames, low_frames)
        video_item["fps"] = float(low_fps)
        video_item["max_frames"] = low_frames
        video_item["use_max_frames"] = False
        video_item["low_fps_degenerated"] = True
        height = int(rnd.get("video_height") or 0)
        width = int(rnd.get("video_width") or 0)
        rnd["is_focus"] = False
        rnd["focus_frames"] = 0
        rnd["video_frames"] = low_frames
        rnd["video_pixels"] = low_frames * height * width
        return True
    return False


def _compact_native_aura_rounds(
    rounds: List[Dict[str, Any]],
    pending: Optional[Dict[str, Any]],
    *,
    proactive_focus: bool,
    max_focus_context_frames: int,
    time_compress: bool,
    pixel_budget: Optional[int],
    low_fps_degeneration: bool,
    low_fps: float,
) -> None:
    """Apply the Qwen focus-frame cap and pixel budget to AURA video turns.

    The turns stay AURA video messages. Overflow first samples the oldest
    high-fps chunk back down to the normal fps, then drops the oldest video.
    """
    focus_limit = max(0, int(max_focus_context_frames)) if proactive_focus else 0
    use_pixel_budget = bool(time_compress) and pixel_budget is not None
    if focus_limit <= 0 and not use_pixel_budget:
        return
    inserted = False
    if pending is not None:
        rounds.append(pending)
        inserted = True

    def focus_frames() -> int:
        return sum(int(rnd.get("focus_frames") or 0) for rnd in rounds if rnd.get("is_focus"))

    def total_pixels() -> int:
        return sum(int(rnd.get("video_pixels") or 0) for rnd in rounds)

    try:
        if not proactive_focus:
            while use_pixel_budget and total_pixels() > int(pixel_budget):
                if not _native_aura_drop_oldest_video(rounds, focus_only=False):
                    break
            return
        while True:
            focus_exceeded = focus_limit > 0 and focus_frames() > focus_limit
            pixel_exceeded = use_pixel_budget and total_pixels() > int(pixel_budget)
            if not focus_exceeded and not pixel_exceeded:
                break
            if focus_exceeded and low_fps_degeneration:
                if _native_aura_degrade_oldest_focus(rounds, low_fps):
                    continue
            if not _native_aura_drop_oldest_video(
                rounds,
                focus_only=focus_exceeded and not pixel_exceeded,
            ):
                break
    finally:
        if not inserted:
            return
        if rounds and rounds[-1] is pending:
            rounds.pop()
            return
        pending["content"] = []
        pending["is_focus"] = False
        pending["focus_frames"] = 0
        pending["video_frames"] = 0
        pending["video_pixels"] = 0


def _native_aura_messages(
    context_qas: List[List[Dict[str, str]]],
    rounds: List[Dict[str, Any]],
    new_user_content: Optional[List[Dict[str, Any]]] = None,
    system_prompt: str = NATIVE_AURA_SYSTEM,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]
    for qa in context_qas:
        for item in qa:
            messages.append(
                {"role": item["role"], "content": [{"type": "text", "text": item.get("content", "")}]}
            )
    for rnd in rounds:
        messages.append({"role": "user", "content": rnd["content"]})
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": rnd["assistant"]}]}
        )
    if new_user_content:
        messages.append({"role": "user", "content": new_user_content})
    return messages


class NativeVideoChat3History:
    """Sliding 32-round history used by VideoChat3's streaming demo."""

    def __init__(self, question: str, max_rounds: int = NATIVE_VIDEOCHAT3_MAX_USER_ROUNDS):
        self.question = question
        self.max_rounds = max_rounds
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": NATIVE_VIDEOCHAT3_SYSTEM}]}
        ]
        self.last_answer: Optional[str] = None
        self.window_start_round = 0
        self.question_rounds: List[Tuple[int, str]] = []

    def _user_content(
        self,
        image_items: List[Dict[str, Any]],
        time_start: float,
        time_end: float,
        question_texts: List[str],
    ) -> List[Dict[str, Any]]:
        time_tag = f"<{time_start:g}s-{time_end:g}s>"
        text_parts = [text for text in question_texts if text]
        text_parts.append(time_tag)
        return list(image_items) + [{"type": "text", "text": "\n".join(text_parts)}]

    def _inject_question(self, message: Dict[str, Any], question_text: str) -> None:
        if not question_text:
            return
        content = message.get("content", [])
        for index, item in enumerate(content):
            if item.get("type") != "text":
                continue
            if question_text not in str(item.get("text", "")):
                updated = dict(item)
                updated["text"] = question_text + "\n" + str(updated.get("text", ""))
                message["content"] = list(content)
                message["content"][index] = updated
            break

    def append_turn(
        self,
        image_items: List[Dict[str, Any]],
        round_idx: int,
        time_start: float,
        time_end: float,
        extra_questions: List[str],
    ) -> None:
        question_texts = [text for text in extra_questions if text]
        for question_text in question_texts:
            self.question_rounds.append((round_idx, question_text))

        if not any(message.get("role") == "user" for message in self.messages):
            self.messages.append(
                {
                    "role": "user",
                    "content": self._user_content(
                        image_items,
                        time_start,
                        time_end,
                        question_texts,
                    ),
                }
            )
            self.window_start_round = round_idx
            return

        if self.last_answer is not None:
            self.messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": self.last_answer}]}
            )
        self.messages.append(
            {
                "role": "user",
                "content": self._user_content(image_items, time_start, time_end, question_texts),
            }
        )

        user_indexes = [index for index, message in enumerate(self.messages) if message.get("role") == "user"]
        overflow = len(user_indexes) - self.max_rounds
        if overflow <= 0:
            return
        sys_offset = 1 if self.messages and self.messages[0].get("role") == "system" else 0
        for _ in range(overflow):
            if sys_offset < len(self.messages) and self.messages[sys_offset].get("role") == "user":
                del self.messages[sys_offset]
            if sys_offset < len(self.messages) and self.messages[sys_offset].get("role") == "assistant":
                del self.messages[sys_offset]
        self.window_start_round += overflow
        for message in self.messages:
            if message.get("role") != "user":
                continue
            for question_round, question_text in self.question_rounds:
                if question_round <= self.window_start_round:
                    self._inject_question(message, question_text)
            break


@dataclass
class InferenceOptions:
    sparse_mode: bool
    active_window: int
    max_retries: int
    chunk_seconds: float
    model_video_fps: float
    trim_fps: Optional[float]
    force_focus: bool
    focus_window_seconds: float
    high_fps: Optional[float]
    full_video_high_fps: bool
    max_context_frames: int
    dialog_dump_root: Optional[Path]
    chunk_cache_root: Optional[Path]
    proactive_focus: bool = False
    max_focus_context_frames: int = 90
    resolution_compress: bool = False
    time_compress: bool = False
    also_compress_focus: bool = False
    low_fps_degeneration: bool = False
    visual_token_budget: int = QWEN3_VL_DEFAULT_VISUAL_TOKEN_BUDGET
    pixel_budget: Optional[int] = None
    native_videochat3: bool = False
    native_moss: bool = False
    native_joyai: bool = False
    native_aura: bool = False


class StreamingState:
    def __init__(
        self,
        sample: Dict[str, Any],
        chunk_seconds: float,
        trim_fps: Optional[float],
        model_video_fps: float,
        force_focus: bool,
        focus_window_seconds: float,
        high_fps: Optional[float],
        full_video_high_fps: bool,
        chunk_cache_root: Optional[Path],
    ):
        self.sample_id = str(sample.get("id", "")).strip() or str(sample.get("uuid", "")).strip() or "unknown"
        self.video_uuid = sample.get("uuid", "")
        self.video_path = sample.get("video", "")
        self.events = sorted(sample.get("sqa", []), key=lambda x: time_to_seconds(x["timestamp"]))
        self.events_queue = deque(self.events)
        self.current_time = 0.0
        self.last_event_time = float(time_to_seconds(self.events[-1]["timestamp"])) if self.events else 0.0
        self.duration = float(sample.get("video_info", {}).get("duration", 0.0))
        if self.duration <= 0:
            self.duration = self.last_event_time + 10.0
        self.chunk_seconds = chunk_seconds
        self.trim_fps = trim_fps
        self.default_model_video_fps = float(model_video_fps)
        self.source_video_fps = self._resolve_source_video_fps(sample)
        self.force_focus = bool(force_focus)
        self.focus_window_seconds = max(0.0, float(focus_window_seconds))
        self.high_fps = high_fps
        self.full_video_high_fps = bool(full_video_high_fps)
        self.proactive_focus_active = False
        self.focus_windows = [
            (focus_start, focus_start + self.focus_window_seconds)
            for focus_start in self._resolve_focus_start_seconds(sample)
            if self.focus_window_seconds > 0
        ]
        self.chunk_cache_dir = self._resolve_chunk_cache_dir(sample, chunk_cache_root)
        self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_source_video_fps(self, sample: Dict[str, Any]) -> Optional[float]:
        video_info = sample.get("video_info")
        if not isinstance(video_info, dict):
            return None
        stream_info = video_info.get("video")
        if not isinstance(stream_info, dict):
            return None
        fps = stream_info.get("fps")
        try:
            fps_value = float(fps)
        except Exception:
            return None
        return fps_value if fps_value > 0 else None

    def _resolve_focus_start_seconds(self, sample: Dict[str, Any]) -> List[float]:
        raw_focuses: List[Any] = []
        configured_focuses = sample.get("timestamp_focuses")
        if isinstance(configured_focuses, (list, tuple)):
            raw_focuses.extend(configured_focuses)
        for key in ("timestamp_focus", "target_timestamp", "focus_timestamp"):
            raw = sample.get(key)
            if raw is not None:
                raw_focuses.append(raw)
        verified = sample.get("verified_responses")
        if isinstance(verified, list):
            raw_focuses.extend(
                item.get("timestamp_focus")
                for item in verified
                if isinstance(item, dict) and item.get("timestamp_focus") is not None
            )

        focus_starts: List[float] = []
        for raw in raw_focuses:
            text = str(raw).strip()
            if not text:
                continue
            try:
                focus_start = float(time_to_seconds(text))
            except Exception:
                try:
                    focus_start = float(text)
                except Exception:
                    continue
            if focus_start >= 0 and focus_start not in focus_starts:
                focus_starts.append(focus_start)
        return sorted(focus_starts)

    def _chunk_overlaps_focus_window(self, start_sec: float, end_sec: float) -> bool:
        return self.proactive_focus_active or (
            self.force_focus
            and any(
                start_sec < focus_end and end_sec > focus_start
                for focus_start, focus_end in self.focus_windows
            )
        )

    def set_proactive_focus(self, active: bool) -> None:
        self.proactive_focus_active = bool(active)

    def _resolve_chunk_policy(
        self,
        start_sec: float,
        end_sec: float,
    ) -> Tuple[Optional[float], float, bool, bool]:
        is_focus_chunk = self._chunk_overlaps_focus_window(start_sec, end_sec)
        if self.full_video_high_fps or is_focus_chunk:
            if self.high_fps is not None:
                return self.high_fps, self.high_fps, True, is_focus_chunk
            if self.source_video_fps is not None:
                return None, self.source_video_fps, True, is_focus_chunk
            # If source fps is unavailable, still avoid trim downsampling.
            return None, self.default_model_video_fps, True, is_focus_chunk
        return self.trim_fps, self.default_model_video_fps, False, False

    def _resolve_chunk_cache_dir(self, sample: Dict[str, Any], chunk_cache_root: Optional[Path]) -> Path:
        if chunk_cache_root is not None:
            return chunk_cache_root / sample.get("uuid", "unknown")
        stream_addr = sample.get("stream_addr")
        if stream_addr:
            return Path(stream_addr)
        return SCRIPT_DIR / "cache" / "stream_chunk_cache" / sample.get("uuid", "unknown")

    def _collect_current_events(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        while self.events_queue and time_to_seconds(self.events_queue[0]["timestamp"]) <= self.current_time:
            out.append(self.events_queue.popleft())
        return out

    def _get_or_create_chunk(
        self,
        chunk_path: Path,
        start_time: str,
        end_time: str,
        trim_fps_for_chunk: Optional[float],
    ) -> str:
        if chunk_path.exists():
            info = probe_video(str(chunk_path))
            if isinstance(info, dict) and isinstance(info.get("video"), dict):
                return str(chunk_path)
            try:
                chunk_path.unlink()
            except Exception:
                pass
        try:
            trim_video(
                video_path=self.video_path,
                trim_path=str(chunk_path),
                start_time=start_time,
                end_time=end_time,
                fps=trim_fps_for_chunk,
            )
        except Exception as exc:
            print(f"[WARN] chunk trim failed for {chunk_path}: {exc}")
            return str(chunk_path)
        return str(chunk_path)

    def step(self) -> Dict[str, Any]:
        prev_time = self.current_time
        valid_chunk = prev_time < self.duration
        prev_ts = seconds_to_time(int(prev_time))
        self.current_time += self.chunk_seconds
        current_ts = seconds_to_time(int(self.current_time))
        chunk_start_sec = float(prev_time)
        chunk_end_sec = min(float(self.current_time), float(self.duration))
        chunk_trim_fps, chunk_model_fps, use_max_frames, is_focus_chunk = self._resolve_chunk_policy(
            chunk_start_sec,
            chunk_end_sec,
        )
        # Version downsampled cache names so files produced by the old output
        # `-r` implementation are never silently reused.
        fps_tag = (
            "orig"
            if chunk_trim_fps is None
            else f"{str(chunk_trim_fps).replace('.', 'p')}_selectv1"
        )
        chunk_file = f"video_{self.video_uuid}_{prev_ts}_{current_ts}_trim{fps_tag}.mp4".replace(":", "")

        new_events = self._collect_current_events()
        is_finished = (self.current_time >= int(self.duration)) or (self.current_time >= self.last_event_time + 10.0)
        return {
            "current_timestamp": current_ts,
            "stream_chunk": self._get_or_create_chunk(
                self.chunk_cache_dir / chunk_file,
                prev_ts,
                current_ts,
                chunk_trim_fps,
            )
            if valid_chunk
            else None,
            "chunk_start_seconds": chunk_start_sec,
            "chunk_duration_seconds": max(0.0, chunk_end_sec - chunk_start_sec),
            "sample_id": self.sample_id,
            "chunk_model_fps": chunk_model_fps,
            "use_max_frames": use_max_frames,
            "is_focus_chunk": is_focus_chunk,
            "new_events": new_events,
            "is_finished": is_finished,
        }


def render_user_prompt(event: Dict[str, Any]) -> str:
    prompt = ""
    if event.get("question"):
        prompt += f"Question: {event['question']}\n"
    if event.get("options") is not None:
        prompt += f"Options: {event['options']}\n"
    return prompt


class UnifiedInferenceRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        config_dir: Path,
        model_name: str,
        prompts_name: str,
        model_path_override: str,
        options: InferenceOptions,
        context_window_seconds: Optional[float],
        video_root: Optional[Path],
        stream_addr_root: Optional[Path],
    ):
        self.config = config
        self.config_dir = config_dir
        self.model_name = model_name
        self.prompts_name = prompts_name
        self.options = options
        self.video_root = video_root
        self.stream_addr_root = stream_addr_root

        model_cfg = dict(config["models"][model_name])
        if model_path_override:
            model_cfg["model"] = model_path_override
        if self.options.dialog_dump_root is not None:
            model_cfg["dialog_dump_root"] = str(self.options.dialog_dump_root)
        backend = build_backend(model_name, model_cfg)
        context_window = (
            float(context_window_seconds)
            if context_window_seconds is not None
            else float(model_cfg.get("max_video_length_in_seconds", 60.2))
        )
        self.model_max_new_tokens = int(model_cfg.get("max_new_tokens", 1024))
        self.model = SessionModel(
            backend=backend,
            max_video_length_in_seconds=context_window,
            max_context_frames=self.options.max_context_frames,
            default_video_fps=self.options.model_video_fps,
            max_focus_context_frames=(
                self.options.max_focus_context_frames
                if self.options.proactive_focus or self.options.force_focus
                else 0
            ),
            resolution_compress=self.options.resolution_compress,
            time_compress=self.options.time_compress,
            also_compress_focus=self.options.also_compress_focus,
            low_fps_degeneration=self.options.low_fps_degeneration,
            visual_token_budget=self.options.visual_token_budget,
            pixel_budget=self.options.pixel_budget,
        )

        prompt_cfg_raw = config.get("prompts", {}).get(prompts_name)
        if not isinstance(prompt_cfg_raw, dict):
            raise ValueError(f"Prompt preset '{prompts_name}' not found in config.prompts")
        prompt_cfg = dict(prompt_cfg_raw)
        if "system_prompt" not in prompt_cfg:
            raise ValueError(f"Prompt preset '{prompts_name}' missing 'system_prompt'")
        self.system_prompt_text = str(prompt_cfg["system_prompt"])
        self.silent_word = str(prompt_cfg.get("silent_word", "silent")).strip().lower()
        if self.options.native_videochat3:
            self.system_prompt_text = NATIVE_VIDEOCHAT3_SYSTEM
            self.silent_word = NATIVE_VIDEOCHAT3_SILENCE.strip().lower()
        elif self.options.native_joyai:
            self.system_prompt_text = NATIVE_JOYAI_SYSTEM
            if self.options.proactive_focus:
                self.system_prompt_text = (
                    NATIVE_JOYAI_SYSTEM + "\n\n" + NATIVE_JOYAI_FOCUS_RULES
                )
        elif self.options.native_aura:
            self.system_prompt_text = (
                NATIVE_AURA_FOCUS_SYSTEM if self.options.proactive_focus else NATIVE_AURA_SYSTEM
            )

    def _is_silent_response(self, text: str) -> bool:
        return (text or "").strip().lower() == self.silent_word

    @staticmethod
    def _parse_focus_action(text: str) -> Optional[str]:
        normalized = (text or "").strip().lower()
        if normalized == "focus_start":
            return "start"
        if normalized == "focus_end":
            return "end"
        return None

    def _build_user_content(
        self,
        stream_chunk: Optional[str],
        sample_id: str,
        chunk_start_seconds: float,
        chunk_duration_seconds: float,
        chunk_model_fps: float,
        use_max_frames: bool,
        is_focus_chunk: bool,
        new_events: List[Dict[str, Any]],
        answer_window: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        content: List[Dict[str, Any]] = []
        if stream_chunk:
            content.append(
                {
                    "type": "video",
                    "video": stream_chunk,
                    "fps": float(chunk_model_fps),
                    "use_max_frames": bool(use_max_frames),
                    "is_focus_chunk": bool(is_focus_chunk),
                    "sample_id": str(sample_id).strip(),
                    "chunk_start_seconds": float(chunk_start_seconds),
                    "chunk_duration_seconds": float(chunk_duration_seconds),
                }
            )

        if new_events:
            act_w = -1
            for event in new_events:
                prompt = render_user_prompt(event)
                if prompt:
                    content.append({"type": "text", "text": prompt})
                if event.get("question") and event.get("response"):
                    act_w = max(1, act_w)
                elif event.get("question") or event.get("response"):
                    act_w = max(self.options.active_window, act_w)
            answer_window = max(answer_window, act_w)
        return content, answer_window

    def _generate_once(self, current_time: float) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw = self.model.generate(max_new_tokens=self.model_max_new_tokens)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        response = str(raw.get("response", ""))
        return {
            "timestamp": seconds_to_time(int(current_time)),
            "response": response,
            "raw_response": str(raw.get("raw_response", response)),
            "status_code": int(raw.get("status_code", 200)),
            "runtime_ms": round(runtime_ms, 2),
        }

    def _generate_with_retry(self, current_time: float, retries_left: int) -> Tuple[Dict[str, Any], int]:
        while True:
            try:
                result = self._generate_once(current_time)
            except Exception as exc:
                result = {
                    "timestamp": seconds_to_time(int(current_time)),
                    "response": f"[ERROR] {exc}",
                    "raw_response": f"[ERROR] {exc}",
                    "status_code": 502,
                    "runtime_ms": 0.0,
                }
            if result["status_code"] == 200:
                return result, retries_left
            if retries_left <= 0:
                return result, retries_left
            retries_left -= 1

    def _native_benchmark_answer_window(self, answer_window: int, new_events: List[Dict[str, Any]]) -> int:
        """Match UnifiedInferenceRunner._build_user_content plus the pre-generate decrement."""
        if new_events:
            act_w = -1
            for event in new_events:
                if event.get("question") and event.get("response"):
                    act_w = max(1, act_w)
                elif event.get("question") or event.get("response"):
                    act_w = max(self.options.active_window, act_w)
            answer_window = max(answer_window, act_w)
        return max(-1, answer_window - 1)

    def _run_sample_native_videochat3(self, sample: Dict[str, Any], bench_name: str) -> Dict[str, Any]:
        events = sorted(sample.get("sqa", []) or [], key=lambda event: time_to_seconds(event.get("timestamp", "00:00")))
        sample_id = str(sample.get("id", "")).strip() or str(sample.get("uuid", "")).strip() or "unknown"
        cache_root = self.options.chunk_cache_root
        if cache_root is None:
            stream_addr = str(sample.get("stream_addr") or "").strip()
            cache_root = Path(stream_addr) if stream_addr else (SCRIPT_DIR / "cache" / "native_videochat3")
        frame_dir = cache_root / str(sample.get("uuid", sample_id)) / "native_videochat3_frames"
        source_frames = native_videochat3_extract_frames(
            str(sample.get("video", "")),
            frame_dir / "source",
            NATIVE_VIDEOCHAT3_FPS,
        )
        frames_per_round = max(1, int(round(NATIVE_VIDEOCHAT3_FPS)))
        history = NativeVideoChat3History("")
        responses: List[Dict[str, Any]] = []
        standby_remaining = 0
        resized_dir = frame_dir / "resized"
        duration = float(sample.get("video_info", {}).get("duration", 0.0) or 0.0)
        last_event_time = max((time_to_seconds(event.get("timestamp", "00:00")) for event in events), default=0)
        if duration <= 0:
            duration = float(last_event_time + 10)
        chunk_seconds = float(self.options.chunk_seconds)
        events_queue = deque(events)
        current_time = 0.0
        answer_window = -1
        round_idx = 0

        while True:
            prev_time = current_time
            if prev_time >= duration:
                break
            current_time += chunk_seconds
            raw_start = int(prev_time) * frames_per_round
            raw_end = int(current_time) * frames_per_round
            if raw_start >= len(source_frames):
                break
            raw_frames = source_frames[raw_start:min(raw_end, len(source_frames))]
            if not raw_frames:
                break
            high_res = standby_remaining > 0
            if standby_remaining > 0:
                standby_remaining -= 1
            max_pixels = NATIVE_VIDEOCHAT3_MAX_PIXELS * (4 if high_res else 1)
            image_items: List[Dict[str, Any]] = []
            for frame_offset, src in enumerate(raw_frames):
                dst = resized_dir / f"r{round_idx:04d}_f{frame_offset:02d}_{max_pixels}.jpg"
                if dst.exists():
                    from PIL import Image

                    with Image.open(dst) as existing:
                        resized_width, resized_height = existing.size
                else:
                    resized_height, resized_width = native_videochat3_resize_image(src, dst, max_pixels)
                image_items.append(
                    {
                        "type": "image",
                        "image": str(dst),
                        "sample_id": sample_id,
                        "min_pixels": 28 * 28,
                        "max_pixels": max_pixels,
                        "resized_height": resized_height,
                        "resized_width": resized_width,
                    }
                )
            time_start = float(prev_time)
            time_end = float(current_time)
            new_events: List[Dict[str, Any]] = []
            while events_queue and time_to_seconds(events_queue[0].get("timestamp", "00:00")) <= current_time:
                new_events.append(events_queue.popleft())
            question_texts = [render_user_prompt(event).strip() for event in new_events]
            question_texts = [text for text in question_texts if text]
            history.append_turn(image_items, round_idx, time_start, time_end, question_texts)
            answer_window = self._native_benchmark_answer_window(answer_window, new_events)
            should_generate = (not self.options.sparse_mode) or answer_window >= 0
            if should_generate:
                t0 = time.perf_counter()
                generated = self.model.backend.generate(
                    history.messages,
                    max_new_tokens=NATIVE_VIDEOCHAT3_MAX_NEW_TOKENS,
                    temperature=NATIVE_VIDEOCHAT3_TEMPERATURE,
                )
                answer = str(generated.get("response", ""))
                runtime_ms = (time.perf_counter() - t0) * 1000.0
                history.last_answer = answer
                if NATIVE_VIDEOCHAT3_STANDBY in answer:
                    standby_remaining = 1
                status_code = int(generated.get("status_code", 200))
                responses.append(
                    {
                        "timestamp": seconds_to_time(int(current_time)),
                        "response": answer,
                        "raw_response": str(generated.get("raw_response", answer)),
                        "status_code": status_code,
                        "runtime_ms": round(runtime_ms, 2),
                    }
                )
                if status_code != 200:
                    break
            else:
                history.last_answer = None
            finished = current_time >= int(duration) or (events and current_time >= last_event_time + 10.0)
            if finished or (not events_queue and answer_window < 0):
                break
            round_idx += 1
        return sample | {"responses": responses, "bench": bench_name}

    def _run_sample_native_moss(self, sample: Dict[str, Any], bench_name: str) -> Dict[str, Any]:
        """Keep the 1-second chunk clock, and feed each chunk as native MOSS frames.

        Question timestamps, sparse windows, and response timestamps stay on the same
        ``StreamingState`` clock as Qwen / VideoChat. Only the pixels change: each chunk
        is decoded with MOSS-VL's fps timestamp sampler and pushed into one realtime session.
        """
        streaming = StreamingState(
            sample=sample,
            chunk_seconds=self.options.chunk_seconds,
            trim_fps=self.options.trim_fps,
            model_video_fps=self.options.model_video_fps,
            force_focus=self.options.force_focus,
            focus_window_seconds=self.options.focus_window_seconds,
            high_fps=self.options.high_fps,
            full_video_high_fps=self.options.full_video_high_fps,
            chunk_cache_root=self.options.chunk_cache_root,
        )
        sample_id = str(sample.get("id", "")).strip() or str(sample.get("uuid", "")).strip() or "unknown"
        frame_root = streaming.chunk_cache_dir / "native_moss_frames"
        responses: List[Dict[str, Any]] = []
        answer_window = -1
        retries_left = int(self.options.max_retries)
        reset_session = True
        pending_frames: List[Dict[str, Any]] = []
        pending_prompts: List[str] = []
        while True:
            step = streaming.step()
            has_chunk = bool(step.get("stream_chunk"))
            new_events = step.get("new_events") or []
            if has_chunk or new_events:
                if has_chunk:
                    chunk_start = float(step.get("chunk_start_seconds", 0.0))
                    chunk_end = chunk_start + float(
                        step.get("chunk_duration_seconds", self.options.chunk_seconds)
                    )
                    extracted = native_moss_extract_chunk_frames(
                        video_path=streaming.video_path,
                        start=chunk_start,
                        end=chunk_end,
                        sample_fps=float(step["chunk_model_fps"]),
                        out_dir=frame_root,
                    )
                    pending_frames.extend(
                        {"image": image_path, "timestamp": timestamp}
                        for image_path, timestamp in extracted
                    )
                prompts = [render_user_prompt(event).strip() for event in new_events]
                pending_prompts.extend(text for text in prompts if text)
                answer_window = self._native_benchmark_answer_window(answer_window, new_events)
                should_generate = (
                    (not self.options.sparse_mode)
                    or answer_window >= 0
                    or self.options.proactive_focus
                )
                if should_generate and (pending_frames or pending_prompts):
                    message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "moss_realtime",
                                "sample_id": sample_id,
                                "reset": reset_session,
                                "system_prompt": self.system_prompt_text,
                                "frames": list(pending_frames),
                                "prompts": list(pending_prompts),
                            }
                        ],
                    }
                    record, retries_left = self._generate_native_moss_step(
                        message,
                        float(streaming.current_time),
                        retries_left,
                    )
                    if record["status_code"] != 200:
                        responses.append(record)
                        break
                    reset_session = False
                    pending_frames.clear()
                    pending_prompts.clear()
                    focus_action = (
                        self._parse_focus_action(record["response"])
                        if self.options.proactive_focus
                        else None
                    )
                    if focus_action is not None:
                        record["focus_action"] = focus_action
                    responses.append(record)
                    if focus_action == "start":
                        streaming.set_proactive_focus(True)
                    elif focus_action == "end":
                        streaming.set_proactive_focus(False)
            if step["is_finished"]:
                break
        return sample | {"responses": responses, "bench": bench_name}

    def _generate_native_moss_step(
        self,
        message: Dict[str, Any],
        current_time: float,
        retries_left: int,
    ) -> Tuple[Dict[str, Any], int]:
        while True:
            t0 = time.perf_counter()
            try:
                generated = self.model.backend.generate(
                    [message],
                    max_new_tokens=self.model_max_new_tokens,
                )
                runtime_ms = (time.perf_counter() - t0) * 1000.0
                answer = str(generated.get("response", ""))
                result = {
                    "timestamp": seconds_to_time(int(current_time)),
                    "response": prepare_native_moss_scored_text(answer),
                    "raw_response": str(generated.get("raw_response", answer)),
                    "status_code": int(generated.get("status_code", 200)),
                    "runtime_ms": round(runtime_ms, 2),
                }
            except Exception as exc:
                result = {
                    "timestamp": seconds_to_time(int(current_time)),
                    "response": f"[ERROR] {exc}",
                    "raw_response": f"[ERROR] {exc}",
                    "status_code": 502,
                    "runtime_ms": 0.0,
                }
            if result["status_code"] == 200:
                return result, retries_left
            if retries_left <= 0:
                return result, retries_left
            retries_left -= 1
            message["content"][0]["reset"] = True

    def _compact_native_joyai_history(
        self,
        messages: List[Dict[str, Any]],
        turn_meta: List[Dict[str, Any]],
        video_path: str,
        frame_root: Path,
    ) -> None:
        """Apply the same focus-frame and pixel budgets used by the video-chunk path.

        Without proactive focus, --time-compress --pixel-budget still drops the oldest
        turns. Focus-frame limits and low-FPS degeneration stay on the proactive path.
        """
        if not self.options.proactive_focus:
            if self.options.time_compress and self.options.pixel_budget is not None:
                _evict_native_joyai_turns_over_pixel_budget(
                    messages,
                    turn_meta,
                    int(self.options.pixel_budget),
                )
            return
        max_focus = int(self.options.max_focus_context_frames)
        pixel_budget = self.options.pixel_budget

        def focus_frames() -> int:
            return sum(int(meta.get("n_frames", 0) or 0) for meta in turn_meta if meta.get("is_focus"))

        def total_pixels() -> int:
            return sum(int(meta.get("pixels", 0) or 0) for meta in turn_meta)

        while turn_meta:
            focus_exceeded = max_focus > 0 and focus_frames() > max_focus
            pixel_exceeded = (
                self.options.time_compress
                and pixel_budget is not None
                and total_pixels() > int(pixel_budget)
            )
            if not focus_exceeded and not pixel_exceeded:
                return
            if focus_exceeded and self.options.low_fps_degeneration:
                idx = next((i for i, meta in enumerate(turn_meta) if meta.get("is_focus")), None)
                if idx is None:
                    return
                meta = turn_meta[idx]
                low_fps = float(self.options.model_video_fps)
                if float(meta.get("sample_fps", low_fps)) <= low_fps + 1e-6:
                    meta["is_focus"] = False
                    continue
                frames = native_joyai_extract_chunk_frames(
                    video_path=video_path,
                    start=float(meta["start"]),
                    end=float(meta["end"]),
                    sample_fps=low_fps,
                    out_dir=frame_root,
                )
                message = _native_joyai_user_message(messages, idx)
                if message is None or not frames:
                    meta["is_focus"] = False
                    continue
                sample_id = ""
                for item in message.get("content", []):
                    if item.get("sample_id"):
                        sample_id = str(item["sample_id"])
                        break
                _replace_native_joyai_images(message, frames, sample_id)
                meta["n_frames"] = len(frames)
                meta["pixels"] = _native_joyai_frame_pixels(frames)
                meta["sample_fps"] = low_fps
                meta["is_focus"] = False
                continue
            drop_index = 0
            if focus_exceeded and not pixel_exceeded:
                focus_index = next((i for i, meta in enumerate(turn_meta) if meta.get("is_focus")), None)
                if focus_index is None:
                    return
                drop_index = focus_index
            if not _drop_native_joyai_turn(messages, turn_meta, drop_index):
                return

    def _run_sample_native_joyai(self, sample: Dict[str, Any], bench_name: str) -> Dict[str, Any]:
        """Keep the 1-second chunk clock, and encode each chunk as JoyAI frames.

        Question timestamps, sparse windows, and response timestamps stay on the same
        StreamingState clock as Qwen / VideoChat. Each second is one JoyAI turn:
        ``<T seconds>`` plus the chunk's frames. Before a question, and outside the
        sparse window, the turn is recorded as ``</silence>`` without a model call.
        """
        streaming = StreamingState(
            sample=sample,
            chunk_seconds=self.options.chunk_seconds,
            trim_fps=self.options.trim_fps,
            model_video_fps=self.options.model_video_fps,
            force_focus=self.options.force_focus,
            focus_window_seconds=self.options.focus_window_seconds,
            high_fps=self.options.high_fps,
            full_video_high_fps=self.options.full_video_high_fps,
            chunk_cache_root=self.options.chunk_cache_root,
        )
        sample_id = str(sample.get("id", "")).strip() or str(sample.get("uuid", "")).strip() or "unknown"
        frame_root = streaming.chunk_cache_dir / "native_joyai_frames"
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt_text}]}
        ]
        turn_meta: List[Dict[str, Any]] = []
        responses: List[Dict[str, Any]] = []
        answer_window = -1
        retries_left = int(self.options.max_retries)
        current_query = ""
        reinject_query = False
        while True:
            step = streaming.step()
            has_chunk = bool(step.get("stream_chunk"))
            new_events = step.get("new_events") or []
            if has_chunk or new_events:
                frames: List[Tuple[str, float]] = []
                chunk_start = float(step.get("chunk_start_seconds", streaming.current_time))
                if has_chunk:
                    chunk_end = chunk_start + float(
                        step.get("chunk_duration_seconds", self.options.chunk_seconds)
                    )
                    frames = native_joyai_extract_chunk_frames(
                        video_path=streaming.video_path,
                        start=chunk_start,
                        end=chunk_end,
                        sample_fps=float(step["chunk_model_fps"]),
                        out_dir=frame_root,
                    )
                prompts = [render_user_prompt(event).strip() for event in new_events]
                prompts = [text for text in prompts if text]
                introduced_query = ""
                if prompts:
                    introduced_query = "\n".join(prompts)
                    current_query = introduced_query
                    reinject_query = False
                answer_window = self._native_benchmark_answer_window(answer_window, new_events)
                user_turns = sum(1 for message in messages if message.get("role") == "user")
                if user_turns >= NATIVE_JOYAI_CHUNK_TURNS:
                    messages = [messages[0]]
                    turn_meta.clear()
                    reinject_query = bool(current_query) and not introduced_query
                query_text = introduced_query or (current_query if reinject_query else "")
                if query_text:
                    reinject_query = False
                if frames or query_text:
                    content: List[Dict[str, Any]] = []
                    if query_text:
                        content.append(
                            {
                                "type": "text",
                                "text": NATIVE_JOYAI_QUERY_HEADER + "\n" + query_text,
                            }
                        )
                    content.append({"type": "text", "text": f"<{chunk_start:.1f} seconds>"})
                    for offset, (image_path, _timestamp) in enumerate(frames):
                        image_item: Dict[str, Any] = {
                            "type": "image",
                            "image": image_path,
                            "max_pixels": NATIVE_JOYAI_MAX_PIXELS,
                        }
                        if offset == 0:
                            image_item["sample_id"] = sample_id
                        content.append(image_item)
                    messages.append({"role": "user", "content": content})
                    turn_meta.append(
                        {
                            "n_frames": len(frames),
                            "is_focus": bool(step.get("is_focus_chunk")),
                            "pixels": _native_joyai_frame_pixels(frames),
                            "start": chunk_start,
                            "end": chunk_start
                            + float(step.get("chunk_duration_seconds", self.options.chunk_seconds)),
                            "sample_fps": float(step.get("chunk_model_fps", self.options.model_video_fps)),
                        }
                    )
                    self._compact_native_joyai_history(
                        messages,
                        turn_meta,
                        video_path=streaming.video_path,
                        frame_root=frame_root,
                    )
                    should_generate = (
                        (not self.options.sparse_mode)
                        or answer_window >= 0
                        or self.options.proactive_focus
                    )
                    # Proactive focus has to see every second, including the
                    # seconds before the first question, so it can emit Focus_Start.
                    has_user = any(message.get("role") == "user" for message in messages)
                    if has_user and should_generate and (current_query or self.options.proactive_focus):
                        record, retries_left = self._generate_native_joyai_step(
                            messages,
                            float(streaming.current_time),
                            retries_left,
                        )
                        if record["status_code"] != 200:
                            record.pop("_history_text", None)
                            responses.append(record)
                            break
                        focus_action = (
                            self._parse_focus_action(record["response"])
                            if self.options.proactive_focus
                            else None
                        )
                        history_text = str(record.pop("_history_text"))
                        if focus_action == "start":
                            history_text = "Focus_Start"
                        elif focus_action == "end":
                            history_text = "Focus_End"
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": history_text}],
                            }
                        )
                        if focus_action is not None:
                            record["focus_action"] = focus_action
                        responses.append(record)
                        if focus_action == "start":
                            streaming.set_proactive_focus(True)
                        elif focus_action == "end":
                            streaming.set_proactive_focus(False)
                    else:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": NATIVE_JOYAI_SILENCE}],
                            }
                        )
            if step["is_finished"]:
                break
        return sample | {"responses": responses, "bench": bench_name}

    def _generate_native_joyai_step(
        self,
        messages: List[Dict[str, Any]],
        current_time: float,
        retries_left: int,
    ) -> Tuple[Dict[str, Any], int]:
        while True:
            t0 = time.perf_counter()
            try:
                generated = self.model.backend.generate(
                    messages,
                    max_new_tokens=self.model_max_new_tokens,
                    skip_special_tokens=False,
                )
                runtime_ms = (time.perf_counter() - t0) * 1000.0
                answer = str(generated.get("response", ""))
                history_text = normalize_native_joyai_output(answer)
                result = {
                    "timestamp": seconds_to_time(int(current_time)),
                    "response": prepare_native_joyai_scored_text(history_text),
                    "raw_response": str(generated.get("raw_response", answer)),
                    "_history_text": history_text,
                    "status_code": int(generated.get("status_code", 200)),
                    "runtime_ms": round(runtime_ms, 2),
                }
            except Exception as exc:
                result = {
                    "timestamp": seconds_to_time(int(current_time)),
                    "response": f"[ERROR] {exc}",
                    "raw_response": f"[ERROR] {exc}",
                    "_history_text": NATIVE_JOYAI_SILENCE,
                    "status_code": 502,
                    "runtime_ms": 0.0,
                }
            if result["status_code"] == 200:
                return result, retries_left
            if retries_left <= 0:
                return result, retries_left
            retries_left -= 1

    def _run_sample_native_aura(self, sample: Dict[str, Any], bench_name: str) -> Dict[str, Any]:
        """Keep the 1-second chunk clock, and send each chunk as an AURA video turn.

        Question timestamps, sparse windows, and response timestamps stay on the same
        StreamingState clock as Qwen / VideoChat. Each second is one user video at that
        chunk's fps. Outside the sparse window the turn is recorded as <|silent|>
        without a model call. A question is attached only to the second it arrives.
        Proactive focus keeps that video-turn format: Focus_Start makes the next chunk
        use the source frame rate, and the focus-frame / pixel budget trims old videos.
        """
        streaming = StreamingState(
            sample=sample,
            chunk_seconds=self.options.chunk_seconds,
            trim_fps=self.options.trim_fps,
            model_video_fps=self.options.model_video_fps,
            force_focus=self.options.force_focus,
            focus_window_seconds=self.options.focus_window_seconds,
            high_fps=self.options.high_fps,
            full_video_high_fps=self.options.full_video_high_fps,
            chunk_cache_root=self.options.chunk_cache_root,
        )
        sample_id = str(sample.get("id", "")).strip() or str(sample.get("uuid", "")).strip() or "unknown"
        responses: List[Dict[str, Any]] = []
        answer_window = -1
        retries_left = int(self.options.max_retries)
        rounds: List[Dict[str, Any]] = []
        context_qas: List[List[Dict[str, str]]] = []
        while True:
            step = streaming.step()
            has_chunk = bool(step.get("stream_chunk"))
            new_events = step.get("new_events") or []
            if has_chunk or new_events:
                prompts = [render_user_prompt(event).strip() for event in new_events]
                question_text = "\n".join(text for text in prompts if text)
                answer_window = self._native_benchmark_answer_window(answer_window, new_events)
                content: List[Dict[str, Any]] = []
                if has_chunk:
                    content.append(
                        {
                            "type": "video",
                            "video": native_aura_ensure_min_frames(str(step["stream_chunk"])),
                            "fps": float(step["chunk_model_fps"]),
                            "sample_id": sample_id,
                            "chunk_start_seconds": float(step.get("chunk_start_seconds", 0.0)),
                            "chunk_duration_seconds": float(
                                step.get("chunk_duration_seconds", self.options.chunk_seconds)
                            ),
                        }
                    )
                if question_text:
                    content.append({"type": "text", "text": question_text})
                if content:
                    pending = _native_aura_describe_round(
                        content,
                        "",
                        question_text,
                        bool(step.get("is_focus_chunk")),
                    )
                    _compact_native_aura_rounds(
                        rounds,
                        pending,
                        proactive_focus=self.options.proactive_focus,
                        max_focus_context_frames=self.options.max_focus_context_frames,
                        time_compress=self.options.time_compress,
                        pixel_budget=self.options.pixel_budget,
                        low_fps_degeneration=self.options.low_fps_degeneration,
                        low_fps=float(self.options.model_video_fps),
                    )
                    if pending["content"]:
                        should_generate = (
                            (not self.options.sparse_mode)
                            or answer_window >= 0
                            or self.options.proactive_focus
                        )
                        if should_generate:
                            messages = _native_aura_messages(
                                context_qas,
                                rounds,
                                pending["content"],
                                system_prompt=self.system_prompt_text,
                            )
                            record, retries_left = self._generate_native_aura_step(
                                messages,
                                float(streaming.current_time),
                                retries_left,
                            )
                            history_text = str(record.pop("_history_text", NATIVE_AURA_SILENCE))
                            if record["status_code"] != 200:
                                responses.append(record)
                                break
                            focus_action = (
                                self._parse_focus_action(record["response"])
                                if self.options.proactive_focus
                                else None
                            )
                            if focus_action == "start":
                                history_text = "Focus_Start"
                            elif focus_action == "end":
                                history_text = "Focus_End"
                            pending["assistant"] = history_text
                            rounds.append(pending)
                            rounds, context_qas = _native_aura_prune_rounds(rounds, context_qas)
                            if focus_action is not None:
                                record["focus_action"] = focus_action
                            responses.append(record)
                            if focus_action == "start":
                                streaming.set_proactive_focus(True)
                            elif focus_action == "end":
                                streaming.set_proactive_focus(False)
                        else:
                            pending["assistant"] = NATIVE_AURA_SILENCE
                            rounds.append(pending)
                            rounds, context_qas = _native_aura_prune_rounds(rounds, context_qas)
            if step["is_finished"]:
                break
        return sample | {"responses": responses, "bench": bench_name}

    def _generate_native_aura_step(
        self,
        messages: List[Dict[str, Any]],
        current_time: float,
        retries_left: int,
    ) -> Tuple[Dict[str, Any], int]:
        while True:
            t0 = time.perf_counter()
            try:
                generated = self.model.backend.generate(
                    messages,
                    max_new_tokens=self.model_max_new_tokens,
                    skip_special_tokens=False,
                )
                runtime_ms = (time.perf_counter() - t0) * 1000.0
                answer = str(generated.get("response", ""))
                history_text = prepare_native_aura_scored_text(answer)
                result = {
                    "timestamp": seconds_to_time(int(current_time)),
                    "response": history_text,
                    "raw_response": str(generated.get("raw_response", answer)),
                    "_history_text": history_text,
                    "status_code": int(generated.get("status_code", 200)),
                    "runtime_ms": round(runtime_ms, 2),
                }
            except Exception as exc:
                result = {
                    "timestamp": seconds_to_time(int(current_time)),
                    "response": f"[ERROR] {exc}",
                    "raw_response": f"[ERROR] {exc}",
                    "_history_text": NATIVE_AURA_SILENCE,
                    "status_code": 502,
                    "runtime_ms": 0.0,
                }
            if result["status_code"] == 200:
                return result, retries_left
            if retries_left <= 0:
                return result, retries_left
            retries_left -= 1

    def run_sample(self, sample: Dict[str, Any], bench_name: str) -> Dict[str, Any]:
        if self.options.native_videochat3:
            return self._run_sample_native_videochat3(sample, bench_name)
        if self.options.native_moss:
            return self._run_sample_native_moss(sample, bench_name)
        if self.options.native_joyai:
            return self._run_sample_native_joyai(sample, bench_name)
        if self.options.native_aura:
            return self._run_sample_native_aura(sample, bench_name)
        streaming = StreamingState(
            sample=sample,
            chunk_seconds=self.options.chunk_seconds,
            trim_fps=self.options.trim_fps,
            model_video_fps=self.options.model_video_fps,
            force_focus=self.options.force_focus,
            focus_window_seconds=self.options.focus_window_seconds,
            high_fps=self.options.high_fps,
            full_video_high_fps=self.options.full_video_high_fps,
            chunk_cache_root=self.options.chunk_cache_root,
        )
        sys_msg = {"role": "system", "content": [{"type": "text", "text": self.system_prompt_text}]}
        self.model.new_session(sys_msg)

        responses: List[Dict[str, Any]] = []
        answer_window = -1
        retries_left = int(self.options.max_retries)
        while True:
            step = streaming.step()
            content, answer_window = self._build_user_content(
                stream_chunk=step["stream_chunk"],
                sample_id=str(step.get("sample_id", "")).strip(),
                chunk_start_seconds=float(step.get("chunk_start_seconds", 0.0)),
                chunk_duration_seconds=float(step.get("chunk_duration_seconds", self.options.chunk_seconds)),
                chunk_model_fps=float(step["chunk_model_fps"]),
                use_max_frames=bool(step["use_max_frames"]),
                is_focus_chunk=bool(step["is_focus_chunk"]),
                new_events=step["new_events"],
                answer_window=answer_window,
            )
            if content:
                self.model.add_chunk({"role": "user", "content": content})
                answer_window = max(-1, answer_window - 1)
                if (
                    (not self.options.sparse_mode)
                    or (answer_window >= 0)
                    # Proactive focus must inspect every chunk so it can start
                    # before a question or other trigger event arrives.
                    or self.options.proactive_focus
                ):
                    record, retries_left = self._generate_with_retry(streaming.current_time, retries_left)
                    focus_action = (
                        self._parse_focus_action(record["response"])
                        if self.options.proactive_focus
                        else None
                    )
                    if focus_action is not None:
                        record["focus_action"] = focus_action
                    responses.append(record)
                    if record["status_code"] != 200:
                        break
                    if focus_action == "start":
                        streaming.set_proactive_focus(True)
                    elif focus_action == "end":
                        streaming.set_proactive_focus(False)
                    if not self._is_silent_response(record["response"]):
                        self.model.add_chunk(
                            {"role": "assistant", "content": [{"type": "text", "text": record["response"]}]}
                        )
            if step["is_finished"]:
                break
        return sample | {"responses": responses, "bench": bench_name}

    def benchmark_path(self, benchmark_name_or_path: str) -> Path:
        bench_cfg = self.config.get("benchmarks", {}).get(benchmark_name_or_path, {})
        if isinstance(bench_cfg, dict) and "path" in bench_cfg:
            return resolve_path(str(bench_cfg["path"]), [self.config_dir, SCRIPT_DIR, Path.cwd()])
        return resolve_path(benchmark_name_or_path, [Path.cwd(), self.config_dir, SCRIPT_DIR])

    def run_benchmarks(self, benchmark_list: List[str], output_dir: Path) -> List[Dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_records: List[Dict[str, Any]] = []
        for bench in benchmark_list:
            bench_path = self.benchmark_path(bench)
            samples = load_samples_any_format(
                benchmark_path=bench_path,
                bench_name=bench,
                video_root=self.video_root,
                stream_addr_root=self.stream_addr_root,
                need_video_info=True,
            )
            print(f"\n[Inference] benchmark={bench} path={bench_path} samples={len(samples)}")
            for sample in tqdm(samples, desc=bench):
                record = self.run_sample(sample, bench)
                append_jsonl(output_dir / f"{bench}.jsonl", record)
                all_records.append(record)
        return all_records


def parse_logical_questions(raw_sqa: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logical: List[Dict[str, Any]] = []
    i = 0
    while i < len(raw_sqa):
        item = raw_sqa[i]
        if "question" in item and "response" in item:
            t_question = time_to_seconds(item["timestamp"])
            task_type = normalize_task_type_preserving_legacy(
                item.get("type", item.get("task_type", "DefaultType")),
                default="DefaultType",
            )
            logical.append(
                {
                    "question_time_sec": t_question,
                    "answer_event_time_sec": t_question,
                    "question": item["question"],
                    "ground_truth": item["response"],
                    "is_objective": "options" in item,
                    "options": item.get("options"),
                    "task_type": task_type,
                }
            )
            i += 1
        elif "question" in item and "response" not in item:
            t_question = time_to_seconds(item["timestamp"])
            t_answer_event = t_question
            ground_truth = ""
            task_type = normalize_task_type_preserving_legacy(
                item.get("type", item.get("task_type", "DefaultType")),
                default="DefaultType",
            )
            if i + 1 < len(raw_sqa):
                next_item = raw_sqa[i + 1]
                if "response" in next_item and "question" not in next_item:
                    t_answer_event = time_to_seconds(next_item["timestamp"])
                    ground_truth = next_item["response"]
                    i += 2
                else:
                    i += 1
            else:
                i += 1
            logical.append(
                {
                    "question_time_sec": t_question,
                    "answer_event_time_sec": t_answer_event,
                    "question": item["question"],
                    "ground_truth": ground_truth,
                    "is_objective": "options" in item,
                    "options": item.get("options"),
                    "task_type": task_type,
                }
            )
        else:
            i += 1
    return logical


class LLMJudger:
    def __init__(self, backend: Any, prompt_template: str):
        self.backend = backend
        self.prompt_template = prompt_template

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        if repair_json is not None:
            parsed = json.loads(repair_json(raw))
        else:
            parsed = json.loads(raw)
        score = max(0.0, min(5.0, float(parsed["score"])))
        explanation = str(parsed.get("explanation", ""))
        return {"score": score, "explanation": explanation}

    def judge(self, question: str, model_output: str, reference: str, retries: int = 5) -> Dict[str, Any]:
        import time as _time
        prompt = self.prompt_template.format(
            question=question,
            model_output=model_output,
            reference_answer=reference,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        last_err = None
        for attempt in range(retries):
            if attempt > 0:
                _time.sleep(min(2 ** attempt, 16))
            result = self.backend.generate(messages, max_new_tokens=256)
            if int(result.get("status_code", 500)) != 200:
                last_err = result.get("response", "")
                continue
            try:
                return self._parse_response(str(result.get("response", "")).strip())
            except Exception as exc:
                last_err = exc
                continue
        return {"score": 0.0, "explanation": f"Judger parse error: {last_err}"}


def load_llm_judger(config: Dict[str, Any]) -> Optional[LLMJudger]:
    judger_cfg = config.get("judger")
    if not isinstance(judger_cfg, dict):
        return None
    backend = build_backend("judger", judger_cfg)
    if "prompt_template" not in judger_cfg:
        raise ValueError("config.judger.prompt_template is required")
    prompt_template = str(judger_cfg["prompt_template"])
    return LLMJudger(backend, prompt_template)


def evaluate_sample(
    sample: Dict[str, Any],
    llm_judger: Optional[LLMJudger],
    time_window: float,
    penalize_early_response: bool = True,
    early_window_seconds: float = 2.0,
    native_videochat3: bool = False,
    native_moss: bool = False,
    native_joyai: bool = False,
    native_aura: bool = False,
) -> List[Dict[str, Any]]:
    if early_window_seconds < 0:
        raise ValueError("early_window_seconds must be non-negative")
    logical_questions = parse_logical_questions(sample.get("sqa", []))
    if not logical_questions:
        return []
    model_responses: List[Tuple[int, str]] = []
    for response in sample.get("responses", []):
        t = time_to_seconds(response["timestamp"])
        model_responses.append((t, response.get("response", "")))
    model_responses.sort(key=lambda x: x[0])

    used_indices = set()
    results: List[Dict[str, Any]] = []
    for q in logical_questions:
        t_question = q["question_time_sec"]
        t_answer = q["answer_event_time_sec"]
        ground_truth = q["ground_truth"]

        window_start = t_question
        window_end = t_answer + time_window
        correct_time_end = t_answer if t_question == t_answer else (t_answer + time_window)
        if native_videochat3:
            score_placeholders = NATIVE_VIDEOCHAT3_SCORE_PLACEHOLDERS
        elif native_moss:
            score_placeholders = NATIVE_MOSS_SCORE_PLACEHOLDERS
        elif native_joyai:
            score_placeholders = NATIVE_JOYAI_SCORE_PLACEHOLDERS
        elif native_aura:
            score_placeholders = NATIVE_AURA_SCORE_PLACEHOLDERS
        else:
            score_placeholders = None
        for idx, (t_resp, model_text) in enumerate(model_responses):
            if idx in used_indices:
                continue
            if not (window_start <= t_resp <= window_end):
                continue
            if native_videochat3:
                scored_text = prepare_native_videochat3_scored_text(model_text)
            elif native_moss:
                scored_text = prepare_native_moss_scored_text(model_text)
            elif native_joyai:
                scored_text = prepare_native_joyai_scored_text(model_text)
            elif native_aura:
                scored_text = prepare_native_aura_scored_text(model_text)
            else:
                scored_text = model_text
            if scored_text == ground_truth or (not is_placeholder(scored_text, score_placeholders)):
                if (
                    penalize_early_response
                    and is_forward_task(q["task_type"])
                    and t_resp == t_question
                    and t_resp < t_answer - early_window_seconds
                ):
                    continue
                q["model_response_time_sec"] = t_resp
                q["model_response_content"] = scored_text
                used_indices.add(idx)
                break

        t_model = q.get("model_response_time_sec", t_answer)
        c_model = q.get("model_response_content", "")
        explanation = ""
        if t_model < t_answer - early_window_seconds and penalize_early_response:
            score_100 = 0.0
            category = "EarlyResponse"
        elif c_model != ground_truth and is_placeholder(c_model, score_placeholders):
            score_100 = 0.0
            category = "NoResponse"
        elif t_model > correct_time_end:
            score_100 = 0.0
            category = "LateResponse"
        elif q["is_objective"]:
            clean_up = lambda x: x.strip().replace(".", "")[:1]
            if c_model.lower() == ground_truth.lower() or clean_up(c_model).lower() == ground_truth.lower():
                score_100 = 100.0
                category = "Correct"
            else:
                score_100 = 0.0
                category = "WrongAnswer"
        else:
            if llm_judger is None:
                score_100 = 0.0
                category = "Error (no LLM)"
            else:
                judged = llm_judger.judge(q["question"], c_model, ground_truth)
                score_100 = judged["score"] * 20.0
                explanation = judged.get("explanation", "")
                category = "PartlyCorrect"

        results.append(
            {
                "sample_id": sample["id"],
                "question_time": seconds_to_time(q["question_time_sec"]),
                "question": q["question"],
                "answer_time": seconds_to_time(q["answer_event_time_sec"]),
                "answer": ground_truth,
                "response_time": seconds_to_time(int(t_model)),
                "response": c_model,
                "score": score_100,
                "category": category,
                "task_type": q["task_type"],
                "is_objective": q["is_objective"],
                "explanation": explanation,
            }
        )
    return results


def build_summary(df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    total = len(df)
    if total == 0:
        return {"model_name": model_name, "#samples": 0, "final_score": 0.0}
    summary: Dict[str, Any] = {"model_name": model_name, "#samples": total, "final_score": df["score"].mean()}
    for kind, mask in [("objective", df["is_objective"]), ("subjective", ~df["is_objective"])]:
        subset = df[mask]
        summary[kind] = round(subset["score"].mean() if len(subset) else 0.0, 1)
    task_types = df["task_type"].dropna().unique()
    for task_type in task_types:
        subset_obj = df[df["is_objective"] & (df["task_type"] == task_type)]
        subset_sub = df[(~df["is_objective"]) & (df["task_type"] == task_type)]
        summary[f"{task_type}(objective)"] = round(subset_obj["score"].mean() if len(subset_obj) else 0.0, 1)
        summary[f"{task_type}(subjective)"] = round(subset_sub["score"].mean() if len(subset_sub) else 0.0, 1)

    categories = df["category"].unique()
    forward_mask = df["task_type"].astype(str).str.lower().isin(FORWARD_TASK_TYPES)
    forward_obj = df[forward_mask & (df["is_objective"])]
    forward_sub = df[forward_mask & (~df["is_objective"])]
    for cat in categories:
        subset = forward_obj[forward_obj["category"] == cat]
        percent = round(len(subset) / len(forward_obj) * 100, 1) if len(forward_obj) else 0.0
        score = round(subset["score"].mean(), 1) if len(subset) else 0.0
        summary[f"{cat}(objective-future)"] = f"{percent}%({score})"
    for cat in categories:
        subset = forward_sub[forward_sub["category"] == cat]
        percent = round(len(subset) / len(forward_sub) * 100, 1) if len(forward_sub) else 0.0
        score = round(subset["score"].mean(), 1) if len(subset) else 0.0
        summary[f"{cat}(subjective-future)"] = f"{percent}%({score})"
    return summary


def run_scoring(
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    model_name: str,
    output_dir: Path,
    collection_name: str,
    time_window: float,
    disable_llm_judge: bool,
    penalize_early_response: bool = True,
    early_window_seconds: float = 2.0,
    native_videochat3: bool = False,
    native_moss: bool = False,
    native_joyai: bool = False,
    native_aura: bool = False,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    judger = None if disable_llm_judge else load_llm_judger(config)
    all_rows: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="Scoring"):
        all_rows.extend(
            evaluate_sample(
                sample,
                judger,
                time_window,
                penalize_early_response=penalize_early_response,
                early_window_seconds=early_window_seconds,
                native_videochat3=native_videochat3,
                native_moss=native_moss,
                native_joyai=native_joyai,
                native_aura=native_aura,
            )
        )
    details_path = output_dir / f"{collection_name}_details.jsonl"
    write_jsonl(details_path, all_rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
    else:
        df = pd.DataFrame(
            columns=[
                "sample_id",
                "question_time",
                "question",
                "answer_time",
                "answer",
                "response_time",
                "response",
                "score",
                "category",
                "task_type",
                "is_objective",
                "explanation",
            ]
        )
    summary = build_summary(df, model_name)
    summary_df = pd.DataFrame([summary]).round(1)

    db_path = output_dir / f"{collection_name}.db"
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(model_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()
    csv_path = output_dir / f"{collection_name}.csv"
    if csv_path.exists():
        merged = pd.concat([pd.read_csv(csv_path), summary_df], ignore_index=True)
        merged.to_csv(csv_path, index=False)
    else:
        summary_df.to_csv(csv_path, index=False)
    summary_json = output_dir / f"{collection_name}_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {
        "details_jsonl": details_path,
        "sqlite_db": db_path,
        "summary_csv": csv_path,
        "summary_json": summary_json,
    }


def resolve_model_output_input(path_str: str) -> List[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"model output not found: {path}")
    files = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no json/jsonl files under: {path}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-contained StreamEval inference + scoring")
    parser.add_argument("--config", default="", help="Optional local config yaml")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", default="", help="Override model id/path for API payload")
    parser.add_argument("--benchmarks", nargs="+", default=[], help="Benchmark names or paths")
    parser.add_argument("--video-root", default="", help="Root dir for relative video_path in new-format datasets")
    parser.add_argument("--stream-addr-root", default="", help="Root dir for generated stream chunk cache")
    parser.add_argument(
        "--prompts",
        default="",
        help="Prompt preset name. Empty means use config default_prompt.",
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "runs"))
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))

    parser.add_argument("--sparse-mode", type=int, default=1)
    parser.add_argument("--active-window", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--model-video-fps", type=float, default=2.0)
    parser.add_argument("--trim-fps", type=float, default=None)
    parser.add_argument(
        "--force-focus",
        action="store_true",
        help="Disable chunk FPS downsampling near sample.timestamp_focus for a short window.",
    )
    parser.add_argument(
        "--focus-window-seconds",
        type=float,
        default=0.0,
        help="Window size in seconds after timestamp_focus where original FPS is kept.",
    )
    parser.add_argument(
        "--high-fps",
        type=parse_high_fps,
        default=None,
        metavar="FPS|original",
        help="FPS for high-FPS chunks; 'original' (default) retains source FPS.",
    )
    parser.add_argument(
        "--full-video-high-fps",
        type=int,
        choices=(0, 1),
        default=0,
        metavar="{0,1}",
        help="Use high-FPS policy for every video chunk (default: 0).",
    )
    parser.add_argument(
        "--proactive-focus",
        action="store_true",
        help="Allow model Focus_Start/Focus_End outputs to control high-FPS sampling.",
    )
    parser.add_argument(
        "--native-videochat3",
        action="store_true",
        help=(
            "Use VideoChat3's native streaming protocol: 4 images per second, "
            "224x224 pixel budget, </Silence>/</Standby>/</Response>, and a 32-round window."
        ),
    )
    parser.add_argument(
        "--native-moss",
        action="store_true",
        help=(
            "Feed each 1-second chunk through MOSS-VL's native timestamped frame sampler "
            "and realtime session. <|silence|>, <|response|>, and round tokens are scored "
            "as silence or stripped before the existing timestamp window."
        ),
    )
    parser.add_argument(
        "--native-joyai",
        action="store_true",
        help=(
            "Feed each 1-second chunk as JoyAI image frames with a <T seconds> tag. "
            "</silence> is scored as silence and </response> is stripped before the "
            "existing timestamp window."
        ),
    )
    parser.add_argument(
        "--native-aura",
        action="store_true",
        help=(
            "Feed each 1-second chunk as an AURA video turn. <|silent|> is scored as "
            "silence before the existing timestamp window."
        ),
    )
    parser.add_argument("--chunk-cache-root", type=str, default=None)
    parser.add_argument(
        "--dialog-dump-dir",
        type=str,
        default="",
        help="Directory to dump per-request prompt/images/response artifacts.",
    )
    parser.add_argument("--context-window-seconds", type=float, default=None)
    parser.add_argument(
        "--max-context-frames",
        type=int,
        default=120,
        help=(
            "Maximum total source frames retained across video chunks (default: 120). "
            "Set to 0 to disable this limit and use only the time window."
        ),
    )
    parser.add_argument(
        "--max-focus-context-frames",
        type=int,
        default=90,
        help=(
            "Maximum high-FPS source frames retained across focus chunks (default: 90). "
            "Set to 0 to disable the dedicated focus-frame limit."
        ),
    )
    parser.add_argument(
        "--resolution-compress",
        action="store_true",
        help=(
            "Keep the time window but replace total-frame eviction with dynamic "
            "resolution compression of non-focus chunks under a Qwen3-VL visual-token budget."
        ),
    )
    parser.add_argument(
        "--time-compress",
        action="store_true",
        help=(
            "Keep native spatial resolution and evict the oldest video chunks "
            "when the selected visual-token or source-pixel budget is exceeded."
        ),
    )
    parser.add_argument(
        "--also-compress-focus",
        action="store_true",
        help=(
            "Allow focus chunks to participate in dynamic resolution compression. "
            "Has effect only with --resolution-compress."
        ),
    )
    parser.add_argument(
        "--visual-token-budget",
        type=int,
        default=None,
        help=(
            "Visual-token budget used by --resolution-compress or --time-compress. "
            f"Defaults to {QWEN3_VL_DEFAULT_VISUAL_TOKEN_BUDGET} unless "
            "--pixel-budget is selected."
        ),
    )
    parser.add_argument(
        "--pixel-budget",
        nargs="?",
        const=QWEN3_VL_DEFAULT_PIXEL_BUDGET,
        type=int,
        default=None,
        help=(
            "Use source pixels before native Qwen preprocessing as the context "
            "budget. With no value, defaults to 120 * 1920 * 1080 = "
            f"{QWEN3_VL_DEFAULT_PIXEL_BUDGET} pixels."
        ),
    )
    parser.add_argument(
        "--low-fps-degeneration",
        action="store_true",
        help=(
            "When the focus-frame window overflows, demote the oldest focus chunk "
            "to the normal model video FPS instead of discarding it."
        ),
    )

    parser.add_argument("--time-window", type=float, default=2.0)
    parser.add_argument(
        "--early-window-seconds",
        type=float,
        default=2.0,
        help="Allow responses up to this many seconds before answer_time (default: 2.0).",
    )
    parser.add_argument(
        "--allow-early-correct",
        action="store_true",
        help="Do not force early responses to 0; score them normally if content is correct.",
    )
    parser.add_argument("--collection", default="result")
    parser.add_argument("--disable-llm-judge", action="store_true")

    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--model-output", default="")
    return parser.parse_args()


def parse_high_fps(value: str) -> Optional[float]:
    """Parse a fixed high FPS or ``original`` to retain source frame rate."""
    if str(value).strip().lower() == "original":
        return None
    try:
        fps = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--high-fps must be a positive number or 'original'"
        ) from exc
    if fps <= 0:
        raise argparse.ArgumentTypeError("--high-fps must be positive")
    return fps


def main() -> None:
    args = parse_args()
    if args.skip_inference and not args.model_output:
        raise ValueError("--model-output is required when --skip-inference is set")
    if args.skip_inference and args.skip_scoring:
        raise ValueError("both --skip-inference and --skip-scoring are set, nothing to do")
    if args.max_context_frames < 0:
        raise ValueError("--max-context-frames must be non-negative")
    if args.early_window_seconds < 0:
        raise ValueError("--early-window-seconds must be non-negative")
    if args.max_focus_context_frames < 0:
        raise ValueError("--max-focus-context-frames must be non-negative")
    if args.visual_token_budget is not None and args.visual_token_budget <= 0:
        raise ValueError("--visual-token-budget must be positive")
    if args.pixel_budget is not None and args.pixel_budget <= 0:
        raise ValueError("--pixel-budget must be positive")
    if args.pixel_budget is not None and args.visual_token_budget is not None:
        raise ValueError("--pixel-budget and --visual-token-budget are mutually exclusive")
    if args.resolution_compress and args.time_compress:
        raise ValueError("--resolution-compress and --time-compress are mutually exclusive")
    visual_token_budget = (
        args.visual_token_budget
        if args.visual_token_budget is not None
        else QWEN3_VL_DEFAULT_VISUAL_TOKEN_BUDGET
    )

    config_arg = Path(args.config).resolve() if args.config else DEFAULT_CONFIG_PATH
    config, config_path = load_release_config(config_arg)
    init_runtime_constants(config)
    config_dir = config_path.parent if config_path.exists() else SCRIPT_DIR
    prompt_name = str(args.prompts).strip() or str(config.get("default_prompt", "streaming"))
    bench_list = args.benchmarks or list(config.get("default_benchmarks", []))
    if not bench_list and not args.skip_inference:
        raise ValueError("No benchmarks provided and no default_benchmarks found in config")

    config_video_root = str(config.get("video_root", "") or "").strip()
    config_stream_addr_root = str(config.get("stream_addr_root", "") or "").strip()
    video_root = Path(args.video_root) if args.video_root else (Path(config_video_root) if config_video_root else None)
    stream_addr_root = (
        Path(args.stream_addr_root)
        if args.stream_addr_root
        else (Path(config_stream_addr_root) if config_stream_addr_root else None)
    )

    run_root = Path(args.output_dir) / f"{args.model_name}_{args.run_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    dialog_dump_root = Path(args.dialog_dump_dir) if args.dialog_dump_dir else (run_root / "dialog_dumps" / args.model_name)
    dialog_dump_root.mkdir(parents=True, exist_ok=True)
    with open(run_root / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "config_path": str(config_path),
                "default_prompt": prompt_name,
                "created_at": datetime.now().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    inference_samples: List[Dict[str, Any]] = []
    inference_inputs: List[Path] = []
    if not args.skip_inference:
        options = InferenceOptions(
            sparse_mode=bool(args.sparse_mode),
            active_window=args.active_window,
            max_retries=args.max_retries,
            chunk_seconds=args.chunk_seconds,
            model_video_fps=args.model_video_fps,
            trim_fps=args.trim_fps,
            force_focus=bool(args.force_focus),
            focus_window_seconds=float(args.focus_window_seconds),
            high_fps=args.high_fps,
            full_video_high_fps=bool(args.full_video_high_fps),
            max_context_frames=args.max_context_frames,
            dialog_dump_root=dialog_dump_root,
            chunk_cache_root=Path(args.chunk_cache_root) if args.chunk_cache_root else None,
            proactive_focus=bool(args.proactive_focus),
            max_focus_context_frames=args.max_focus_context_frames,
            resolution_compress=bool(args.resolution_compress),
            time_compress=bool(args.time_compress),
            also_compress_focus=bool(args.also_compress_focus),
            low_fps_degeneration=bool(args.low_fps_degeneration),
            visual_token_budget=visual_token_budget,
            pixel_budget=args.pixel_budget,
            native_videochat3=bool(args.native_videochat3),
            native_moss=bool(args.native_moss),
            native_joyai=bool(args.native_joyai),
            native_aura=bool(args.native_aura),
        )
        runner = UnifiedInferenceRunner(
            config=config,
            config_dir=config_dir,
            model_name=args.model_name,
            prompts_name=prompt_name,
            model_path_override=args.model_path,
            options=options,
            context_window_seconds=args.context_window_seconds,
            video_root=video_root,
            stream_addr_root=stream_addr_root,
        )
        with open(run_root / "system_prompt.txt", "w", encoding="utf-8") as f:
            f.write(runner.system_prompt_text + "\n")
        inference_dir = run_root / "inference"
        inference_samples = runner.run_benchmarks(bench_list, inference_dir)
        merged_jsonl = inference_dir / "all_benchmarks.jsonl"
        write_jsonl(merged_jsonl, inference_samples)
        inference_inputs.append(merged_jsonl)
    else:
        for path in resolve_model_output_input(args.model_output):
            inference_inputs.append(path)
            inference_samples.extend(
                load_samples_any_format(
                    benchmark_path=path,
                    bench_name=path.stem,
                    video_root=video_root,
                    stream_addr_root=stream_addr_root,
                    need_video_info=False,
                )
            )

    scoring_outputs = {}
    if not args.skip_scoring:
        scoring_outputs = run_scoring(
            samples=inference_samples,
            config=config,
            model_name=args.model_name,
            output_dir=run_root / "scoring",
            collection_name=args.collection,
            time_window=args.time_window,
            disable_llm_judge=args.disable_llm_judge,
            penalize_early_response=not args.allow_early_correct,
            early_window_seconds=args.early_window_seconds,
            native_videochat3=bool(args.native_videochat3),
            native_moss=bool(args.native_moss),
            native_joyai=bool(args.native_joyai),
            native_aura=bool(args.native_aura),
        )

    print("\n=== StreamEval Release Completed ===")
    print(f"Run root: {run_root}")
    print(f"Dialog dumps: {dialog_dump_root}")
    if inference_inputs:
        print("Inference inputs/outputs:")
        for path in inference_inputs:
            print(f"  - {path}")
    if scoring_outputs:
        print("Scoring outputs:")
        for key, value in scoring_outputs.items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
