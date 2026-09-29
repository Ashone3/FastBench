#!/usr/bin/env python3
"""
HF OpenAI-compatible server.

Start a local OpenAI-compatible chat endpoint backed by HuggingFace models.
Designed for StreamEval standalone usage where we want HF inference behavior.

Example:
  python hf_openai_server.py --model-path Qwen/Qwen3-VL-8B-Instruct

Then use:
  api_base=http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def now_ts() -> int:
    return int(time.time())


def read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def write_json(handler: BaseHTTPRequestHandler, status_code: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def normalize_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            if text:
                out.append({"type": "text", "text": str(text)})
        elif item_type in {"video", "input_video"}:
            # stream eval uses {"type":"video","video":"..."}
            video_path = item.get("video")
            if not video_path:
                video_url = item.get("video_url")
                if isinstance(video_url, str) and video_url.startswith("data:video/"):
                    video_path = decode_data_url_to_temp_file(video_url)
                else:
                    video_path = video_url
            if video_path:
                video_item: Dict[str, Any] = {
                    "type": "video",
                    "video": str(video_path),
                    "fps": item.get("fps", 2.0),
                }
                # Keep legacy requests byte-for-byte equivalent. Extra fields
                # are only honored for the opt-in context-window modes.
                resolution_compress = bool(item.get("resolution_compress", False))
                low_fps_degenerated = bool(item.get("low_fps_degenerated", False))
                if resolution_compress:
                    video_item["resolution_compress"] = True
                    for key in ("resized_height", "resized_width"):
                        if key in item:
                            video_item[key] = int(item[key])
                if low_fps_degenerated:
                    video_item["low_fps_degenerated"] = True
                if (resolution_compress or low_fps_degenerated) and "max_frames" in item:
                    video_item["max_frames"] = int(item["max_frames"])
                out.append(video_item)
        elif item_type == "image":
            image_path = item.get("image")
            if image_path:
                image_item: Dict[str, Any] = {"type": "image", "image": str(image_path)}
                for key in ("min_pixels", "max_pixels", "resized_height", "resized_width"):
                    if key in item:
                        image_item[key] = int(item[key])
                out.append(image_item)
        elif item_type == "image_url":
            # Optional pass-through for image messages.
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict) and image_url.get("url"):
                out.append({"type": "image", "image": image_url["url"]})
        elif item_type == "moss_realtime":
            frames = []
            for frame in item.get("frames") or []:
                if not isinstance(frame, dict):
                    continue
                image_path = frame.get("image")
                if not image_path:
                    continue
                frames.append(
                    {
                        "image": str(image_path),
                        "timestamp": float(frame.get("timestamp", 0.0)),
                    }
                )
            prompts = [str(prompt) for prompt in (item.get("prompts") or []) if str(prompt).strip()]
            out.append(
                {
                    "type": "moss_realtime",
                    "sample_id": str(item.get("sample_id", "")).strip(),
                    "reset": bool(item.get("reset", False)),
                    "system_prompt": str(item.get("system_prompt", "") or ""),
                    "frames": frames,
                    "prompts": prompts,
                    "drain_seconds": float(item.get("drain_seconds", 30.0) or 30.0),
                }
            )
    return out


def normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    out: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        out.append({"role": role, "content": normalize_content(message.get("content", []))})
    return out


def decode_data_url_to_temp_file(data_url: str) -> str:
    # format: data:video/mp4;base64,xxxx
    _, payload = data_url.split(",", 1)
    raw = base64.b64decode(payload)
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="hf_openai_upload_")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


class BaseEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        raise NotImplementedError


class TextEngine(BaseEngine):
    def __init__(self, model_path: str, device_map: str, torch_dtype: str):
        super().__init__(model_path)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = resolve_torch_dtype(torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        msgs = []
        for msg in messages:
            text_chunks = [x.get("text", "") for x in msg.get("content", []) if x.get("type") == "text"]
            text = "\n".join([x for x in text_chunks if x]).strip()
            if text:
                msgs.append({"role": msg.get("role", "user"), "content": text})

        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else None,
            )
        input_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, input_len:]
        return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


class Qwen3VLEngine(BaseEngine):
    def __init__(self, model_path: str, device_map: str, torch_dtype: str, attn_implementation: str):
        super().__init__(model_path)
        from transformers import AutoProcessor
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

        dtype = resolve_torch_dtype(torch_dtype)
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except OSError as exc:
            # Some Qwen3-VL checkpoints advertise a remote processor module that is
            # not in the snapshot. The library processor still loads their tokenizer.
            if "does not appear to have a file named" not in str(exc):
                raise
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
        if "A3B" in model_path:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        self.model.eval()

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "qwen_vl_utils is required for Qwen3-VL models. "
                "Please install qwen-vl-utils."
            ) from exc
        self.process_vision_info = process_vision_info

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        skip_special_tokens: bool = True,
    ) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = self.process_vision_info(
            messages,
            return_video_kwargs=True,
        )

        if video_kwargs:
            # qwen_vl_utils returns kwargs for the video processor (e.g. fps/do_sample_frames).
            # These should be passed through `videos_kwargs` instead of `video_metadata`.
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                return_tensors="pt",
                videos_kwargs=video_kwargs,
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                return_tensors="pt",
            ).to(self.model.device)

        do_sample = temperature > 0
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else None,
            )
        input_len = inputs.input_ids.shape[1]
        trimmed = generated_ids[:, input_len:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False
        )[0].strip()


class Qwen3OmniEngine(BaseEngine):
    def __init__(
        self,
        model_path: str,
        device_map: str,
        torch_dtype: str,
        attn_implementation: str,
        use_audio_in_video: bool,
    ):
        super().__init__(model_path)
        from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

        dtype = resolve_torch_dtype(torch_dtype)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        self.model.eval()
        self.use_audio_in_video = use_audio_in_video

        try:
            from qwen_omni_utils import process_mm_info
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "qwen_omni_utils is required for Qwen3-Omni models. "
                "Please install qwen-omni-utils."
            ) from exc
        self.process_mm_info = process_mm_info

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = self.process_mm_info(
            messages, use_audio_in_video=self.use_audio_in_video
        )
        inputs = self.processor(
            text=text_prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        ).to(device=self.model.device, dtype=self.model.dtype)

        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else None,
                speaker="Ethan",
                thinker_return_dict_in_generate=True,
                use_audio_in_video=self.use_audio_in_video,
            )

        if isinstance(out, tuple):
            seq = out[0].sequences if hasattr(out[0], "sequences") else out[0]
        elif hasattr(out, "sequences"):
            seq = out.sequences
        else:
            seq = out

        input_len = inputs["input_ids"].shape[1]
        trimmed = seq[:, input_len:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


def _count_video_frames(path: str) -> Optional[int]:
    try:
        import decord

        return int(len(decord.VideoReader(path, num_threads=1)))
    except Exception:
        pass
    try:
        from torchvision.io import read_video

        video, _, _ = read_video(path, pts_unit="sec")
        return int(video.shape[0]) if video is not None and video.ndim >= 1 else 0
    except Exception:
        return None


def _load_video_frames_as_images(path: str) -> List[Any]:
    from PIL import Image

    try:
        import decord

        reader = decord.VideoReader(path, num_threads=1)
        return [Image.fromarray(reader[i].asnumpy()) for i in range(len(reader))]
    except Exception:
        from torchvision.io import read_video

        video, _, _ = read_video(path, pts_unit="sec")
        if video is None or video.numel() == 0:
            return []
        return [
            Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
            for frame in video
        ]


def pad_short_videochat3_videos(
    messages: List[Dict[str, Any]],
    min_frames: int = 2,
) -> List[Dict[str, Any]]:
    """Rewrite VideoChat3 clips with fewer than min_frames by repeating the last frame.

    qwen_vl_utils.smart_nframes requires at least FRAME_FACTOR=2 frames. Some
    1-second trim-fps clips only contain one frame and would otherwise 500.
    """
    padded_messages: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            padded_messages.append(message)
            continue
        new_content: List[Any] = []
        changed = False
        for item in content:
            if not isinstance(item, dict) or item.get("type") not in {"video", "input_video"}:
                new_content.append(item)
                continue
            video = item.get("video")
            if not isinstance(video, str) or not video:
                new_content.append(item)
                continue
            frame_count = _count_video_frames(video)
            if frame_count is None or frame_count >= min_frames or frame_count <= 0:
                new_content.append(item)
                continue
            frames = _load_video_frames_as_images(video)
            if not frames:
                new_content.append(item)
                continue
            last = frames[-1]
            while len(frames) < min_frames:
                frames.append(last.copy())
            new_item = dict(item)
            new_item["video"] = frames
            new_content.append(new_item)
            changed = True
        if changed:
            padded_messages.append({**message, "content": new_content})
        else:
            padded_messages.append(message)
    return padded_messages


class VideoChat3Engine(BaseEngine):
    def __init__(self, model_path: str, device_map: str, torch_dtype: str):
        super().__init__(model_path)
        from transformers import AutoModelForCausalLM, AutoProcessor

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "qwen_vl_utils is required for VideoChat3 models. "
                "Please install qwen-vl-utils."
            ) from exc

        dtype = resolve_torch_dtype(torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.process_vision_info = process_vision_info

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vision_messages = pad_short_videochat3_videos(messages, min_frames=2)
        images, videos, video_kwargs = self.process_vision_info(
            vision_messages,
            image_patch_size=14,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        ).to(self.model.device)

        do_sample = temperature > 0
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else None,
            )

        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


NATIVE_MOSS_CONTROL_TOKENS = (
    "<|round_start|>",
    "<|round_end|>",
    "<|response|>",
    "<|assistant|>",
)
NATIVE_MOSS_SILENCE = "<|silence|>"


def prepare_native_moss_output(text: str) -> str:
    """Drop MOSS realtime control tokens and keep <|silence|> as the idle answer."""
    cleaned = text or ""
    for token in NATIVE_MOSS_CONTROL_TOKENS:
        cleaned = cleaned.replace(token, "")
    answer = cleaned.replace(NATIVE_MOSS_SILENCE, "").strip()
    if not answer:
        return NATIVE_MOSS_SILENCE
    return answer


class MossVLEngine(BaseEngine):
    """HF server for MOSS-VL-Realtime.

    Ordinary chat requests use the checkpoint's offline video processor so the
    existing 1-second chunk protocol keeps working. Requests that carry a
    ``moss_realtime`` item stay on one ``create_realtime_session`` and push
    timestamped PIL frames, which is the native streaming interface.
    """

    def __init__(self, model_path: str, device_map: str, torch_dtype: str, attn_implementation: str):
        super().__init__(model_path)
        from transformers import AutoModelForCausalLM, AutoProcessor

        dtype = resolve_torch_dtype(torch_dtype)
        if str(torch_dtype).lower() in {"auto", ""}:
            dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            frame_extract_num_threads=1,
        )
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except Exception as exc:
            attention_error = "flash" in str(exc).lower() or "attn" in str(exc).lower()
            if load_kwargs.get("attn_implementation") == "eager" or not attention_error:
                raise
            load_kwargs["attn_implementation"] = "eager"
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        self._session = None
        self._session_sample_id = ""
        self._lock = threading.Lock()

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        realtime = _find_moss_realtime_item(messages)
        if realtime is not None:
            return self._generate_realtime(realtime, max_new_tokens, temperature)
        return self._generate_offline(messages, max_new_tokens, temperature)

    def _generate_offline(self, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        video_fps_values = [
            float(item.get("fps", 1.0))
            for message in messages
            for item in message.get("content", [])
            if isinstance(item, dict) and item.get("type") == "video"
        ]
        video_fps = video_fps_values[-1] if video_fps_values else 1.0
        do_sample = temperature > 0
        query = {
            "messages": messages,
            "media_kwargs": {
                "video_fps": video_fps,
                "min_frames": 1,
                "max_frames": 256,
            },
            "generate_kwargs": {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": max(temperature, 1e-6) if do_sample else 1.0,
            },
        }
        prepared = self.model.offline_prepare_query_cpu(self.processor, query)
        result = self.model.offline_generate_from_prepared(self.processor, prepared)
        return str(result.get("text", "")).strip()

    def _reset_session(self, system_prompt: str, max_new_tokens: int, temperature: float) -> None:
        if self._session is not None:
            try:
                self._session.close(timeout=10.0)
            except Exception:
                self.model.stop_real_time_generate()
            self._session = None
        do_sample = temperature > 0
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = max(temperature, 1e-6)
        self._session = self.model.create_realtime_session(
            self.processor,
            initial_prompt="",
            system_prompt=system_prompt or None,
            frame_queue_size=8192,
            max_tokens_per_turn=86400,
            **generate_kwargs,
        )
        self._session.start()

    def _generate_realtime(self, item: Dict[str, Any], max_new_tokens: int, temperature: float) -> str:
        from PIL import Image

        sample_id = str(item.get("sample_id", "")).strip()
        with self._lock:
            if item.get("reset") or self._session is None or sample_id != self._session_sample_id:
                self._reset_session(str(item.get("system_prompt", "") or ""), max_new_tokens, temperature)
                self._session_sample_id = sample_id
            session = self._session
            if session is None:
                raise RuntimeError("MOSS-VL realtime session was not created")

            import queue as queue_module

            frames = item.get("frames") or []
            prompts = item.get("prompts") or []
            for frame in frames:
                image = Image.open(frame["image"]).convert("RGB")
                timestamp = float(frame["timestamp"])
                while True:
                    try:
                        session.push_frame(image, timestamp=timestamp, drop_oldest=False)
                        break
                    except queue_module.Full:
                        session.poll_output(timeout=0.05)
            for prompt in prompts:
                session.push_prompt(str(prompt))

            deadline = time.monotonic() + max(1.0, float(item.get("drain_seconds", 30.0)))
            parts: List[str] = []
            # A prompt drained together with frames is prefixed by a synthetic <|silence|>
            # before vision encoding finishes. That token is not the end of the turn.
            synthetic_silence_pending = bool(frames) and bool(prompts)
            idle_after_ingest = False
            last_output_at = time.monotonic()
            while time.monotonic() < deadline:
                queues_empty = session.pending_frames == 0 and session._prompt_queue.empty()
                chunk = session.poll_output(timeout=0.1)
                now = time.monotonic()
                if chunk is not None:
                    text = str(chunk)
                    last_output_at = now
                    if (
                        synthetic_silence_pending
                        and queues_empty
                        and text.strip() == NATIVE_MOSS_SILENCE
                    ):
                        synthetic_silence_pending = False
                        parts.append(text)
                        continue
                    parts.append(text)
                    if (
                        queues_empty
                        and not synthetic_silence_pending
                        and NATIVE_MOSS_SILENCE in text
                    ):
                        idle_after_ingest = True
                    continue
                if (
                    idle_after_ingest
                    and queues_empty
                    and now - last_output_at >= 0.3
                ):
                    break
                if not frames and not prompts and now - last_output_at >= 0.3:
                    break
            return prepare_native_moss_output("".join(parts))


def _find_moss_realtime_item(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for message in messages:
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("type") == "moss_realtime":
                return item
    return None


def resolve_torch_dtype(name: str) -> torch.dtype:
    text = str(name).lower()
    if text in {"auto", ""}:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16"}:
        return torch.float16
    if text in {"fp32", "float32"}:
        return torch.float32
    return torch.float16 if torch.cuda.is_available() else torch.float32


def build_engine(
    model_path: str,
    model_type: str,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
    use_audio_in_video: bool,
) -> BaseEngine:
    if model_type == "auto":
        lowered = model_path.lower()
        if "moss" in lowered:
            model_type = "moss_vl"
        elif "omni" in lowered:
            model_type = "qwen3_omni"
        elif "vl" in lowered:
            model_type = "qwen3_vl"
        elif "videochat" in lowered or "vc3" in lowered:
            model_type = "videochat3"
        else:
            model_type = "text"

    if model_type == "qwen3_omni":
        return Qwen3OmniEngine(
            model_path=model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            use_audio_in_video=use_audio_in_video,
        )
    if model_type == "qwen3_vl":
        return Qwen3VLEngine(
            model_path=model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
    if model_type == "videochat3":
        return VideoChat3Engine(
            model_path=model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    if model_type == "moss_vl":
        return MossVLEngine(
            model_path=model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
    return TextEngine(model_path=model_path, device_map=device_map, torch_dtype=torch_dtype)


def build_openai_response(model: str, content: str) -> Dict[str, Any]:
    created = now_ts()
    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class AppHandler(BaseHTTPRequestHandler):
    server_version = "hf-openai/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            write_json(self, HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/v1/models":
            model_id = self.server.app_state["served_model_name"]  # type: ignore[attr-defined]
            write_json(
                self,
                HTTPStatus.OK,
                {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]},
            )
            return
        write_json(self, HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            write_json(self, HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})
            return

        try:
            payload = read_json_body(self)
        except Exception as exc:
            write_json(self, HTTPStatus.BAD_REQUEST, {"error": {"message": f"Invalid JSON: {exc}"}})
            return

        if payload.get("stream"):
            write_json(
                self,
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": "stream=true is not supported in this server."}},
            )
            return

        model_name = str(payload.get("model") or self.server.app_state["served_model_name"])  # type: ignore[attr-defined]
        messages = normalize_messages(payload.get("messages", []))
        max_tokens = int(payload.get("max_tokens", payload.get("max_new_tokens", 1024)))
        temperature = float(payload.get("temperature", 0.0))

        if not messages:
            write_json(self, HTTPStatus.BAD_REQUEST, {"error": {"message": "messages is empty"}})
            return

        try:
            engine: BaseEngine = self.server.app_state["engine"]  # type: ignore[attr-defined]
            # Default stays skip_special_tokens=True. Only an explicit false keeps
            # control tokens such as JoyAI's </silence> and </response>.
            generate_kwargs: Dict[str, Any] = {}
            if payload.get("skip_special_tokens") is False:
                generate_kwargs["skip_special_tokens"] = False
            text = engine.generate(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **generate_kwargs,
            )
            write_json(self, HTTPStatus.OK, build_openai_response(model_name, text))
        except Exception as exc:
            write_json(
                self,
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"message": f"Inference failed: {type(exc).__name__}: {exc}"}},
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        # concise logs
        print(f"[hf-openai] {self.address_string()} - {fmt % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF OpenAI-compatible chat server")
    parser.add_argument("--model-path", required=True, help="HuggingFace model path or repo id")
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "qwen3_omni", "qwen3_vl", "videochat3", "moss_vl", "text"],
    )
    parser.add_argument("--served-model-name", default="", help="Model name returned by /v1/models")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", help="auto|bf16|fp16|fp32")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--use-audio-in-video", type=int, default=1, help="1 for true, 0 for false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    served_model_name = args.served_model_name or Path(args.model_path).name

    print(f"[hf-openai] Loading model: {args.model_path}")
    engine = build_engine(
        model_path=args.model_path,
        model_type=args.model_type,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        use_audio_in_video=bool(args.use_audio_in_video),
    )
    print(f"[hf-openai] Loaded. Serving as model: {served_model_name}")

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    server.app_state = {"engine": engine, "served_model_name": served_model_name}  # type: ignore[attr-defined]

    print(f"[hf-openai] Listening on http://{args.host}:{args.port}")
    print("[hf-openai] Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[hf-openai] Stopped")


if __name__ == "__main__":
    main()
