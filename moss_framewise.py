# MIT License
#
# Copyright (c) 2026 Xudong Lu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Opt-in MOSS queue synchronization, adapted from ShacklesLay/FastBench.

Reference: 959a4779905257a6314bfad1ab056937183035a2, moss_native/{transport,gate,server}.py.
Unlike that adapter, questions precede the LAST permitted frame of the current
benchmark chunk, and every input waits for a fresh generated silence. No next
chunk can supply pixels while the current question is still being answered.
This module does not implement scoring or change model/checkpoint code.
"""

import math
import queue
import threading
import time
import types


PROTOCOL = "moss-framewise-v1"
SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in real-time video analysis. "
    "The video streams to you frame by frame. At every frame, you decide "
    "independently whether to respond or stay silent — output <|silence|> when "
    "nothing relevant has happened, and respond when the visual content warrants it."
)


def verify_endpoint(api_base, expected_pid=None):
    """Reject old servers, other models, and health replies from an older PID."""
    import requests

    endpoint = f"{api_base.rstrip('/').removesuffix('/v1')}/moss-framewise/protocol"
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        info = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(
            f"MOSS framewise preflight failed at {endpoint}: {error}. "
            "This port may belong to an old server or another model. "
            "Deploy this repository's hf_openai_server.py using "
            "scripts/deploy_moss_vl_realtime_framewise.sh and use matching BASE_PORT."
        ) from error
    if not isinstance(info, dict) or info.get("protocol") != PROTOCOL or info.get("model_type") != "moss_vl":
        raise RuntimeError(f"{endpoint} is not a compatible MOSS-VL framewise server: {info!r}")
    if expected_pid is not None and info.get("pid") != int(expected_pid):
        raise RuntimeError(
            f"{endpoint} is served by PID {info.get('pid')}, not newly launched PID {expected_pid}; "
            "another server owns this port"
        )
    return info


def input_groups(frames, prompts, chunk_end):
    """Same pixels, text before the last legal frame; never borrow the Q frame.

    Reorder input delivery within this chunk only. Keep image timestamps and
    benchmark question/answer timestamps intact. With no pixels, ask text only.
    """
    frames = list(frames)
    timestamps = [float(frame["timestamp"]) for frame in frames]
    if not math.isfinite(float(chunk_end)) or any(
        not math.isfinite(t) or t < 0 or t >= float(chunk_end) for t in timestamps
    ):
        raise ValueError("Framewise MOSS requires frame timestamps strictly before chunk_end")
    if timestamps != sorted(timestamps):
        raise ValueError("Framewise MOSS requires ordered frame timestamps")
    split = len(frames) - 1 if frames and prompts else len(frames)
    groups = [([frame], []) for frame in frames[:split]]
    groups.extend(([], [str(prompt)]) for prompt in prompts)
    groups.extend(([frame], []) for frame in frames[split:])
    return groups


class SilenceGate:
    """Publish one input atomically; acknowledge only silence generated AFTER ingest.

    Observe generated tokens at the native updater, not the output queue (which
    may also contain synthetic silence). Pause at silence before calling the
    updater with local queues, so its internal idle loop cannot hide new inputs.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.silence_id = tokenizer.convert_tokens_to_ids("<|silence|>")
        if self.silence_id is None or self.silence_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError("Checkpoint tokenizer does not define <|silence|>")
        self.cv = threading.Condition(threading.RLock())
        self.pending = None
        self.current = None
        self.completed = None
        self.startup = False
        self.closed = False
        self.error = None
        self.sequence = 0
        self.original = model._update_model_kwargs_for_real_time_generation

    def install(self):
        gate = self

        def update(bound_model, *args, **kwargs):
            try:
                if args:
                    raise TypeError("Unsupported MOSS native updater calling convention")
                with gate.cv:
                    # Output is queued by the native loop before entering this hook.
                    if int(kwargs["input_ids"][0, -1].item()) == gate.silence_id:
                        if gate.current is None:
                            gate.startup = True
                        else:
                            gate.completed = gate.current
                        gate.cv.notify_all()
                    while (
                        kwargs["should_wait_for_new_input"]
                        and gate.pending is None
                        and not gate.closed
                        and bound_model.continue_generating
                    ):
                        gate.cv.wait(0.05)
                    event = gate.pending
                    gate.pending = None
                    local_frames, local_prompts = queue.Queue(), queue.Queue()
                    if event is not None:
                        for source, target in (
                            (kwargs["new_video_frames"], local_frames),
                            (kwargs["new_prompts"], local_prompts),
                        ):
                            while True:
                                try:
                                    target.put_nowait(source.get_nowait())
                                except queue.Empty:
                                    break
                        if local_frames.qsize() != event[1] or local_prompts.qsize() != event[2]:
                            raise RuntimeError("MOSS input queues do not match published input")
                result = gate.original(**dict(
                    kwargs, new_video_frames=local_frames, new_prompts=local_prompts
                ))
                if event is not None:
                    with gate.cv:
                        gate.current = event[0]
                        gate.cv.notify_all()
                return result
            except BaseException as error:
                with gate.cv:
                    gate.error = error
                    gate.cv.notify_all()
                raise

        self.model._update_model_kwargs_for_real_time_generation = types.MethodType(update, self.model)

    def enqueue(self, session, decoded, prompts):
        with self.cv:
            if self.closed or self.pending is not None or (
                self.current is not None and self.completed != self.current
            ):
                raise RuntimeError("Previous MOSS input has not completed")
            if session.pending_frames or not session._prompt_queue.empty():
                raise RuntimeError("MOSS external input queues are not empty")
            self.sequence += 1
            for image, timestamp in decoded:
                session.push_frame(image, timestamp=timestamp, drop_oldest=False)
            for prompt in prompts:
                session.push_prompt(prompt)
            self.pending = (self.sequence, len(decoded), len(prompts))
            self.cv.notify_all()
            return self.sequence

    def wait(self, session, sequence=None, timeout=60.0):
        deadline = time.monotonic() + timeout
        with self.cv:
            while not (self.startup if sequence is None else self.completed == sequence):
                if self.error is not None:
                    raise RuntimeError("MOSS native updater failed") from self.error
                session._raise_if_failed()
                if self.closed or session._done.is_set():
                    raise RuntimeError("MOSS session stopped before generated silence")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No fresh MOSS generated silence for input {sequence}")
                self.cv.wait(min(remaining, 0.1))


def collect_outputs(session):
    parts = []
    while True:
        chunk = session.poll_output(timeout=0.0)
        if chunk is None:
            return "".join(parts)
        parts.append(str(chunk))


class FramewiseSession:
    """Own the temporary native hook; restore it before any legacy session starts."""

    def __init__(self, model, processor, max_new_tokens, temperature):
        self.model = model
        self.session = None
        self.gate = SilenceGate(model, processor.tokenizer)
        self.gate.install()
        try:
            kwargs = {"max_new_tokens": int(max_new_tokens), "do_sample": temperature > 0}
            if temperature > 0:
                kwargs["temperature"] = float(temperature)
            self.session = model.create_realtime_session(
                processor, initial_prompt="", system_prompt=SYSTEM_PROMPT,
                frame_queue_size=8192, max_tokens_per_turn=86400, **kwargs,
            )
            self.session.start()
            self.gate.wait(self.session)
            collect_outputs(self.session)  # Discard only pre-input startup output.
        except BaseException:
            self.close()
            raise

    def generate(self, item):
        from PIL import Image

        groups = input_groups(item.get("frames") or [], item.get("prompts") or [], item["chunk_end"])
        parts = []
        for frames, prompts in groups:
            decoded = []
            for frame in frames:
                with Image.open(frame["image"]) as image:
                    decoded.append((image.convert("RGB"), float(frame["timestamp"])))
            sequence = self.gate.enqueue(self.session, decoded, prompts)
            self.gate.wait(self.session, sequence)
            parts.append(collect_outputs(self.session))
        return "".join(parts)

    def close(self):
        self.model.stop_real_time_generate()
        with self.gate.cv:
            self.gate.closed = True
            self.gate.cv.notify_all()
        try:
            if self.session is not None:
                self.session.close(timeout=10.0)
        except Exception:
            # Native close re-raises a prior generation error even after joining.
            # A joined failed generator is safe to clean up; a live one is not.
            thread = getattr(self.session, "_thread", None)
            if thread is not None and thread.is_alive():
                raise
        finally:
            # Never install a legacy updater while an old generator is alive.
            thread = getattr(self.session, "_thread", None)
            if thread is None or not thread.is_alive():
                self.model._update_model_kwargs_for_real_time_generation = self.gate.original
