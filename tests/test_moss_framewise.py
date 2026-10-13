"""CPU protocol regressions; no checkpoint, CUDA or judge service required."""
import importlib.util
import hashlib
import os
import queue
import re
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

import moss_framewise as mf
import run_stream_eval as runner_module
import run_stream_eval_parallel as parallel_module


ROOT = Path(__file__).resolve().parents[1]


class Token:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return self

    def item(self):
        return self.value


class FakeSession:
    def __init__(self, model):
        self.model = model
        self._prompt_queue = queue.Queue()
        self.frames = queue.Queue()
        self._output_queue = queue.Queue()
        self._done = threading.Event()
        self._error = None
        self._thread = None

    @property
    def pending_frames(self):
        return self.frames.qsize()

    def start(self):
        self.model.continue_generating = True
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def run(self):
        tokens = deque([(7, "<|silence|>")])
        try:
            while self.model.continue_generating:
                token_id, text = tokens.popleft() if tokens else (3, "tail")
                self._output_queue.put(text)
                result = self.model._update_model_kwargs_for_real_time_generation(
                    input_ids=Token(token_id), should_wait_for_new_input=token_id == 7,
                    new_video_frames=self.frames, new_prompts=self._prompt_queue,
                    output_text_queue=self._output_queue,
                )
                if result:
                    tokens = deque([(2, "answer first"), (3, " answer last"), (7, "<|silence|>")])
                    while not self.model.allow_generate.wait(0.01):
                        if not self.model.continue_generating:
                            return
        except BaseException as error:
            self._error = error
        finally:
            self._done.set()

    def push_frame(self, image, timestamp, drop_oldest=False):
        self.frames.put((image, timestamp))

    def push_prompt(self, prompt):
        self._prompt_queue.put(prompt)

    def poll_output(self, timeout=0.0):
        try:
            return self._output_queue.get_nowait()
        except queue.Empty:
            self._raise_if_failed()
            return None

    def _raise_if_failed(self):
        if self._done.is_set() and self._error:
            raise RuntimeError("fake generator failed") from self._error

    def close(self, timeout=10):
        self.model.stop_real_time_generate()
        self._thread.join(timeout)
        assert not self._thread.is_alive()
        self._raise_if_failed()


class FakeModel:
    def __init__(self):
        self.continue_generating = False
        self.inputs = []
        self.allow_generate = threading.Event()
        self.allow_generate.set()

    def stop_real_time_generate(self):
        self.continue_generating = False

    def create_realtime_session(self, processor, **kwargs):
        self.settings = kwargs
        return FakeSession(self)

    def _update_model_kwargs_for_real_time_generation(self, **kwargs):
        frames, prompts = [], []
        for source, target in ((kwargs["new_video_frames"], frames), (kwargs["new_prompts"], prompts)):
            while not source.empty():
                target.append(source.get_nowait())
        if frames or prompts:
            self.inputs.append(([t for _, t in frames], prompts))
            # This is NOT a generated silence and must not release the gate.
            kwargs["output_text_queue"].put("<|silence|>")
            return True
        return False


def processor():
    return SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 7, unk_token_id=0))


def test_same_pixels_text_before_last_legal_frame():
    frames = [{"timestamp": 9.0}, {"timestamp": 9.5}]
    groups = mf.input_groups(frames, ["Q"], 10.0)
    assert groups == [([frames[0]], []), ([], ["Q"]), ([frames[1]], [])]
    assert [frame for group, _ in groups for frame in group] == frames
    assert mf.input_groups([], ["Q"], 10) == [([], ["Q"])]


@pytest.mark.parametrize("timestamps", [[10], [9.5, 9], [float("nan")], [float("inf")]])
def test_reject_boundary_and_invalid_pixels(timestamps):
    with pytest.raises(ValueError):
        mf.input_groups([{"timestamp": t} for t in timestamps], ["Q"], 10)


def test_startup_and_synthetic_silence_cannot_complete_new_input():
    model = FakeModel()
    adapter = mf.FramewiseSession(model, processor(), 1024, 0)
    original = adapter.gate.original
    try:
        model.allow_generate.clear()
        sequence = adapter.gate.enqueue(adapter.session, [], ["Q"])
        with pytest.raises(TimeoutError):
            adapter.gate.wait(adapter.session, sequence, timeout=0.03)
        assert adapter.gate.completed != sequence
        model.allow_generate.set()
        adapter.gate.wait(adapter.session, sequence, timeout=1)
        output = mf.collect_outputs(adapter.session)
        assert "answer first answer last" in output
    finally:
        adapter.close()
    assert model._update_model_kwargs_for_real_time_generation == original


def test_full_question_answer_collected_before_next_frame(tmp_path):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (2, 2)).save(image_path)
    model = FakeModel()
    adapter = mf.FramewiseSession(model, processor(), 1024, 0)
    try:
        result = adapter.generate({
            "frames": [{"image": str(image_path), "timestamp": t} for t in [9, 9.5]],
            "prompts": ["Q"], "chunk_end": 10,
        })
        assert model.inputs == [([9], []), ([], ["Q"]), ([9.5], [])]
        assert result.count("answer first answer last") == 3
        assert mf.collect_outputs(adapter.session) == ""
    finally:
        adapter.close()


@pytest.fixture
def server_module():
    # Import server control flow without requiring torch in the CPU test environment.
    spec = importlib.util.spec_from_file_location("framewise_test_server", ROOT / "hf_openai_server.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
        spec.loader.exec_module(module)
    return module


def test_server_preserves_legacy_normalization_and_opt_in_fields(server_module):
    item = {"type": "moss_realtime", "frames": [], "prompts": ["Q"]}
    old = server_module.normalize_content([item])[0]
    assert "framewise_protocol" not in old and "chunk_end" not in old
    new = server_module.normalize_content([dict(item, framewise_protocol=mf.PROTOCOL, chunk_end=10)])[0]
    assert new["framewise_protocol"] == mf.PROTOCOL and new["chunk_end"] == 10


def test_mode_switch_stops_session_and_restores_updater(server_module):
    engine = server_module.MossVLEngine.__new__(server_module.MossVLEngine)
    engine.model = FakeModel()
    engine.processor = processor()
    engine._lock = threading.Lock()
    engine._session = None
    engine._session_sample_id = ""
    engine._framewise = None
    engine._framewise_sample_id = ""
    original = engine.model._update_model_kwargs_for_real_time_generation
    item = {"sample_id": "s", "reset": True, "framewise_protocol": mf.PROTOCOL,
            "frames": [], "prompts": ["Q"], "chunk_end": 10}
    engine._generate_realtime(item, 1024, 0)
    first = engine._framewise
    engine._generate_realtime(dict(item, sample_id="other"), 1024, 0)
    assert first.gate.closed and first.session._done.is_set()
    engine._reset_session = lambda *args: setattr(engine, "_session", FakeSession(engine.model))
    engine._generate_realtime({"sample_id": "legacy", "frames": [], "prompts": []}, 1024, 0)
    assert engine._framewise is None
    assert engine.model._update_model_kwargs_for_real_time_generation == original


@pytest.mark.parametrize("module", [runner_module, parallel_module])
def test_opt_in_cli_and_legacy_defaults(module):
    with patch.object(sys, "argv", ["runner", "--model-name", "moss", "--native-moss"]):
        args = module.parse_args()
        assert args.native_moss and not args.native_moss_framewise
    with patch.object(sys, "argv", ["runner", "--model-name", "moss", "--native-moss-framewise"]):
        args = module.parse_args()
        assert args.native_moss and args.native_moss_framewise
    with patch.object(sys, "argv", ["runner", "--model-name", "moss", "--native-moss-framewise", "--native-aura"]):
        with pytest.raises(SystemExit):
            module.parse_args()


@pytest.mark.parametrize("response_time,category", [(10, "Correct"), (11, "LateResponse"), (14, "LateResponse")])
def test_no_four_second_grace_for_instant_backward(response_time, category):
    sample = {"id": "s", "sqa": [
        {"timestamp": "00:00:10", "question": "Q", "options": ["A", "B"], "type": "Instant"},
        {"timestamp": "00:00:10", "response": "A"},
    ], "responses": [{"timestamp": runner_module.seconds_to_time(response_time), "response": "A"}]}
    row = runner_module.evaluate_sample(sample, None, 4, native_moss=True)[0]
    assert row["category"] == category


def test_runner_keeps_chunk_clock_continuous_pixels_and_no_retries(tmp_path):
    class Stream:
        def __init__(self, **kwargs):
            self.current_time = 0
            self.video_path = "video.mp4"
            self.chunk_cache_dir = tmp_path

        def step(self):
            start = self.current_time
            self.current_time += 1
            return {"stream_chunk": "chunk.mp4", "chunk_start_seconds": start,
                    "chunk_duration_seconds": 1, "chunk_model_fps": 2,
                    "new_events": [{"question": "Q"}] if self.current_time == 2 else [],
                    "is_finished": self.current_time == 2}

    class Backend:
        max_retries, timeout, api_base = 3, 120, "http://localhost:8000/v1"
        items = []

        def generate(self, messages, **kwargs):
            assert self.max_retries == 1
            self.items.append(messages[0]["content"][0])
            return {"status_code": 200, "response": "A"}

    runner = runner_module.UnifiedInferenceRunner.__new__(runner_module.UnifiedInferenceRunner)
    runner.model = SimpleNamespace(backend=Backend())
    runner.model_max_new_tokens = 1024
    runner.system_prompt_text = mf.SYSTEM_PROMPT
    runner.options = SimpleNamespace(chunk_seconds=1, trim_fps=2, model_video_fps=2,
        force_focus=False, focus_window_seconds=0, high_fps=None, full_video_high_fps=False,
        chunk_cache_root=tmp_path, sparse_mode=True)
    capability = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
        "protocol": mf.PROTOCOL, "model_type": "moss_vl",
    })
    extract = lambda path, start, end, fps, root: [("frame.jpg", start), ("frame.jpg", start + .5)]
    with patch.object(runner_module, "StreamingState", Stream), \
         patch.object(runner_module, "native_moss_extract_chunk_frames", extract), \
         patch.object(runner_module.requests, "get", return_value=capability):
        result = runner._run_sample_native_moss_framewise({"id": "s"}, "bench")
    assert [r["timestamp"] for r in result["responses"]] == ["00:01", "00:02"]
    assert [[f["timestamp"] for f in i["frames"]] for i in runner.model.backend.items] == [[0, .5], [1, 1.5]]
    assert [i["reset"] for i in runner.model.backend.items] == [True, False]
    assert runner.model.backend.max_retries == 3


def test_failed_generator_wakes_gate_and_restores_hook():
    class BrokenModel(FakeModel):
        def _update_model_kwargs_for_real_time_generation(self, **kwargs):
            if not kwargs["new_prompts"].empty():
                raise ValueError("test ingest failure")
            return super()._update_model_kwargs_for_real_time_generation(**kwargs)

    model = BrokenModel()
    original = model._update_model_kwargs_for_real_time_generation
    adapter = mf.FramewiseSession(model, processor(), 1024, 0)
    try:
        with pytest.raises(RuntimeError, match="updater failed"):
            adapter.generate({"frames": [], "prompts": ["Q"], "chunk_end": 10})
    finally:
        adapter.close()
    assert model._update_model_kwargs_for_real_time_generation == original



def test_native_error_chain_logged_before_cleanup(server_module, capsys):
    engine = server_module.MossVLEngine.__new__(server_module.MossVLEngine)
    engine._lock = threading.Lock()
    engine._session = None
    engine._framewise_sample_id = "7"
    class FailedSession:
        closed = False
        def generate(self, item):
            try:
                raise ValueError("fixture native tensor error")
            except ValueError as error:
                raise RuntimeError("MOSS-VL realtime generation failed") from error
        def close(self):
            self.closed = True
    session = FailedSession()
    engine._framewise = session
    with pytest.raises(RuntimeError, match="MOSS-VL realtime generation failed"):
        engine._generate_framewise({"framewise_protocol": mf.PROTOCOL, "sample_id": "7", "chunk_end": 71}, 1024, 0)
    error_log = capsys.readouterr().err
    assert "fixture native tensor error" in error_log
    assert "sample_id='7'" in error_log and "chunk_end=71" in error_log
    assert session.closed and engine._framewise is None
