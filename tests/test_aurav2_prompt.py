"""Check prompt opt-in and request wiring without loading models or calling APIs."""
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

import run_stream_eval as serial
import run_stream_eval_parallel as parallel


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "test_aurav2_proactive_perception_benchmark_local_20260923_0236_pure_en_notallowearly.sh"


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("focus", [False, True])
def test_prompt_selection_and_request_system_message(enabled, focus):
    options = serial.InferenceOptions(
        sparse_mode=True, active_window=4, max_retries=2, chunk_seconds=1,
        model_video_fps=2, trim_fps=2, force_focus=False, focus_window_seconds=0,
        high_fps=None, full_video_high_fps=False, max_context_frames=120,
        dialog_dump_root=None, chunk_cache_root=None, native_aura=True,
        native_aurav2_prompt=enabled, proactive_focus=focus,
    )
    config = {"models": {"aura": {}}, "prompts": {"p": {"system_prompt": "configured prompt"}}}
    with patch.object(serial, "build_backend"), patch.object(serial, "SessionModel"):
        runner = serial.UnifiedInferenceRunner(
            config, ROOT, "aura", "p", "", options, None, None, None,
        )
    expected = serial.NATIVE_AURA_FOCUS_SYSTEM if focus else serial.NATIVE_AURA_SYSTEM
    if enabled:
        expected += "\n\n" + serial.NATIVE_AURAV2_RESPONSE_RULES
        if not focus:
            assert expected == serial.NATIVE_AURAV2_SYSTEM
    assert runner.system_prompt_text == expected
    messages = serial._native_aura_messages([], [], system_prompt=runner.system_prompt_text)
    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": expected}]}
    # Callers that use the original helper default still receive the original prompt.
    assert serial._native_aura_messages([], [])[0]["content"][0]["text"] == serial.NATIVE_AURA_SYSTEM


@pytest.mark.parametrize("module", [serial, parallel])
@pytest.mark.parametrize("enabled", [False, True])
def test_cli_opt_in_defaults_off(module, enabled):
    argv = ["eval", "--model-name", "aura", "--native-aura"]
    if enabled:
        argv.append("--native-aurav2-prompt")
    with patch.object(sys, "argv", argv):
        assert module.parse_args().native_aurav2_prompt is enabled


@pytest.mark.parametrize("module", [serial, parallel])
@pytest.mark.parametrize("flags,reason", [
    ([], "requires --native-aura"),
    (["--native-aura", "--native-videochat3"], "another native model"),
    (["--native-aura", "--native-moss"], "another native model"),
    (["--native-aura", "--native-moss-framewise"], "another native model"),
    (["--native-aura", "--native-joyai"], "another native model"),
])
def test_cli_rejects_wrong_native_branch(module, flags, reason, capsys):
    with patch.object(sys, "argv", ["eval", "--model-name", "aura", "--native-aurav2-prompt", *flags]):
        with pytest.raises(SystemExit) as error:
            module.parse_args()
    assert error.value.code == 2
    assert reason in capsys.readouterr().err
