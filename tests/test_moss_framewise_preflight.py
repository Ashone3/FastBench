"""Deployment regressions: stale /health must never validate another model."""
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import requests

import moss_framewise as mf
import run_stream_eval as runner_module
import run_stream_eval_parallel as parallel_module


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moss_preflight", ROOT / "scripts/moss_framewise_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_404_is_a_deployment_error_with_endpoint_and_remedy():
    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    with patch.object(requests, "get", return_value=response):
        with pytest.raises(RuntimeError, match="deploy_moss_vl_realtime_framewise.sh") as error:
            mf.verify_endpoint("http://127.0.0.1:8004/v1")
    assert "8004/moss-framewise/protocol" in str(error.value)


@pytest.mark.parametrize("info", [
    {"protocol": mf.PROTOCOL, "model_type": "qwen3_vl", "pid": 10},
    {"protocol": "old", "model_type": "moss_vl", "pid": 10},
    {"protocol": mf.PROTOCOL, "model_type": "moss_vl", "pid": 9},
])
def test_old_model_or_different_pid_cannot_validate_new_deployment(info):
    response = MagicMock()
    response.json.return_value = info
    with patch.object(requests, "get", return_value=response):
        with pytest.raises(RuntimeError):
            mf.verify_endpoint("http://127.0.0.1:8004/v1", expected_pid=10)


def test_matching_model_protocol_and_pid_pass():
    info = {"protocol": mf.PROTOCOL, "model_type": "moss_vl", "pid": 10}
    response = MagicMock()
    response.json.return_value = info
    with patch.object(requests, "get", return_value=response):
        assert mf.verify_endpoint("http://127.0.0.1:8004/v1/", expected_pid=10) == info


@pytest.mark.parametrize("enabled", [False, True])
def test_check_happens_before_sample_distribution_only_for_opt_in(enabled):
    options = runner_module.InferenceOptions(
        sparse_mode=True, active_window=4, max_retries=2, chunk_seconds=1,
        model_video_fps=2, trim_fps=2, force_focus=False, focus_window_seconds=0,
        high_fps=None, full_video_high_fps=False, max_context_frames=120,
        dialog_dump_root=None, chunk_cache_root=None,
        native_moss=True, native_moss_framewise=enabled,
    )
    config = {"models": {"moss": {"api_base": "http://127.0.0.1:8000/v1", "model": "MOSS"}},
              "prompts": {"p": {"system_prompt": "legacy prompt"}}}
    with patch.object(mf, "verify_endpoint", side_effect=RuntimeError("wrong server")) as verify:
        kwargs = dict(num_workers=8, config=config, config_dir=ROOT, model_name="moss", prompt_name="p",
            model_path="", options=options, context_window_seconds=None, video_root=None,
            stream_addr_root=None, api_bases=[f"http://127.0.0.1:{port}/v1" for port in range(8000, 8008)])
        if enabled:
            with pytest.raises(RuntimeError, match="wrong server"):
                parallel_module._build_runners(**kwargs)
            assert verify.call_count == 1
        else:
            assert len(parallel_module._build_runners(**kwargs)) == 8
            verify.assert_not_called()


def test_occupied_port_is_rejected_before_model_loading():
    probe = MagicMock()
    probe.__enter__.return_value.bind.side_effect = OSError("Address already in use")
    with patch.object(socket, "socket", return_value=probe):
        with pytest.raises(RuntimeError, match="Ports unavailable before model loading") as error:
            preflight.check_ports("127.0.0.1", 8000, 2)
    assert "8000" in str(error.value) and "8001" in str(error.value)


def test_every_launched_pid_is_verified(tmp_path):
    pid_file = tmp_path / "pids"
    pid_file.write_text("101\n102\n")
    with patch.object(preflight, "verify_endpoint") as verify:
        preflight.check_servers("127.0.0.1", 8000, 2, pid_file)
    assert verify.call_args_list == [
        call("http://127.0.0.1:8000/v1", expected_pid=101),
        call("http://127.0.0.1:8001/v1", expected_pid=102),
    ]



def test_protocol_endpoint_rejects_other_engine_and_identifies_process():
    spec = importlib.util.spec_from_file_location("protocol_test_server", ROOT / "hf_openai_server.py")
    server = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
        spec.loader.exec_module(server)
    handler = SimpleNamespace(path="/moss-framewise/protocol",
        server=SimpleNamespace(app_state={"engine": object()}))
    with patch.object(server, "write_json") as write:
        server.AppHandler.do_GET(handler)
        assert write.call_args.args[1] == 409
        handler.server.app_state["engine"] = server.MossVLEngine.__new__(server.MossVLEngine)
        server.AppHandler.do_GET(handler)
        assert write.call_args.args[1] == 200
        assert write.call_args.args[2]["pid"] == os.getpid()
        assert write.call_args.args[2]["model_type"] == "moss_vl"

@pytest.mark.parametrize("ports_status,start_status,servers_status,expected,events", [
    (1, 0, 0, 1, ["stop", "ports"]),
    (0, 1, 0, 1, ["stop", "ports", "start", "stop"]),
    (0, 0, 1, 1, ["stop", "ports", "start", "servers", "stop"]),
    (0, 0, 0, 0, ["stop", "ports", "start", "servers"]),
])
def test_launcher_failure_cleanup_and_opt_in_isolation(
    tmp_path, ports_status, start_status, servers_status, expected, events,
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / "deploy_moss_vl_realtime_framewise.sh"
    shutil.copy(ROOT / "scripts" / launcher.name, launcher)
    starter = scripts / "serve_model.sh"
    starter.write_text(
        '#!/bin/bash\n'
        'echo "${1:-start}" >> "$MOSS_TEST_EVENTS"\n'
        '[[ "$MODEL_TYPE" == moss_vl && "$LOG_DIR" == */logs/servers/moss_framewise ]] || exit 9\n'
        '[[ "${1:-}" == stop ]] && exit 0\n'
        'exit "$MOSS_TEST_START_STATUS"\n'
    )
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    python_stub = binary_dir / "python"
    python_stub.write_text(
        '#!/bin/bash\necho "$2" >> "$MOSS_TEST_EVENTS"\n'
        '[[ "$2" == ports ]] && exit "$MOSS_TEST_PORTS_STATUS"\n'
        'exit "$MOSS_TEST_SERVERS_STATUS"\n'
    )
    python_stub.chmod(0o755)
    event_file = tmp_path / "events"
    env = dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ["PATH"],
        MOSS_TEST_EVENTS=str(event_file), MOSS_TEST_PORTS_STATUS=str(ports_status),
        MOSS_TEST_START_STATUS=str(start_status), MOSS_TEST_SERVERS_STATUS=str(servers_status))
    env.pop("LOG_DIR", None)
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True)
    assert result.returncode == expected, result.stderr.decode()
    assert event_file.read_text().splitlines() == events
