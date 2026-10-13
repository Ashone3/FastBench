import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hf_control', ROOT / 'scripts/hf_server_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


@pytest.mark.parametrize('ignore_term', [False, True])
def test_stop_waits_for_exit_and_escalates_only_if_needed(ignore_term):
    code = 'import signal,time; '
    if ignore_term:
        code += 'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
    code += 'print("ready",flush=True); time.sleep(60)'
    process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        with patch.object(control, 'targets', side_effect=lambda ports: {process.pid} if process.poll() is None else set()):
            control.stop({8000}, 0.3)
        assert process.wait(timeout=2) == -(signal.SIGKILL if ignore_term else signal.SIGTERM)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_hidden_socket_owner_fails_instead_of_claiming_cleanup():
    result = SimpleNamespace(stdout='LISTEN 0 5 127.0.0.1:8000 0.0.0.0:*\n')
    with patch.object(subprocess, 'run', return_value=result):
        with pytest.raises(RuntimeError, match='Cannot identify owner'):
            control.targets({8000})


def test_stop_retries_transient_ownerless_listener_during_cleanup(capsys):
    with patch.object(control, 'targets', side_effect=[
        set(), control.UnidentifiedListener('Cannot identify owner of port 8002'), set(),
    ]), patch.object(control.time, 'sleep') as sleep:
        control.stop({8002}, 0)
    sleep.assert_called_once_with(0.2)
    assert '[OK]' in capsys.readouterr().out


def test_stop_does_not_claim_success_for_persistent_ownerless_listener(capsys):
    with patch.object(control, 'targets', side_effect=[
        set(), control.UnidentifiedListener('Cannot identify owner of port 8002'),
    ]), patch.object(control.time, 'monotonic', side_effect=[0, 0, 11]):
        with pytest.raises(RuntimeError, match='Still unresolved'):
            control.stop({8002}, 0)
    assert '[OK]' not in capsys.readouterr().out


@pytest.mark.parametrize('pid,path,kind,passes', [
    (123, '/models/aura', 'qwen3_vl', True),
    (124, '/models/aura', 'qwen3_vl', False),
    (123, '/models/qwen', 'qwen3_vl', False),
    (123, '/models/aura', 'moss_vl', False),
])
def test_verify_rejects_old_pid_or_wrong_checkpoint(tmp_path, pid, path, kind, passes):
    pid_file = tmp_path / 'pids'
    pid_file.write_text('123\n')
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({
        'pid': pid, 'model_path': path, 'model_type': kind, 'engine_class': 'Qwen3VLEngine',
    }).encode()
    argv = ['python', 'hf_openai_server.py', '--model-path', '/models/aura', '--model-type', 'qwen3_vl', '--port', '8000']
    with patch.object(control, 'command', return_value=argv), \
         patch.object(control, 'alive', return_value=True), \
         patch.object(control.os, 'readlink', return_value=str(tmp_path)), \
         patch.object(control.urllib.request, 'urlopen', return_value=response):
        if passes:
            control.verify('127.0.0.1', {8000}, pid_file, 0)
        else:
            with pytest.raises(RuntimeError):
                control.verify('127.0.0.1', {8000}, pid_file, 0)


def test_server_info_uses_loaded_engine_path_instead_of_requested_model_name():
    spec = importlib.util.spec_from_file_location('identity_server', ROOT / 'hf_openai_server.py')
    server = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': SimpleNamespace()}):
        spec.loader.exec_module(server)
    handler = SimpleNamespace(path='/server-info', server=SimpleNamespace(app_state={
        'engine': SimpleNamespace(model_path='/models/actual-checkpoint'),
        'model_type': 'qwen3_vl', 'served_model_name': 'misleading-name',
    }))
    with patch.object(server, 'write_json') as write:
        server.AppHandler.do_GET(handler)
    assert write.call_args.args[1] == 200
    assert write.call_args.args[2]['model_path'] == '/models/actual-checkpoint'


def test_port_scan_selects_only_requested_listeners():
    output = '\n'.join([
        'LISTEN 0 5 127.0.0.1:8000 0.0.0.0:* users:(("python",pid=123,fd=7))',
        'LISTEN 0 5 [::]:8001 [::]:* users:(("python",pid=124,fd=7))',
        'LISTEN 0 5 127.0.0.1:9000 0.0.0.0:* users:(("python",pid=125,fd=7))',
    ])
    with patch.object(subprocess, 'run', return_value=SimpleNamespace(stdout=output)), \
         patch.object(Path, 'iterdir', return_value=iter([])):
        assert control.targets({8000, 8001}) == {123, 124}


@pytest.mark.parametrize('deploy_status,verify_status,events', [
    (0, 0, ['stop', 'deploy', 'verify']),
    (1, 0, ['stop', 'deploy', 'stop']),
    (0, 1, ['stop', 'deploy', 'verify', 'stop']),
])
def test_restart_orders_cleanup_deployment_verification(tmp_path, deploy_status, verify_status, events):
    import os
    import shutil
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('restart_hf_servers_clean.sh', 'stop_hf_servers_clean.sh'):
        shutil.copy(ROOT / 'scripts' / name, scripts / name)
    (scripts / 'deploy_aurav2.sh').write_text(
        'echo deploy >> "$TEST_EVENTS"\n'
        '[[ "$LOG_DIR" == */logs/servers/clean_aurav2_* ]] || exit 8\n'
        'exit "$TEST_DEPLOY_STATUS"\n'
    )
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    stub = binaries / 'python'
    stub.write_text('#!/bin/bash\necho "$2" >> "$TEST_EVENTS"\n'
                    '[[ "$2" == stop ]] && exit 0\nexit "$TEST_VERIFY_STATUS"\n')
    stub.chmod(0o755)
    event_file = tmp_path / 'events'
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ['PATH'],
               TEST_EVENTS=str(event_file), TEST_DEPLOY_STATUS=str(deploy_status),
               TEST_VERIFY_STATUS=str(verify_status))
    result = subprocess.run(['bash', str(scripts / 'restart_hf_servers_clean.sh'), 'aurav2'],
                            env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == (deploy_status == verify_status == 0), result.stderr
    assert event_file.read_text().splitlines() == events
