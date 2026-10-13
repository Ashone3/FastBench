#!/usr/bin/env python3
"""Stop selected ports and verify the checkpoint of newly launched HF servers."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request


def command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1]


def option(argv, name):
    for i, value in enumerate(argv):
        if value == name and i + 1 < len(argv):
            return argv[i + 1]
        if value.startswith(name + '='):
            return value.split('=', 1)[1]
    return None


class UnidentifiedListener(RuntimeError):
    """A listener was observed, but ss could not associate it with a PID."""


def targets(ports):
    result = subprocess.run(['ss', '-H', '-ltnp'], check=True, capture_output=True, text=True)
    pids = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            port = int(fields[3].rsplit(':', 1)[1])
        except (ValueError, IndexError):
            continue
        if port in ports:
            owners = re.findall(r'pid=(\d+)', line)
            if not owners:
                raise UnidentifiedListener(
                    f'Cannot identify owner of port {port}; the process may be exiting '
                    'or its PID may not be visible to this account.'
                )
            pids.update(map(int, owners))
    # Also stop HF servers still loading weights, before they start listening.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = command(entry.name)
            if any(Path(arg).name == 'hf_openai_server.py' for arg in argv):
                if int(option(argv, '--port') or 8000) in ports:
                    pids.add(int(entry.name))
        except (OSError, ValueError):
            continue
    return pids


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def process_birth(pid):
    # Field 22, after removing the parenthesized command name.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


class ProcessHandle:
    def __init__(self, pid):
        self.pid = pid
        self.birth = process_birth(pid)
        self.fd = None
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            self.fd = os.pidfd_open(pid)

    def send(self, sig):
        if self.fd is not None:
            signal.pidfd_send_signal(self.fd, sig)
        elif process_birth(self.pid) == self.birth:
            # Compatibility for Python builds without pidfd support.
            os.kill(self.pid, sig)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)


def stop(ports, timeout):
    owners = targets(ports)
    handles = {}
    try:
        # Prefer pidfds; otherwise check process start time before signaling.
        for pid in sorted(owners):
            try:
                handle = ProcessHandle(pid)
                handles[pid] = handle
                print(f'[STOP] PID {pid}: {" ".join(command(pid))}', flush=True)
                handle.send(signal.SIGTERM)
            except ProcessLookupError:
                continue
            except FileNotFoundError:
                continue
        deadline = time.monotonic() + timeout
        while any(alive(pid) for pid in handles) and time.monotonic() < deadline:
            time.sleep(0.2)
        for pid, handle in handles.items():
            if alive(pid):
                print(f'[KILL] PID {pid} did not exit within {timeout}s', flush=True)
                try:
                    handle.send(signal.SIGKILL)
                except (ProcessLookupError, FileNotFoundError):
                    pass
        deadline = time.monotonic() + 10
        while True:
            try:
                remaining = targets(ports)
            except UnidentifiedListener as error:
                # ss observes sockets and owners separately. During exit it can
                # briefly report a listener whose owning process has disappeared.
                # Never claim success for that snapshot, but allow it to settle.
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f'{error} Still unresolved after cleanup waiting period. '
                        'Inspect with ss -ltnp; use the service owner account or '
                        'sudo only if the listener belongs to another account.'
                    ) from error
                time.sleep(0.2)
                continue
            if not remaining and not any(alive(pid) for pid in handles):
                print(f'[OK] Ports {min(ports)}–{max(ports)} released; matched HF loading processes exited.')
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Servers remain or have been restarted automatically: {sorted(remaining)}')
            time.sleep(0.2)
    finally:
        for handle in handles.values():
            handle.close()


def canonical(path, cwd):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    return str(candidate.resolve())


def verify(host, ports, pid_file, timeout):
    pids = Path(pid_file).read_text().splitlines()
    if len(pids) != len(ports) or any(not p.isdigit() for p in pids):
        raise RuntimeError(f'Expected {len(ports)} PIDs in {pid_file}')
    for port, pid_text in zip(sorted(ports), pids):
        pid = int(pid_text)
        argv = command(pid)
        expected_path = option(argv, '--model-path')
        expected_type = option(argv, '--model-type') or 'auto'
        if not expected_path or not any(Path(arg).name == 'hf_openai_server.py' for arg in argv):
            raise RuntimeError(f'PID {pid} is not the expected HF server')
        if int(option(argv, '--port') or 8000) != port:
            raise RuntimeError(f'PID {pid} has unexpected port')
        cwd = os.readlink(f'/proc/{pid}/cwd')
        deadline = time.monotonic() + timeout
        while True:
            if not alive(pid):
                raise RuntimeError(f'New server PID {pid} exited. Check deployment logs.')
            try:
                with urllib.request.urlopen(f'http://{host}:{port}/server-info', timeout=3) as response:
                    info = json.load(response)
            except Exception as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Cannot verify {host}:{port}/server-info: {error}') from error
                time.sleep(0.5)
                continue
            if info.get('pid') != pid:
                raise RuntimeError(f'Port {port} belongs to PID {info.get("pid")}, expected {pid}')
            if canonical(info.get('model_path', ''), cwd) != canonical(expected_path, cwd):
                raise RuntimeError(f'Port {port} loaded unexpected checkpoint: {info!r}')
            if info.get('model_type') != expected_type:
                raise RuntimeError(f'Port {port} loaded unexpected model type: {info!r}')
            print(f'[OK] port={port} PID={pid} checkpoint={info["model_path"]} engine={info["engine_class"]}')
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stop', 'verify'))
    parser.add_argument('--base-port', type=int, default=8000)
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--pid-file')
    args = parser.parse_args()
    if args.num_gpus < 1 or args.base_port < 1 or args.base_port + args.num_gpus > 65536 or args.timeout < 0:
        parser.error('Invalid port range or timeout')
    if args.action == 'verify' and not args.pid_file:
        parser.error('verify requires --pid-file')
    ports = set(range(args.base_port, args.base_port + args.num_gpus))
    try:
        if args.action == 'stop':
            stop(ports, args.timeout)
        else:
            verify(args.host, ports, args.pid_file, args.timeout)
    except Exception as error:
        parser.exit(1, f'[ERROR] {error}\n')


if __name__ == '__main__':
    main()
