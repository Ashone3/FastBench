#!/usr/bin/env python3
"""CPU-only deployment checks for the optional framewise MOSS launcher."""
import argparse
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moss_framewise import verify_endpoint


def check_ports(host, base_port, num_gpus, wait_seconds=0):
    deadline = time.monotonic() + wait_seconds
    while True:
        failures = []
        for port in range(base_port, base_port + num_gpus):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    probe.bind((host, port))
            except OSError as error:
                failures.append(f"{host}:{port}: {error}")
        if not failures:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "Ports unavailable before model loading:\n  " + "\n  ".join(failures)
                + "\nStop the service that owns these ports, or set BASE_PORT to a free range "
                "in BOTH deployment and evaluation. Changing ports does not free GPU memory."
            )
        time.sleep(min(0.2, remaining))


def check_servers(host, base_port, num_gpus, pid_file):
    pids = Path(pid_file).read_text().splitlines()
    if len(pids) != num_gpus or any(not pid.isdigit() for pid in pids):
        raise RuntimeError(f"Expected {num_gpus} valid process IDs in {pid_file}")
    for offset, pid in enumerate(pids):
        port = base_port + offset
        verify_endpoint(f"http://{host}:{port}/v1", expected_pid=int(pid))
        print(f"[OK] {host}:{port}: MOSS framewise, PID {pid}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ports", "servers"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--pid-file")
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.num_gpus < 1 or args.base_port < 1 or args.base_port + args.num_gpus - 1 > 65535:
        parser.error("Invalid port range or number of GPUs")
    if args.mode == "servers" and not args.pid_file:
        parser.error("servers requires --pid-file")
    try:
        if args.mode == "ports":
            check_ports(args.host, args.base_port, args.num_gpus, max(0, args.wait_seconds))
        else:
            check_servers(args.host, args.base_port, args.num_gpus, args.pid_file)
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
