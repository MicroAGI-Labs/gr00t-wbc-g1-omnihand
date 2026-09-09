"""Run a standalone DEX 1 hand service alongside local or remote teleop."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from .backends.dex1 import DEFAULT_WORKER, USB_PORTS
from .protocol import DEFAULT_HAND_CONTROL_PORT, DEFAULT_HAND_INTENT_PORT, DEFAULT_HAND_STATE_PORT
from .supervisor import HandWorkerSupervisor


def host(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError("use an IPv4 address or hostname")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teleop-host", type=host, default="127.0.0.1")
    parser.add_argument("--state-bind-host", type=host, default="0.0.0.0")
    parser.add_argument("--intent-port", type=int, default=DEFAULT_HAND_INTENT_PORT)
    parser.add_argument("--state-port", type=int, default=DEFAULT_HAND_STATE_PORT)
    parser.add_argument("--control-port", type=int, default=DEFAULT_HAND_CONTROL_PORT)
    parser.add_argument("--dex1-worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--dex1-transition-duration", type=float, default=1.5)
    parser.add_argument("--enable-command", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Run the offline worker self-test; no motor I/O")
    return parser


def worker_command(args: argparse.Namespace) -> list[str]:
    ports = (args.intent_port, args.state_port, args.control_port)
    if any(not 1 <= port <= 65535 for port in ports) or len(set(ports)) != len(ports):
        raise ValueError("intent, state, and control ports must be distinct and within 1..65535")
    if not 1.35 <= args.dex1_transition_duration <= 30:
        raise ValueError("DEX 1 transition duration must be within 1.35..30 seconds")
    return [
        sys.executable,
        "-m",
        "gear_sonic.end_effectors.controller",
        "run",
        "--backend",
        "dex1",
        "--sides",
        "both",
        "--frequency",
        "50",
        "--intent-endpoint",
        f"tcp://{args.teleop_host}:{args.intent_port}",
        "--state-endpoint",
        f"tcp://{args.state_bind_host}:{args.state_port}",
        "--dex1-worker",
        str(args.dex1_worker),
        "--dex1-transition-duration",
        str(args.dex1_transition_duration),
        *(["--enable-command"] if args.enable_command else []),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        command = worker_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.check_only and not args.enable_command:
        parser.error("--enable-command is required to start physical hand control")
    if not os.access(args.dex1_worker, os.X_OK):
        parser.error("DEX 1 worker unavailable; run bash install_scripts/install_hand_server.sh")
    control_endpoint = f"tcp://{args.teleop_host}:{args.control_port}"
    print(f"Hands: {USB_PORTS}", flush=True)
    print(f"Reconnect endpoint: {control_endpoint}", flush=True)
    print(shlex.join(command), flush=True)
    if args.check_only:
        return subprocess.run([str(args.dex1_worker), "--self-test"], check=False).returncode
    return HandWorkerSupervisor(command, control_endpoint=control_endpoint).run()


if __name__ == "__main__":
    raise SystemExit(main())
