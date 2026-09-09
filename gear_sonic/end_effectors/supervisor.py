"""Restart the external-hand worker without depending on a responsive vendor SDK."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
import signal
import subprocess
import time
import uuid

import zmq

from .controller import RECOVERABLE_DISCONNECT_EXIT_CODE
from .protocol import DEFAULT_HAND_CONTROL_PORT, HAND_CONTROL_TOPIC, decode_control


class HandWorkerSupervisor:
    """Own a hand worker and replace it after failure or a UI request."""

    def __init__(
        self,
        worker_command: Sequence[str],
        *,
        control_endpoint: str,
        restart_delay_s: float = 1.0,
        terminate_timeout_s: float = 2.0,
    ) -> None:
        if not worker_command:
            raise ValueError("hand worker command is required")
        if restart_delay_s < 0 or terminate_timeout_s <= 0:
            raise ValueError("invalid hand-worker restart timing")
        self.worker_command = list(worker_command)
        self.control_endpoint = control_endpoint
        self.restart_delay_s = restart_delay_s
        self.terminate_timeout_s = terminate_timeout_s

    def _stop_worker(self, worker: subprocess.Popen[bytes]) -> None:
        if worker.poll() is not None:
            return
        worker.terminate()
        try:
            worker.wait(timeout=self.terminate_timeout_s)
        except subprocess.TimeoutExpired:
            print("[Hands supervisor] Worker did not terminate; killing it")
            worker.kill()
            worker.wait(timeout=self.terminate_timeout_s)

    def run(self) -> int:
        os.environ.setdefault("SONIC_HAND_SESSION_ID", uuid.uuid4().hex)
        context = zmq.Context()
        control = context.socket(zmq.SUB)
        control.setsockopt(zmq.RCVHWM, 1)
        control.setsockopt(zmq.CONFLATE, 1)
        control.setsockopt(zmq.SUBSCRIBE, HAND_CONTROL_TOPIC)
        control.connect(self.control_endpoint)
        worker: subprocess.Popen[bytes] | None = None
        try:
            while True:
                if worker is None:
                    worker = subprocess.Popen(self.worker_command)
                    print(f"[Hands supervisor] Worker started with PID {worker.pid}")

                if control.poll(100):
                    latest = control.recv()
                    while control.poll(0):
                        latest = control.recv(zmq.NOBLOCK)
                    try:
                        request = decode_control(latest)
                    except Exception as exc:
                        print(f"[Hands supervisor] Ignoring invalid control request: {exc}")
                    else:
                        if request["action"] == "reconnect":
                            print("[Hands supervisor] Manual clean reconnect requested")
                            self._stop_worker(worker)
                            worker = None
                            if self.restart_delay_s:
                                time.sleep(self.restart_delay_s)
                            continue

                status = worker.poll()
                if status is None:
                    continue
                worker = None
                if status == RECOVERABLE_DISCONNECT_EXIT_CODE:
                    print(
                        "[Hands supervisor] Transport failed; "
                        f"restarting in {self.restart_delay_s:.1f}s"
                    )
                    if self.restart_delay_s:
                        time.sleep(self.restart_delay_s)
                    continue
                if status in {0, 128 + signal.SIGINT, -signal.SIGINT}:
                    return 0
                print(f"[Hands supervisor] Worker stopped with unexpected status {status}")
                return status
        except KeyboardInterrupt:
            return 0
        finally:
            if worker is not None:
                self._stop_worker(worker)
            control.close(linger=0)
            context.term()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-endpoint", default=f"tcp://localhost:{DEFAULT_HAND_CONTROL_PORT}"
    )
    parser.add_argument("--restart-delay", type=float, default=1.0)
    parser.add_argument("--terminate-timeout", type=float, default=2.0)
    parser.add_argument("worker_command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.worker_command)
    if command[:1] == ["--"]:
        command = command[1:]
    supervisor = HandWorkerSupervisor(
        command,
        control_endpoint=args.control_endpoint,
        restart_delay_s=args.restart_delay,
        terminate_timeout_s=args.terminate_timeout,
    )
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
