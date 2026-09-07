"""Own the hand worker, reconnect endpoint, and independent 50 Hz status clock."""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import time
import uuid

import zmq

from .protocol import (
    HAND_CONFIG_TOPIC,
    HAND_STATE_SCHEMA,
    HAND_STATE_TOPIC,
    RECOVERABLE_DISCONNECT_EXIT_CODE,
    decode_config,
    decode_control,
    decode_state,
    encode,
)


class HandWorkerSupervisor:
    """Keep SDK calls in a replaceable child; never block status on child exit."""

    def __init__(
        self,
        worker_command: list[str],
        *,
        state_endpoint: str = "tcp://*:5570",
        control_endpoint: str = "tcp://127.0.0.1:5572",
        frequency: float = 50.0,
        restart_delay_s: float = 1.0,
        terminate_timeout_s: float = 2.0,
        worker_timeout_s: float = 10.0,
    ):
        if not worker_command:
            raise ValueError("a hand worker command is required")
        if not all(math.isfinite(v) and v > 0 for v in (frequency, terminate_timeout_s, worker_timeout_s)):
            raise ValueError("frequency and timeouts must be finite and positive")
        if not math.isfinite(restart_delay_s) or restart_delay_s < 0:
            raise ValueError("restart delay must be finite and nonnegative")
        self.command = worker_command
        self.state_endpoint, self.control_endpoint = state_endpoint, control_endpoint
        self.period = 1.0 / frequency
        self.restart_delay_s, self.terminate_timeout_s = restart_delay_s, terminate_timeout_s
        self.worker_timeout_s = worker_timeout_s
        self.session_id = uuid.uuid4().hex
        self.worker_id = ""
        self.worker: subprocess.Popen | None = None
        self.state: dict | None = None
        self.config: dict | None = None
        self.error: str | None = None
        self.next_start: float | None = 0.0
        self.stop_deadline: float | None = None
        self.last_update = 0.0
        self.publish_sequence = 0

    def _start_worker(self, endpoint: str, now: float) -> None:
        self.worker_id = uuid.uuid4().hex
        environment = dict(
            os.environ,
            SONIC_HAND_SESSION_ID=self.session_id,
            SONIC_HAND_WORKER_ID=self.worker_id,
            SONIC_HAND_STATE_SINK=endpoint,
        )
        self.worker = subprocess.Popen(self.command, env=environment, start_new_session=True)
        self.state = self.config = None
        self.last_update = now
        self.next_start = None
        self.stop_deadline = None
        print(f"[Hands] Started worker {self.worker.pid}", flush=True)

    def request_reconnect(self, now: float) -> None:
        if self.stop_deadline is not None or self.next_start is not None:
            return  # Coalesce button presses while a restart is already pending.
        if self.worker is None or self.worker.poll() is not None:
            self.worker = None
            self.next_start = now + self.restart_delay_s
            return
        try:
            self.worker.terminate()
        except ProcessLookupError:
            pass
        self.stop_deadline = now + self.terminate_timeout_s

    def _accept_snapshot(self, raw: bytes, now: float) -> None:
        is_state = raw.startswith(HAND_STATE_TOPIC + b" ")
        payload = decode_state(raw) if is_state else decode_config(raw)
        if payload.get("session_id") != self.session_id or payload.get("worker_id") != self.worker_id:
            return  # Late IPC from a replaced worker must not revive old feedback.
        if is_state:
            self.state = payload
            self.last_update = now
            if payload.get("mode") != "disconnected":
                self.error = None
        else:
            self.config = payload

    def snapshot(self, now: float) -> dict:
        state = dict(self.state or {})
        stamp = state.get("monotonic_ns")
        age = (now - stamp / 1e9) if isinstance(stamp, int) and not isinstance(stamp, bool) else None
        restarting = self.stop_deadline is not None or self.next_start is not None
        live = self.worker is not None and self.worker.poll() is None and not restarting
        ready = live and age is not None and 0 <= age <= self.worker_timeout_s
        if not ready:
            state.update(
                schema=HAND_STATE_SCHEMA,
                mode="fault" if state.get("mode") == "fault" else "disconnected",
                input_stale=True,
                intent_sequence=None,
                sides={
                    side: {"valid": False, "connected": False}
                    for side in (self.config or {}).get("selected_sides", ("left", "right"))
                },
            )
        state.update(
            session_id=self.session_id,
            worker_id=self.worker_id,
            profile=state.get("profile", "omnihand_o10.v1"),
            publish_sequence=self.publish_sequence,
            published_monotonic_ns=int(now * 1e9),
            state_age_s=age if age is not None and age >= 0 else None,
            supervisor_error=self.error,
        )
        return state

    def _tick_worker(self, endpoint: str, now: float) -> bool:
        if self.worker is not None:
            status = self.worker.poll()
            if status is not None:
                restarting = self.stop_deadline is not None
                self.worker = None
                self.stop_deadline = None
                recoverable = (
                    status == RECOVERABLE_DISCONNECT_EXIT_CODE and (self.state or {}).get("mode") != "fault"
                )
                if restarting or recoverable:
                    self.next_start = now + self.restart_delay_s
                    self.error = f"Worker stopped ({status}); reconnecting"
                elif status in (0, -signal.SIGINT, 128 + signal.SIGINT):
                    return False
                else:
                    self.error = f"Worker exited with status {status}; manual reconnect required"
            elif self.stop_deadline is not None:
                if now >= self.stop_deadline:
                    try:
                        self.worker.kill()
                    except ProcessLookupError:
                        pass
                    self.stop_deadline = now + self.terminate_timeout_s
            elif now - self.last_update > self.worker_timeout_s:
                # A known motor fault stays latched, even if its worker later stalls.
                if (self.state or {}).get("mode") != "fault":
                    self.error = "Hand I/O stopped responding; reconnecting"
                    self.request_reconnect(now)
        if self.worker is None and self.next_start is not None and now >= self.next_start:
            self._start_worker(endpoint, now)
        return True

    def run(self) -> int:
        context = zmq.Context()
        updates = context.socket(zmq.PULL)
        updates.setsockopt(zmq.RCVHWM, 8)
        port = updates.bind_to_random_port("tcp://127.0.0.1")
        endpoint = f"tcp://127.0.0.1:{port}"
        publisher = context.socket(zmq.PUB)
        publisher.setsockopt(zmq.SNDHWM, 4)
        control = context.socket(zmq.REP)
        try:
            publisher.bind(self.state_endpoint)
            control.bind(self.control_endpoint)
            deadline = time.monotonic()
            last_config = -math.inf
            while True:
                now = time.monotonic()
                if control.poll(0):
                    try:
                        decode_control(control.recv())
                        self.request_reconnect(now)
                        reply = {"accepted": True}
                    except ValueError as exc:
                        reply = {"accepted": False, "error": str(exc)}
                    control.send_json(reply)
                for _ in range(16):
                    if not updates.poll(0):
                        break
                    try:
                        self._accept_snapshot(updates.recv(), now)
                    except ValueError as exc:
                        self.error = f"Invalid worker status: {exc}"
                now = time.monotonic()
                if not self._tick_worker(endpoint, now):
                    return 0
                state = self.snapshot(now)
                publisher.send(encode(HAND_STATE_TOPIC, state), flags=zmq.NOBLOCK)
                self.publish_sequence += 1
                if self.config is not None and now - last_config >= 2.0:
                    publisher.send(encode(HAND_CONFIG_TOPIC, self.config), flags=zmq.NOBLOCK)
                    last_config = now
                deadline = max(deadline + self.period, now)
                time.sleep(max(0.0, deadline - time.monotonic()))
        except KeyboardInterrupt:
            return 0
        finally:
            if self.worker is not None and self.worker.poll() is None:
                try:
                    self.worker.terminate()
                except ProcessLookupError:
                    pass
                try:
                    self.worker.wait(timeout=self.terminate_timeout_s)
                except subprocess.TimeoutExpired:
                    try:
                        self.worker.kill()
                    except ProcessLookupError:
                        pass
                    self.worker.wait(timeout=self.terminate_timeout_s)
            for sock in (updates, publisher, control):
                sock.close(linger=0)
            context.term()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-endpoint", default="tcp://*:5570")
    parser.add_argument("--control-endpoint", default="tcp://127.0.0.1:5572")
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--restart-delay", type=float, default=1.0)
    parser.add_argument("--terminate-timeout", type=float, default=2.0)
    parser.add_argument("--worker-timeout", type=float, default=10.0)
    parser.add_argument("worker_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.worker_command
    if command[:1] == ["--"]:
        command = command[1:]

    # Keep terminal shutdown in the supervisor so its child is always reaped.
    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    return HandWorkerSupervisor(
        command,
        state_endpoint=args.state_endpoint,
        control_endpoint=args.control_endpoint,
        frequency=args.frequency,
        restart_delay_s=args.restart_delay,
        terminate_timeout_s=args.terminate_timeout,
        worker_timeout_s=args.worker_timeout,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
