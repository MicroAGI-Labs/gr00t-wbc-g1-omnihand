from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time

import pytest
import zmq

from gear_sonic.end_effectors.backends.mujoco import MuJoCoHandTransportError
from gear_sonic.end_effectors.backends.omnihand import OmniHandHardwareError
from gear_sonic.end_effectors.controller import HandControllerError, _worker_failure_exit_code
from gear_sonic.end_effectors.protocol import (
    HAND_STATE_SCHEMA,
    HAND_STATE_TOPIC,
    RECOVERABLE_DISCONNECT_EXIT_CODE,
    decode_state,
    encode,
    hand_state_age_s,
)
from gear_sonic.end_effectors.supervisor import HandWorkerSupervisor
from gear_sonic.scripts.run_camera_web_viewer import CameraWebViewerConfig, RecorderControlHub


class Worker:
    def __init__(self):
        self.status = None
        self.signals = []

    def poll(self):
        return self.status

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.signals.append("kill")


def supervisor_with_worker():
    supervisor = HandWorkerSupervisor(["unused"], worker_timeout_s=1, terminate_timeout_s=0.2)
    supervisor.worker = Worker()
    supervisor.worker_id = "first"
    supervisor.next_start = None
    return supervisor


def test_restart_waits_for_old_worker_exit_before_replacement(monkeypatch):
    supervisor = supervisor_with_worker()
    worker = supervisor.worker
    supervisor.request_reconnect(10.0)
    supervisor.request_reconnect(10.1)
    assert worker.signals == ["terminate"]
    supervisor._tick_worker("unused", 10.3)
    assert worker.signals == ["terminate", "kill"]
    assert supervisor.worker is worker
    worker.status = -signal.SIGKILL
    supervisor._tick_worker("unused", 10.4)
    assert supervisor.next_start == pytest.approx(11.4)
    starts = []
    monkeypatch.setattr(supervisor, "_start_worker", lambda endpoint, now: starts.append(now))
    supervisor._tick_worker("unused", 11.3)
    assert starts == []
    supervisor._tick_worker("unused", 11.5)
    assert starts == [11.5]


@pytest.mark.parametrize("mode,expected", [("tracking", ["terminate"]), ("fault", [])])
def test_watchdog_restarts_stalled_io_but_preserves_motor_fault_latch(mode, expected):
    supervisor = supervisor_with_worker()
    supervisor.state = {"mode": mode}
    supervisor._tick_worker("unused", 2.0)
    assert supervisor.worker.signals == expected
    status = supervisor.snapshot(2.0)
    assert status["mode"] == ("fault" if mode == "fault" else "disconnected")
    assert not status["sides"]["left"]["valid"]


@pytest.mark.parametrize("exit_code,fault,restarts", [(75, False, True), (1, False, False), (75, True, False)])
def test_only_recoverable_exit_is_automatically_restarted(exit_code, fault, restarts):
    supervisor = supervisor_with_worker()
    supervisor.state = {"mode": "fault" if fault else "tracking"}
    supervisor.worker.status = exit_code
    supervisor._tick_worker("unused", 10.0)
    assert (supervisor.next_start is not None) is restarts
    if not restarts:
        supervisor.request_reconnect(11.0)
        assert supervisor.next_start == 12.0


@pytest.mark.parametrize(
    "error,recoverable",
    [
        (OmniHandHardwareError(), True),
        (MuJoCoHandTransportError(), True),
        (HandControllerError(), False),
        (ValueError(), False),
    ],
)
def test_worker_retries_backend_errors_but_not_bugs_or_latched_faults(error, recoverable):
    assert (_worker_failure_exit_code(error) == RECOVERABLE_DISCONNECT_EXIT_CODE) is recoverable
    assert _worker_failure_exit_code(error, fault_latched=True) == 1


def test_old_worker_packets_are_rejected_and_republished_state_keeps_source_age():
    supervisor = supervisor_with_worker()
    payload = dict(
        schema=HAND_STATE_SCHEMA,
        session_id=supervisor.session_id,
        worker_id="old",
        monotonic_ns=1_000_000_000,
        sequence=3,
        mode="tracking",
    )
    supervisor._accept_snapshot(encode(HAND_STATE_TOPIC, payload), 1.0)
    assert supervisor.state is None
    payload["worker_id"] = "first"
    supervisor._accept_snapshot(encode(HAND_STATE_TOPIC, payload), 1.0)
    first, later = supervisor.snapshot(1.1), supervisor.snapshot(1.3)
    assert first["sequence"] == later["sequence"] == 3
    assert first["monotonic_ns"] == later["monotonic_ns"] == 1_000_000_000
    assert hand_state_age_s(later, 0.1) == pytest.approx(0.4)


@pytest.mark.parametrize("age", [None, -1, float("nan"), float("inf"), True])
def test_supervised_state_requires_a_valid_age(age):
    assert hand_state_age_s({"worker_id": "worker", "state_age_s": age}) is None
    assert hand_state_age_s({}, 0.1) == 0.1  # Direct publisher compatibility.


def test_browser_status_expires_even_with_fresh_relay_messages(monkeypatch):
    hub = RecorderControlHub(CameraWebViewerConfig(hand_controls=True))
    assert hub.hands_status()["feedback_age_s"] is None
    hub._hand_received_at = 10.0
    hub._hand_status = {
        "worker_id": "worker",
        "state_age_s": 0.15,
        "sides": {"left": {"valid": True, "connected": True}},
    }
    monkeypatch.setattr(time, "monotonic", lambda: 10.04)
    assert hub.hands_status()["connected"]
    monkeypatch.setattr(time, "monotonic", lambda: 10.06)
    assert not hub.hands_status()["connected"]
    hub.config.hand_controls = False
    assert not hub.reconnect_hands()["accepted"]


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_real_supervisor_keeps_status_alive_and_browser_can_replace_hung_worker():
    # A CPU-only stand-in for an SDK that blocks and ignores termination.
    worker_script = """
import os, signal, time, zmq
from gear_sonic.end_effectors.protocol import HAND_STATE_TOPIC, HAND_STATE_SCHEMA, encode
signal.signal(signal.SIGTERM, signal.SIG_IGN)
ctx = zmq.Context()
out = ctx.socket(zmq.PUSH)
out.connect(os.environ['SONIC_HAND_STATE_SINK'])
out.send(encode(HAND_STATE_TOPIC, dict(
    schema=HAND_STATE_SCHEMA, session_id=os.environ['SONIC_HAND_SESSION_ID'],
    worker_id=os.environ['SONIC_HAND_WORKER_ID'], sequence=1,
    monotonic_ns=time.monotonic_ns(), mode='tracking', pid=os.getpid())))
while True:
    time.sleep(1)
"""
    state_endpoint = f"tcp://127.0.0.1:{unused_port()}"
    control_endpoint = f"tcp://127.0.0.1:{unused_port()}"
    with zmq.Context() as context, context.socket(zmq.SUB) as status:
        status.setsockopt(zmq.LINGER, 0)
        status.setsockopt(zmq.SUBSCRIBE, HAND_STATE_TOPIC)
        status.connect(state_endpoint)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gear_sonic.end_effectors.supervisor",
                "--state-endpoint",
                state_endpoint,
                "--control-endpoint",
                control_endpoint,
                "--terminate-timeout",
                "0.1",
                "--restart-delay",
                "0.05",
                "--",
                sys.executable,
                "-c",
                worker_script,
            ]
        )

        def receive(predicate):
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if status.poll(100):
                    message = decode_state(status.recv())
                    if predicate(message):
                        return message
            pytest.fail("supervisor did not publish expected status")

        try:
            first = receive(lambda message: message.get("mode") == "tracking")
            later = receive(lambda message: message.get("state_age_s", 0) > 0.15)
            assert later["sequence"] == first["sequence"]
            assert later["publish_sequence"] > first["publish_sequence"]
            hub = RecorderControlHub(
                CameraWebViewerConfig(
                    hand_controls=True,
                    hand_control_endpoint=control_endpoint,
                )
            )
            assert hub.reconnect_hands() == {"accepted": True}
            second = receive(
                lambda message: (
                    message.get("mode") == "tracking" and message.get("worker_id") != first["worker_id"]
                )
            )
            assert first["session_id"] == second["session_id"]
            assert first["pid"] != second["pid"]
            with pytest.raises(ProcessLookupError):
                os.kill(first["pid"], 0)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
                pytest.fail("supervisor did not clean up its worker")
