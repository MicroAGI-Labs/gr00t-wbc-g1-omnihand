from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
import tty
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np
import pytest
import zmq

from gear_sonic.data.features_sonic_vla import (
    assemble_dataset_configuration,
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
)
from gear_sonic.end_effectors.backends import dex1
from gear_sonic.end_effectors.backends.sim import SimHandBackend
from gear_sonic.end_effectors import controller as hand_controller
from gear_sonic.end_effectors.controller import SafeHandController
from gear_sonic.end_effectors.profiles import DEX1, HandSide, dataset_robot_type
from gear_sonic.scripts import launch_data_collection as launcher
from gear_sonic.scripts.launch_data_collection import DataCollectionLaunchConfig, _hand_worker_command
from gear_sonic.scripts.run_data_exporter import resolve_hand_profile


@pytest.fixture(scope="module")
def fake_worker(tmp_path_factory):
    if not shutil.which("g++"):
        pytest.skip("g++ is needed for native motor-worker contract tests")
    tests = Path(__file__).resolve().parent
    binary = tmp_path_factory.mktemp("dex1-native") / "dex1_worker"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            "-DDEX1_FAKE_MOTOR_FOR_TEST",
            f"-I{tests / 'fakes'}",
            str(tests.parent / "native/dex1_worker.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


@pytest.fixture
def motor(fake_worker, tmp_path, monkeypatch):
    master, slave = os.openpty()
    port = os.ttyname(slave)
    stable_port = tmp_path / "serial-by-id"
    stable_port.symlink_to(port)
    log = tmp_path / "motor.log"
    monkeypatch.setenv("DEX1_TEST_MOTOR_LOG", str(log))
    monkeypatch.setitem(dex1.USB_PORTS, "left", str(stable_port))
    backend = dex1.Dex1Backend(HandSide.LEFT, DEX1.left, worker=fake_worker, command_enabled=True)
    assert backend._process.args[1] == port
    try:
        yield backend, log
    finally:
        backend.close()
        os.close(slave)
        os.close(master)


def run_for(backend, duration):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        backend.read_positions()
        time.sleep(0.02)


def intent(sequence, *, closed=True, hold=False):
    return {
        "sequence": sequence,
        "hold": hold,
        "left": {"valid": True, "closed": closed},
    }


def assert_stopped(log):
    modes = [int(line.split()[0]) for line in log.read_text().splitlines()]
    assert modes[-3:] == [0, 0, 0]


@pytest.mark.parametrize(
    "side,raw,expected,offset",
    [
        ("right", 3.90671, 3.90671 - 2 * np.pi, 2 * np.pi),
        ("right", 2.84 + 2 * np.pi, 2.84, 2 * np.pi),
        ("right", -2.36 - 2 * np.pi, -2.36, -2 * np.pi),
        ("right", -2.36, -2.36, 0),
        ("left", 0.12 + 2 * np.pi, 0.12, 2 * np.pi),
        ("left", 5.30 - 2 * np.pi, 5.30, -2 * np.pi),
        ("left", 5.30, 5.30, 0),
    ],
)
def test_startup_turn_preserves_profile_positions_and_measured_hold(
    fake_worker, tmp_path, monkeypatch, side, raw, expected, offset
):
    master, slave = os.openpty()
    log = tmp_path / "motor.log"
    monkeypatch.setenv("DEX1_TEST_FEEDBACK_POSITION", str(raw))
    monkeypatch.setenv("DEX1_TEST_MOTOR_LOG", str(log))
    monkeypatch.setitem(dex1.USB_PORTS, side, os.ttyname(slave))
    backend = None
    try:
        backend = dex1.Dex1Backend(side, DEX1.side(side), worker=fake_worker, command_enabled=True)
        measured = backend.read_positions()
        assert measured == pytest.approx([expected], abs=2e-6)
        assert backend.read_health()["encoder_position_rad"] == pytest.approx([raw], abs=2e-6)
        assert backend.read_health()["encoder_offset_rad"] == pytest.approx([offset], abs=2e-6)
        backend.write_positions(measured)
        run_for(backend, 0.12)
        assert backend.applied_positions == pytest.approx(measured, abs=2e-6)
        commands = [line.split() for line in log.read_text().splitlines()]
        assert any(int(mode) == 1 for mode, _ in commands)
        assert max(abs(float(torque)) for _, torque in commands) < 1e-5
    finally:
        if backend is not None:
            backend.close()
            assert_stopped(log)
        os.close(slave)
        os.close(master)


@pytest.mark.parametrize(
    "raw,extra,detail,active_request",
    [
        (3.3, {}, "position limit", False),
        (-2.8, {}, "position limit", False),
        (3.3 + 2 * np.pi, {}, "position limit", False),
        (-2.36 + 4 * np.pi, {}, "position limit", False),
        (float("nan"), {}, "nonfinite feedback", False),
        (3.90671, {"DEX1_TEST_LOW_VOLTAGE": "1"}, "supply outside", False),
        (3.90671, {"DEX1_TEST_STARTUP_ENABLED": "1"}, "drive not disabled", False),
        (3.90671, {"DEX1_TEST_ACTIVE_POSITION": str(3.90671 + 2 * np.pi)}, "position limit", True),
        (3.90671, {"DEX1_TEST_ACTIVE_POSITION": str(3.90671 - 2 * np.pi)}, "position limit", True),
        (-2.36, {"DEX1_TEST_ACTIVE_POSITION": "3.2"}, "position limit", True),
    ],
)
def test_startup_turn_keeps_health_and_runtime_position_guards(
    fake_worker, tmp_path, raw, extra, detail, active_request
):
    master, slave = os.openpty()
    log = tmp_path / "exchanges.log"
    try:
        result = subprocess.run(
            [str(fake_worker), os.ttyname(slave), "1", "-2.60", "3.10", "1.35", "1"],
            input="C 1 0 -2.36\n", capture_output=True, text=True, timeout=3,
            env={**os.environ, "DEX1_TEST_FEEDBACK_POSITION": str(raw),
                 "DEX1_TEST_EXCHANGE_LOG": str(log), **extra},
        )
        assert result.returncode == 2, result.stderr
        assert detail in result.stderr
        modes = [int(line) for line in log.read_text().splitlines()]
        assert modes == ([0, 1, 0, 0, 0] if active_request else [0, 0, 0, 0])
        if not active_request:
            assert "startup position reference" not in result.stderr
    finally:
        os.close(slave)
        os.close(master)


@pytest.mark.parametrize(
    "fault,value,success,detail",
    [
        ("DEX1_TEST_STARTUP_FAILURES", "3", True, "startup recovered"),
        ("DEX1_TEST_STARTUP_FAILURES", "1000", False, "startup timed out with drive disabled"),
        ("DEX1_TEST_LOW_VOLTAGE", "1", False, "supply outside 24-64 V"),
        ("DEX1_TEST_ACTIVE_FAILURE", "1", False, "invalid motor response"),
    ],
)
def test_native_startup_retries_only_disabled_communication(fake_worker, tmp_path, fault, value, success, detail):
    master, slave = os.openpty()
    log = tmp_path / "exchanges.log"
    try:
        result = subprocess.run(
            [str(fake_worker), os.ttyname(slave), "0", "-0.1", "5.75", "1.5", "1"],
            input="C 1 0 2.5\n", capture_output=True, text=True, timeout=3,
            env={**os.environ, fault: value, "DEX1_TEST_EXCHANGE_LOG": str(log)},
        )
        expected_exit = 0 if success else 2 if fault == "DEX1_TEST_LOW_VOLTAGE" else dex1.TRANSPORT_EXIT_CODE
        assert result.returncode == expected_exit, result.stderr
        assert detail in result.stderr
        modes = [int(line) for line in log.read_text().splitlines()]
        if fault == "DEX1_TEST_STARTUP_FAILURES" and success:
            assert modes[:4] == [0, 0, 0, 0]
            assert modes.count(1) == 1
        elif fault == "DEX1_TEST_ACTIVE_FAILURE":
            assert modes == [0, 1, 0, 0, 0]  # one failed active request, then stops
        else:
            assert set(modes) == {0}
        if fault == "DEX1_TEST_LOW_VOLTAGE":
            assert len(modes) == 4  # health fault is never retried
    finally:
        os.close(slave)
        os.close(master)


@pytest.mark.parametrize("during_startup", [True, False])
def test_native_receive_backlog_is_cleared_only_while_disabled(fake_worker, tmp_path, during_startup):
    master, slave = os.openpty()
    tty.setraw(slave)
    log = tmp_path / "exchanges.log"
    env = {**os.environ, "DEX1_TEST_RX_MASTER_FD": str(master), "DEX1_TEST_EXCHANGE_LOG": str(log)}
    if during_startup:
        env["DEX1_TEST_RX_AT_STARTUP"] = "1"
    try:
        result = subprocess.run(
            [str(fake_worker), os.ttyname(slave), "0", "-0.1", "5.75", "1.5", "1"],
            input="C 1 0 2.5\n", capture_output=True, text=True, timeout=3,
            env=env, pass_fds=(master,),
        )
        modes = [int(line) for line in log.read_text().splitlines()]
        if during_startup:
            assert result.returncode == 0, result.stderr
            assert "startup recovered" in result.stderr
            assert modes[:3] == [0, 0, 1]
        else:
            assert result.returncode == dex1.TRANSPORT_EXIT_CODE, result.stderr
            assert "stale motor feedback" in result.stderr
            assert modes == [0, 1, 0, 0, 0]
        assert modes[-3:] == [0, 0, 0]
    finally:
        os.close(slave)
        os.close(master)


def test_native_pending_reply_prevents_next_command(fake_worker, tmp_path):
    master, slave = os.openpty()
    tty.setraw(slave)
    log = tmp_path / "exchanges.log"
    process = subprocess.Popen(
        [str(fake_worker), os.ttyname(slave), "0", "-0.1", "5.75", "1.5", "1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "DEX1_TEST_EXCHANGE_LOG": str(log)},
    )
    try:
        # First feedback establishes that startup is complete and the worker
        # is waiting for an operator command with its motor disabled.
        assert json.loads(process.stdout.readline())["mode"] == -1
        os.write(master, b"\0" * 130)
        process.wait(timeout=2)
        assert process.returncode == dex1.TRANSPORT_EXIT_CODE
        assert "stale motor feedback" in process.stderr.read()
        assert set(log.read_text().splitlines()) == {"0"}
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=2)
        os.close(slave)
        os.close(master)


@pytest.mark.parametrize(
    "side,adapter,motor_id,lower,upper,closed,opened",
    [
        ("left", "FTBQ776H", 0, -0.10, 5.75, 0.12, 5.30),
        ("right", "FTBWJBC1", 1, -2.60, 3.10, -2.36, 2.84),
    ],
)
def test_physical_side_keeps_adapter_id_and_calibration_together(
    fake_worker, monkeypatch, side, adapter, motor_id, lower, upper, closed, opened
):
    assert dex1.USB_PORTS[side].endswith(f"{adapter}-if00-port0")
    master, slave = os.openpty()
    backend = None
    try:
        port = os.ttyname(slave)
        monkeypatch.setitem(dex1.USB_PORTS, side, port)
        backend = dex1.Dex1Backend(side, DEX1.side(side), worker=fake_worker, command_enabled=True)
        assert backend._process.args[1:5] == [port, str(motor_id), str(lower), str(upper)]
        assert backend.read_health()["motor_id"] == motor_id
        controller = SafeHandController(DEX1, {side: backend}, backend_name="dex1")
        for sequence, (is_closed, target) in enumerate(((True, closed), (False, opened)), start=1):
            controller.accept_intent(
                {"sequence": sequence, "hold": False, side: {"valid": True, "closed": is_closed}}
            )
            controller.step()
            assert backend._target == pytest.approx(target)
    finally:
        if backend is not None:
            backend.close()
        os.close(slave)
        os.close(master)


def test_contact_holds_at_capped_torque_and_accepts_release(motor):
    backend, log = motor
    backend.write_positions(np.array([0.12]))
    backend.set_control_mode("tracking")
    run_for(backend, 1.7)
    assert backend.applied_positions[0] == pytest.approx(0.12, abs=0.02)
    assert backend.read_health()["torque_nm"][0] == pytest.approx(-1)
    # Stationary test motor emulates a grasped object, not successful travel.
    assert backend.read_positions()[0] == 2.5
    backend.write_positions(np.array([2.6]))
    run_for(backend, 0.8)
    assert backend.read_health()["torque_nm"][0] > 0
    commands = [float(line.split()[1]) for line in log.read_text().splitlines()]
    assert max(map(abs, commands)) <= 1
    assert len(commands) > 250  # Motor I/O continues between 50 Hz caller updates.


@pytest.mark.parametrize(
    "variable,value,health_key",
    [
        ("DEX1_TEST_FEEDBACK_SPEED", "12", "velocity_rad_s"),
        ("DEX1_TEST_FEEDBACK_SPEED", "-12", "velocity_rad_s"),
        ("DEX1_TEST_FEEDBACK_TORQUE", "2", "torque_nm"),
        ("DEX1_TEST_FEEDBACK_TORQUE", "-2", "torque_nm"),
    ],
)
def test_high_speed_or_torque_feedback_keeps_hand_connected(monkeypatch, request, variable, value, health_key):
    monkeypatch.setenv(variable, value)
    backend, log = request.getfixturevalue("motor")
    controller = SafeHandController(DEX1, {"left": backend}, backend_name="dex1")
    controller.accept_intent(intent(1))
    for _ in range(6):
        state = controller.step()
        assert state["sides"]["left"]["connected"] is True
        assert state["sides"]["left"][health_key] == pytest.approx([float(value)])
        time.sleep(0.02)
    assert backend._process.poll() is None
    commands = [line.split() for line in log.read_text().splitlines()]
    assert any(int(mode) == 1 for mode, _ in commands)
    assert max(abs(float(torque)) for _, torque in commands) <= 1


def test_native_watchdog_stops_without_python_cleanup(motor):
    backend, log = motor
    backend.write_positions(np.array([0.12]))
    backend.set_control_mode("tracking")
    run_for(backend, 0.1)
    assert backend._process.wait(timeout=1) == 2
    assert_stopped(log)
    with pytest.raises(dex1.Dex1SafetyError, match="exited"):
        backend.read_health()


@pytest.mark.parametrize("read_method", ["read_positions", "read_health"])
def test_native_usb_failure_is_recoverable_and_stops_motor(monkeypatch, request, read_method):
    monkeypatch.setenv("DEX1_TEST_ACTIVE_FAILURE", "1")
    backend, _ = request.getfixturevalue("motor")
    backend.write_positions(np.array([2.5]))
    # A failed enabled exchange must stop the worker, not retry torque commands.
    with pytest.raises(dex1.Dex1TransportError, match="USB communication lost"):
        backend.read_positions()
        backend._process.wait(timeout=1)
        getattr(backend, read_method)()
    assert backend._process.returncode == dex1.TRANSPORT_EXIT_CODE


def test_missing_adapter_is_recoverable(tmp_path, monkeypatch):
    monkeypatch.setitem(dex1.USB_PORTS, "left", str(tmp_path / "absent-adapter"))
    with pytest.raises(dex1.Dex1TransportError, match="adapter unavailable"):
        dex1.Dex1Backend(HandSide.LEFT, DEX1.left)


@pytest.mark.parametrize("exit_code,error", [(75, dex1.Dex1TransportError), (2, dex1.Dex1SafetyError)])
def test_command_pipe_exit_race_preserves_native_fault(motor, monkeypatch, exit_code, error):
    backend, _ = motor
    backend.write_positions(np.array([2.5]))
    calls = [0]
    def poll():
        calls[0] += 1
        return None if calls[0] == 1 else exit_code
    def broken_pipe(*args):
        raise BrokenPipeError()
    with monkeypatch.context() as patch:
        patch.setattr(backend._process, "poll", poll)
        patch.setattr(backend._process, "wait", lambda **kwargs: exit_code)
        patch.setattr(dex1.os, "write", broken_pipe)
        with pytest.raises(error):
            backend.read_positions()


@pytest.mark.parametrize("phase", ["startup", "runtime"])
@pytest.mark.parametrize("recoverable", [True, False])
def test_service_retries_usb_failures_but_latches_motor_faults(monkeypatch, phase, recoverable):
    attempts, devices, states, queue = [], [], [], []
    clock = [0.]
    fault = dex1.Dex1TransportError if recoverable else dex1.Dex1SafetyError

    class Device(SimHandBackend):
        def __init__(self, fails):
            super().__init__(DEX1.left, initial=np.array([2.5]))
            self.fails = fails
            self.reads = 0

        def read_health(self):
            self.reads += 1
            if self.fails and self.reads > 1:
                raise fault("injected runtime fault")
            return super().read_health()

    def connect(*args):
        attempts.append(clock[0])
        assert len(attempts) <= 3
        queue.append(b"old queued intent must be discarded before decoding")
        if phase == "startup" and len(attempts) <= 2:
            raise fault("injected startup fault")
        device = Device(phase == "runtime" and len(attempts) <= 2)
        devices.append(device)
        return {"left": device}

    class Publisher:
        def __init__(self, *args): pass
        def start(self): pass
        def close(self): pass
        def update_config(self, config): pass
        def update_state(self, state):
            states.append(state)
            if state["sides"]["left"]["connected"]:
                assert state["mode"] == "hold"
                assert state["sides"]["left"]["measured_position_rad"] == [2.5]
                raise KeyboardInterrupt
            if sum(s["mode"] == "fault" for s in states) >= 100:
                raise KeyboardInterrupt  # More than one retry interval elapsed.

    subscriber = SimpleNamespace(setsockopt=lambda *a: None, connect=lambda *a: None,
                                 poll=lambda *a: bool(queue), recv=lambda *a: queue.pop(0),
                                 close=lambda **kw: None)
    context = SimpleNamespace(socket=lambda *a: subscriber, term=lambda: None)
    monkeypatch.setattr(hand_controller.zmq, "Context", lambda: context)
    monkeypatch.setattr(hand_controller, "FixedRateHandStatePublisher", Publisher)
    monkeypatch.setattr(hand_controller, "_make_devices", connect)
    monkeypatch.setattr(hand_controller, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
    args = hand_controller.build_parser().parse_args(["run", "--backend", "dex1", "--sides", "left",
                                                      "--enable-command"])
    assert hand_controller.run(args) == 0
    assert len(attempts) == (3 if recoverable else 1)
    assert all(b - a >= args.reconnect_interval for a, b in zip(attempts, attempts[1:]))
    assert all(device.closed for device in devices)
    for device in devices:
        np.testing.assert_array_equal(device.commands, [[2.5]])
    assert any(state["mode"] == "fault" for state in states) == (not recoverable)


def test_parent_pipe_eof_stops_motor(motor):
    backend, log = motor
    backend.write_positions(np.array([0.12]))
    backend.set_control_mode("tracking")
    run_for(backend, 0.1)
    backend._process.stdin.close()
    assert backend._process.wait(timeout=1) == 0
    assert_stopped(log)


def test_stale_teleop_cancels_motion_at_measured_position(motor):
    backend, _ = motor
    controller = SafeHandController(DEX1, {"left": backend}, backend_name="dex1", target_timeout_s=0.1)
    controller.accept_intent(intent(1))
    controller.step()
    time.sleep(0.12)
    controller.step()  # Worker still has a heartbeat, but source intent is stale.
    time.sleep(0.04)
    state = controller.step()
    assert state["mode"] == "hold"
    assert state["sides"]["left"]["applied_position_rad"] == pytest.approx([2.5])
    controller.accept_intent(intent(2, closed=False))
    controller.step()
    controller.accept_intent(intent(3, hold=True))
    controller.step()
    time.sleep(0.04)
    assert controller.step()["sides"]["left"]["applied_position_rad"] == pytest.approx([2.5])


@pytest.mark.parametrize("packet", [b"C 0 1 2.5\n", b"C 1 1 nan\n", b"C 1 1 8\n", b"C 1 9 2.5\n"])
def test_invalid_native_command_stops_motor(motor, packet):
    backend, log = motor
    os.write(backend._process.stdin.fileno(), packet)
    assert backend._process.wait(timeout=1) == 2
    assert_stopped(log)


def test_opening_obstruction_latches_fault(motor):
    backend, log = motor
    backend.write_positions(np.array([5.3]))
    backend.set_control_mode("tracking")
    with pytest.raises(dex1.Dex1SafetyError):
        run_for(backend, 2)
    backend._process.wait(timeout=1)
    assert_stopped(log)


def test_second_worker_cannot_claim_same_serial_port(motor, fake_worker):
    backend, _ = motor
    with pytest.raises(dex1.Dex1SafetyError):
        dex1.Dex1Backend(HandSide.LEFT, DEX1.left, worker=fake_worker)
    assert backend._process.poll() is None


def test_probe_backend_never_enables_motor(motor):
    backend, log = motor
    backend.read_positions()
    backend.close()
    assert all(line.split()[0] == "0" for line in log.read_text().splitlines())


def test_stale_native_snapshot_is_not_refreshed_by_read(motor):
    backend, _ = motor
    backend._drain = lambda: None
    backend._latest["monotonic_ns"] = time.monotonic_ns() - 1_000_000_000
    with pytest.raises(dex1.Dex1SafetyError, match="stale"):
        backend.read_health()


def test_dex1_dataset_uses_two_native_channels_and_correct_offsets():
    model = get_g1_robot_model()
    features = get_features_sonic_vla(model, DEX1)
    assert features["observation.state"]["shape"] == (31,)
    assert features["action.wbc"]["shape"] == (31,)
    assert features["control.hand_applied_position"]["shape"] == (2,)
    assert features["observation.dex1_left_raw"]["shape"] == (1,)
    assert "observation.omnihand_left_raw" not in features
    joints = assemble_dataset_configuration(model, np.arange(29), DEX1.left.closed_rad, DEX1.right.closed_rad, DEX1)
    assert joints[22] == pytest.approx(0.12)
    assert joints[30] == pytest.approx(-2.36)
    assert get_modality_config_sonic_vla(model, DEX1)["state"]["right_hand"] == {"start": 30, "end": 31}
    assert dataset_robot_type(DEX1) == "unitree_g1_dex1_sonic"
    assert resolve_hand_profile("auto", {"profile": "dex1.v1"}) is DEX1
    with pytest.raises(RuntimeError, match="controller reports"):
        resolve_hand_profile("omnihand_o10.v1", {"profile": "dex1.v1"})


def test_launcher_selects_dex1_worker_and_preserves_omnihand():
    config = DataCollectionLaunchConfig(hand_backend="dex1", remote_ui=True)
    venv, command = _hand_worker_command(config)
    assert venv == ".venv_data_collection"
    assert "--backend dex1" in command and "--enable-command" in command
    assert "--dex1-transition-duration 1.35" in command
    venv, command = _hand_worker_command(DataCollectionLaunchConfig(hand_backend="omnihand"))
    assert venv == ".venv_omnihand"
    assert "--transition-duration 0.2" in command


def test_complete_launcher_connects_external_hands_and_browser(monkeypatch):
    commands = {}
    for name in ("_check_prerequisites", "_switch_camera_source", "_create_tmux_session"):
        monkeypatch.setattr(launcher, name, lambda config: None)
    for name in ("_kill_existing_session", "_select_dashboard"):
        monkeypatch.setattr(launcher, name, lambda: None)
    monkeypatch.setattr(launcher, "_send_to_pane", lambda pane, cmd, **kw: commands.update({pane: cmd}))
    monkeypatch.setattr(launcher, "_check_pane_alive", lambda pane: True)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kw: SimpleNamespace(returncode=0))
    config = DataCollectionLaunchConfig(hand_backend="dex1", remote_ui=True)
    launcher.main(config)
    assert "--hand-control external" in commands[0]
    assert "--enable-hand-controls" in commands[3]
    assert "--backend dex1" in commands[4]
    assert "end_effectors.supervisor" in commands[4]
    assert "--hand-state-port 5570" in commands[2]


def test_panes_use_worktree_source_even_with_existing_tmux(monkeypatch):
    commands = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    launcher._send_to_pane(4, "python -m gear_sonic.end_effectors.controller", wait=0)
    assert commands[0][4].startswith(f"export PYTHONPATH={launcher._SOURCE_ROOT} && ")


def test_partial_bilateral_start_closes_first_motor(monkeypatch):
    from gear_sonic.end_effectors import controller

    first = SimpleNamespace(close=lambda: closed.append(True))
    closed = []

    def make(args, side, profile):
        if side == "left":
            return first
        raise dex1.Dex1SafetyError("right unavailable")

    monkeypatch.setattr(controller, "_make_hardware_device", make)
    with pytest.raises(dex1.Dex1SafetyError):
        controller._make_devices(SimpleNamespace(sides="both", backend="dex1"), DEX1, None)
    assert closed == [True]


def test_browser_displays_dex1_state_and_sends_reconnect(motor):
    from gear_sonic.end_effectors.protocol import HAND_STATE_TOPIC, decode_control, encode
    from gear_sonic.scripts.run_camera_web_viewer import CameraWebViewerConfig, HandControlHub, make_handler

    backend, _ = motor
    controller = SafeHandController(DEX1, {"left": backend}, backend_name="dex1")
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    state_port = publisher.bind_to_random_port("tcp://127.0.0.1")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        control_port = reservation.getsockname()[1]
    hub = HandControlHub(
        CameraWebViewerConfig(
            enable_hand_controls=True,
            hand_state_port=state_port,
            hand_control_port=control_port,
        )
    )
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"hand_control")
    subscriber.connect(f"tcp://127.0.0.1:{control_port}")
    hub.start()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            SimpleNamespace(health=lambda: {"streaming": False}),
            SimpleNamespace(status=lambda: {"connected": False}),
            hub,
            SimpleNamespace(),
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    pump_stop = threading.Event()
    pump_errors = []

    def publish_state():
        try:
            while not pump_stop.is_set():
                publisher.send(encode(HAND_STATE_TOPIC, controller.step()))
                pump_stop.wait(0.02)
        except Exception as exc:
            pump_errors.append(exc)

    pump = threading.Thread(target=publish_state, daemon=True)
    pump.start()
    try:
        deadline = time.monotonic() + 2
        while not hub.status()["connected"] and time.monotonic() < deadline:
            time.sleep(0.02)
        # Both ZMQ directions must establish their independent subscriptions.
        time.sleep(0.2)
        with urlopen(endpoint + "/", timeout=2) as response:
            assert b"hand-reconnect" in response.read()
        with urlopen(endpoint + "/hands/status", timeout=2) as response:
            status = json.load(response)
            assert status["connected"] and status["backend"] == "dex1"
            assert status["profile"] == "dex1.v1"
        with urlopen(Request(endpoint + "/hands/reconnect", method="POST"), timeout=2) as response:
            assert json.load(response)["accepted"]
        assert subscriber.poll(1000)
        assert decode_control(subscriber.recv())["action"] == "reconnect"
        assert not pump_errors
    finally:
        pump_stop.set()
        pump.join(timeout=2)
        backend.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        hub.close()
        subscriber.close(linger=0)
        publisher.close(linger=0)
        context.term()
