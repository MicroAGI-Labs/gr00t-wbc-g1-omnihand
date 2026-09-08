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
    monkeypatch.setitem(dex1.USB_PORTS, "right", str(stable_port))
    backend = dex1.Dex1Backend(HandSide.RIGHT, DEX1.right, worker=fake_worker, command_enabled=True)
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
        "right": {"valid": True, "closed": closed},
    }


def assert_stopped(log):
    modes = [int(line.split()[0]) for line in log.read_text().splitlines()]
    assert modes[-3:] == [0, 0, 0]


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


def test_native_watchdog_stops_without_python_cleanup(motor):
    backend, log = motor
    backend.write_positions(np.array([0.12]))
    backend.set_control_mode("tracking")
    run_for(backend, 0.1)
    assert backend._process.wait(timeout=1) == 2
    assert_stopped(log)
    with pytest.raises(dex1.Dex1SafetyError, match="exited"):
        backend.read_health()


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
    controller = SafeHandController(DEX1, {"right": backend}, backend_name="dex1", target_timeout_s=0.1)
    controller.accept_intent(intent(1))
    controller.step()
    time.sleep(0.12)
    controller.step()  # Worker still has a heartbeat, but source intent is stale.
    time.sleep(0.04)
    state = controller.step()
    assert state["mode"] == "hold"
    assert state["sides"]["right"]["applied_position_rad"] == pytest.approx([2.5])
    controller.accept_intent(intent(2, closed=False))
    controller.step()
    controller.accept_intent(intent(3, hold=True))
    controller.step()
    time.sleep(0.04)
    assert controller.step()["sides"]["right"]["applied_position_rad"] == pytest.approx([2.5])


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
        dex1.Dex1Backend(HandSide.RIGHT, DEX1.right, worker=fake_worker)
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
    joints = assemble_dataset_configuration(model, np.arange(29), [-2.36], [0.12], DEX1)
    assert joints[22] == pytest.approx(-2.36)
    assert joints[30] == pytest.approx(0.12)
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
    assert "--dex1-transition-duration 1.5" in command
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
    controller = SafeHandController(DEX1, {"right": backend}, backend_name="dex1")
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
