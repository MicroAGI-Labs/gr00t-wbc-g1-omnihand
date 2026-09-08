from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import zmq

from gear_sonic.end_effectors import controller as controller_module
from gear_sonic.end_effectors.backends.omnihand import (
    OmniHandBackend,
    OmniHandHardwareError,
    _canfd_link_mismatches,
)
from gear_sonic.end_effectors.backends.sim import SimHandBackend
from gear_sonic.end_effectors.controller import (
    HandControllerError,
    SafeHandController,
    TriggerHysteresis,
    _has_hard_motor_error,
)
from gear_sonic.end_effectors.profiles import OMNIHAND_O10, HandSide
from gear_sonic.end_effectors.protocol import (
    HAND_CONTROL_SCHEMA,
    HAND_CONTROL_TOPIC,
    HAND_INTENT_SCHEMA,
    HAND_INTENT_TOPIC,
    HAND_STATE_TOPIC,
    HandProtocolError,
    decode_intent,
    decode_state,
    encode,
)


def _intent(sequence: int, *, left_closed: bool, right_closed: bool, valid: bool = True):
    return {
        "schema": HAND_INTENT_SCHEMA,
        "sequence": sequence,
        "monotonic_ns": sequence,
        "source": "pico",
        "left": {"valid": valid, "closed": left_closed, "trigger": 0.8 if left_closed else 0.2},
        "right": {"valid": valid, "closed": right_closed, "trigger": 0.8 if right_closed else 0.2},
    }


def test_o10_profile_has_exact_bilateral_contract():
    assert OMNIHAND_O10.width == 10
    assert OMNIHAND_O10.right.joint_names[0] == "R_thumb_roll_joint"
    assert OMNIHAND_O10.left.joint_names[-1] == "L_pinky_pip_joint"
    expected_right_open = np.asarray(
        (
            0.5525805,
            -0.7803525,
            0.0013675,
            -0.0759570,
            0.0,
            0.0,
            0.0836715,
            0.0004200,
            0.0936660,
            0.0004200,
        )
    )
    mirror_sign = np.asarray((-1, -1, -1, -1, 1, 1, -1, 1, -1, 1))
    np.testing.assert_allclose(OMNIHAND_O10.right.open_rad, expected_right_open)
    np.testing.assert_allclose(
        OMNIHAND_O10.left.open_rad,
        expected_right_open * mirror_sign,
    )
    for side in (OMNIHAND_O10.left, OMNIHAND_O10.right):
        assert np.all(np.asarray(side.closed_rad) >= np.asarray(side.lower_rad))
        assert np.all(np.asarray(side.closed_rad) <= np.asarray(side.upper_rad))
        assert np.all(np.asarray(side.open_rad) >= np.asarray(side.lower_rad))
        assert np.all(np.asarray(side.open_rad) <= np.asarray(side.upper_rad))


def test_protocol_rejects_unknown_schema_and_bad_trigger():
    good = encode(HAND_INTENT_TOPIC, _intent(1, left_closed=False, right_closed=True))
    assert decode_intent(good)["right"]["closed"] is True
    bad_schema = dict(_intent(1, left_closed=False, right_closed=False), schema="sonic.hand_intent.v2")
    with pytest.raises(HandProtocolError):
        decode_intent(encode(HAND_INTENT_TOPIC, bad_schema))
    bad_trigger = _intent(2, left_closed=False, right_closed=False)
    bad_trigger["left"]["trigger"] = 1.1
    with pytest.raises(HandProtocolError):
        decode_intent(encode(HAND_INTENT_TOPIC, bad_trigger))


def test_trigger_hysteresis_retains_state_and_ignores_invalid_input():
    hysteresis = TriggerHysteresis()
    assert hysteresis.update("left", 0.61, True) is True
    assert hysteresis.update("left", 0.50, True) is True
    assert hysteresis.update("left", 0.0, False) is True
    assert hysteresis.update("left", 0.39, True) is False


def test_controller_holds_first_feedback_then_slews_independent_targets():
    now = [10.0]
    left = SimHandBackend(OMNIHAND_O10.left)
    right = SimHandBackend(OMNIHAND_O10.right)
    controller = SafeHandController(
        OMNIHAND_O10,
        {"left": left, "right": right},
        backend_name="sim",
        close_scale=0.5,
        clock=lambda: now[0],
    )
    assert len(left.commands) == 1  # zero-delta startup hold
    assert controller.accept_intent(_intent(1, left_closed=True, right_closed=False), now=10.0)
    now[0] = 10.1
    state = controller.step(now=now[0])
    np.testing.assert_allclose(
        state["sides"]["left"]["applied_position_rad"][0],
        OMNIHAND_O10.left.open_rad[0] - 0.1 * OMNIHAND_O10.left.velocity_rad_s[0],
    )
    np.testing.assert_allclose(
        state["sides"]["right"]["applied_position_rad"], OMNIHAND_O10.right.open_rad
    )
    assert state["sides"]["left"]["requested_position_rad"][0] == pytest.approx(
        OMNIHAND_O10.left.target(True, close_scale=0.5)[0]
    )


@pytest.mark.parametrize("close_scale", [0.35, 1.0])
def test_controller_completes_scaled_transition_in_one_second(close_scale):
    now = [10.0]
    left = SimHandBackend(OMNIHAND_O10.left)
    controller = SafeHandController(
        OMNIHAND_O10,
        {"left": left},
        backend_name="sim",
        close_scale=close_scale,
        target_timeout_s=2.0,
        transition_duration_s=1.0,
        clock=lambda: now[0],
    )
    assert controller.accept_intent(_intent(1, left_closed=True, right_closed=False), now=now[0])
    for step in range(1, 11):
        now[0] = 10.0 + step / 10.0
        controller.step(now=now[0])
    np.testing.assert_allclose(
        controller.applied["left"],
        OMNIHAND_O10.left.target(True, close_scale),
        atol=1e-12,
    )


def test_communication_bit_is_reported_without_becoming_a_motor_fault():
    assert not _has_hard_motor_error([16] * 10)
    assert _has_hard_motor_error([0, 1])
    assert _has_hard_motor_error([16, 18])

    class CommunicationWarningBackend(SimHandBackend):
        def read_health(self):
            health = super().read_health()
            health["error_masks"][0] = 16
            return health

    now = [0.0]
    left = CommunicationWarningBackend(OMNIHAND_O10.left)
    controller = SafeHandController(
        OMNIHAND_O10,
        {"left": left},
        backend_name="sim",
        target_timeout_s=1.0,
        clock=lambda: now[0],
    )
    assert not controller.fault_latched
    assert controller.accept_intent(
        _intent(1, left_closed=True, right_closed=False), now=now[0]
    )
    now[0] = 0.2
    state = controller.step(now=now[0])
    assert state["mode"] == "tracking"
    assert state["sides"]["left"]["error_masks"][0] == 16


def test_controller_rejects_replayed_sequence_and_watchdog_holds():
    now = [0.0]
    left = SimHandBackend(OMNIHAND_O10.left)
    controller = SafeHandController(OMNIHAND_O10, {"left": left}, backend_name="sim", clock=lambda: now[0])
    assert controller.accept_intent(_intent(2, left_closed=True, right_closed=False), now=0.0)
    assert not controller.accept_intent(_intent(2, left_closed=False, right_closed=False), now=0.1)
    assert not controller.accept_intent(_intent(3, left_closed=False, right_closed=False, valid=False), now=0.4)
    now[0] = 0.6
    before = controller.applied["left"].copy()
    state = controller.step(now=now[0])
    np.testing.assert_array_equal(controller.applied["left"], before)
    assert state["mode"] == "hold"
    assert state["input_stale"] is True


def test_startup_feedback_far_outside_limits_fails_closed():
    backend = SimHandBackend(OMNIHAND_O10.right)
    backend._positions[0] = OMNIHAND_O10.right.upper_rad[0] + 0.02
    with pytest.raises(HandControllerError, match="startup feedback"):
        SafeHandController(OMNIHAND_O10, {"right": backend}, backend_name="sim")


def test_motor_error_latches_fault_and_stops_future_tracking():
    class FaultableBackend(SimHandBackend):
        fault = False

        def read_health(self):
            result = super().read_health()
            result["error_masks"][0] = int(self.fault)
            return result

    now = [0.0]
    backend = FaultableBackend(OMNIHAND_O10.right)
    controller = SafeHandController(OMNIHAND_O10, {"right": backend}, backend_name="sim", clock=lambda: now[0])
    controller.accept_intent(_intent(1, left_closed=False, right_closed=True), now=0.0)
    backend.fault = True
    now[0] = 0.2
    state = controller.step(now=0.2)
    assert state["mode"] == "fault"
    command_count = len(backend.commands)
    assert not controller.accept_intent(_intent(2, left_closed=False, right_closed=False), now=0.3)
    now[0] = 0.4
    controller.step(now=0.4)
    assert len(backend.commands) == command_count


def test_bilateral_health_gate_precedes_every_write():
    class FaultableBackend(SimHandBackend):
        fault = False

        def read_health(self):
            result = super().read_health()
            result["error_masks"][0] = int(self.fault)
            return result

    startup_left = FaultableBackend(OMNIHAND_O10.left)
    startup_right = FaultableBackend(OMNIHAND_O10.right)
    startup_right.fault = True
    with pytest.raises(HandControllerError, match="startup motor error"):
        SafeHandController(
            OMNIHAND_O10,
            {"left": startup_left, "right": startup_right},
            backend_name="sim",
        )
    assert startup_left.commands == []
    assert startup_right.commands == []

    now = [0.0]
    left = FaultableBackend(OMNIHAND_O10.left)
    right = FaultableBackend(OMNIHAND_O10.right)
    controller = SafeHandController(
        OMNIHAND_O10,
        {"left": left, "right": right},
        backend_name="sim",
        clock=lambda: now[0],
    )
    controller.accept_intent(_intent(1, left_closed=True, right_closed=True), now=0.0)
    right.fault = True
    controller.step(now=0.2)
    assert len(left.commands) == 1  # startup hold only
    assert len(right.commands) == 1


def test_hardware_adapter_probes_without_writing_and_uses_explicit_joint_command(tmp_path):
    class FakeHand:
        def __init__(self):
            self.commands = []
            self.request_interval_ms = None
            self.frame_recv_timeout_ms = None

        def init(self):
            return True

        def get_product_type(self):
            return 7

        def set_request_interval(self, value):
            self.request_interval_ms = value

        def get_request_interval(self):
            return self.request_interval_ms

        def set_frame_recv_timeout(self, value):
            self.frame_recv_timeout_ms = value

        def get_frame_recv_timeout(self):
            return self.frame_recv_timeout_ms

        def get_joint_names(self):
            return list(OMNIHAND_O10.right.joint_names)

        def get_all_active_joint_angles(self):
            return [0.0] * 10

        def set_all_active_joint_angles(self, values):
            self.commands.append(values)

        def get_all_error_reports(self):
            return [
                SimpleNamespace(
                    stalled=False,
                    overheat=False,
                    over_current=False,
                    motor_except=False,
                    commu_except=False,
                )
                for _ in range(10)
            ]

        def get_all_temperature_reports(self):
            return [25.0] * 10

        def get_all_current_reports(self):
            return [0.0] * 10

    hand = FakeHand()
    sdk = SimpleNamespace(
        HandType=SimpleNamespace(LEFT=1, RIGHT=2),
        ProductType=SimpleNamespace(OMNIHAND_2025=7),
        OmniHand2025=SimpleNamespace(
            kDegreesOfActiveFreedom=10,
            create_hand_socketcan=lambda *_: hand,
        ),
    )
    net_class = tmp_path / "net"
    driver = tmp_path / "drivers" / "gs_usb"
    driver.mkdir(parents=True)
    device = net_class / "can10" / "device"
    device.mkdir(parents=True)
    (device / "driver").symlink_to(driver, target_is_directory=True)
    validated_interfaces = []
    backend = OmniHandBackend(
        HandSide.RIGHT,
        OMNIHAND_O10.right,
        "can10",
        command_enabled=True,
        sdk=sdk,
        net_class=net_class,
        interface_index=lambda _: 10,
        serial_reader=lambda *_: "2082395E534B50052",
        link_validator=validated_interfaces.append,
    )
    assert hand.commands == []
    assert hand.request_interval_ms == 0
    assert hand.frame_recv_timeout_ms == 50
    assert validated_interfaces == ["can10"]
    backend.write_positions(np.zeros(10))
    assert hand.commands == [[0.0] * 10]
    assert backend.read_health()["error_masks"] == [0] * 10
    assert validated_interfaces == ["can10"]

    with pytest.raises(OmniHandHardwareError, match="between 10 and 1000"):
        backend._configure_transport_timing(0, 9)


def test_canfd_admission_uses_machine_readable_link_state():
    document = [
        {
            "link_type": "can",
            "mtu": 72,
            "linkinfo": {
                "info_data": {
                    "ctrlmode": ["FD"],
                    "state": "ERROR-ACTIVE",
                    "berr_counter": {"tx": 0, "rx": 0},
                    "bittiming": {"bitrate": 1_000_000, "sample_point": "0.800"},
                    "data_bittiming": {"bitrate": 5_000_000, "sample_point": "0.750"},
                }
            },
            "operstate": "UP",
            "flags": ["UP", "LOWER_UP"],
        }
    ]
    assert _canfd_link_mismatches(document) == []

    document[0]["linkinfo"]["info_data"]["ctrlmode"] = []
    assert _canfd_link_mismatches(document) == ["FD mode"]

    document[0]["linkinfo"]["info_data"]["ctrlmode"] = ["FD"]
    document[0]["linkinfo"]["info_data"]["berr_counter"]["rx"] = 1
    assert _canfd_link_mismatches(document) == []

    assert _canfd_link_mismatches([]) == ["valid link data"]
    document[0]["flags"] = None
    assert "UP interface" in _canfd_link_mismatches(document)


def test_partial_connection_closes_opened_hand(monkeypatch):
    left = SimHandBackend(OMNIHAND_O10.left)

    def connect(args, side, profile):
        if side == "right":
            raise OmniHandHardwareError("right disconnected")
        return left

    monkeypatch.setattr(controller_module, "_make_hardware_device", connect)
    with pytest.raises(OmniHandHardwareError):
        controller_module._make_devices(SimpleNamespace(backend="omnihand", sides="both"), OMNIHAND_O10, None)
    assert left.closed


def test_server_reconnects_in_process_and_preserves_fault_latch(monkeypatch):
    stop, disconnected, fault = (threading.Event() for _ in range(3))
    devices, results = [], []

    class Backend(SimHandBackend):
        def read_positions(self):
            if disconnected.is_set():
                raise OmniHandHardwareError("test disconnect")
            return super().read_positions()

        def read_health(self):
            return {"error_masks": [int(fault.is_set())] * 10}

    def connect(*args):
        backend = Backend(OMNIHAND_O10.left)
        devices.append(backend)
        return {"left": backend}

    def sleep(seconds):
        if stop.wait(seconds):
            raise KeyboardInterrupt

    def endpoint():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return f"tcp://127.0.0.1:{listener.getsockname()[1]}"

    monkeypatch.setattr(controller_module, "_make_devices", connect)
    monkeypatch.setattr(controller_module.time, "sleep", sleep)
    state_endpoint, control_endpoint = endpoint(), endpoint()
    with (
        zmq.Context() as context,
        context.socket(zmq.SUB) as state,
        context.socket(zmq.REQ) as control,
        context.socket(zmq.PUB) as intent,
    ):
        for sock in (state, control, intent):
            sock.setsockopt(zmq.LINGER, 0)
        intent_port = intent.bind_to_random_port("tcp://127.0.0.1")
        state.setsockopt(zmq.SUBSCRIBE, HAND_STATE_TOPIC)
        state.connect(state_endpoint)
        control.setsockopt(zmq.RCVTIMEO, 1000)
        control.connect(control_endpoint)
        args = controller_module.build_parser().parse_args(
            [
                "run",
                "--sides",
                "left",
                "--state-endpoint",
                state_endpoint,
                "--control-endpoint",
                control_endpoint,
                "--reconnect-interval",
                "0.05",
                "--intent-endpoint",
                f"tcp://127.0.0.1:{intent_port}",
            ]
        )
        server = threading.Thread(target=lambda: results.append(controller_module.run(args)), daemon=True)
        server.start()
        last_sequence = 0

        def receive(predicate):
            nonlocal last_sequence
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if state.poll(50):
                    payload = decode_state(state.recv())
                    assert payload["sequence"] > last_sequence
                    last_sequence = payload["sequence"]
                    if predicate(payload):
                        return payload
            pytest.fail("hand server did not publish expected state")

        def reconnect():
            control.send(encode(HAND_CONTROL_TOPIC, {"schema": HAND_CONTROL_SCHEMA, "action": "reconnect"}))
            assert control.recv_json() == {"accepted": True}

        try:
            first = receive(lambda message: message["mode"] == "hold")
            control.send(encode(HAND_CONTROL_TOPIC, {"schema": HAND_CONTROL_SCHEMA, "action": "invalid"}))
            assert not control.recv_json()["accepted"]
            intent.send(HAND_INTENT_TOPIC + b" invalid")
            unchanged = receive(lambda message: message["sequence"] > first["sequence"] + 3)
            assert unchanged["connection_id"] == first["connection_id"]
            intent.send(encode(HAND_INTENT_TOPIC, _intent(1, left_closed=True, right_closed=False)))
            receive(lambda message: message["mode"] == "tracking")
            reconnect()
            second = receive(lambda message: message["connection_id"] != first["connection_id"])
            assert second["session_id"] == first["session_id"]
            assert second["mode"] == "hold" and second["intent_sequence"] is None
            assert devices[0].closed and len(devices) == 2
            disconnected.set()
            receive(lambda message: message["mode"] == "disconnected")
            disconnected.clear()
            third = receive(lambda message: message["mode"] == "hold")
            assert third["connection_id"] != second["connection_id"]
            fault.set()
            receive(lambda message: message["mode"] == "fault")
            disconnected.set()
            failed = receive(lambda message: message.get("connection_error") == "test disconnect")
            attempts = len(devices)
            disconnected.clear()
            fault.clear()
            receive(lambda message: message["monotonic_ns"] > failed["monotonic_ns"] + 150_000_000)
            assert len(devices) == attempts  # Transport loss cannot clear a latched fault.
            reconnect()
            recovered = receive(lambda message: message["mode"] == "hold")
            assert recovered["connection_id"] != third["connection_id"]
            assert recovered["session_id"] == first["session_id"]
        finally:
            stop.set()
            server.join(timeout=2)
            assert not server.is_alive()
        assert results == [0]
        assert all(device.closed for device in devices)
