from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.end_effectors.backends.omnihand import OmniHandBackend
from gear_sonic.end_effectors.backends.sim import SimHandBackend
from gear_sonic.end_effectors.controller import (
    HandControllerError,
    SafeHandController,
    TriggerHysteresis,
)
from gear_sonic.end_effectors.profiles import OMNIHAND_O10, HandSide
from gear_sonic.end_effectors.protocol import (
    HAND_INTENT_SCHEMA,
    HAND_INTENT_TOPIC,
    HandProtocolError,
    decode_intent,
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
    np.testing.assert_allclose(
        OMNIHAND_O10.right.open_rad[:3], (0.549773, -0.757866, 0.002735)
    )
    np.testing.assert_allclose(
        OMNIHAND_O10.left.open_rad[:3], (-0.555388, 0.802839, 0.0)
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

        def init(self):
            return True

        def get_product_type(self):
            return 7

        def get_joint_names(self):
            return list(OMNIHAND_O10.right.joint_names)

        def get_all_active_joint_angles(self):
            return [0.0] * 10

        def set_all_active_joint_angles(self, values):
            self.commands.append(values)

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
    backend = OmniHandBackend(
        HandSide.RIGHT,
        OMNIHAND_O10.right,
        "can10",
        command_enabled=True,
        sdk=sdk,
        net_class=net_class,
        interface_index=lambda _: 10,
        serial_reader=lambda *_: "2082395E534B50052",
        link_validator=lambda _: None,
    )
    assert hand.commands == []
    backend.write_positions(np.zeros(10))
    assert hand.commands == [[0.0] * 10]
