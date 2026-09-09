"""Safe external open/close controller for simulated or physical OmniHand O10."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np
import zmq

from .backends.base import HandBackend
from .backends.mujoco import MuJoCoHandTransport, MuJoCoHandTransportError, MuJoCoSimHandBackend
from .backends.omnihand import OmniHandBackend, OmniHandHardwareError, vendor_output_to_stderr
from .profiles import HandProfile, HandSide, get_hand_profile
from .protocol import (
    HAND_CONFIG_SCHEMA,
    HAND_CONFIG_TOPIC,
    HAND_INTENT_TOPIC,
    HAND_STATE_SCHEMA,
    HAND_STATE_TOPIC,
    HandProtocolError,
    decode_control,
    decode_intent,
    encode,
)

STARTUP_FEEDBACK_TOLERANCE_RAD = 0.01
HARD_MOTOR_ERROR_MASK = 0x0F


def _has_hard_motor_error(masks: Sequence[int]) -> bool:
    """Treat bits 0-3 as motor faults; bit 4 is communication telemetry."""
    return any(int(mask) & HARD_MOTOR_ERROR_MASK for mask in masks)


class HandControllerError(RuntimeError):
    pass


class TriggerHysteresis:
    def __init__(self, close_threshold: float = 0.60, open_threshold: float = 0.40) -> None:
        if not 0 <= open_threshold < close_threshold <= 1:
            raise ValueError("expected 0 <= open threshold < close threshold <= 1")
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.closed = {side.value: False for side in HandSide}

    def update(self, side: HandSide | str, trigger: float, valid: bool) -> bool:
        name = HandSide(side).value
        if not valid:
            return self.closed[name]
        if trigger >= self.close_threshold:
            self.closed[name] = True
        elif trigger <= self.open_threshold:
            self.closed[name] = False
        return self.closed[name]


class SafeHandController:
    """State machine shared by the real and deterministic simulation runtimes."""

    def __init__(
        self,
        profile: HandProfile,
        devices: Mapping[str, HandBackend],
        *,
        backend_name: str,
        close_scale: float = 0.35,
        target_timeout_s: float = 0.5,
        transition_duration_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        session_id: str | None = None,
    ) -> None:
        if not devices:
            raise HandControllerError("at least one side must be selected")
        if not 0 <= close_scale <= 1:
            raise ValueError("close_scale must be in [0, 1]")
        if target_timeout_s <= 0:
            raise ValueError("target_timeout_s must be positive")
        if transition_duration_s is not None and transition_duration_s <= 0:
            raise ValueError("transition_duration_s must be positive")
        self.profile, self.devices, self.backend_name = profile, dict(devices), backend_name
        self.close_scale, self.target_timeout_s, self.clock = close_scale, target_timeout_s, clock
        self.transition_duration_s = transition_duration_s
        self.session_id = uuid.uuid4().hex if session_id is None else session_id
        self.sequence = 0
        self.last_intent_sequence: int | None = None
        self.last_intent_at: float | None = None
        self.last_valid_intent_at: dict[str, float | None] = {side: None for side in self.devices}
        self.last_step_at = clock()
        self.mode = "hold"
        self.explicit_hold = False
        self.requested: dict[str, np.ndarray] = {}
        self.applied: dict[str, np.ndarray] = {}
        self.velocity_scale: dict[str, float] = {}
        self.measured: dict[str, np.ndarray] = {}
        self.health: dict[str, dict[str, Any]] = {}
        self.last_health_at = -math.inf
        self.intent_closed: dict[str, bool | None] = {side: None for side in self.devices}
        self.errors: dict[str, str | None] = {side: None for side in self.devices}
        self.fault_latched = False
        self._connect_hold()

    def _connect_hold(self) -> None:
        startup_positions: dict[str, np.ndarray] = {}
        startup_health: dict[str, dict[str, Any]] = {}
        for side, device in self.devices.items():
            side_profile = self.profile.side(side)
            measured = np.asarray(device.read_positions(), dtype=np.float64).reshape(-1)
            if measured.shape != (side_profile.width,) or not np.all(np.isfinite(measured)):
                raise HandControllerError(f"{side} startup feedback is invalid")
            bounded = np.clip(measured, side_profile.lower_rad, side_profile.upper_rad)
            if np.any(np.abs(bounded - measured) > STARTUP_FEEDBACK_TOLERANCE_RAD):
                raise HandControllerError(f"{side} startup feedback exceeds admitted limits")
            health = device.read_health()
            if _has_hard_motor_error(health.get("error_masks", ())):
                raise HandControllerError(f"{side} reported a startup motor error")
            startup_positions[side] = bounded
            startup_health[side] = health

        # Validate every selected side before the first physical write. The
        # initial command is a zero-delta measured hold, never an open pose.
        for side, device in self.devices.items():
            measured = startup_positions[side]
            device.write_positions(measured)
            self.measured[side] = measured.copy()
            self.requested[side] = measured.copy()
            self.applied[side] = measured.copy()
            self.velocity_scale[side] = 1.0
            self.health[side] = startup_health[side]
        self.mode = "hold"

    def accept_intent(self, payload: Mapping[str, Any], *, now: float | None = None) -> bool:
        sequence = int(payload["sequence"])
        if self.last_intent_sequence is not None and sequence <= self.last_intent_sequence:
            return False
        if self.fault_latched:
            return False
        hold = payload.get("hold", False)
        if hold and not self.explicit_hold:
            # Reuse the bilateral feedback/health admission and measured hold.
            self._connect_hold()
        accepted = False
        received_at = self.clock() if now is None else now
        for side in self.devices:
            side_intent = payload[side]
            if not side_intent["valid"]:
                continue
            side_profile = self.profile.side(side)
            target = self.requested[side] if hold else side_profile.target(side_intent["closed"], self.close_scale)
            if not np.array_equal(target, self.requested[side]):
                self.requested[side] = target
                if self.transition_duration_s is not None:
                    base_velocity = np.asarray(side_profile.velocity_rad_s, dtype=np.float64)
                    nominal_duration = float(np.max(np.abs(target - self.applied[side]) / base_velocity))
                    self.velocity_scale[side] = nominal_duration / self.transition_duration_s
            self.intent_closed[side] = bool(side_intent["closed"])
            self.last_valid_intent_at[side] = received_at
            accepted = True
        self.last_intent_sequence = sequence
        if accepted:
            self.last_intent_at = received_at
        if accepted or hold:
            self.explicit_hold = hold
            self.mode = "hold" if hold else "tracking"
        return accepted or hold

    def step(self, *, now: float | None = None) -> dict[str, Any]:
        checked_at = self.clock() if now is None else now
        dt = min(max(0.0, checked_at - self.last_step_at), 0.1)
        stale_by_side = {
            side: received is None or checked_at - received > self.target_timeout_s
            for side, received in self.last_valid_intent_at.items()
        }
        stale = any(stale_by_side.values())
        if all(stale_by_side.values()) and not self.fault_latched:
            self.mode = "hold"
        poll_health = checked_at - self.last_health_at >= 0.2
        if poll_health:
            # Health is a bilateral gate: discover all selected-side faults
            # before allowing a write to either hand in this cycle.
            for side, device in self.devices.items():
                self.health[side] = device.read_health()
                if _has_hard_motor_error(self.health[side].get("error_masks", ())):
                    self.fault_latched = True
                    self.mode = "fault"
                    self.errors[side] = "non-zero motor error mask (latched)"
        for side, device in self.devices.items():
            side_profile = self.profile.side(side)
            if not stale_by_side[side] and self.mode == "tracking" and not self.fault_latched:
                target = np.clip(self.requested[side], side_profile.lower_rad, side_profile.upper_rad)
                maximum = np.asarray(side_profile.velocity_rad_s) * self.velocity_scale[side] * dt
                safe = self.applied[side] + np.clip(target - self.applied[side], -maximum, maximum)
                device.write_positions(safe)
                self.applied[side] = safe
            self.measured[side] = np.asarray(device.read_positions(), dtype=np.float64).copy()
            if self.measured[side].shape != (side_profile.width,) or not np.all(np.isfinite(self.measured[side])):
                raise HandControllerError(f"{side} feedback became invalid")
        if poll_health:
            self.last_health_at = checked_at
        self.last_step_at = checked_at
        self.sequence += 1
        return self.state_payload(checked_at, stale=stale, stale_by_side=stale_by_side)

    def config_payload(self) -> dict[str, Any]:
        sides: dict[str, Any] = {}
        for side, device in self.devices.items():
            p = self.profile.side(side)
            sides[side] = {
                "joint_names": list(p.joint_names),
                "position_lower_rad": list(p.lower_rad),
                "position_upper_rad": list(p.upper_rad),
                "velocity_limit_rad_s": list(p.velocity_rad_s),
                "open_rad": list(p.open_rad),
                "closed_rad": list(p.closed_rad),
                "sdk_version": device.sdk_version,
                "sdk_commit": device.sdk_commit,
            }
        return {
            "schema": HAND_CONFIG_SCHEMA,
            "session_id": self.session_id,
            "backend": self.backend_name,
            "profile": self.profile.name,
            "units": "rad",
            "selected_sides": list(self.devices),
            "close_scale": self.close_scale,
            "target_timeout_s": self.target_timeout_s,
            "transition_duration_s": self.transition_duration_s,
            "sides": sides,
        }

    def state_payload(self, now: float, *, stale: bool, stale_by_side: Mapping[str, bool]) -> dict[str, Any]:
        return {
            "schema": HAND_STATE_SCHEMA,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "monotonic_ns": int(now * 1e9),
            "backend": self.backend_name,
            "profile": self.profile.name,
            "mode": self.mode,
            "target_source": "pico_open_close",
            "intent_sequence": self.last_intent_sequence,
            "input_stale": stale,
            "input_age_s": None if self.last_intent_at is None else max(0.0, now - self.last_intent_at),
            "sides": {
                side: {
                    "requested_position_rad": self.requested[side].tolist(),
                    "applied_position_rad": self.applied[side].tolist(),
                    "measured_position_rad": self.measured[side].tolist(),
                    "valid": True,
                    "connected": True,
                    "error": self.errors[side],
                    "intent_closed": self.intent_closed[side],
                    "input_stale": stale_by_side[side],
                    "input_age_s": (
                        None
                        if self.last_valid_intent_at[side] is None
                        else max(0.0, now - self.last_valid_intent_at[side])
                    ),
                    "feedback_age_s": 0.0,
                    **self.health.get(side, {}),
                }
                for side in self.devices
            },
        }

    def close(self) -> None:
        for device in self.devices.values():
            device.close()
        self.devices.clear()


def _selected_sides(value: str) -> tuple[str, ...]:
    return ("left", "right") if value == "both" else (value,)


def _make_hardware_device(args: argparse.Namespace, side: str, profile: HandProfile) -> HandBackend:
    p = profile.side(side)
    interface = args.left_interface if side == "left" else args.right_interface
    return OmniHandBackend(HandSide(side), p, interface, command_enabled=bool(args.enable_command))


def _make_devices(
    args: argparse.Namespace,
    profile: HandProfile,
    context: zmq.Context,
) -> dict[str, HandBackend]:
    sides = _selected_sides(args.sides)
    if args.backend == "sim":
        transport = MuJoCoHandTransport(
            args.sim_feedback_endpoint,
            sides,
            startup_timeout_s=args.sim_feedback_timeout,
            max_age_s=args.sim_feedback_max_age,
            context=context,
        )
        return {side: MuJoCoSimHandBackend(side, profile.side(side), transport) for side in sides}
    devices: dict[str, HandBackend] = {}
    try:
        for side in sides:
            devices[side] = _make_hardware_device(args, side, profile)
        return devices
    except Exception:
        for device in devices.values():
            device.close()
        raise


def probe(args: argparse.Namespace) -> int:
    if args.backend == "sim":
        try:
            import mujoco

            scene = (
                Path(__file__).resolve().parents[2]
                / "gear_sonic/data/robot_model/model_data/g1_omnihand/scene_49dof.xml"
            )
            model = mujoco.MjModel.from_xml_path(str(scene))
            passed = model.nu == 49 and model.neq == 12
            print(
                json.dumps(
                    {
                        "passed": passed,
                        "backend": "sim",
                        "scene": str(scene),
                        "body_actuators": 29,
                        "hand_actuators": model.nu - 29,
                        "passive_equalities": model.neq,
                    },
                    indent=2,
                )
            )
            return 0 if passed else 1
        except Exception as exc:
            print(json.dumps({"passed": False, "backend": "sim", "error": str(exc)}, indent=2))
            return 1
    profile = get_hand_profile("omnihand_o10.v1")
    devices: dict[str, HandBackend] = {}
    try:
        with vendor_output_to_stderr():
            for side in _selected_sides(args.sides):
                devices[side] = _make_hardware_device(args, side, profile)
        report = {
            side: {"positions_rad": device.read_positions().tolist(), "health": device.read_health()}
            for side, device in devices.items()
        }
        print(json.dumps({"passed": True, "profile": profile.name, "sides": report}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 1
    finally:
        for device in devices.values():
            device.close()


def run(args: argparse.Namespace) -> int:
    if args.backend == "omnihand" and not args.enable_command:
        raise HandControllerError("--enable-command is required for physical OmniHand writes")
    profile = get_hand_profile("omnihand_o10.v1")
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    # This endpoint is shared with pose/planner/manager topics. SUB filtering
    # leaves only hand intent, while a small HWM bounds latency. ZMQ_CONFLATE
    # is deliberately avoided because it can retain an unrelated last message
    # before topic filtering and intermittently starve this subscriber.
    subscriber.setsockopt(zmq.RCVHWM, 4)
    subscriber.setsockopt(zmq.SUBSCRIBE, HAND_INTENT_TOPIC)
    subscriber.connect(args.intent_endpoint)
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 2)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(args.state_endpoint)
    control = context.socket(zmq.REP)
    connection_id: str | None = None
    state_sequence = 0

    def publish(topic: bytes, payload: dict) -> None:
        nonlocal state_sequence
        payload = dict(payload, connection_id=connection_id)
        if topic == HAND_STATE_TOPIC:
            # Wire sequences belong to the server session, not SDK connections.
            state_sequence += 1
            payload["sequence"] = state_sequence
        try:
            publisher.send(encode(topic, payload), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass  # A slow status consumer must not block hand I/O.

    selected_sides = _selected_sides(args.sides)
    session_id = uuid.uuid4().hex
    controller: SafeHandController | None = None
    next_reconnect_at = 0.0
    last_error: str | None = None
    try:
        control.bind(args.control_endpoint)
        period = 1.0 / args.frequency
        last_config = -math.inf
        while True:
            started = time.monotonic()
            just_connected = False
            if control.poll(0):
                try:
                    decode_control(control.recv())
                except HandProtocolError as exc:
                    control.send_json({"accepted": False, "error": str(exc)})
                else:
                    # Acknowledge the request, not completion of SDK reconnect.
                    control.send_json({"accepted": True})
                    if controller is not None:
                        controller.close()
                        controller = None
                    next_reconnect_at = started
            if controller is None and started >= next_reconnect_at:
                devices: dict[str, HandBackend] = {}
                try:
                    devices = _make_devices(args, profile, context)
                    controller = SafeHandController(
                        profile,
                        devices,
                        backend_name=args.backend,
                        close_scale=args.close_scale,
                        target_timeout_s=args.target_timeout,
                        transition_duration_s=args.transition_duration,
                        session_id=session_id,
                    )
                    connection_id = uuid.uuid4().hex
                    last_error = None
                    last_config = -math.inf
                    # A reconnect is admitted from measured feedback only. Drop
                    # anything queued while disconnected and require a target
                    # published after this hold was established.
                    while subscriber.poll(0):
                        subscriber.recv(zmq.NOBLOCK)
                    just_connected = True
                except Exception as exc:
                    for device in devices.values():
                        device.close()
                    controller = None
                    last_error = str(exc)
                    retryable = isinstance(exc, (OmniHandHardwareError, MuJoCoHandTransportError))
                    next_reconnect_at = started + args.reconnect_interval if retryable else math.inf
                    print(f"[Hands] Connection failed: {exc}", flush=True)

            if controller is not None:
                try:
                    if not just_connected:
                        # A shared high-rate publisher can leave a backlog
                        # across streamer reconnects. Drain it in one cycle
                        # and apply only the newest intent; otherwise stale
                        # pre-restart sequences can take minutes to clear one
                        # message at a time while the hands remain in hold.
                        latest_raw = None
                        while subscriber.poll(0):
                            latest_raw = subscriber.recv(zmq.NOBLOCK)
                        if latest_raw is not None:
                            try:
                                intent = decode_intent(latest_raw)
                            except HandProtocolError as exc:
                                print(f"[Hands] Rejected intent: {exc}")
                            else:
                                controller.accept_intent(intent, now=started)
                    state = controller.step(now=started)
                    publish(HAND_STATE_TOPIC, state)
                    if started - last_config >= 2.0:
                        publish(HAND_CONFIG_TOPIC, controller.config_payload())
                        last_config = started
                except Exception as exc:
                    last_error = str(exc)
                    retryable = not controller.fault_latched and isinstance(
                        exc, (OmniHandHardwareError, MuJoCoHandTransportError)
                    )
                    controller.close()
                    controller = None
                    next_reconnect_at = started + args.reconnect_interval if retryable else math.inf
                    print(f"[Hands] Connection lost: {exc}", flush=True)
            if controller is None:
                publish(
                    HAND_STATE_TOPIC,
                    {
                        "schema": HAND_STATE_SCHEMA,
                        "session_id": session_id,
                        "monotonic_ns": int(started * 1e9),
                        "backend": args.backend,
                        "profile": profile.name,
                        "mode": "fault" if math.isinf(next_reconnect_at) else "disconnected",
                        "connection_error": last_error,
                        "target_source": "pico_open_close",
                        "intent_sequence": None,
                        "input_stale": True,
                        "input_age_s": None,
                        "sides": {
                            side: {"valid": False, "connected": False, "error": last_error}
                            for side in selected_sides
                        },
                    },
                )
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        return 0
    finally:
        if controller is not None:
            controller.close()
        subscriber.close(linger=0)
        publisher.close(linger=0)
        control.close(linger=0)
        context.term()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "run"):
        command = sub.add_parser(name)
        command.add_argument("--backend", choices=("sim", "omnihand"), default="sim")
        command.add_argument("--sides", choices=("left", "right", "both"), default="both")
        command.add_argument("--left-interface", default="can11")
        command.add_argument("--right-interface", default="can10")
        command.add_argument("--enable-command", action="store_true")
    runner = sub.choices["run"]
    runner.add_argument("--intent-endpoint", default="tcp://localhost:5556")
    runner.add_argument("--state-endpoint", default="tcp://*:5570")
    runner.add_argument("--control-endpoint", default="tcp://127.0.0.1:5572")
    runner.add_argument("--frequency", type=float, default=50.0)
    runner.add_argument("--target-timeout", type=float, default=0.5)
    runner.add_argument("--transition-duration", type=float, default=1.0)
    runner.add_argument("--close-scale", type=float, default=0.35)
    runner.add_argument("--reconnect-interval", type=float, default=1.0)
    runner.add_argument("--sim-feedback-endpoint", default="tcp://localhost:5571")
    runner.add_argument("--sim-feedback-timeout", type=float, default=5.0)
    runner.add_argument("--sim-feedback-max-age", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return probe(args) if args.command == "probe" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
