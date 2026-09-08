"""External open/close controller for OmniHand O10 and DEX 1."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
from typing import Any
import uuid

import numpy as np
import zmq

from .backends.base import HandBackend
from .backends.dex1 import DEFAULT_WORKER, Dex1Backend, Dex1SafetyError
from .backends.mujoco import MuJoCoHandTransport, MuJoCoSimHandBackend
from .backends.omnihand import OmniHandBackend, vendor_output_to_stderr
from .profiles import HandProfile, HandSide, get_hand_profile
from .protocol import (
    DEFAULT_HAND_INTENT_PORT,
    DEFAULT_HAND_STATE_PORT,
    HAND_CONFIG_SCHEMA,
    HAND_CONFIG_TOPIC,
    HAND_INTENT_TOPIC,
    HAND_STATE_SCHEMA,
    HAND_STATE_TOPIC,
    decode_intent,
    encode,
)

STARTUP_FEEDBACK_TOLERANCE_RAD = 0.01
HOLD_ERROR_LIMIT_RAD = 0.02
HARD_MOTOR_ERROR_MASK = 0x0F
COMMUNICATION_ERROR_MASK = 0x10
RECOVERABLE_DISCONNECT_EXIT_CODE = 75


def _has_hard_motor_error(masks: Sequence[int]) -> bool:
    """Treat bits 0-3 as motor faults; bit 4 is transport telemetry."""
    return any(int(mask) & HARD_MOTOR_ERROR_MASK for mask in masks)


class HandControllerError(RuntimeError):
    pass


class FixedRateHandStatePublisher:
    """Publish the latest controller snapshot outside the hand-I/O process.

    Physical feedback and health queries may hold the Python GIL while waiting
    for CAN replies. A spawned publisher keeps its 50 Hz clock independent.
    The controller timestamp and sequence are deliberately retained so a
    consumer can distinguish a fresh control update from a held snapshot.
    """

    def __init__(self, endpoint: str, frequency: float) -> None:
        if frequency <= 0:
            raise ValueError("hand-state publish frequency must be positive")
        self.endpoint = endpoint
        self.frequency = float(frequency)
        self.period = 1.0 / self.frequency
        context = mp.get_context("spawn")
        self._stop = context.Event()
        self._ready = context.Event()
        self._updates_rx, self._updates_tx = context.Pipe(duplex=False)
        self._error_rx, self._error_tx = context.Pipe(duplex=False)
        self._parent_pid = os.getpid()
        self._process: mp.Process | None = None
        self._state: dict[str, Any] | None = None
        self._config: dict[str, Any] | None = None
        self._startup_error: BaseException | None = None
        self._publish_sequence = 0
        self._deadline_misses = 0

    def update_state(self, state: Mapping[str, Any]) -> None:
        self._state = copy.deepcopy(dict(state))
        if self._process is not None:
            self._updates_tx.send(("state", self._state))

    def update_config(self, config: Mapping[str, Any]) -> None:
        payload = copy.deepcopy(dict(config))
        payload["state_publish_frequency_hz"] = self.frequency
        self._config = payload
        if self._process is not None:
            self._updates_tx.send(("config", payload))

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("hand-state publisher is already started")
        process = mp.get_context("spawn").Process(
            target=self._run,
            name="hand-state-publisher",
            daemon=True,
        )
        process.start()
        self._process = process
        if not self._ready.wait(timeout=5.0):
            raise HandControllerError("timed out starting hand-state publisher")
        if self._error_rx.poll():
            self._startup_error = RuntimeError(self._error_rx.recv())
        if self._startup_error:
            raise HandControllerError(
                f"could not start hand-state publisher: {self._startup_error}"
            ) from self._startup_error

    def close(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
        self._process = None

    def _snapshot(self, published_at: float) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        state = copy.deepcopy(self._state)
        config = copy.deepcopy(self._config)
        if state is not None:
            control_ns = state.get("monotonic_ns")
            control_at = (
                float(control_ns) / 1e9
                if isinstance(control_ns, int) and not isinstance(control_ns, bool)
                else None
            )
            state["publish_sequence"] = self._publish_sequence
            state["published_monotonic_ns"] = int(published_at * 1e9)
            state["state_age_s"] = None if control_at is None else max(0.0, published_at - control_at)
            state["publisher_target_frequency_hz"] = self.frequency
            state["publisher_deadline_misses"] = self._deadline_misses
        return state, config

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 2)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(self.endpoint)
        except BaseException as exc:
            self._error_tx.send(str(exc))
            self._ready.set()
            socket.close(linger=0)
            context.term()
            return

        self._ready.set()
        deadline = time.monotonic()
        last_config_at = -math.inf
        report_started = deadline
        report_count = 0
        try:
            while not self._stop.is_set() and os.getppid() == self._parent_pid:
                while self._updates_rx.poll():
                    kind, payload = self._updates_rx.recv()
                    if kind == "state":
                        self._state = payload
                    else:
                        self._config = payload
                remaining = deadline - time.monotonic()
                if remaining > 0 and self._stop.wait(remaining):
                    break

                published_at = time.monotonic()
                lateness = published_at - deadline
                if lateness >= self.period:
                    skipped = int(lateness / self.period)
                    self._deadline_misses += skipped
                    deadline += skipped * self.period

                state, config = self._snapshot(published_at)
                if state is not None:
                    try:
                        socket.send(encode(HAND_STATE_TOPIC, state), flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
                    self._publish_sequence += 1
                    report_count += 1
                if config is not None and published_at - last_config_at >= 2.0:
                    try:
                        socket.send(encode(HAND_CONFIG_TOPIC, config), flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
                    last_config_at = published_at

                report_elapsed = published_at - report_started
                if report_elapsed >= 5.0:
                    print(
                        f"[Hands] state publisher: {report_count / report_elapsed:.2f} Hz, "
                        f"deadline misses: {self._deadline_misses}"
                    )
                    report_started = published_at
                    report_count = 0
                deadline += self.period
        finally:
            socket.close(linger=0)
            context.term()


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
        close_scale: float = 1.0,
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
        self.last_intent_received_at: float | None = None
        self.last_intent_source_monotonic_ns: int | None = None
        self.last_valid_intent_at: dict[str, float | None] = {side: None for side in self.devices}
        self.last_step_at = clock()
        self.mode = "hold"
        self.requested: dict[str, np.ndarray] = {}
        self.applied: dict[str, np.ndarray] = {}
        self.velocity_scale: dict[str, float] = {}
        self.measured: dict[str, np.ndarray] = {}
        self.health: dict[str, dict[str, Any]] = {}
        self.last_health_at = -math.inf
        self.intent_closed: dict[str, bool | None] = {side: None for side in self.devices}
        self.errors: dict[str, str | None] = {side: None for side in self.devices}
        self.fault_latched = False
        self.explicit_hold = False
        self._hold_write_pending: set[str] = set()
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
        accepted = False
        received_at = self.clock() if now is None else now
        source_monotonic_ns = payload.get("monotonic_ns")
        if isinstance(source_monotonic_ns, int) and not isinstance(source_monotonic_ns, bool):
            self.last_intent_source_monotonic_ns = source_monotonic_ns
        self.last_intent_received_at = received_at
        hold = bool(payload["hold"])
        for side in self.devices:
            side_intent = payload[side]
            if not side_intent["valid"]:
                continue
            self.last_valid_intent_at[side] = received_at
            accepted = True
            if hold:
                if not self.explicit_hold:
                    # Cancel an in-progress open/close slew at the latest
                    # measured position. Merely stopping new intent would
                    # leave the previous motor setpoint active.
                    side_profile = self.profile.side(side)
                    frozen = np.clip(
                        self.measured[side], side_profile.lower_rad, side_profile.upper_rad
                    )
                    self.requested[side] = frozen.copy()
                    self.applied[side] = frozen.copy()
                    self._hold_write_pending.add(side)
                continue
            side_profile = self.profile.side(side)
            target = side_profile.target(bool(side_intent["closed"]), self.close_scale)
            if not np.array_equal(target, self.requested[side]):
                self.requested[side] = target
                if self.transition_duration_s is not None:
                    base_velocity = np.asarray(side_profile.velocity_rad_s, dtype=np.float64)
                    nominal_duration = float(np.max(np.abs(target - self.applied[side]) / base_velocity))
                    self.velocity_scale[side] = nominal_duration / self.transition_duration_s
            self.intent_closed[side] = bool(side_intent["closed"])
        self.last_intent_sequence = sequence
        if accepted:
            self.last_intent_at = received_at
            self.explicit_hold = hold
            self.mode = "hold" if hold else "tracking"
        return accepted

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
            if side in self._hold_write_pending and not self.fault_latched:
                # Send the captured measured pose once so the device abandons
                # its earlier open/close target immediately.
                device.write_positions(self.applied[side])
                self._hold_write_pending.remove(side)
            elif not stale_by_side[side] and self.mode == "tracking" and not self.fault_latched:
                target = np.clip(self.requested[side], side_profile.lower_rad, side_profile.upper_rad)
                maximum = np.asarray(side_profile.velocity_rad_s) * self.velocity_scale[side] * dt
                safe = self.applied[side] + np.clip(target - self.applied[side], -maximum, maximum)
                if getattr(device, "owns_trajectory", False):
                    safe = target  # The motor worker generates the 200 Hz trajectory.
                # The AGILINK all-joint call expands into multiple CAN
                # transactions. Re-sending an already reached setpoint every
                # cycle can fill the vendor queue with old poses, making a new
                # trigger command appear many seconds late.
                if not np.array_equal(safe, self.applied[side]):
                    device.write_positions(safe)
                    self.applied[side] = safe
            set_control_mode = getattr(device, "set_control_mode", None)
            if set_control_mode is not None:
                set_control_mode(
                    "fault" if self.fault_latched
                    else "hold" if stale_by_side[side] or self.mode == "hold"
                    else "tracking"
                )
            self.measured[side] = np.asarray(device.read_positions(), dtype=np.float64).copy()
            if getattr(device, "owns_trajectory", False):
                self.applied[side] = np.asarray(getattr(device, "applied_positions"), dtype=np.float64).copy()
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
                "request_interval_ms": getattr(device, "request_interval_ms", None),
                "frame_recv_timeout_ms": getattr(device, "frame_recv_timeout_ms", None),
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
            "explicit_hold": self.explicit_hold,
            "target_source": "pico_open_close",
            "intent_sequence": self.last_intent_sequence,
            "intent_source_monotonic_ns": self.last_intent_source_monotonic_ns,
            "intent_received_monotonic_ns": (
                None
                if self.last_intent_received_at is None
                else int(self.last_intent_received_at * 1e9)
            ),
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
    if args.backend == "dex1":
        return Dex1Backend(
            HandSide(side), p, worker=args.dex1_worker,
            transition_duration=args.dex1_transition_duration,
            command_enabled=bool(args.enable_command),
        )
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
    except BaseException:
        for device in devices.values():
            device.close()
        raise


def _profile_for_backend(backend: str) -> HandProfile:
    return get_hand_profile("dex1.v1" if backend == "dex1" else "omnihand_o10.v1")


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
    profile = _profile_for_backend(args.backend)
    devices: dict[str, HandBackend] = {}
    try:
        with vendor_output_to_stderr():
            for side in _selected_sides(args.sides):
                devices[side] = _make_hardware_device(args, side, profile)
        report = {
            side: {"positions_rad": device.read_positions().tolist(), "health": device.read_health()}
            for side, device in devices.items()
        }
        faulted = {
            side: [int(mask) for mask in side_report["health"].get("error_masks", ())]
            for side, side_report in report.items()
            if _has_hard_motor_error(side_report["health"].get("error_masks", ()))
        }
        communication_warnings = {
            side: [int(mask) for mask in side_report["health"].get("error_masks", ())]
            for side, side_report in report.items()
            if any(
                int(mask) & COMMUNICATION_ERROR_MASK
                for mask in side_report["health"].get("error_masks", ())
            )
        }
        passed = not faulted
        payload = {"passed": passed, "profile": profile.name, "sides": report}
        if faulted:
            payload["error"] = f"non-zero motor error masks: {faulted}"
        if communication_warnings:
            payload["warnings"] = {
                "commu_except_masks": communication_warnings,
                "policy": "reported but not latched while SocketCAN remains ERROR-ACTIVE with zero counters",
            }
        print(json.dumps(payload, indent=2))
        return 0 if passed else 1
    except Exception as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 1
    finally:
        for device in devices.values():
            device.close()


def hold(args: argparse.Namespace) -> int:
    """Send one measured-position hold after all physical safety gates pass."""
    if args.backend != "omnihand":
        raise HandControllerError("hold is only supported by the physical OmniHand backend")
    if not args.enable_command:
        raise HandControllerError("--enable-command is required for the physical hold")

    profile = get_hand_profile("omnihand_o10.v1")
    devices: dict[str, HandBackend] = {}
    controller: SafeHandController | None = None
    try:
        with vendor_output_to_stderr():
            for side in _selected_sides(args.sides):
                devices[side] = _make_hardware_device(args, side, profile)
        controller = SafeHandController(
            profile,
            devices,
            backend_name="omnihand",
            session_id=uuid.uuid4().hex,
        )
        # The constructor writes exactly the measured startup pose. Re-read
        # after a short settle and reject any unexpected movement.
        time.sleep(0.25)
        state = controller.step()
        following_error = {
            side: float(
                np.max(
                    np.abs(
                        np.asarray(side_state["measured_position_rad"])
                        - np.asarray(side_state["applied_position_rad"])
                    )
                )
            )
            for side, side_state in state["sides"].items()
        }
        passed = all(error <= HOLD_ERROR_LIMIT_RAD for error in following_error.values())
        payload = {
            "passed": passed,
            "operation": "zero_delta_hold",
            "profile": profile.name,
            "following_error_rad": following_error,
            "limit_rad": HOLD_ERROR_LIMIT_RAD,
            "sides": state["sides"],
        }
        if not passed:
            payload["error"] = "zero-delta hold following error exceeded its limit"
        print(json.dumps(payload, indent=2))
        return 0 if passed else 1
    except Exception as exc:
        print(json.dumps({"passed": False, "operation": "zero_delta_hold", "error": str(exc)}, indent=2))
        return 1
    finally:
        if controller is not None:
            controller.close()
        else:
            for device in devices.values():
                device.close()


def run(args: argparse.Namespace) -> int:
    if args.backend in {"omnihand", "dex1"} and not args.enable_command:
        raise HandControllerError("--enable-command is required for physical hand writes")
    profile = _profile_for_backend(args.backend)
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    # Hand intent owns a dedicated endpoint, so CONFLATE safely guarantees
    # latest-only behavior while bilateral SDK calls are busy.
    subscriber.setsockopt(zmq.RCVHWM, 1)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt(zmq.SUBSCRIBE, HAND_INTENT_TOPIC)
    subscriber.connect(args.intent_endpoint)
    selected_sides = _selected_sides(args.sides)
    session_id = os.environ.get("SONIC_HAND_SESSION_ID") or uuid.uuid4().hex
    controller: SafeHandController | None = None
    next_reconnect_at = 0.0
    last_error: str | None = None
    fault_latched = False
    state_publisher = FixedRateHandStatePublisher(args.state_endpoint, args.frequency)

    def handle_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm_handler = signal.signal(signal.SIGTERM, handle_sigterm)

    def disconnected_state(now: float) -> dict[str, Any]:
        return {
            "schema": HAND_STATE_SCHEMA,
            "session_id": session_id,
            "sequence": 0,
            "monotonic_ns": int(now * 1e9),
            "backend": args.backend,
            "profile": profile.name,
            "mode": "fault" if fault_latched else "disconnected",
            "target_source": "pico_open_close",
            "intent_sequence": None,
            "intent_source_monotonic_ns": None,
            "intent_received_monotonic_ns": None,
            "input_stale": True,
            "input_age_s": None,
            "sides": {
                side: {"valid": False, "connected": False, "error": last_error}
                for side in selected_sides
            },
        }

    state_publisher.update_state(disconnected_state(time.monotonic()))
    try:
        state_publisher.start()
        period = 1.0 / args.frequency
        while True:
            started = time.monotonic()
            just_connected = False
            if controller is None and not fault_latched and started >= next_reconnect_at:
                devices: dict[str, HandBackend] = {}
                try:
                    devices = _make_devices(args, profile, context)
                    controller = SafeHandController(
                        profile,
                        devices,
                        backend_name=args.backend,
                        close_scale=args.close_scale,
                        target_timeout_s=args.target_timeout,
                        transition_duration_s=(
                            args.dex1_transition_duration if args.backend == "dex1" else args.transition_duration
                        ),
                        session_id=session_id,
                    )
                    last_error = None
                    state_publisher.update_config(controller.config_payload())
                    # A reconnect is admitted from measured feedback only. Drop
                    # anything queued while disconnected and require a target
                    # published after this hold was established.
                    while subscriber.poll(0):
                        subscriber.recv(zmq.NOBLOCK)
                    just_connected = True
                    print(f"[Hands] Connected: {', '.join(selected_sides)}")
                except Exception as exc:
                    for device in devices.values():
                        device.close()
                    last_error = str(exc)
                    fault_latched = isinstance(exc, Dex1SafetyError)
                    next_reconnect_at = started + args.reconnect_interval
                    recovery = (
                        "use Reconnect hands after checking the fault" if fault_latched
                        else f"retrying in {args.reconnect_interval:.1f}s"
                    )
                    print(f"[Hands] Connection attempt failed: {last_error}; {recovery}")

            if controller is not None:
                try:
                    if not just_connected:
                        # CONFLATE normally leaves one frame. Drain defensively
                        # across reconnects and apply only the newest intent.
                        latest_raw = None
                        while subscriber.poll(0):
                            latest_raw = subscriber.recv(zmq.NOBLOCK)
                        if latest_raw is not None:
                            controller.accept_intent(decode_intent(latest_raw), now=started)
                    state = controller.step(now=started)
                    state_publisher.update_state(state)
                except Exception as exc:
                    last_error = str(exc)
                    print(
                        f"[Hands] Transport failed: {last_error}; "
                        "requesting a clean worker restart"
                    )
                    controller.close()
                    controller = None
                    if isinstance(exc, Dex1SafetyError):
                        fault_latched = True
                        state_publisher.update_state(disconnected_state(time.monotonic()))
                        print("[Hands] DEX 1 stopped; use Reconnect hands after checking the fault")
                        continue
                    state_publisher.update_state(disconnected_state(time.monotonic()))
                    return RECOVERABLE_DISCONNECT_EXIT_CODE
            else:
                state_publisher.update_state(disconnected_state(started))
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if controller is not None:
            controller.close()
        subscriber.close(linger=0)
        state_publisher.close()
        context.term()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "hold", "run"):
        command = sub.add_parser(name)
        command.add_argument("--backend", choices=("sim", "omnihand", "dex1"), default="sim")
        command.add_argument("--sides", choices=("left", "right", "both"), default="both")
        command.add_argument("--left-interface", default="can11")
        command.add_argument("--right-interface", default="can10")
        command.add_argument("--enable-command", action="store_true")
        command.add_argument("--dex1-worker", default=str(DEFAULT_WORKER))
        command.add_argument("--dex1-transition-duration", type=float, default=1.5)
    runner = sub.choices["run"]
    runner.add_argument(
        "--intent-endpoint", default=f"tcp://localhost:{DEFAULT_HAND_INTENT_PORT}"
    )
    runner.add_argument("--state-endpoint", default=f"tcp://*:{DEFAULT_HAND_STATE_PORT}")
    runner.add_argument("--frequency", type=float, default=50.0)
    runner.add_argument("--target-timeout", type=float, default=0.5)
    runner.add_argument("--transition-duration", type=float, default=0.2)
    runner.add_argument("--close-scale", type=float, default=1.0)
    runner.add_argument("--reconnect-interval", type=float, default=1.0)
    runner.add_argument("--sim-feedback-endpoint", default="tcp://localhost:5571")
    runner.add_argument("--sim-feedback-timeout", type=float, default=5.0)
    runner.add_argument("--sim-feedback-max-age", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        return probe(args)
    if args.command == "hold":
        return hold(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
