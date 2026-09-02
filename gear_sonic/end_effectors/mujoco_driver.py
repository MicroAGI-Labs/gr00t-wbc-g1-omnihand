"""MuJoCo-side O10 actuator and feedback bridge for the combined G1 model."""

from __future__ import annotations

import time
from typing import Any

import mujoco
import numpy as np
import zmq

from .profiles import OMNIHAND_O10
from .protocol import (
    HAND_SIM_FEEDBACK_SCHEMA,
    HAND_SIM_FEEDBACK_TOPIC,
    HAND_STATE_TOPIC,
    decode_state,
    encode,
)

BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _actuator_ids_for_joints(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    by_joint: dict[int, int] = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id >= 0:
            if joint_id in by_joint:
                raise RuntimeError(f"multiple actuators drive MuJoCo joint id {joint_id}")
            by_joint[joint_id] = actuator_id
    result: list[int] = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or joint_id not in by_joint:
            raise RuntimeError(f"no actuator drives required joint {name}")
        result.append(by_joint[joint_id])
    return np.asarray(result, dtype=np.int64)


def _joint_addresses(model: mujoco.MjModel, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
        dtype=np.int64,
    )
    if np.any(ids < 0):
        missing = [name for name, joint_id in zip(names, ids, strict=True) if joint_id < 0]
        raise RuntimeError(f"MuJoCo model is missing joints: {missing}")
    return model.jnt_qposadr[ids].copy(), model.jnt_dofadr[ids].copy()


class OmniHandMuJoCoDriver:
    """Drive both Atlas O10 models from validated controller state messages."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        state_endpoint: str,
        feedback_endpoint: str,
        kp: float = 4.0,
        kd: float = 0.08,
        state_timeout_s: float = 0.5,
        context: zmq.Context | None = None,
    ) -> None:
        if kp < 0 or kd < 0 or state_timeout_s <= 0:
            raise ValueError("invalid OmniHand MuJoCo gains or timeout")
        self.model = model
        self.data = data
        self.kp = kp
        self.kd = kd
        self.state_timeout_s = state_timeout_s
        self.body_qpos, self.body_qvel = _joint_addresses(model, BODY_JOINT_NAMES)
        self.body_actuators = _actuator_ids_for_joints(model, BODY_JOINT_NAMES)
        self.hand_qpos: dict[str, np.ndarray] = {}
        self.hand_qvel: dict[str, np.ndarray] = {}
        self.hand_actuators: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            names = OMNIHAND_O10.side(side).joint_names
            qpos, qvel = _joint_addresses(model, names)
            self.hand_qpos[side] = qpos
            self.hand_qvel[side] = qvel
            self.hand_actuators[side] = _actuator_ids_for_joints(model, names)
        self.targets = {side: data.qpos[self.hand_qpos[side]].copy() for side in ("left", "right")}
        self._session_id: str | None = None
        self._state_sequence: int | None = None
        self._state_received_at: float | None = None
        self._feedback_sequence = 0
        self._context = zmq.Context.instance() if context is None else context
        self._state_socket = self._context.socket(zmq.SUB)
        self._state_socket.setsockopt(zmq.RCVHWM, 1)
        self._state_socket.setsockopt(zmq.CONFLATE, 1)
        self._state_socket.setsockopt(zmq.SUBSCRIBE, HAND_STATE_TOPIC)
        self._state_socket.connect(state_endpoint)
        self._feedback_socket = self._context.socket(zmq.PUB)
        self._feedback_socket.setsockopt(zmq.SNDHWM, 1)
        self._feedback_socket.bind(feedback_endpoint)
        self._closed = False

    def _accept_state(self, raw: bytes) -> None:
        payload = decode_state(raw)
        if payload.get("profile") != OMNIHAND_O10.name or payload.get("backend") != "sim":
            return
        if payload.get("mode") in {"disconnected", "fault"}:
            return
        session_id = payload.get("session_id")
        sequence = payload.get("sequence")
        if not isinstance(session_id, str) or isinstance(sequence, bool) or not isinstance(sequence, int):
            return
        if session_id != self._session_id:
            self._session_id = session_id
            self._state_sequence = None
        if self._state_sequence is not None and sequence <= self._state_sequence:
            return
        sides = payload.get("sides")
        if not isinstance(sides, dict):
            return
        candidates: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            side_payload = sides.get(side)
            if not isinstance(side_payload, dict):
                return
            if not side_payload.get("valid") or not side_payload.get("connected"):
                return
            values = np.asarray(side_payload.get("applied_position_rad"), dtype=np.float64).reshape(-1)
            profile = OMNIHAND_O10.side(side)
            if values.shape != (profile.width,) or not np.all(np.isfinite(values)):
                return
            lower = np.asarray(profile.lower_rad)
            upper = np.asarray(profile.upper_rad)
            if np.any(values < lower) or np.any(values > upper):
                return
            candidates[side] = values
        self.targets = candidates
        self._state_sequence = sequence
        self._state_received_at = time.monotonic()

    def _drain_state(self) -> None:
        while self._state_socket.poll(0):
            self._accept_state(self._state_socket.recv(zmq.NOBLOCK))

    def compute_torques(self) -> np.ndarray:
        self._drain_state()
        # A stale controller retains the last bounded position target. This is
        # the same fail-safe hold used by the physical controller.
        torque: list[np.ndarray] = []
        for side in ("left", "right"):
            q = self.data.qpos[self.hand_qpos[side]]
            qd = self.data.qvel[self.hand_qvel[side]]
            torque.append(self.kp * (self.targets[side] - q) - self.kd * qd)
        return np.concatenate(torque)

    def publish_feedback(self) -> None:
        now = time.monotonic()
        payload: dict[str, Any] = {
            "schema": HAND_SIM_FEEDBACK_SCHEMA,
            "sequence": self._feedback_sequence,
            "monotonic_ns": int(now * 1e9),
            "profile": OMNIHAND_O10.name,
            "controller_state_stale": (
                self._state_received_at is None or now - self._state_received_at > self.state_timeout_s
            ),
            "sides": {
                side: {
                    "position_rad": self.data.qpos[self.hand_qpos[side]].tolist(),
                    "velocity_rad_s": self.data.qvel[self.hand_qvel[side]].tolist(),
                }
                for side in ("left", "right")
            },
        }
        try:
            self._feedback_socket.send(encode(HAND_SIM_FEEDBACK_TOPIC, payload), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        self._feedback_sequence += 1

    def reset_targets(self) -> None:
        self.targets = {side: self.data.qpos[self.hand_qpos[side]].copy() for side in ("left", "right")}
        self._session_id = None
        self._state_sequence = None
        self._state_received_at = None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._state_socket.close(linger=0)
            self._feedback_socket.close(linger=0)
