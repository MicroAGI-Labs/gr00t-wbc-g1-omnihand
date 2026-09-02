"""Controller-side view of the real bilateral OmniHand MuJoCo model."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import zmq

from ..profiles import HandSide, SideProfile
from ..protocol import HAND_SIM_FEEDBACK_TOPIC, decode_sim_feedback

ATLAS_ASSET_COMMIT = "14b1406c8d3069870e298341be0cd204cee5515d"


class MuJoCoHandTransportError(RuntimeError):
    pass


class MuJoCoHandTransport:
    """Latest-only feedback transport shared by the left/right backend views."""

    def __init__(
        self,
        endpoint: str,
        selected_sides: tuple[str, ...],
        *,
        startup_timeout_s: float = 5.0,
        max_age_s: float = 0.25,
        context: zmq.Context | None = None,
    ) -> None:
        if startup_timeout_s <= 0 or max_age_s <= 0:
            raise ValueError("MuJoCo feedback timeouts must be positive")
        self.selected_sides = selected_sides
        self.startup_timeout_s = startup_timeout_s
        self.max_age_s = max_age_s
        self._context = zmq.Context.instance() if context is None else context
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, HAND_SIM_FEEDBACK_TOPIC)
        self._socket.connect(endpoint)
        self._latest: dict[str, np.ndarray] = {}
        self._received_at: float | None = None
        self._sequence: int | None = None
        self._open_sides = set(selected_sides)

    @property
    def feedback_age_s(self) -> float | None:
        if self._received_at is None:
            return None
        return max(0.0, time.monotonic() - self._received_at)

    def _accept(self, raw: bytes) -> None:
        payload = decode_sim_feedback(raw)
        if payload.get("profile") != "omnihand_o10.v1":
            raise MuJoCoHandTransportError("MuJoCo feedback profile is not omnihand_o10.v1")
        sequence = payload.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise MuJoCoHandTransportError("MuJoCo feedback sequence is invalid")
        if self._sequence is not None and sequence <= self._sequence:
            return
        sides = payload.get("sides")
        if not isinstance(sides, dict):
            raise MuJoCoHandTransportError("MuJoCo feedback has no side map")
        latest: dict[str, np.ndarray] = {}
        for side in self.selected_sides:
            side_payload = sides.get(side)
            if not isinstance(side_payload, dict):
                raise MuJoCoHandTransportError(f"MuJoCo feedback is missing {side}")
            values = np.asarray(side_payload.get("position_rad"), dtype=np.float64).reshape(-1)
            if values.shape != (10,) or not np.all(np.isfinite(values)):
                raise MuJoCoHandTransportError(f"MuJoCo {side} feedback is invalid")
            latest[side] = values
        self._latest = latest
        self._sequence = sequence
        self._received_at = time.monotonic()

    def _refresh(self, *, wait: bool) -> None:
        timeout_ms = int(self.startup_timeout_s * 1000) if wait else 0
        if self._socket.poll(timeout_ms):
            self._accept(self._socket.recv())
            while self._socket.poll(0):
                self._accept(self._socket.recv(zmq.NOBLOCK))

    def read(self, side: str) -> np.ndarray:
        self._refresh(wait=self._received_at is None)
        age = self.feedback_age_s
        if age is None:
            raise MuJoCoHandTransportError("timed out waiting for OmniHand MuJoCo feedback")
        if age > self.max_age_s:
            self._refresh(wait=False)
            age = self.feedback_age_s
        if age is None or age > self.max_age_s:
            raise MuJoCoHandTransportError(f"OmniHand MuJoCo feedback is stale ({age!r}s)")
        return self._latest[side].copy()

    def close_side(self, side: str) -> None:
        self._open_sides.discard(side)
        if not self._open_sides:
            self._socket.close(linger=0)


class MuJoCoSimHandBackend:
    """A side-specific backend whose measured state comes from G1 MuJoCo."""

    sdk_version = "mujoco"
    sdk_commit = ATLAS_ASSET_COMMIT

    def __init__(
        self,
        side: HandSide | str,
        profile: SideProfile,
        transport: MuJoCoHandTransport,
    ) -> None:
        self.side = HandSide(side)
        self.profile = profile
        self.transport = transport
        self._closed = False

    @property
    def feedback_age_s(self) -> float | None:
        return self.transport.feedback_age_s

    def read_positions(self) -> np.ndarray:
        if self._closed:
            raise MuJoCoHandTransportError("MuJoCo hand backend is closed")
        return self.transport.read(self.side.value)

    def write_positions(self, positions: np.ndarray) -> None:
        """Validate the target; the controller's hand_state PUB carries it to MuJoCo."""
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        lower = np.asarray(self.profile.lower_rad)
        upper = np.asarray(self.profile.upper_rad)
        if values.shape != (self.profile.width,) or not np.all(np.isfinite(values)):
            raise MuJoCoHandTransportError("MuJoCo target must contain ten finite positions")
        if np.any(values < lower) or np.any(values > upper):
            raise MuJoCoHandTransportError("MuJoCo target exceeds admitted O10 limits")

    def read_health(self) -> dict[str, Any]:
        self.read_positions()
        return {
            "error_masks": [0] * self.profile.width,
            "temperature_c": None,
            "current_ma": None,
            "feedback_age_s": self.feedback_age_s,
        }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.transport.close_side(self.side.value)
