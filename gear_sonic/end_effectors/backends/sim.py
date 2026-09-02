"""Deterministic no-I/O backend for controller and collection simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..profiles import SideProfile


class SimHandBackend:
    sdk_version = "sim.v1"
    sdk_commit = None

    def __init__(self, profile: SideProfile, initial: np.ndarray | None = None) -> None:
        self.profile = profile
        self._positions = np.asarray(profile.open_rad if initial is None else initial, dtype=np.float64).copy()
        if self._positions.shape != (profile.width,) or not np.all(np.isfinite(self._positions)):
            raise ValueError("initial simulated hand state is invalid")
        self.commands: list[np.ndarray] = []
        self.closed = False

    def read_positions(self) -> np.ndarray:
        if self.closed:
            raise RuntimeError("simulated hand is closed")
        return self._positions.copy()

    def write_positions(self, positions: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("simulated hand is closed")
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if values.shape != (self.profile.width,) or not np.all(np.isfinite(values)):
            raise ValueError("simulated hand command is invalid")
        lower = np.asarray(self.profile.lower_rad)
        upper = np.asarray(self.profile.upper_rad)
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("simulated hand command exceeds limits")
        self._positions = values.copy()
        self.commands.append(values.copy())

    def read_health(self) -> dict[str, Any]:
        return {"error_masks": [0] * self.profile.width, "temperature_c": None, "current_ma": None}

    def close(self) -> None:
        self.closed = True
