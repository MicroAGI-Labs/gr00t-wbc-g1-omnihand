"""Minimal backend boundary kept independent of vendor SDK imports."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class HandBackend(Protocol):
    """Position/health boundary.

    A backend that generates trajectories internally may also expose
    ``owns_trajectory = True``, ``set_control_mode(mode)``, and
    ``applied_positions`` for heartbeat/mode forwarding and recording.
    """

    sdk_version: str
    sdk_commit: str | None

    def read_positions(self) -> np.ndarray: ...
    def write_positions(self, positions: np.ndarray) -> None: ...
    def read_health(self) -> dict[str, Any]: ...
    def close(self) -> None: ...
