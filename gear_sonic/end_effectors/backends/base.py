"""Minimal backend boundary kept independent of vendor SDK imports."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class HandBackend(Protocol):
    sdk_version: str
    sdk_commit: str | None

    def read_positions(self) -> np.ndarray: ...
    def write_positions(self, positions: np.ndarray) -> None: ...
    def read_health(self) -> dict[str, Any]: ...
    def close(self) -> None: ...
