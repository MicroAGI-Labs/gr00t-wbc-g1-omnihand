"""DEX 1 adapter: one isolated 200 Hz M4010 worker per USB gripper."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import select
import subprocess
import time
from typing import Any

import numpy as np

from ..profiles import HandSide, SideProfile

SDK_COMMIT = "2986d26eefa4136d4777493e6fd0b8bac7a4c6ae"
DEFAULT_WORKER = Path(__file__).resolve().parents[3] / "build/dex1/dex1_worker"
USB_PORTS = {
    "left": "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBQ776H-if00-port0",
    "right": "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBWJBC1-if00-port0",
}
MOTOR_IDS = {"left": 0, "right": 1}


class Dex1SafetyError(RuntimeError):
    """A failed motor worker requires an explicit operator reconnect."""


class Dex1Backend:
    sdk_version = "unitree-m4010"
    sdk_commit = SDK_COMMIT
    owns_trajectory = True

    def __init__(
        self,
        side: HandSide,
        profile: SideProfile,
        *,
        worker: str | Path = DEFAULT_WORKER,
        transition_duration: float = 1.5,
        command_enabled: bool = False,
    ) -> None:
        self.side, self.profile = HandSide(side), profile
        self.port = USB_PORTS[self.side.value]
        self.motor_id = MOTOR_IDS[self.side.value]
        self._command_enabled = command_enabled
        self._mode = "hold"
        self._target: float | None = None
        self._sequence = 0
        self._buffer = b""
        self._latest: dict[str, Any] | None = None
        self._closed = False
        if not math.isfinite(transition_duration) or not 1.35 <= transition_duration <= 30:
            raise ValueError("DEX 1 transition duration must be between 1.35 and 30 seconds")
        if profile.width != 1:
            raise ValueError("DEX 1 requires one motor per hand")
        try:
            serial_device = str(Path(self.port).resolve(strict=True))
        except OSError as exc:
            raise Dex1SafetyError(f"DEX 1 adapter unavailable: {self.port}") from exc
        self._process = subprocess.Popen(
            [
                str(worker),
                serial_device,  # libserialport expects the canonical tty device.
                str(self.motor_id),
                str(profile.lower_rad[0]),
                str(profile.upper_rad[0]),
                str(transition_duration),
                str(int(command_enabled)),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        assert self._process.stdin is not None and self._process.stdout is not None
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
        try:
            deadline = time.monotonic() + 3.0
            while self._latest is None and time.monotonic() < deadline:
                select.select([self._process.stdout], [], [], 0.05)
                self._drain()
            self._snapshot()
        except BaseException:
            self.close()
            raise

    def _drain(self) -> None:
        assert self._process.stdout is not None
        # Drain the bounded pipe to the newest sample; receipt time must never
        # disguise an old sample. The worker uses this host's CLOCK_MONOTONIC.
        while True:
            try:
                data = os.read(self._process.stdout.fileno(), 65536)
            except BlockingIOError:
                break
            if not data:
                break
            self._buffer += data
            if len(self._buffer) > 131072:
                raise Dex1SafetyError("DEX 1 feedback overflow")
            while b"\n" in self._buffer:
                raw, self._buffer = self._buffer.split(b"\n", 1)
                try:
                    sample = json.loads(raw)
                except (ValueError, UnicodeError) as exc:
                    raise Dex1SafetyError("invalid DEX 1 feedback JSON") from exc
                if (
                    not isinstance(sample, dict)
                    or sample.get("version") != 1
                    or any(
                        not isinstance(sample.get(key), (float, int)) or not math.isfinite(sample[key])
                        for key in (
                            "monotonic_ns",
                            "sequence",
                            "mode",
                            "q",
                            "dq",
                            "tau",
                            "applied_q",
                            "voltage",
                            "temperature",
                            "motor_error",
                        )
                    )
                ):
                    raise Dex1SafetyError("invalid DEX 1 feedback")
                if self._latest is not None and sample["monotonic_ns"] <= self._latest["monotonic_ns"]:
                    raise Dex1SafetyError("replayed DEX 1 feedback")
                self._latest = sample
        status = self._process.poll()
        if status is not None and not self._closed:
            raise Dex1SafetyError(
                f"{self.side.value} DEX 1 worker exited ({status}); inspect Hands pane, then reconnect"
            )

    def _snapshot(self) -> dict[str, Any]:
        self._drain()
        if self._latest is None:
            raise Dex1SafetyError("no DEX 1 feedback")
        age = (time.monotonic_ns() - self._latest["monotonic_ns"]) / 1e9
        if not 0 <= age <= 0.15:
            raise Dex1SafetyError(f"stale DEX 1 feedback ({age:.3f}s)")
        return self._latest

    def set_control_mode(self, mode: str) -> None:
        if mode not in {"tracking", "hold", "fault"}:
            raise ValueError("invalid DEX 1 control mode")
        self._mode = mode

    def read_positions(self) -> np.ndarray:
        if self._target is not None:
            self._sequence += 1
            mode = {"hold": 0, "tracking": 1, "fault": 2}[self._mode]
            packet = f"C {self._sequence} {mode} {self._target:.9g}\n".encode()
            assert self._process.stdin is not None
            try:
                if os.write(self._process.stdin.fileno(), packet) != len(packet):
                    raise Dex1SafetyError("partial DEX 1 command")
            except OSError as exc:
                raise Dex1SafetyError("DEX 1 command pipe unavailable") from exc
        return np.array([self._snapshot()["q"]], dtype=np.float64)

    @property
    def applied_positions(self) -> np.ndarray:
        return np.array([self._snapshot()["applied_q"]], dtype=np.float64)

    def write_positions(self, positions: np.ndarray) -> None:
        if not self._command_enabled:
            raise Dex1SafetyError("DEX 1 commands disabled")
        values = np.asarray(positions, dtype=np.float64)
        if values.shape != (1,) or not np.all(np.isfinite(values)):
            raise ValueError("DEX 1 requires one finite position")
        target = float(values[0])
        if not self.profile.lower_rad[0] <= target <= self.profile.upper_rad[0]:
            raise ValueError("DEX 1 target outside measured range")
        self._target = target

    def read_health(self) -> dict[str, Any]:
        sample = self._snapshot()
        return {
            "error_masks": [int(sample["motor_error"])],
            "temperature_c": [sample["temperature"]],
            "current_ma": None,
            "voltage_v": [sample["voltage"]],
            "torque_nm": [sample["tau"]],
            "velocity_rad_s": [sample["dq"]],
            "feedback_age_s": (time.monotonic_ns() - sample["monotonic_ns"]) / 1e9,
            "serial_port": self.port,
            "motor_id": self.motor_id,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()  # EOF stops the motor on normal exit and parent loss.
        try:
            self._process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1)
        if self._process.stdout is not None:
            self._process.stdout.close()
