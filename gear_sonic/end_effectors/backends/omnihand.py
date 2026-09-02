"""Fail-closed AGILINK O10 SocketCAN adapter with lazy SDK loading."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Callable, Iterator

import numpy as np

from ..profiles import HandSide, SideProfile

SDK_VERSION = "1.1.8"
SDK_COMMIT = "026740d9fdd8ba32b0605fa702a992b322076f1b"
EXPECTED_DRIVER = "gs_usb"
EXPECTED_SERIAL = {
    HandSide.RIGHT: "2082395E534B50052",
    HandSide.LEFT: "205B3973534B50042",
}


class OmniHandHardwareError(RuntimeError):
    pass


@contextlib.contextmanager
def vendor_output_to_stderr() -> Iterator[None]:
    """Prevent native SDK banners from corrupting JSON/protocol stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _load_sdk() -> Any:
    try:
        import omnihand
    except ImportError as exc:
        raise OmniHandHardwareError(
            "AGILINK omnihand 1.1.8 is unavailable; run install_scripts/install_omnihand.sh"
        ) from exc
    return omnihand


def _usb_serial(interface: str, net_class: Path) -> str:
    node = (net_class / interface / "device").resolve(strict=True)
    while node != node.parent:
        serial_file = node / "serial"
        if serial_file.is_file():
            serial = serial_file.read_text().strip()
            if serial:
                return serial
        node = node.parent
    raise OmniHandHardwareError(f"cannot identify USB serial for {interface}")


def _validate_canfd_link(interface: str) -> None:
    try:
        details = subprocess.run(
            ["ip", "-details", "link", "show", interface],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OmniHandHardwareError(f"cannot inspect CAN-FD settings for {interface}") from exc
    required = ("bitrate 1000000", "sample-point 0.800", "dbitrate 5000000", "dsample-point 0.750", "fd on")
    missing = [value for value in required if value not in details]
    if missing:
        raise OmniHandHardwareError(
            f"{interface} does not match the admitted 1M/5M CAN-FD link: missing {missing}"
        )


class OmniHandBackend:
    sdk_version = SDK_VERSION
    sdk_commit = SDK_COMMIT

    def __init__(
        self,
        side: HandSide,
        profile: SideProfile,
        interface: str,
        device_id: int = 1,
        *,
        command_enabled: bool = False,
        sdk: Any | None = None,
        net_class: Path = Path("/sys/class/net"),
        interface_index: Callable[[str], int] = socket.if_nametoindex,
        serial_reader: Callable[[str, Path], str] = _usb_serial,
        link_validator: Callable[[str], None] = _validate_canfd_link,
    ) -> None:
        self.side = HandSide(side)
        self.profile = profile
        self.interface = interface
        self.command_enabled = command_enabled
        try:
            interface_index(interface)
        except OSError as exc:
            raise OmniHandHardwareError(f"SocketCAN interface is missing: {interface}") from exc
        try:
            driver = (net_class / interface / "device" / "driver").resolve(strict=True).name
        except OSError as exc:
            raise OmniHandHardwareError(f"cannot identify driver for {interface}") from exc
        if driver != EXPECTED_DRIVER:
            raise OmniHandHardwareError(f"{interface} uses {driver}, expected {EXPECTED_DRIVER}")
        serial = serial_reader(interface, net_class)
        if serial != EXPECTED_SERIAL[self.side]:
            raise OmniHandHardwareError(
                f"{interface} serial {serial} is not the configured {self.side.value} adapter"
            )
        link_validator(interface)

        self._sdk = _load_sdk() if sdk is None else sdk
        hand_type = self._sdk.HandType.LEFT if self.side is HandSide.LEFT else self._sdk.HandType.RIGHT
        with vendor_output_to_stderr():
            self._hand = self._sdk.OmniHand2025.create_hand_socketcan(hand_type, device_id, interface)
            if not self._hand.init():
                raise OmniHandHardwareError(f"could not initialize {self.side.value} O10 on {interface}")
        if int(self._hand.get_product_type()) != int(self._sdk.ProductType.OMNIHAND_2025):
            raise OmniHandHardwareError("connected device is not OMNIHAND_2025")
        if int(self._sdk.OmniHand2025.kDegreesOfActiveFreedom) != profile.width:
            raise OmniHandHardwareError("SDK active-DOF count does not match omnihand_o10.v1")
        get_names = getattr(self._hand, "get_joint_names", None)
        if callable(get_names):
            actual = tuple(get_names())
            if actual != profile.joint_names:
                raise OmniHandHardwareError(f"SDK joint order mismatch: {actual}")
        self.read_positions()

    def read_positions(self) -> np.ndarray:
        if self._hand is None:
            raise OmniHandHardwareError("OmniHand transport is closed")
        try:
            values = np.asarray(self._hand.get_all_active_joint_angles(), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise OmniHandHardwareError(f"feedback read failed: {exc}") from exc
        if values.shape != (self.profile.width,) or not np.all(np.isfinite(values)):
            raise OmniHandHardwareError(f"invalid feedback shape {values.shape}")
        return values

    def write_positions(self, positions: np.ndarray) -> None:
        if not self.command_enabled:
            raise OmniHandHardwareError("physical OmniHand commands are disabled")
        if self._hand is None:
            raise OmniHandHardwareError("OmniHand transport is closed")
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        lower, upper = np.asarray(self.profile.lower_rad), np.asarray(self.profile.upper_rad)
        if values.shape != (self.profile.width,) or not np.all(np.isfinite(values)):
            raise OmniHandHardwareError("command must contain ten finite positions")
        if np.any(values < lower) or np.any(values > upper):
            raise OmniHandHardwareError("command exceeds admitted O10 limits")
        try:
            self._hand.set_all_active_joint_angles(values.tolist())
        except Exception as exc:
            raise OmniHandHardwareError(f"position write failed: {exc}") from exc

    def read_health(self) -> dict[str, Any]:
        if self._hand is None:
            raise OmniHandHardwareError("OmniHand transport is closed")
        reports = list(self._hand.get_all_error_reports())
        if len(reports) != self.profile.width:
            raise OmniHandHardwareError("invalid error report width")
        fields = ("stalled", "overheat", "over_current", "motor_except", "commu_except")
        masks = [
            sum((1 << bit) for bit, field in enumerate(fields) if bool(getattr(report, field)))
            for report in reports
        ]
        temperature = np.asarray(self._hand.get_all_temperature_reports(), dtype=np.float64).reshape(-1)
        current = np.asarray(self._hand.get_all_current_reports(), dtype=np.float64).reshape(-1)
        if (
            temperature.shape != (self.profile.width,)
            or current.shape != (self.profile.width,)
            or not np.all(np.isfinite(temperature))
            or not np.all(np.isfinite(current))
        ):
            raise OmniHandHardwareError("invalid health telemetry")
        return {"error_masks": masks, "temperature_c": temperature.tolist(), "current_ma": current.tolist()}

    def close(self) -> None:
        self._hand = None
