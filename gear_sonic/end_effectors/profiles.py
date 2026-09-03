"""Canonical hand joint orders, limits, and open/close poses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class SideProfile:
    joint_names: tuple[str, ...]
    lower_rad: tuple[float, ...]
    upper_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    open_rad: tuple[float, ...]
    closed_rad: tuple[float, ...]

    @property
    def width(self) -> int:
        return len(self.joint_names)

    def target(self, closed: bool, close_scale: float = 1.0) -> np.ndarray:
        if not 0.0 <= close_scale <= 1.0:
            raise ValueError("close_scale must be between zero and one")
        opened = np.asarray(self.open_rad, dtype=np.float64)
        if not closed:
            return opened
        shut = np.asarray(self.closed_rad, dtype=np.float64)
        return opened + close_scale * (shut - opened)


@dataclass(frozen=True)
class HandProfile:
    name: str
    left: SideProfile
    right: SideProfile

    @property
    def width(self) -> int:
        if self.left.width != self.right.width:
            raise ValueError(f"asymmetric profile width: {self.name}")
        return self.left.width

    def side(self, side: HandSide | str) -> SideProfile:
        return self.left if HandSide(side) is HandSide.LEFT else self.right


_O10_RIGHT_NAMES = (
    "R_thumb_roll_joint",
    "R_thumb_abad_joint",
    "R_thumb_mcp_joint",
    "R_index_abad_joint",
    "R_index_pip_joint",
    "R_middle_pip_joint",
    "R_ring_abad_joint",
    "R_ring_pip_joint",
    "R_pinky_abad_joint",
    "R_pinky_pip_joint",
)
_O10_LEFT_NAMES = tuple(name.replace("R_", "L_", 1) for name in _O10_RIGHT_NAMES)
_O10_RIGHT_LOWER = (-0.0296, -1.6423, 0.0, -0.1640, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_O10_RIGHT_UPPER = (1.1213, 0.0453, 0.8412, 0.0, 1.4835, 1.4835, 0.1692, 1.4835, 0.1850, 1.4835)
_O10_LEFT_LOWER = (-1.1213, -0.0453, -0.8412, 0.0, 0.0, 0.0, -0.1692, 0.0, -0.1850, 0.0)
_O10_LEFT_UPPER = (0.0296, 1.6423, 0.0, 0.1640, 1.4835, 1.4835, 0.0, 1.4835, 0.0, 1.4835)
_O10_VELOCITY = (0.164, 0.164, 0.308, 0.164, 0.308, 0.308, 0.164, 0.308, 0.164, 0.308)
# Keep the thumb MCP slightly flexed in the nominal open pose.  A fully
# extended thumb can cause an electrical short on this hardware revision.
_O10_THUMB_MCP_OPEN_FLEX_RAD = 0.08
_O10_RIGHT_OPEN = (0.0, 0.0, _O10_THUMB_MCP_OPEN_FLEX_RAD, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_O10_LEFT_OPEN = (0.0, 0.0, -_O10_THUMB_MCP_OPEN_FLEX_RAD, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_O10_RIGHT_CLOSED = (0.728845, -0.903265, 0.757080, 0.0, 1.335150, 1.335150, 0.0, 1.335150, 0.0, 1.335150)
_O10_LEFT_CLOSED = (-0.728845, 0.903265, -0.757080, 0.0, 1.335150, 1.335150, 0.0, 1.335150, 0.0, 1.335150)

OMNIHAND_O10 = HandProfile(
    "omnihand_o10.v1",
    SideProfile(
        _O10_LEFT_NAMES,
        _O10_LEFT_LOWER,
        _O10_LEFT_UPPER,
        _O10_VELOCITY,
        _O10_LEFT_OPEN,
        _O10_LEFT_CLOSED,
    ),
    SideProfile(
        _O10_RIGHT_NAMES,
        _O10_RIGHT_LOWER,
        _O10_RIGHT_UPPER,
        _O10_VELOCITY,
        _O10_RIGHT_OPEN,
        _O10_RIGHT_CLOSED,
    ),
)

_DEX_NAMES = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
DEX3 = HandProfile(
    "dex3.v1",
    SideProfile(
        tuple(f"left_hand_{n}_joint" for n in _DEX_NAMES),
        (-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75),
        (1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0),
        (0.25,) * 7,
        (0.0,) * 7,
        (0.0, 0.163, 0.875, -0.785, -0.875, -0.785, -0.875),
    ),
    SideProfile(
        tuple(f"right_hand_{n}_joint" for n in _DEX_NAMES),
        (-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0),
        (1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75),
        (0.25,) * 7,
        (0.0,) * 7,
        (0.0, -0.154, -0.875, 0.785, 0.875, 0.785, 0.875),
    ),
)

PROFILES = {profile.name: profile for profile in (DEX3, OMNIHAND_O10)}


def get_hand_profile(name: str) -> HandProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown hand profile {name!r}; expected one of {tuple(PROFILES)}") from exc
