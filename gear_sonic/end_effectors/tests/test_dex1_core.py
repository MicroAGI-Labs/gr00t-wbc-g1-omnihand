import numpy as np
import pytest

from gear_sonic.end_effectors.controller import build_parser
from gear_sonic.end_effectors.profiles import DEX1, dataset_robot_type, get_hand_profile, raw_hand_name


def test_dex1_profile_is_one_motor_per_side():
    assert get_hand_profile("dex1.v1") is DEX1
    assert DEX1.left.width == DEX1.right.width == 1
    assert np.all(DEX1.left.target(False) == DEX1.left.open_rad)
    assert dataset_robot_type(DEX1) == "unitree_g1_dex1_sonic"
    assert raw_hand_name(DEX1) == "dex1"


def test_dex1_controller_backend_options_are_explicit():
    args = build_parser().parse_args(["run", "--backend", "dex1", "--dex1-transition-duration", "2.0"])
    assert args.backend == "dex1"
    assert args.dex1_transition_duration == 2.0


@pytest.mark.parametrize("duration", [0.0, 1.0, 30.1])
def test_dex1_transition_duration_is_checked_by_backend(duration):
    from gear_sonic.end_effectors.backends.dex1 import Dex1Backend

    with pytest.raises(ValueError):
        Dex1Backend("left", DEX1.left, worker="/missing", transition_duration=duration)
