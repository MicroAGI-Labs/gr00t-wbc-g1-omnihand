from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.scripts import run_data_exporter
from gear_sonic.scripts.run_data_exporter import (
    GrootDataCollector,
    _episode_hand_motion_range,
    _recording_mode_ready,
    _required_vector,
)
from gear_sonic.utils.data_collection.episode_state import EpisodeState


def test_required_vector_accepts_exact_finite_shape_as_float32():
    value = _required_vector({"token": np.arange(64, dtype=np.float64)}, "token", 64)
    assert value.shape == (64,)
    assert value.dtype == np.float32


@pytest.mark.parametrize(
    "payload,error",
    [
        ({}, "missing"),
        ({"token": np.zeros(63)}, "shape"),
        ({"token": np.full(64, np.nan)}, "NaN or Inf"),
    ],
)
def test_required_vector_rejects_untrainable_values(payload, error):
    with pytest.raises(ValueError, match=error):
        _required_vector(payload, "token", 64)


def test_episode_hand_motion_range_detects_a_command_transition():
    still = np.zeros(10, dtype=np.float32)
    closed = still.copy()
    closed[3] = 0.8
    episode = {
        "teleop.left_hand_joints": [still, closed],
        "teleop.right_hand_joints": [still, still],
    }
    assert _episode_hand_motion_range(episode) == pytest.approx(0.8)


def test_recording_mode_requires_exact_launch_selected_mode():
    assert _recording_mode_ready(5, 5)
    assert not _recording_mode_ready(2, 5)
    assert not _recording_mode_ready(1, 5)


def test_recording_start_is_blocked_outside_required_mode():
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector.required_stream_mode = 5
    collector.current_stream_mode = 2
    collector._episode_state = EpisodeState()
    collector._manager_toggle_dc = False
    collector._manager_toggle_da = False
    collector._manager_discard_reason = None
    collector._recording_message = "Ready to record"
    collector._keyboard_listener = type(
        "_Keyboard", (), {"read_msg": lambda self: "c"}
    )()
    collector._print_and_say = lambda *args, **kwargs: None

    collector._check_recording_commands()

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert collector._recording_message == "Enter VR3PT with A+X before recording"


def test_manager_mode_exit_requests_automatic_discard(monkeypatch):
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector.required_stream_mode = 5
    collector.current_stream_mode = 5
    collector._episode_state = EpisodeState()
    collector._episode_state.change_state()
    collector._manager_toggle_dc = False
    collector._manager_toggle_da = False
    collector._manager_discard_reason = None
    collector._recording_message = "Recording episode 0"

    class _Rates:
        def observe(self, *args, **kwargs):
            pass

    collector.stream_rates = _Rates()
    monkeypatch.setattr(
        run_data_exporter,
        "unpack_pose_message",
        lambda raw, topic: {"stream_mode": np.asarray([2], dtype=np.int32)},
    )

    collector._handle_manager_state(b"manager_state")

    assert collector.current_stream_mode == 2
    assert collector._manager_toggle_da
    assert collector._manager_discard_reason.startswith("required_teleop_mode_exited")
