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


@pytest.mark.parametrize("passed", [True, False])
def test_recording_stop_reports_validation_outcome_and_returns_to_idle(passed):
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_state = EpisodeState()
    collector._episode_state.change_state()
    collector._manager_toggle_dc = False
    collector._manager_toggle_da = False
    collector._manager_discard_reason = None
    collector._keyboard_listener = type(
        "_Keyboard", (), {"read_msg": lambda self: "c"}
    )()

    class _Exporter:
        def __init__(self):
            self.episode_buffer = {"episode_index": 0, "size": 10}

        def detach_episode(self):
            completed = self.episode_buffer
            self.episode_buffer = {"episode_index": 1, "size": 0}
            return completed, {"camera": object()}

    class _Finalizer:
        def __init__(self):
            self.jobs = []

        def enqueue(self, **job):
            self.jobs.append(job)

    collector.data_exporter = _Exporter()
    collector.episode_finalizer = _Finalizer()
    collector._episode_validation = lambda: {
        "passed": passed,
        "errors": [] if passed else ["hand commands did not move enough"],
    }
    collector.sonic_timing_monitor = type("_Monitor", (), {"reset": lambda self: None})()
    collector._episode_input_errors = set()
    collector._initial_yaw = 1.0
    collector._print_and_say = lambda *args, **kwargs: None
    audio_events = []
    collector._set_recording_audio_event = audio_events.append

    collector._check_recording_commands()

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert collector.current_episode_index == 1
    assert audio_events == ["saved" if passed else "validation_failed"]
    assert collector.episode_finalizer.jobs[0]["episode_index"] == 0
    assert collector.episode_finalizer.jobs[0]["success"] is passed
    if not passed:
        assert "failed validation: hand commands did not move enough" in collector._recording_message
        assert "Preserved as unsuccessful" in collector._recording_message
        assert "upload continue in background" in collector._recording_message
        assert "discarded" not in collector._recording_message


def test_recording_discard_acknowledges_and_finalizes_in_background():
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_state = EpisodeState()
    collector._episode_state.change_state()
    collector.data_exporter = type(
        "_Exporter",
        (),
        {
            "episode_buffer": {"episode_index": 3, "size": 5},
            "detach_episode": lambda self: (
                self.episode_buffer,
                {"camera": object()},
            ),
        },
    )()
    jobs = []
    collector.episode_finalizer = type(
        "_Finalizer", (), {"enqueue": lambda self, **job: jobs.append(job)}
    )()
    collector.sonic_timing_monitor = type("_Monitor", (), {"reset": lambda self: None})()
    collector._episode_input_errors = set()
    collector._initial_yaw = 1.0
    collector._print_and_say = lambda *args, **kwargs: None
    audio_events = []
    collector._set_recording_audio_event = audio_events.append

    collector._finish_recording(save=False, discard_reason="operator_discarded")

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert audio_events == ["discard"]
    assert jobs[0]["episode_index"] == 3
    assert jobs[0]["success"] is False
    assert jobs[0]["validation"]["errors"] == ["operator_discarded"]


def test_empty_recording_emits_discard_outcome():
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_state = EpisodeState()
    collector._episode_state.state = collector._episode_state.NEED_TO_SAVE
    collector.data_exporter = type(
        "_Exporter", (), {"episode_buffer": {"episode_index": 0, "size": 0}}
    )()
    collector.frequency = 50.0
    collector._print_and_say = lambda *args, **kwargs: None
    audio_events = []
    collector._set_recording_audio_event = audio_events.append

    collector._finalize_frame(run_data_exporter.time.monotonic())

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert collector._recording_message == "Nothing saved: no frames collected"
    assert audio_events == ["discard"]


def test_manager_mode_exit_does_not_request_automatic_discard(monkeypatch):
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
    assert not collector._manager_toggle_da
    assert collector._manager_discard_reason is None
    assert collector._recording_message == "Recording paused: return to VR3PT mode"


def test_rolling_stream_rate_is_diagnostic_not_a_discard_criterion():
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_input_errors = set()
    collector.data_exporter = type(
        "_Exporter",
        (),
        {"episode_buffer": {"teleop.stream_mode": [np.asarray([5])]}},
    )()
    collector.hand_config = None
    collector.require_hand_activity = True
    collector.minimum_hand_motion_rad = 0.02
    collector.required_stream_mode = 5
    collector.minimum_recording_rate_hz = 45.0

    class _Rates:
        def snapshot(self, streams):
            return {stream: {"sent_hz": 1.0} for stream in streams}

    collector.stream_rates = _Rates()
    validation = collector._episode_validation()

    assert validation["rate_check_enforced"] is False
