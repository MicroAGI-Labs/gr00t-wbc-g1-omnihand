from __future__ import annotations

import time

import numpy as np
import pytest

from gear_sonic.data.causal_sync import (
    CausalSelection,
    CausalSynchronizer,
    TimedSample,
)
from gear_sonic.data.episode_finalizer import EpisodeFinalizationResult
from gear_sonic.scripts.run_data_exporter import GrootDataCollector
from gear_sonic.utils.data_collection.episode_state import EpisodeState


class FakeImageSubscriber:
    def __init__(self, stats=None):
        self._stats = stats or {
            "capacity": 5,
            "depth": 1,
            "received": 10,
            "overflow_dropped": 2,
            "latency_dropped": 3,
            "publisher_gap_dropped": 0,
            "received_hz": 50.0,
            "publisher_hz": 60.0,
        }

    def buffer_stats(self):
        return dict(self._stats)


class FakeFinalizer:
    def __init__(self):
        self.jobs = []
        self.results = []

    def can_accept(self):
        return True

    def enqueue(self, **job):
        self.jobs.append(job)

    def drain_results(self):
        results = self.results
        self.results = []
        return results


class FakeExporter:
    def __init__(self):
        self.episode_buffer = {"episode_index": 0, "size": 8}
        self.features = {"observation.images.ego_view": {"dtype": "video"}}

    def detach_episode(self):
        completed = self.episode_buffer
        self.episode_buffer = {"episode_index": 1, "size": 0}
        return completed, {"observation.images.ego_view": object()}


def _recording_collector():
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_state = EpisodeState()
    collector._episode_state.change_state()
    collector._manager_toggle_dc = False
    collector._manager_toggle_da = False
    collector._keyboard_listener = type("Keyboard", (), {"read_msg": lambda self: "c"})()
    collector.data_exporter = FakeExporter()
    collector.episode_finalizer = FakeFinalizer()
    collector._image_subscriber = FakeImageSubscriber()
    collector._episode_camera_stats_start = {
        "received": 4,
        "overflow_dropped": 1,
        "latency_dropped": 1,
        "publisher_gap_dropped": 0,
        "publisher_resets": 0,
    }
    collector.camera_max_age = 0.25
    collector.minimum_camera_rate_hz = 25.0
    collector.hand_config = None
    collector.hand_state_max_age = 0.2
    collector.frequency = 50
    collector.loop_period = 0.02
    collector.loop_period_ns = 20_000_000
    collector.synchronization_delay_ns = 100_000_000
    collector.synchronization_wait_timeout_ns = 250_000_000
    collector.proprio_max_age_ns = 100_000_000
    collector.teleop_max_age_ns = 200_000_000
    collector._synchronizer = CausalSynchronizer()
    collector._recording_start_target_ns = time.monotonic_ns()
    collector._next_target_ns = collector._recording_start_target_ns
    collector._recording_stop_target_ns = None
    collector._synchronization_errors = []
    collector._synchronization_skipped_targets = 0
    collector.latest_image_msg = None
    collector.latest_image_received_at = None
    collector.sonic_timing_monitor = type("Monitor", (), {"reset": lambda self: None})()
    collector._initial_yaw = 1.0
    collector._recording_message = "Recording episode 0"
    collector._last_finalization = None
    collector._print_and_say = lambda *args, **kwargs: None
    return collector


def test_recording_stop_reports_queued_until_background_commit_completes():
    collector = _recording_collector()

    collector._check_recording_commands()

    assert collector._episode_state.get_state() == collector._episode_state.NEED_TO_SAVE
    assert collector.episode_finalizer.jobs == []
    assert "Draining synchronized episode" in collector._recording_message

    collector._next_target_ns = collector._recording_stop_target_ns + 1
    collector._add_data_frame()

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert collector.current_episode_index == 1
    assert "queued for background save" in collector._recording_message
    [job] = collector.episode_finalizer.jobs
    assert job["episode_index"] == 0
    assert job["discarded"] is False
    assert job["validation"]["camera_buffer"]["episode_deltas"] == {
        "received": 6,
        "overflow_dropped": 1,
        "latency_dropped": 2,
        "publisher_gap_dropped": 0,
        "publisher_resets": 0,
    }


def test_saved_message_is_emitted_only_from_a_completed_result():
    collector = _recording_collector()
    collector._episode_state.reset_state()
    collector.episode_finalizer.results.append(EpisodeFinalizationResult(episode_index=0, discarded=False))
    messages = []
    collector._print_and_say = lambda message, **kwargs: messages.append(message)

    collector._consume_finalizer_results()

    assert collector._recording_message == "Episode 0 saved"
    assert messages == ["Episode 0 saved"]


def test_camera_health_checks_each_required_camera_independently():
    collector = _recording_collector()
    collector.data_exporter.features = {
        "observation.images.head": {"dtype": "video"},
        "observation.images.wrist": {"dtype": "video"},
    }
    now = time.monotonic()
    collector.latest_image_received_at = now
    collector.latest_image_msg = {
        "images": {
            "head": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        "capture_monotonic_ns": {
            "head": 10_000_000_000,
            "wrist": 9_000_000_000,
        },
        "publisher_monotonic_ns": 10_000_000_000,
        "receiver_monotonic_ns": time.monotonic_ns(),
    }

    health = collector._camera_health()

    assert health["ready"] is False
    assert health["missing"] == []
    assert health["stale"] == ["wrist"]


def test_stalled_camera_packet_is_not_recorded():
    collector = _recording_collector()
    target_ns = time.monotonic_ns()
    camera_message = {
        "images": {"ego_view": np.zeros((2, 2, 3))},
        "capture_monotonic_ns": {"ego_view": 9_000_000_000},
        "publisher_monotonic_ns": 10_000_000_000,
        "receiver_monotonic_ns": target_ns - 10_000_000,
    }
    selection = CausalSelection(
        target_ns=target_ns,
        samples={
            "camera": TimedSample(
                timestamp_ns=target_ns - 10_000_000,
                value=camera_message,
            )
        },
    )

    assert collector._selected_camera_errors(selection) == [
        "camera ego_view is 1010.0 ms old"
    ]


def test_camera_health_rejects_a_low_source_rate():
    collector = _recording_collector()
    now = time.monotonic()
    collector.latest_image_received_at = now
    collector.latest_image_msg = {
        "images": {"ego_view": np.zeros((2, 2, 3))},
        "capture_monotonic_ns": {"ego_view": 10_000_000_000},
        "publisher_monotonic_ns": 10_000_000_000,
        "receiver_monotonic_ns": time.monotonic_ns(),
    }
    collector._image_subscriber = FakeImageSubscriber(
        {
            "capacity": 5,
            "depth": 1,
            "received": 10,
            "overflow_dropped": 0,
            "latency_dropped": 0,
            "publisher_gap_dropped": 0,
            "received_hz": 8.0,
            "publisher_hz": 8.0,
        }
    )

    health = collector._camera_health()

    assert health["ready"] is False
    assert health["rate_ready"] is False


def test_collector_selects_only_past_samples_using_past_manager_mode():
    collector = _recording_collector()
    target = time.monotonic_ns()
    for stream, past, future in (
        ("proprio", {"value": "state-past"}, {"value": "state-future"}),
        ("camera", {"value": "camera-past"}, {"value": "camera-future"}),
        (
            "manager",
            {"stream_mode": 1, "value": "manager-past"},
            {"stream_mode": 5, "value": "manager-future"},
        ),
        ("sonic", {"value": "sonic-past"}, {"value": "sonic-future"}),
    ):
        collector._synchronizer.observe(stream, past, target - 1_000_000)
        collector._synchronizer.observe(stream, future, target + 1_000_000)

    selection = collector._selection_for_target(target)

    assert selection.ready
    assert set(selection.samples) == {"proprio", "camera", "manager", "sonic"}
    assert all(sample.timestamp_ns <= target for sample in selection.samples.values())
    assert selection.samples["manager"].value["stream_mode"] == 1
    assert selection.samples["sonic"].value["value"] == "sonic-past"


@pytest.mark.parametrize("stream_mode", [2, 3, 5])
def test_collector_requires_planner_for_each_active_planner_mode(stream_mode):
    collector = _recording_collector()
    target = time.monotonic_ns()
    for stream, past, future in (
        ("proprio", {}, {}),
        ("camera", {}, {}),
        ("manager", {"stream_mode": stream_mode}, {"stream_mode": stream_mode}),
        ("planner", {"value": "planner-past"}, {"value": "planner-future"}),
    ):
        collector._synchronizer.observe(stream, past, target - 1_000_000)
        collector._synchronizer.observe(stream, future, target + 1_000_000)

    selection = collector._selection_for_target(target)

    assert selection.ready
    assert set(selection.samples) == {"proprio", "camera", "manager", "planner"}
    assert selection.samples["planner"].value["value"] == "planner-past"


def test_collector_does_not_wait_for_sonic_while_pose_is_paused():
    collector = _recording_collector()
    target = time.monotonic_ns()
    for stream, value in (
        ("proprio", {}),
        ("camera", {}),
        ("manager", {"stream_mode": 4}),
    ):
        collector._synchronizer.observe(stream, value, target - 1_000_000)
        collector._synchronizer.observe(stream, value, target + 1_000_000)

    selection = collector._selection_for_target(target)

    assert selection.ready
    assert set(selection.samples) == {"proprio", "camera", "manager"}


def test_standard_planner_mode_records_the_selected_planner_command():
    collector = _recording_collector()
    target = time.monotonic_ns()
    planner = {
        "planner_mode": 2,
        "planner_movement": np.array([0.1, 0.2, 0.0], dtype=np.float32),
        "planner_facing": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "planner_speed": 0.5,
        "planner_height": -1.0,
        "received_monotonic_ns": target - 3_000_000,
    }
    frame = {}

    age_ms = collector._add_sonic_pose_features(
        frame,
        stream_mode=2,
        smpl_msg=None,
        planner_msg=planner,
        target_ns=target,
    )

    assert age_ms == 3.0
    assert frame["teleop.planner_mode"].item() == 2
    assert frame["teleop.planner_movement"].tolist() == pytest.approx(
        [0.1, 0.2, 0.0]
    )


def test_synchronization_metadata_records_nonnegative_selected_ages():
    collector = _recording_collector()
    collector.data_exporter.features.update(
        {
            "capture.sync_target_monotonic_ns": {},
            "capture.camera_sequence": {},
            "capture.camera_capture_age_ms": {},
            "capture.camera_received_monotonic_ns": {},
            "capture.camera_age_ms": {},
            "capture.proprio_received_monotonic_ns": {},
            "capture.proprio_age_ms": {},
            "capture.sonic_received_monotonic_ns": {},
            "capture.sonic_age_ms": {},
        }
    )
    target = time.monotonic_ns()
    camera = {
        "publisher_sequence": 42,
        "images": {"ego_view": np.zeros((2, 2, 3), dtype=np.uint8)},
        "capture_monotonic_ns": {"ego_view": 9_990_000_000},
        "publisher_monotonic_ns": 10_000_000_000,
        "receiver_monotonic_ns": target - 5_000_000,
    }
    selection = CausalSelection(
        target_ns=target,
        samples={
            "camera": TimedSample(target - 5_000_000, camera),
            "proprio": TimedSample(target - 2_000_000, {}),
        },
    )
    frame = {}

    collector._add_synchronization_features(frame, selection)

    assert frame["capture.camera_sequence"].item() == 42
    assert frame["capture.camera_age_ms"].item() == 5.0
    assert frame["capture.proprio_age_ms"].item() == 2.0
    assert frame["capture.camera_capture_age_ms"].tolist() == [15.0, -1.0, -1.0]
    assert frame["capture.sonic_received_monotonic_ns"].item() == -1
    assert frame["capture.sonic_age_ms"].item() == -1.0


def test_synchronization_gap_marks_episode_for_discard():
    collector = _recording_collector()
    collector._synchronization_errors = ["camera did not advance"]

    collector._finish_recording(discarded=False, reason="")

    [job] = collector.episode_finalizer.jobs
    assert job["discarded"] is True
    assert job["validation"]["passed"] is False
    assert job["validation"]["errors"] == ["camera did not advance"]


def test_invalid_selected_hand_state_becomes_a_gap_instead_of_crashing(monkeypatch):
    collector = _recording_collector()
    collector.hand_config = {"session_id": "session"}
    collector.hand_profile = type("Profile", (), {"width": 10})()
    target = collector._next_target_ns
    for stream, value in (
        ("proprio", {}),
        (
            "camera",
            {
                "images": {"ego_view": np.zeros((2, 2, 3), dtype=np.uint8)},
                "receiver_monotonic_ns": target - 1,
            },
        ),
        ("manager", {"stream_mode": 0}),
        ("hand", {"mode": "fault"}),
    ):
        collector._synchronizer.observe(stream, value, target - 1)
        collector._synchronizer.observe(stream, value, target + 1)
    monkeypatch.setattr(
        "gear_sonic.scripts.run_data_exporter.time.monotonic_ns",
        lambda: target + collector.synchronization_delay_ns,
    )
    collector._add_data_frame_sonic = lambda *_args: (_ for _ in ()).throw(
        AssertionError("invalid hand state reached frame assembly")
    )

    assert collector._add_data_frame() is False
    assert "external hand controller is faulted" in collector._synchronization_errors[0]


def test_collector_emits_target_only_after_future_watermarks_without_using_them(
    monkeypatch,
):
    collector = _recording_collector()
    target = collector._next_target_ns
    camera_past = {
        "images": {"ego_view": np.zeros((2, 2, 3), dtype=np.uint8)},
        "capture_monotonic_ns": {"ego_view": 9_990_000_000},
        "publisher_monotonic_ns": 10_000_000_000,
        "publisher_sequence": 7,
        "receiver_monotonic_ns": target - 5_000_000,
    }
    samples = {
        "proprio": ({"value": "state-past"}, {"value": "state-future"}),
        "camera": (camera_past, {"value": "camera-future"}),
        "manager": ({"stream_mode": 0}, {"stream_mode": 0}),
    }
    for stream, (past, future) in samples.items():
        collector._synchronizer.observe(stream, past, target - 5_000_000)
        collector._synchronizer.observe(stream, future, target + 1_000_000)
    selected = []
    collector._add_data_frame_sonic = lambda _start, selection: selected.append(
        selection
    )
    monkeypatch.setattr(
        "gear_sonic.scripts.run_data_exporter.time.monotonic_ns",
        lambda: target + collector.synchronization_delay_ns,
    )

    assert collector._add_data_frame() is True

    [selection] = selected
    assert all(sample.timestamp_ns <= target for sample in selection.samples.values())
    assert selection.samples["proprio"].value["value"] == "state-past"
    assert selection.samples["camera"].value is camera_past
    assert collector._next_target_ns == target + collector.loop_period_ns


def test_collector_waits_for_watermarks_then_records_a_bounded_gap(monkeypatch):
    collector = _recording_collector()
    target = collector._next_target_ns
    for stream, value in (
        ("proprio", {}),
        (
            "camera",
            {
                "images": {"ego_view": np.zeros((2, 2, 3), dtype=np.uint8)},
                "receiver_monotonic_ns": target - 1,
            },
        ),
        ("manager", {"stream_mode": 0}),
    ):
        collector._synchronizer.observe(stream, value, target - 1)

    now = target + collector.synchronization_delay_ns
    monkeypatch.setattr(
        "gear_sonic.scripts.run_data_exporter.time.monotonic_ns",
        lambda: now,
    )
    assert collector._add_data_frame() is False
    assert collector._synchronization_errors == []

    now = (
        target
        + collector.synchronization_delay_ns
        + collector.synchronization_wait_timeout_ns
        + 1
    )
    assert collector._add_data_frame() is False
    assert "streams did not advance" in collector._synchronization_errors[0]
    assert collector._synchronization_skipped_targets > 0
