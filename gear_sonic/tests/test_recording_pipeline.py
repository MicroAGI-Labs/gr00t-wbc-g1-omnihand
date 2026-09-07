from __future__ import annotations

import time

import numpy as np
import pytest

from gear_sonic.data.causal_sync import (
    CausalSelection,
    CausalSynchronizer,
    TimedSample,
)
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


@pytest.mark.parametrize(
    ("stream_mode", "extra_stream"),
    [(2, "planner"), (3, "planner"), (4, None), (5, "planner")],
)
def test_collector_uses_streams_required_by_manager_mode(stream_mode, extra_stream):
    collector = _recording_collector()
    target = time.monotonic_ns()
    samples = [
        ("proprio", {}, {}),
        ("camera", {}, {}),
        ("manager", {"stream_mode": stream_mode}, {"stream_mode": stream_mode}),
    ]
    if extra_stream:
        samples.append((extra_stream, {"value": "past"}, {"value": "future"}))
    for stream, past, future in samples:
        collector._synchronizer.observe(stream, past, target - 1_000_000)
        collector._synchronizer.observe(stream, future, target + 1_000_000)

    selection = collector._selection_for_target(target)

    assert selection.ready
    expected = {"proprio", "camera", "manager"}
    if extra_stream:
        expected.add(extra_stream)
        assert selection.samples[extra_stream].value["value"] == "past"
    assert set(selection.samples) == expected


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
    collector._finish_recording(discarded=False, reason="")
    assert collector.episode_finalizer.jobs[0]["discarded"] is True
