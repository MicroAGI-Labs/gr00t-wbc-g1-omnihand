from __future__ import annotations

import time

import numpy as np

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
    collector.latest_proprio_msg = {"body_q": np.zeros(29)}
    collector.latest_image_msg = {
        "images": {"ego_view": np.zeros((2, 2, 3))},
        "capture_monotonic_ns": {"ego_view": 9_000_000_000},
        "publisher_monotonic_ns": 10_000_000_000,
        "receiver_monotonic_ns": time.monotonic_ns() - 1_000_000_000,
    }
    collector.latest_image_received_at = time.monotonic() - 1.0
    collector._add_data_frame_sonic = lambda _start: (_ for _ in ()).throw(
        AssertionError("stale frame reached exporter")
    )

    assert collector._add_data_frame() is False


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
