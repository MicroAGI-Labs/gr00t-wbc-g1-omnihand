import json
import socket
import threading
import time
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import numpy as np
import pytest

from gear_sonic.utils.teleop.pico_body_diagnostics import PicoBodySnapshot, PicoBodyPublisher, PicoBodySubscriber


def body():
    # Deliberately non-normalized orientations: diagnostics must preserve raw data.
    return np.arange(168, dtype=float).reshape(24, 7) / 100


def test_body_freshness_is_independent_of_controller_packets():
    cache = PicoBodySnapshot()
    assert cache.snapshot(now=0)["state"] == "waiting"
    poses = body()
    cache.record(poses, 100, 10, now=0)
    for i in range(1, 31):
        cache.record(poses, 100 + i, 10, now=i / 10)
    status = cache.snapshot(now=3)
    assert status["packet_live"]
    assert not status["body_live"]
    assert status["state"] == "stale"
    assert status["body_age_s"] == 3
    np.testing.assert_array_equal(status["poses"], poses)
    cache.record(poses, 140, 11, now=3.1)
    assert cache.snapshot(now=3.1)["state"] == "live"


def test_stationary_pose_is_live_when_body_timestamp_advances():
    cache = PicoBodySnapshot()
    for i in range(1, 50):
        cache.record(body(), i, i, now=i)
    assert cache.snapshot(now=49)["body_live"]


def test_missing_body_timestamp_is_not_reported_live():
    cache = PicoBodySnapshot()
    cache.record(body(), 123, 0, now=0)
    status = cache.snapshot(now=0)
    assert status["packet_live"] and not status["body_live"]
    assert status["state"] == "timestamp_unavailable"


def test_raw_read_rate_without_body_timestamp_counts_frames_not_ui_updates():
    cache = PicoBodySnapshot()
    for i in range(181):
        cache.record(body(), i + 1, None, now=i / 90)
        cache.record(body(), i + 1, None, now=i / 90)  # Repeated SDK packet.
    status = cache.snapshot(now=2)
    assert status["body_read_hz"] == pytest.approx(90)
    assert status["body_hz"] is None
    assert not status["body_live"]
    assert cache.snapshot(now=5)["body_read_hz"] == 0


def test_body_rate_is_independent_of_packet_rate_and_stationary_pose():
    cache = PicoBodySnapshot()
    for i in range(181):
        cache.record(body(), i + 1, i // 3 + 1, now=i / 90)
    status = cache.snapshot(now=2)
    assert status["body_read_hz"] == pytest.approx(90)
    assert status["body_hz"] == pytest.approx(30)
    for i in range(181, 451):
        cache.record(body(), i + 1, 61, now=i / 90)
    status = cache.snapshot(now=5)
    assert status["body_hz"] == 0
    assert status["body_read_hz"] == pytest.approx(90)
    assert status["state"] == "stale"


def test_unavailable_body_does_not_count_as_a_body_read():
    cache = PicoBodySnapshot()
    for i in range(100):
        cache.record(None, i + 1, i + 1, available=False, now=i / 90)
    assert cache.snapshot(now=1.1)["body_read_hz"] == 0
    assert cache.snapshot(now=1.1)["body_hz"] == 0


def test_sdk_diagnostics_counts_every_sample_before_publication_throttling(monkeypatch):
    from gear_sonic.scripts import pico_manager_thread_server as manager

    calls = []
    monkeypatch.setattr(manager, "_pico_body_diagnostics", SimpleNamespace(record=lambda *args, **kwargs: calls.append(args)))
    monkeypatch.setattr(manager, "xrt", SimpleNamespace())  # No body timestamp API.
    for i in range(90):
        manager._record_pico_body_diagnostics(available=True, poses=body(), packet_stamp=i + 1)
    assert len(calls) == 90
    assert [args[1] for args in calls] == list(range(1, 91))


def test_outage_preserves_last_raw_frame_with_stale_label():
    cache = PicoBodySnapshot()
    poses = body()
    cache.record(poses, 123, 456, now=0)
    poses[:] = 0
    cache.record(None, 124, 456, available=False, now=1)
    status = cache.snapshot(now=1)
    assert status["state"] == "unavailable"
    np.testing.assert_array_equal(status["poses"], body())
    assert cache.snapshot(now=4)["state"] == "disconnected"


@pytest.mark.parametrize("poses", [np.zeros((3, 7)), np.full((24, 7), np.nan)])
def test_invalid_frame_is_not_serialized_as_a_live_skeleton(poses):
    cache = PicoBodySnapshot()
    cache.record(poses, 123, 456, now=0)
    status = cache.snapshot(now=0)
    assert status["state"] == "unavailable"
    assert status["poses"] is None
    json.dumps(status, allow_nan=False)


def test_live_publisher_to_http_and_transport_loss():
    from gear_sonic.scripts.run_camera_web_viewer import make_handler

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    publisher = PicoBodyPublisher(port)
    subscriber = PicoBodySubscriber(port=port)
    publisher.record(body(), 1234567890123456789, 1234567890123456788)
    publisher.start()
    subscriber.start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(None, None, None, None, subscriber))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while not subscriber.status()["connected"] and time.monotonic() < deadline:
            time.sleep(.02)
        with urlopen(f"http://127.0.0.1:{server.server_port}/pico/body", timeout=2) as response:
            status = json.load(response)
        assert status["connected"]
        assert status["body_timestamp_ns"] == "1234567890123456788"
        np.testing.assert_array_equal(status["poses"], body())
        publisher.close()
        subscriber.close()
        with subscriber._lock:
            subscriber._received_at -= 3
        status = subscriber.status()
        assert status["state"] == "disconnected"
        assert not status["packet_live"] and not status["body_live"]
        assert status["body_read_hz"] == 0
        np.testing.assert_array_equal(status["poses"], body())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        publisher.close()
        subscriber.close()
