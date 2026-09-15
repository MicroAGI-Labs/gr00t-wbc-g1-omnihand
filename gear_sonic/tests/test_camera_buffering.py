from __future__ import annotations

import socket
import time
from collections import deque

import numpy as np
import pytest

from gear_sonic.camera import sensor_server
from gear_sonic.camera.composed_camera import (
    CameraFrameBuffer,
    ComposedCameraClientSensor,
    estimate_camera_age_s,
)
from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_camera_frame_buffer_bounds_memory_and_reports_rates_and_drops(monkeypatch):
    buffer = CameraFrameBuffer(capacity=3, reserve=1)
    for sequence, timestamp in zip(
        (10, 12, 13, 14),
        (1_000_000_000, 1_020_000_000, 1_040_000_000, 1_060_000_000),
        strict=True,
    ):
        buffer.put(
            {"publisher_sequence": sequence, "receiver_monotonic_ns": timestamp}
        )
    monkeypatch.setattr(
        "gear_sonic.camera.composed_camera.time.monotonic",
        lambda: 1.06,
    )

    assert buffer.pop_for_collection()["publisher_sequence"] == 13
    assert buffer.pop_for_collection()["publisher_sequence"] == 14
    stats = buffer.stats()
    assert stats["depth"] == 0
    assert stats["received"] == 4
    assert stats["overflow_dropped"] == 1
    assert stats["latency_dropped"] == 1
    assert stats["publisher_gap_dropped"] == 1
    assert stats["received_hz"] == 50.0
    assert stats["publisher_hz"] == 66.67


def test_sensor_server_preserves_direct_capture_time_and_derives_missing_time(
    monkeypatch,
):
    class CapturingSocket:
        payload = None

        def send(self, payload, flags):
            self.payload = payload

    server = object.__new__(sensor_server.SensorServer)
    server.socket = CapturingSocket()
    server.message_sent = 4
    server.message_dropped = 0
    monkeypatch.setattr(sensor_server.msgpack, "packb", lambda payload, use_bin_type: payload)
    monkeypatch.setattr(sensor_server.time, "time", lambda: 100.0)
    monkeypatch.setattr(sensor_server.time, "monotonic_ns", lambda: 10_000_000_000)

    server.send_message(
        {
            "timestamps": {"head": 99.0, "wrist": 98.0},
            "images": {},
            "capture_monotonic_ns": {"head": 123},
        }
    )

    assert server.socket.payload["capture_monotonic_ns"] == {
        "head": 123,
        "wrist": 8_000_000_000,
    }
    assert server.socket.payload["publisher_sequence"] == 5
    assert server.socket.payload["publisher_monotonic_ns"] == 10_000_000_000


def test_background_camera_client_receives_and_stops_cleanly():
    port = _unused_tcp_port()
    server = SensorServer()
    server.start_server(port)
    client = ComposedCameraClientSensor(
        server_ip="127.0.0.1",
        port=port,
        background=True,
        background_queue_size=3,
        background_reserve=1,
    )
    try:
        payload = ImageMessageSchema(
            timestamps={"head": time.time()},
            images={"head": np.zeros((2, 2, 3), dtype=np.uint8)},
            capture_monotonic_ns={"head": time.monotonic_ns()},
        ).serialize()
        received = None
        deadline = time.monotonic() + 2.0
        while received is None and time.monotonic() < deadline:
            server.send_message(payload)
            time.sleep(0.01)
            pending = client.read_pending()
            received = pending[-1] if pending else None

        assert received is not None
        assert received["images"]["head"].shape == (2, 2, 3)
        assert received["publisher_sequence"] >= 1
        assert received["receiver_monotonic_ns"] > 0
        assert client.buffer_stats()["received"] >= 1
    finally:
        client.close()
        server.stop_server()


def test_camera_age_does_not_compare_monotonic_clocks_across_hosts():
    message = {
        "capture_monotonic_ns": {"head": 9_900_000_000},
        "publisher_monotonic_ns": 10_000_000_000,
        "receiver_monotonic_ns": 999_950_000_000,
    }

    age = estimate_camera_age_s(
        message,
        "head",
        now_monotonic_ns=1_000_000_000_000,
    )

    assert age == 0.15


def test_local_camera_age_includes_time_spent_queued_before_receipt():
    message = {
        "capture_monotonic_ns": {"head": 1_000_000_000},
        "publisher_monotonic_ns": 1_020_000_000,
        "receiver_monotonic_ns": 1_600_000_000,
        "_publisher_clock_is_local": True,
    }
    assert estimate_camera_age_s(message, "head", now_monotonic_ns=1_610_000_000) == 0.61


@pytest.mark.parametrize("reserve", [0, 1])
def test_receiver_skips_encoded_backlog_and_retains_recent_reserve(reserve):
    client = object.__new__(ComposedCameraClientSensor)
    client._background_target_depth = reserve + 1
    client._wire_backlog_dropped = 0
    queued = deque({"publisher_sequence": i} for i in range(1, 21))
    client.receive_message_nonblocking = lambda **kwargs: queued.popleft() if queued else None
    messages = client._receive_recent_messages()
    assert [message["publisher_sequence"] for message, received in messages] == list(range(20 - reserve, 21))
    assert client._wire_backlog_dropped == 19 - reserve
    assert all(received > 0 for message, received in messages)


def test_cached_camera_decode_tracks_each_capture_and_evicts_removed_views(monkeypatch):
    client = object.__new__(ComposedCameraClientSensor)
    client._decoded_image_cache = {}
    original = ImageMessageSchema.deserialize
    decoded_names = []
    def deserialize(message):
        decoded_names.append(set(message["images"]))
        return original(message)
    monkeypatch.setattr(ImageMessageSchema, "deserialize", staticmethod(deserialize))
    head = np.zeros((2, 2, 3), dtype=np.uint8)
    wrist = np.ones_like(head)
    message = {"images": {"head": head, "wrist": wrist},
               "timestamps": {"head": 10.0, "wrist": 10.0},
               "capture_monotonic_ns": {"head": 100, "wrist": 100}}
    first = client._decode_camera_message(message)
    message["capture_monotonic_ns"] = {"head": 100, "wrist": 200}
    message["timestamps"] = {"head": 10.0, "wrist": 10.1}
    second = client._decode_camera_message(message)
    assert decoded_names == [{"head", "wrist"}, {"wrist"}]
    assert second["images"]["head"] is first["images"]["head"]
    assert second["timestamps"] == {"head": 10.0, "wrist": 10.1}
    # A repeated wall timestamp cannot hide a new monotonic capture.
    message["capture_monotonic_ns"] = {"head": 300, "wrist": 200}
    client._decode_camera_message(message)
    assert decoded_names[-1] == {"head"}
    client._decode_camera_message({"images": {"wrist": wrist}, "timestamps": {"wrist": 11.0}})
    assert set(client._decoded_image_cache) == {"wrist"}


def test_images_without_capture_identity_are_always_decoded():
    client = object.__new__(ComposedCameraClientSensor)
    client._decoded_image_cache = {}
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones_like(first)
    client._decode_camera_message({"images": {"head": first}})
    assert client._decode_camera_message({"images": {"head": second}})["images"]["head"] is second
