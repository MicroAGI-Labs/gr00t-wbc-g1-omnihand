from __future__ import annotations

import socket
import time

import numpy as np

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
            received = client.read()

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
