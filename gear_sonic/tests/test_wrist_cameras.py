import queue
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gear_sonic.camera.composed_camera import (
    THOR_JR_WRIST_DEVICE_IDS,
    ComposedCameraClientSensor,
    ComposedCameraConfig,
    ComposedCameraSensor,
)
from gear_sonic.camera.drivers import usb_camera
from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.data.features_sonic_vla import get_wrist_camera_features
from gear_sonic.scripts.run_data_exporter import GrootDataCollector


def sample(name, sequence, received_ns=1_000_000_000):
    return {
        "images": {name: np.full((480, 640, 3), sequence % 255, dtype=np.uint8)},
        "timestamps": {name: float(sequence)},
        "receiver_monotonic_ns": received_ns,
    }


def receiver():
    client = object.__new__(ComposedCameraClientSensor)
    client._background_frames = {}
    client._background_timestamps = {}
    client._selected_frames = {}
    client._background_target_depth = 2
    return client


def test_usb_uses_stable_path_mjpeg_two_buffers_and_resizes(monkeypatch):
    properties = {}
    opened = []
    frame = np.full((720, 1280, 3), (10, 20, 30), dtype=np.uint8)

    class Capture:
        def __init__(self, *args):
            opened.append(args)

        def isOpened(self):
            return True

        def set(self, key, value):
            properties[key] = value
            return True

        def get(self, key):
            return properties[key]

        def read(self):
            return True, frame

        def release(self):
            self.released = True

    monkeypatch.setattr(usb_camera.cv2, "VideoCapture", Capture)
    composed = object.__new__(ComposedCameraSensor)
    composed.config = ComposedCameraConfig(
        usb_camera_fps=60,
        usb_camera_resolution=(1280, 720),
        usb_camera_mjpeg=True,
    )
    path = "/dev/v4l/by-id/usb-JR0001_JR0001_JR0001-video-index0"
    sensor = composed._instantiate_camera("left_wrist", "usb", path)
    assert opened == [(path, cv2.CAP_V4L2)]
    assert properties[cv2.CAP_PROP_FPS] == 60
    assert properties[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    assert properties[cv2.CAP_PROP_BUFFERSIZE] == 2
    result = sensor.read()
    assert result["images"]["left_wrist"].shape == (480, 640, 3)
    np.testing.assert_array_equal(result["images"]["left_wrist"][0, 0], [30, 20, 10])
    assert result["sample_monotonic_ns"] > 0
    assert sensor.serialize(result)["sample_monotonic_ns"] == result["sample_monotonic_ns"]
    sensor.close()
    assert sensor.cap.released


def test_combiner_retains_independent_arrivals_and_original_timestamps():
    server = object.__new__(ComposedCameraSensor)
    server.error_events = {}
    server.camera_queues = {name: queue.Queue() for name in ("ego_view", "left_wrist", "right_wrist")}
    server._latest_frames = {}
    for i, name in enumerate(server.camera_queues, 1):
        server.camera_queues[name].put(sample(name, i))
        message = server.read()
        assert len(message) == i
    decoded = ImageMessageSchema.deserialize(server.serialize_message(message)).asdict()
    assert decoded["timestamps"] == {"ego_view": 1.0, "left_wrist": 2.0, "right_wrist": 3.0}
    assert server.read() is None
    server.camera_queues["left_wrist"].put(sample("left_wrist", 4))
    assert server.read()["ego_view"]["timestamps"]["ego_view"] == 1.0


def test_receiver_samples_three_asynchronous_60hz_streams_at_50hz():
    client = receiver()
    selected = {name: [] for name in ("ego_view", "left_wrist", "right_wrist")}
    # A 300 Hz deterministic clock expresses both 60 and 50 Hz exactly.
    for tick in range(600):
        for phase, name in enumerate(selected):
            if tick % 5 == phase:
                client._buffer_camera_frames(sample(name, tick + 1, (tick + 1) * 1_000_000))
        if tick % 6 == 5:
            message = client._sample_camera_frames()
            for name in selected:
                selected[name].append(message["timestamps"][name])
    for timestamps in selected.values():
        assert len(timestamps) == len(set(timestamps)) == 100
        assert timestamps == sorted(timestamps)
    assert all(len(frames) <= 2 for frames in client._background_frames.values())


def test_repeated_cached_images_do_not_mask_stalled_camera():
    client = receiver()
    first = {**sample("left_wrist", 1), "images": {}, "timestamps": {}}
    for name in ("left_wrist", "right_wrist"):
        first["images"].update(sample(name, 1)["images"])
        first["timestamps"][name] = 1.0
    client._buffer_camera_frames(first)
    client._sample_camera_frames()
    newer = {
        **first,
        "timestamps": {"left_wrist": 1.0, "right_wrist": 2.0},
        "receiver_monotonic_ns": 2_000_000_000,
    }
    client._buffer_camera_frames(newer)
    message = client._sample_camera_frames()
    assert message["camera_received_monotonic_ns"]["left_wrist"] == 1_000_000_000
    assert message["camera_received_monotonic_ns"]["right_wrist"] == 2_000_000_000
    assert message["timestamps"]["left_wrist"] == 1


def collector(wrists):
    obj = object.__new__(GrootDataCollector)
    features = {"observation.images.ego_view": {"dtype": "video", "shape": [480, 640, 3]}}
    if wrists:
        features.update(get_wrist_camera_features())
    obj.data_exporter = SimpleNamespace(features=features)
    obj.camera_max_age = 0.1
    obj.latest_image_msg = sample("ego_view", 1)
    obj.latest_image_msg["camera_received_monotonic_ns"] = {"ego_view": 1_000_000_000}
    return obj


@pytest.mark.parametrize("wrists", [False, True])
def test_recording_includes_wrist_images_and_source_times_only_when_enabled(wrists):
    obj = collector(wrists)
    for name in ("left_wrist", "right_wrist"):
        obj.latest_image_msg["images"].update(sample(name, 2)["images"])
        obj.latest_image_msg["timestamps"][name] = 2.0
        obj.latest_image_msg["camera_received_monotonic_ns"][name] = 1_000_000_000
    obj._validate_camera_inputs(1.01)
    frame = {}
    obj._add_images_to_frame_data(frame)
    assert len(frame) == (5 if wrists else 1)
    if wrists:
        assert frame["capture.left_wrist_source_timestamp_ns"].tolist() == [2_000_000_000]


def test_optional_recording_rejects_missing_stale_and_wrong_size_wrists():
    obj = collector(True)
    with pytest.raises(RuntimeError, match="left_wrist is unavailable"):
        obj._validate_camera_inputs(1.01)
    for name in ("left_wrist", "right_wrist"):
        obj.latest_image_msg["images"].update(sample(name, 1)["images"])
        obj.latest_image_msg["timestamps"][name] = 1.0
    obj.latest_image_msg["camera_received_monotonic_ns"]["left_wrist"] = 100_000_000
    with pytest.raises(RuntimeError, match="left_wrist is stale"):
        obj._validate_camera_inputs(1.01)
    obj.latest_image_msg["camera_received_monotonic_ns"]["left_wrist"] = 1_000_000_000
    obj.latest_image_msg["images"]["right_wrist"] = np.zeros((720, 1280, 3))
    with pytest.raises(RuntimeError, match="right_wrist shape"):
        obj._validate_camera_inputs(1.01)
    obj.data_exporter.features = collector(False).data_exporter.features
    obj._validate_camera_inputs(1.01)


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"capture_dim": (0, 720)}, {"buffer_size": 1}])
def test_invalid_usb_capture_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        usb_camera.USBCameraConfig(**kwargs)


def test_thor_profile_pins_robot_sides_and_capture_settings():
    config = ComposedCameraConfig(ego_view_camera="zed", wrist_camera_profile="thor-jr")
    sensor = object.__new__(ComposedCameraSensor)
    sensor.config = config
    cameras = sensor._get_camera_configs()
    assert cameras["left_wrist"] == {
        "camera_type": "usb",
        "device_id": "/dev/v4l/by-id/usb-JR0001_JR0001_JR0001-video-index0",
    }
    assert cameras["right_wrist"] == {
        "camera_type": "usb",
        "device_id": "/dev/v4l/by-id/usb-JR0002_JR0002_JR0002-video-index0",
    }
    assert config.usb_camera_resolution == (1280, 720)
    assert config.usb_camera_fps == config.fps == 60
    assert config.usb_camera_mjpeg
    assert cameras["ego_view"]["camera_type"] == "zed"


def test_thor_profile_does_not_resolve_or_fallback_when_devices_are_absent(monkeypatch):
    from pathlib import Path

    def no_device_access(*args, **kwargs):
        raise AssertionError("Mapping must not depend on current USB enumeration")

    monkeypatch.setattr(Path, "resolve", no_device_access)
    monkeypatch.setattr(Path, "exists", no_device_access)
    config = ComposedCameraConfig(wrist_camera_profile="thor-jr")
    assert config.left_wrist_device_id == THOR_JR_WRIST_DEVICE_IDS["left_wrist"]
    assert config.right_wrist_device_id == THOR_JR_WRIST_DEVICE_IDS["right_wrist"]


@pytest.mark.parametrize(
    "override",
    [
        {"left_wrist_device_id": "0"},
        {"right_wrist_device_id": THOR_JR_WRIST_DEVICE_IDS["left_wrist"]},
        {"left_wrist_camera": "oak"},
    ],
)
def test_thor_profile_rejects_conflicting_side_assignments(override):
    with pytest.raises(ValueError, match="thor-jr fixes"):
        ComposedCameraConfig(wrist_camera_profile="thor-jr", **override)


def test_default_profile_keeps_wrist_capture_optional():
    config = ComposedCameraConfig()
    assert config.left_wrist_camera is config.right_wrist_camera is None
