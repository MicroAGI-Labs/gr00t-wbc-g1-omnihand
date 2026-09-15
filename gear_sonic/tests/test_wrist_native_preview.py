from http.server import ThreadingHTTPServer
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
import pytest

from gear_sonic.camera.composed_camera import ComposedCameraSensor
from gear_sonic.camera.drivers.usb_camera import USBCameraConfig, USBCameraSensor
from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.scripts.run_camera_web_viewer import CameraFrameHub, CameraWebViewerConfig, make_handler


def frame_hub():
    hub = object.__new__(CameraFrameHub)
    hub.config = CameraWebViewerConfig()
    hub._condition = threading.Condition()
    hub._native_previews = {}
    return hub


def test_native_usb_read_does_not_resize_and_keeps_rgb(monkeypatch):
    frame = np.full((720, 1280, 3), [10, 50, 220], dtype=np.uint8)
    sensor = object.__new__(USBCameraSensor)
    sensor.config = USBCameraConfig(image_dim=(1280, 720), capture_dim=(1280, 720))
    sensor.mount_position = "left_wrist"
    sensor.cap = SimpleNamespace(read=lambda: (True, frame.copy()))
    monkeypatch.setattr(cv2, "resize", lambda *args, **kwargs: pytest.fail("Native image was resized"))
    sample = sensor.read()
    assert sample["images"]["left_wrist"].shape == (720, 1280, 3)
    np.testing.assert_array_equal(sample["images"]["left_wrist"][0, 0], [220, 50, 10])
    assert "native_previews" not in sample


def test_native_recording_and_ui_use_the_same_camera_message():
    server = object.__new__(ComposedCameraSensor)
    image = np.full((720, 1280, 3), [220, 50, 10], dtype=np.uint8)
    wire = server.serialize_message({"left_wrist": {
        "images": {"left_wrist": image}, "timestamps": {"left_wrist": 10.0},
    }})
    assert "native_previews" not in wire
    recorded = ImageMessageSchema.deserialize(wire)
    assert recorded.images["left_wrist"].shape == (720, 1280, 3)
    hub = frame_hub()
    hub._update_native_previews(recorded.images, recorded.timestamps, time.monotonic())
    jpeg = hub.wrist_preview("left_wrist", "native")
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == image.shape
    assert np.abs(decoded.astype(float) - image[..., ::-1]).mean() < 3


def test_http_native_preview_and_stale_camera():
    hub = frame_hub()
    now = time.monotonic()
    images = {"left_wrist": np.full((720, 1280, 3), [220, 50, 10], dtype=np.uint8)}
    timestamps = {"left_wrist": 10.0}
    hub._update_native_previews(images, timestamps, now)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(hub, None, None, None))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/cameras/left_wrist/preview.jpg", timeout=2) as response:
            assert response.headers["Cache-Control"] == "no-store"
            image = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (720, 1280, 3)
        assert image[0, 0, 2] > image[0, 0, 0]
        for path, code in (("/cameras/right_wrist/preview.jpg", 503),
                           ("/cameras/left_wrist/preview.jpg?resolution=640x480", 400)):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path, timeout=2)
            assert error.value.code == code
        hub._native_previews["left_wrist"]["received_at"] = now - 3
        hub._update_native_previews(images, timestamps, now)
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/cameras/left_wrist/preview.jpg", timeout=2)
        assert error.value.code == 503
        with urlopen(base, timeout=2) as response:
            page = response.read().decode()
        assert 'id="wrist-resolution"' not in page
        assert "recordings use native 1280" in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
