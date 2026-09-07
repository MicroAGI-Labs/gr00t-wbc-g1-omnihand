from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.camera import sensor_server
from gear_sonic.camera.composed_camera import ComposedCameraConfig, ComposedCameraSensor
from gear_sonic.camera.drivers import zed as zed_driver


class FakeMat:
    def __init__(self, image):
        self.image = image
        self.freed = False

    def get_data(self):
        return self.image

    def free(self):
        self.freed = True


class FakeInitParameters:
    def __init__(self):
        self.camera_resolution = None
        self.camera_fps = None
        self.depth_mode = None
        self.serial_number = None

    def set_from_serial_number(self, serial_number):
        self.serial_number = serial_number


class FakeRuntimeParameters:
    def __init__(self):
        self.enable_depth = True


class FakeResolution:
    def __init__(self, width, height):
        self.width = width
        self.height = height


class FakeTimestamp:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds

    def get_nanoseconds(self):
        return self.nanoseconds


class FakeCamera:
    def __init__(self, sdk):
        self.sdk = sdk
        self.init_params = None
        self.closed = False
        self.retrieve_args = None

    def open(self, init_params):
        self.init_params = init_params
        return self.sdk.open_status

    def grab(self, runtime_params):
        self.runtime_params = runtime_params
        return self.sdk.grab_status

    def retrieve_image(self, *args):
        self.retrieve_args = args
        return self.sdk.retrieve_status

    def get_timestamp(self, time_reference):
        self.time_reference = time_reference
        return FakeTimestamp(self.sdk.timestamp_ns)

    def get_camera_information(self):
        return SimpleNamespace(
            camera_model="ZED2i",
            serial_number=123456,
            camera_configuration=SimpleNamespace(
                resolution=FakeResolution(1280, 720),
                fps=60,
            ),
        )

    def close(self):
        self.closed = True


class FakeSDK:
    RESOLUTION = SimpleNamespace(HD720="HD720")
    DEPTH_MODE = SimpleNamespace(NONE="NONE")
    ERROR_CODE = SimpleNamespace(SUCCESS=0, FAILURE=1)
    VIEW = SimpleNamespace(LEFT="LEFT")
    MEM = SimpleNamespace(CPU="CPU")
    TIME_REFERENCE = SimpleNamespace(IMAGE="IMAGE")
    InitParameters = FakeInitParameters
    RuntimeParameters = FakeRuntimeParameters
    Resolution = FakeResolution

    def __init__(self, image=None):
        if image is None:
            image = np.zeros((480, 640, 4), dtype=np.uint8)
        self.image = image
        self.open_status = self.ERROR_CODE.SUCCESS
        self.grab_status = self.ERROR_CODE.SUCCESS
        self.retrieve_status = self.ERROR_CODE.SUCCESS
        self.timestamp_ns = 1_700_000_000_250_000_000
        self.camera = FakeCamera(self)
        self.mat = FakeMat(self.image)

    def Camera(self):
        return self.camera

    def Mat(self):
        return self.mat


@pytest.fixture
def fake_sdk(monkeypatch):
    sdk = FakeSDK()
    monkeypatch.setattr(zed_driver, "_load_zed_sdk", lambda: sdk)
    return sdk


def test_zed_sensor_configures_hd720_60fps_without_depth(fake_sdk):
    sensor = zed_driver.ZEDSensor(device_id="123456")

    init_params = fake_sdk.camera.init_params
    assert init_params.camera_resolution == fake_sdk.RESOLUTION.HD720
    assert init_params.camera_fps == 60
    assert init_params.depth_mode == fake_sdk.DEPTH_MODE.NONE
    assert init_params.serial_number == 123456
    assert sensor._runtime_params.enable_depth is False
    assert (sensor._output_resolution.width, sensor._output_resolution.height) == (640, 480)


def test_zed_sensor_returns_owned_rgb_frame_and_host_timestamps(monkeypatch):
    bgra = np.zeros((480, 640, 4), dtype=np.uint8)
    bgra[0, 0] = [10, 20, 30, 255]
    sdk = FakeSDK(image=bgra)
    monkeypatch.setattr(zed_driver, "_load_zed_sdk", lambda: sdk)
    monkeypatch.setattr(zed_driver.time, "time", lambda: 2_000_000_000.0)
    monkeypatch.setattr(zed_driver.time, "monotonic_ns", lambda: 123_456_789)
    sensor = zed_driver.ZEDSensor(mount_position="ego_view")

    sample = sensor.read()

    assert sample is not None
    image = sample["images"]["ego_view"]
    np.testing.assert_array_equal(image[0, 0], np.array([30, 20, 10]))
    assert image.shape == (480, 640, 3)
    assert image.flags.c_contiguous
    assert not np.shares_memory(image, bgra)
    assert sample["timestamps"]["ego_view"] == 2_000_000_000.0
    assert sample["sample_monotonic_ns"] == 123_456_789
    assert sensor.serialize(sample)["sample_monotonic_ns"] == 123_456_789
    assert sdk.camera.retrieve_args[1:] == (
        sdk.VIEW.LEFT,
        sdk.MEM.CPU,
        sensor._output_resolution,
    )


def test_zed_sensor_ignores_sdk_epoch_after_clock_jump(monkeypatch):
    sdk = FakeSDK()
    sdk.timestamp_ns = 1_700_000_000_000_000_000
    monkeypatch.setattr(zed_driver, "_load_zed_sdk", lambda: sdk)
    monkeypatch.setattr(zed_driver.time, "time", lambda: 2_000_000_000.0)
    monkeypatch.setattr(zed_driver.time, "monotonic_ns", lambda: 987_654_321)

    sample = zed_driver.ZEDSensor().read()

    assert sample["timestamps"]["ego_view"] == 2_000_000_000.0
    assert sample["sample_monotonic_ns"] == 987_654_321
    assert not hasattr(sdk.camera, "time_reference")


def test_composed_camera_preserves_source_monotonic_timestamp():
    composed = object.__new__(ComposedCameraSensor)
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    serialized = composed.serialize_message(
        {
            "ego_view": {
                "timestamps": {"ego_view": 2_000_000_000.0},
                "images": {"ego_view": image},
                "sample_monotonic_ns": 123_456_789,
            }
        }
    )

    assert serialized["sample_monotonic_ns"] == 123_456_789


def test_sensor_server_does_not_rederive_supplied_monotonic_timestamp(monkeypatch):
    class CapturingSocket:
        def send(self, payload, flags):
            self.payload = payload

    server = object.__new__(sensor_server.SensorServer)
    server.socket = CapturingSocket()
    server.message_sent = 0
    server.message_dropped = 0
    monkeypatch.setattr(
        sensor_server.msgpack, "packb", lambda payload, use_bin_type: payload
    )
    monkeypatch.setattr(sensor_server.time, "time", lambda: 2_000_000_010.0)
    monkeypatch.setattr(sensor_server.time, "monotonic_ns", lambda: 999_999_999)

    server.send_message(
        {
            "timestamps": {"ego_view": 1_700_000_000.0},
            "images": {},
            "sample_monotonic_ns": 123_456_789,
        }
    )

    assert server.socket.payload["sample_monotonic_ns"] == 123_456_789
    assert server.socket.payload["publisher_monotonic_ns"] == 999_999_999


def test_zed_sensor_rotates_frame_180(monkeypatch):
    bgra = np.zeros((2, 3, 4), dtype=np.uint8)
    bgra[-1, -1] = [10, 20, 30, 255]
    sdk = FakeSDK(image=bgra)
    monkeypatch.setattr(zed_driver, "_load_zed_sdk", lambda: sdk)

    image = zed_driver.ZEDSensor(config=zed_driver.ZEDConfig(rotate_180=True)).read()[
        "images"
    ]["ego_view"]

    np.testing.assert_array_equal(image[0, 0], np.array([30, 20, 10]))
    assert image.flags.c_contiguous


@pytest.mark.parametrize("failure_stage", ["grab", "retrieve"])
def test_zed_sensor_returns_none_on_capture_failure(fake_sdk, failure_stage):
    sensor = zed_driver.ZEDSensor()
    if failure_stage == "grab":
        fake_sdk.grab_status = fake_sdk.ERROR_CODE.FAILURE
    else:
        fake_sdk.retrieve_status = fake_sdk.ERROR_CODE.FAILURE

    assert sensor.read() is None


def test_zed_sensor_closes_camera_and_mat(fake_sdk):
    sensor = zed_driver.ZEDSensor()

    sensor.close()
    sensor.close()

    assert fake_sdk.mat.freed
    assert fake_sdk.camera.closed


def test_zed_sensor_rejects_non_numeric_serial(fake_sdk):
    with pytest.raises(ValueError, match="numeric serial number"):
        zed_driver.ZEDSensor(device_id="/dev/video0")


def test_zed_sensor_closes_camera_when_open_fails(fake_sdk):
    fake_sdk.open_status = fake_sdk.ERROR_CODE.FAILURE

    with pytest.raises(RuntimeError, match="Failed to open ZED camera"):
        zed_driver.ZEDSensor()

    assert fake_sdk.camera.closed


def test_composed_camera_factory_passes_zed_options(monkeypatch):
    captured = {}
    expected_sensor = object()

    def fake_zed_sensor(**kwargs):
        captured.update(kwargs)
        return expected_sensor

    monkeypatch.setattr(zed_driver, "ZEDSensor", fake_zed_sensor)
    composed = object.__new__(ComposedCameraSensor)
    composed.config = ComposedCameraConfig(
        zed_camera_resolution="HD1080",
        zed_camera_fps=30,
    )

    sensor = composed._instantiate_camera("ego_view", "zed", "123456")

    assert sensor is expected_sensor
    assert captured["mount_position"] == "ego_view"
    assert captured["device_id"] == "123456"
    assert captured["config"] == zed_driver.ZEDConfig(
        camera_resolution="HD1080",
        camera_fps=30,
        rotate_180=True,
    )
