from types import SimpleNamespace
import json

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

    def retrieve_measure(self, mat, *args):
        mat.image = self.sdk.depth
        self.retrieve_measure_args = args
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
    DEPTH_MODE = SimpleNamespace(NONE="NONE", PERFORMANCE="PERFORMANCE")
    ERROR_CODE = SimpleNamespace(SUCCESS=0, FAILURE=1)
    VIEW = SimpleNamespace(LEFT="LEFT", RIGHT="RIGHT")
    MEM = SimpleNamespace(CPU="CPU")
    MEASURE = SimpleNamespace(DEPTH="DEPTH")
    TIME_REFERENCE = SimpleNamespace(IMAGE="IMAGE")
    InitParameters = FakeInitParameters
    RuntimeParameters = FakeRuntimeParameters
    Resolution = FakeResolution

    def __init__(self, image=None):
        if image is None:
            image = np.zeros((480, 640, 4), dtype=np.uint8)
        self.image = image
        self.depth = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
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


def test_zed_sensor_configures_hd720_60fps_with_stereo_depth(fake_sdk):
    sensor = zed_driver.ZEDSensor(device_id="123456")

    init_params = fake_sdk.camera.init_params
    assert init_params.camera_resolution == fake_sdk.RESOLUTION.HD720
    assert init_params.camera_fps == 60
    assert init_params.depth_mode == "PERFORMANCE"
    assert init_params.serial_number == 123456
    assert sensor._runtime_params.enable_depth is True
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
    assert sample["images"]["ego_view_left"].shape == (480, 640, 3)
    assert sample["depths"]["ego_view_depth"].shape == (480, 640)
    assert sdk.camera.retrieve_args[1] == sdk.VIEW.RIGHT


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


def test_sensor_server_round_trips_lossless_zed_depth(monkeypatch):
    server = object.__new__(sensor_server.SensorServer)
    server.socket = SimpleNamespace(send=lambda payload, flags: setattr(server, "payload", payload))
    server.message_sent = 0
    server.message_dropped = 0
    depth = np.asarray([[0.5, 1.25]], dtype=np.float32)
    server.send_message({"timestamps": {"ego_view_depth": 2.0}, "images": {}, "depths": {"ego_view_depth": depth}})
    decoded = sensor_server.ImageMessageSchema.deserialize(
        sensor_server.msgpack.unpackb(server.payload, object_hook=sensor_server.m.decode)
    )
    np.testing.assert_array_equal(decoded.depths["ego_view_depth"], depth)


def test_zed_sensor_keeps_native_orientation_by_default(monkeypatch):
    bgra = np.zeros((2, 3, 4), dtype=np.uint8)
    bgra[-1, -1] = [10, 20, 30, 255]
    sdk = FakeSDK(image=bgra)
    monkeypatch.setattr(zed_driver, "_load_zed_sdk", lambda: sdk)

    image = zed_driver.ZEDSensor(config=zed_driver.ZEDConfig(image_dim=(3, 2))).read()[
        "images"
    ]["ego_view"]

    np.testing.assert_array_equal(image[0, 0], np.array([0, 0, 0]))
    assert image.flags.c_contiguous


@pytest.mark.parametrize("failure_stage", ["grab", "retrieve"])
def test_zed_sensor_returns_none_on_capture_failure(fake_sdk, failure_stage):
    sensor = zed_driver.ZEDSensor()
    if failure_stage == "grab":
        fake_sdk.grab_status = fake_sdk.ERROR_CODE.FAILURE
    else:
        fake_sdk.retrieve_status = fake_sdk.ERROR_CODE.FAILURE

    assert sensor.read() is None


@pytest.mark.parametrize("slow_stage", ["grab", "ego_view_left", "ego_view", "depth_view", "depth_measure"])
def test_zed_delay_log_identifies_stage_without_retiming_frame(monkeypatch, capsys, slow_stage):
    sdk = FakeSDK(image=np.zeros((2, 3, 4), dtype=np.uint8))
    sdk.VIEW = SimpleNamespace(LEFT="LEFT", RIGHT="RIGHT", DEPTH="DEPTH")
    sdk.Mat = lambda: FakeMat(sdk.image.copy())
    clock = [1_000_000_000]
    monkeypatch.setattr(zed_driver, "_load_zed_sdk", lambda: sdk)
    monkeypatch.setattr(zed_driver.time, "monotonic_ns", lambda: clock[0])
    def delay(stage):
        if stage == slow_stage:
            clock[0] += 200_000_000
        return sdk.ERROR_CODE.SUCCESS
    sdk.camera.grab = lambda runtime: delay("grab")
    sdk.camera.retrieve_image = lambda mat, view, *args: delay({
        "LEFT": "ego_view_left", "RIGHT": "ego_view", "DEPTH": "depth_view",
    }[view])
    def retrieve_depth(mat, *args):
        mat.image = sdk.depth
        return delay("depth_measure")
    sdk.camera.retrieve_measure = retrieve_depth
    sdk.camera.get_frame_dropped_count = lambda: 42
    sensor = zed_driver.ZEDSensor(config=zed_driver.ZEDConfig(image_dim=(3, 2)))
    sample = sensor.read()
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("[ZED capture delay]"))
    event = json.loads(line.removeprefix("[ZED capture delay] "))
    assert event["stages_ms"][slow_stage] == 200
    assert sum(event["stages_ms"].values()) == 200
    assert event["read_ms"] == 200
    assert event["sdk_dropped_frames"] == 42
    assert sample["sample_monotonic_ns"] == (1_200_000_000 if slow_stage == "grab" else 1_000_000_000)
    assert set(sample["images"]) == {"ego_view", "ego_view_left", "ego_view_depth"}


def test_zed_logs_gap_between_reads_even_when_sdk_calls_are_fast(fake_sdk, monkeypatch, capsys):
    clock = [1_000_000_000]
    monkeypatch.setattr(zed_driver.time, "monotonic_ns", lambda: clock[0])
    sensor = zed_driver.ZEDSensor(config=zed_driver.ZEDConfig(record_depth=False))
    sensor.read()
    assert "[ZED capture delay]" not in capsys.readouterr().out
    clock[0] += 300_000_000
    sensor.read()
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("[ZED capture delay]"))
    event = json.loads(line.removeprefix("[ZED capture delay] "))
    assert event["capture_gap_ms"] == 300
    assert event["read_ms"] == 0
    assert "sdk_dropped_frames" not in event  # Older SDKs may omit the optional counter.


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
        rotate_180=False,
    )
