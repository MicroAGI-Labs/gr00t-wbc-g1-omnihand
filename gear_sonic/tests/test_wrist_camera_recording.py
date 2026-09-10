import pytest

from gear_sonic.camera.composed_camera import (
    THOR_JR_WRIST_DEVICE_IDS,
    ComposedCameraConfig,
    ComposedCameraSensor,
)
from gear_sonic.camera.drivers.usb_camera import USBCameraConfig


def test_thor_profile_pins_sides_and_capture_settings():
    config = ComposedCameraConfig(ego_view_camera="zed", wrist_camera_profile="thor-jr")
    sensor = object.__new__(ComposedCameraSensor)
    sensor.config = config
    cameras = sensor._get_camera_configs()
    assert cameras["left_wrist"]["device_id"] == THOR_JR_WRIST_DEVICE_IDS["left_wrist"]
    assert cameras["right_wrist"]["device_id"] == THOR_JR_WRIST_DEVICE_IDS["right_wrist"]
    assert config.usb_camera_resolution == (1280, 720)
    assert config.usb_camera_fps == config.fps == 60
    assert config.usb_camera_mjpeg is True


@pytest.mark.parametrize(
    "override",
    [
        {"left_wrist_device_id": "0"},
        {"right_wrist_device_id": THOR_JR_WRIST_DEVICE_IDS["left_wrist"]},
        {"left_wrist_camera": "oak"},
    ],
)
def test_thor_profile_rejects_conflicting_assignments(override):
    with pytest.raises(ValueError, match="thor-jr fixes"):
        ComposedCameraConfig(wrist_camera_profile="thor-jr", **override)


def test_default_profile_keeps_wrist_cameras_optional():
    config = ComposedCameraConfig()
    assert config.left_wrist_camera is None
    assert config.right_wrist_camera is None


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"capture_dim": (0, 720)}, {"buffer_size": 1}])
def test_usb_capture_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        USBCameraConfig(**kwargs)
