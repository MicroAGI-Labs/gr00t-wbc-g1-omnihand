"""Stereolabs ZED RGB camera driver.

The ZED SDK and its Python API are system dependencies and are intentionally
loaded lazily. Install the SDK on the camera host, then install ``pyzed`` into
the camera virtual environment with ``/usr/local/zed/get_python_api.py``.

This first integration exposes the rectified left RGB image only. Depth needs
its own lossless wire format and is deliberately kept out of the JPEG pipeline.
"""

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition


def _load_zed_sdk():
    try:
        import pyzed.sl as sl
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "ZED Python API is unavailable. Install the ZED SDK on this host, then run "
            "'.venv_camera/bin/python /usr/local/zed/get_python_api.py'."
        ) from exc
    return sl


@dataclass(frozen=True)
class ZEDConfig:
    """Configuration for a ZED RGB stream."""

    image_dim: tuple[int, int] = (640, 480)
    camera_resolution: str = "HD720"
    camera_fps: int = 60
    rotate_180: bool = False

    def __post_init__(self):
        if len(self.image_dim) != 2 or min(self.image_dim) <= 0:
            raise ValueError(f"image_dim must contain positive width and height, got {self.image_dim}")
        if not self.camera_resolution:
            raise ValueError("camera_resolution must not be empty")
        if self.camera_fps <= 0:
            raise ValueError(f"camera_fps must be positive, got {self.camera_fps}")


class ZEDSensor(Sensor):
    """Rectified left-RGB stream from a Stereolabs ZED camera."""

    def __init__(
        self,
        config: ZEDConfig | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        device_id: str | None = None,
    ):
        self.config = config or ZEDConfig()
        self.mount_position = mount_position
        self.device_id = device_id
        self._sl = _load_zed_sdk()
        self._camera: Any | None = None
        self._image: Any | None = None

        resolution_name = self.config.camera_resolution.upper()
        try:
            camera_resolution = getattr(self._sl.RESOLUTION, resolution_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported ZED camera resolution: {resolution_name}") from exc

        init_params = self._sl.InitParameters()
        init_params.camera_resolution = camera_resolution
        init_params.camera_fps = self.config.camera_fps
        init_params.depth_mode = self._sl.DEPTH_MODE.NONE

        if device_id is not None:
            try:
                serial_number = int(device_id)
            except ValueError as exc:
                raise ValueError(f"ZED device ID must be a numeric serial number, got {device_id!r}") from exc
            init_params.set_from_serial_number(serial_number)

        self._camera = self._sl.Camera()
        open_status = self._camera.open(init_params)
        if open_status != self._sl.ERROR_CODE.SUCCESS:
            self.close()
            raise RuntimeError(f"Failed to open ZED camera for {mount_position}: {open_status}")

        try:
            self._runtime_params = self._sl.RuntimeParameters()
            self._runtime_params.enable_depth = False
            self._image = self._sl.Mat()
            self._output_resolution = self._sl.Resolution(*self.config.image_dim)
            self._print_camera_info()
        except Exception:
            self.close()
            raise

    def _print_camera_info(self) -> None:
        try:
            info = self._camera.get_camera_information()
            camera_config = info.camera_configuration
            resolution = camera_config.resolution
            print(
                f"[{self.mount_position}] ZED opened: model={info.camera_model}, "
                f"serial={info.serial_number}, capture={resolution.width}x{resolution.height}"
                f"@{camera_config.fps}, output={self.config.image_dim[0]}x{self.config.image_dim[1]}"
            )
        except Exception:
            print(
                f"[{self.mount_position}] ZED opened at {self.config.camera_resolution}"
                f"@{self.config.camera_fps}, output={self.config.image_dim[0]}x{self.config.image_dim[1]}"
            )

    def read(self) -> dict[str, Any] | None:
        if self._camera is None or self._image is None:
            return None

        grab_status = self._camera.grab(self._runtime_params)
        if grab_status != self._sl.ERROR_CODE.SUCCESS:
            print(f"[{self.mount_position}] ZED grab failed: {grab_status}")
            return None

        # A successful blocking grab means a new frame is available now.  Use
        # host clocks at that boundary rather than the ZED IMAGE timestamp:
        # the SDK's epoch mapping is established when the camera opens and can
        # retain an old offset if CLOCK_REALTIME is stepped by NTP afterwards.
        capture_time = time.time()
        sample_monotonic_ns = time.monotonic_ns()

        retrieve_status = self._camera.retrieve_image(
            self._image,
            self._sl.VIEW.LEFT,
            self._sl.MEM.CPU,
            self._output_resolution,
        )
        if retrieve_status != self._sl.ERROR_CODE.SUCCESS:
            print(f"[{self.mount_position}] ZED image retrieval failed: {retrieve_status}")
            return None

        image_bgra = np.asarray(self._image.get_data())
        if image_bgra.ndim != 3 or image_bgra.shape[2] < 3 or image_bgra.size == 0:
            print(f"[{self.mount_position}] ZED returned an invalid image shape: {image_bgra.shape}")
            return None

        # VIEW.LEFT is BGRA. Reordering also makes an owned, contiguous RGB copy;
        # the SDK-owned Mat buffer may be overwritten by the next grab().
        image_rgb = np.ascontiguousarray(
            image_bgra[::-1, ::-1, 2::-1] if self.config.rotate_180 else image_bgra[..., 2::-1]
        )

        return {
            "timestamps": {self.mount_position: capture_time},
            "images": {self.mount_position: image_rgb},
            "sample_monotonic_ns": sample_monotonic_ns,
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        from gear_sonic.camera.sensor_server import ImageMessageSchema

        return ImageMessageSchema(
            timestamps=data["timestamps"],
            images=data["images"],
            sample_monotonic_ns=data.get("sample_monotonic_ns"),
        ).serialize()

    def observation_space(self):
        if gym is None:
            return None
        width, height = self.config.image_dim
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(height, width, 3),
                    dtype=np.uint8,
                ),
            }
        )

    def close(self):
        image = self._image
        self._image = None
        if image is not None:
            try:
                image.free()
            except Exception:
                pass

        camera = self._camera
        self._camera = None
        if camera is not None:
            camera.close()
