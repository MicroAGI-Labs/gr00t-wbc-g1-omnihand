"""Stereolabs ZED RGB camera driver.

The ZED SDK and its Python API are system dependencies and are intentionally
loaded lazily. Install the SDK on the camera host, then install ``pyzed`` into
the camera virtual environment with ``/usr/local/zed/get_python_api.py``.

This integration exposes both rectified RGB eyes, the SDK-rendered depth view,
and a float32 depth map.
"""

from dataclasses import dataclass
import json
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
    record_depth: bool = True

    def __post_init__(self):
        if len(self.image_dim) != 2 or min(self.image_dim) <= 0:
            raise ValueError(f"image_dim must contain positive width and height, got {self.image_dim}")
        if not self.camera_resolution:
            raise ValueError("camera_resolution must not be empty")
        if self.camera_fps <= 0:
            raise ValueError(f"camera_fps must be positive, got {self.camera_fps}")


class ZEDSensor(Sensor):
    """Rectified right-RGB stream from a Stereolabs ZED camera."""

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
        self._last_capture_monotonic_ns: int | None = None

        resolution_name = self.config.camera_resolution.upper()
        try:
            camera_resolution = getattr(self._sl.RESOLUTION, resolution_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported ZED camera resolution: {resolution_name}") from exc

        init_params = self._sl.InitParameters()
        init_params.camera_resolution = camera_resolution
        init_params.camera_fps = self.config.camera_fps
        depth_name = "PERFORMANCE" if self.config.record_depth else "NONE"
        init_params.depth_mode = getattr(self._sl.DEPTH_MODE, depth_name)

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
            self._runtime_params.enable_depth = self.config.record_depth
            self._left_image = self._sl.Mat()
            self._right_image = self._sl.Mat()
            self._depth = self._sl.Mat() if self.config.record_depth else None
            self._depth_view = self._sl.Mat() if self.config.record_depth else None
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
        if self._camera is None or self._left_image is None or self._right_image is None:
            return None

        read_started_ns = time.monotonic_ns()
        stage_times_ns = {}
        grab_status = self._camera.grab(self._runtime_params)
        stage_times_ns["grab"] = time.monotonic_ns() - read_started_ns
        if grab_status != self._sl.ERROR_CODE.SUCCESS:
            print(f"[{self.mount_position}] ZED grab failed: {grab_status}")
            return None

        # A successful blocking grab means a new frame is available now.  Use
        # host clocks at that boundary rather than the ZED IMAGE timestamp:
        # the SDK's epoch mapping is established when the camera opens and can
        # retain an old offset if CLOCK_REALTIME is stepped by NTP afterwards.
        capture_time = time.time()
        sample_monotonic_ns = time.monotonic_ns()

        images = {}
        for view, mat, name in ((self._sl.VIEW.LEFT, self._left_image, f"{self.mount_position}_left"),
                                (self._sl.VIEW.RIGHT, self._right_image, self.mount_position)):
            stage_started_ns = time.monotonic_ns()
            retrieve_status = self._camera.retrieve_image(mat, view, self._sl.MEM.CPU, self._output_resolution)
            if retrieve_status != self._sl.ERROR_CODE.SUCCESS:
                print(f"[{self.mount_position}] ZED image retrieval failed: {retrieve_status}")
                return None
            image_bgra = np.asarray(mat.get_data())
            if image_bgra.ndim != 3 or image_bgra.shape[2] < 3 or image_bgra.size == 0:
                print(f"[{self.mount_position}] ZED returned an invalid image shape: {image_bgra.shape}")
                return None
            rgb = image_bgra[..., 2::-1]
            if self.config.rotate_180:
                rgb = rgb[::-1, ::-1]
            images[name] = np.ascontiguousarray(rgb)
            stage_times_ns[name] = time.monotonic_ns() - stage_started_ns

        depths = {}
        if self.config.record_depth and self._depth is not None:
            depth_name = f"{self.mount_position}_depth"
            # Capture the SDK's own filtered/colorized depth view so the
            # recorded video matches the ZED viewer instead of re-encoding
            # noisy float depth ourselves.
            if self._depth_view is not None and hasattr(self._sl.VIEW, "DEPTH"):
                stage_started_ns = time.monotonic_ns()
                view_status = self._camera.retrieve_image(
                    self._depth_view, self._sl.VIEW.DEPTH, self._sl.MEM.CPU, self._output_resolution
                )
                if view_status != self._sl.ERROR_CODE.SUCCESS:
                    print(f"[{self.mount_position}] ZED depth view retrieval failed: {view_status}")
                    return None
                view_bgra = np.asarray(self._depth_view.get_data())
                if view_bgra.ndim != 3 or view_bgra.shape[2] < 3 or view_bgra.size == 0:
                    print(f"[{self.mount_position}] ZED returned an invalid depth view shape: {view_bgra.shape}")
                    return None
                depth_view = np.ascontiguousarray(view_bgra[..., 2::-1])
                if self.config.rotate_180:
                    depth_view = depth_view[::-1, ::-1]
                images[depth_name] = depth_view
                stage_times_ns["depth_view"] = time.monotonic_ns() - stage_started_ns

            stage_started_ns = time.monotonic_ns()
            status = self._camera.retrieve_measure(self._depth, self._sl.MEASURE.DEPTH, self._sl.MEM.CPU,
                                                   self._output_resolution)
            if status != self._sl.ERROR_CODE.SUCCESS:
                print(f"[{self.mount_position}] ZED depth retrieval failed: {status}")
                return None
            depth = np.asarray(self._depth.get_data(), dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.shape != (self.config.image_dim[1], self.config.image_dim[0]):
                print(f"[{self.mount_position}] ZED returned invalid depth shape: {depth.shape}")
                return None
            if self.config.rotate_180:
                depth = depth[::-1, ::-1]
            depths[depth_name] = np.ascontiguousarray(depth)
            stage_times_ns["depth_measure"] = time.monotonic_ns() - stage_started_ns

        self._report_capture_delay(read_started_ns, sample_monotonic_ns, stage_times_ns)
        timestamps = {name: capture_time for name in images}
        timestamps.update({name: capture_time for name in depths})
        return {
            "timestamps": timestamps,
            "images": images,
            "depths": depths,
            "sample_monotonic_ns": sample_monotonic_ns,
        }

    def _report_capture_delay(self, read_started_ns, captured_ns, stage_times_ns) -> None:
        """Log long captures without changing image timestamps or recovery behavior."""
        read_ms = (time.monotonic_ns() - read_started_ns) / 1e6
        capture_gap_ms = (
            None if self._last_capture_monotonic_ns is None
            else (captured_ns - self._last_capture_monotonic_ns) / 1e6
        )
        self._last_capture_monotonic_ns = captured_ns
        if read_ms < 100 and (capture_gap_ms is None or capture_gap_ms < 100):
            return
        event = {
            "camera": self.mount_position,
            "captured_monotonic_ns": captured_ns,
            "read_ms": round(read_ms, 3),
            "capture_gap_ms": None if capture_gap_ms is None else round(capture_gap_ms, 3),
            "stages_ms": {name: round(ns / 1e6, 3) for name, ns in stage_times_ns.items()},
        }
        # SDK's cumulative count includes frames superseded before grab(); it
        # does not by itself diagnose USB loss. Read it only on a slow frame.
        try:
            event["sdk_dropped_frames"] = int(self._camera.get_frame_dropped_count())
        except Exception:
            # Optional diagnostics must not turn a valid frame into a failure.
            pass
        print(f"[ZED capture delay] {json.dumps(event)}", flush=True)

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        from gear_sonic.camera.sensor_server import ImageMessageSchema

        return ImageMessageSchema(
            timestamps=data["timestamps"],
            images=data["images"],
            depths=data.get("depths", {}),
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
        for attr in ("_left_image", "_right_image", "_depth", "_depth_view"):
            image = getattr(self, attr, None)
            setattr(self, attr, None)
            if image is not None:
                try:
                    image.free()
                except Exception:
                    pass

        camera = self._camera
        self._camera = None
        if camera is not None:
            camera.close()
