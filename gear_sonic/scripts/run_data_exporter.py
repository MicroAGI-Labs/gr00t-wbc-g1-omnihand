"""
Sonic VLA data exporter for G1 -- NO ROS 2 DEPENDENCY.

All data sources use ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (port 5557, from C++ zmq_output_handler)
  2. Teleop       -> ZMQ SUB on pose/planner/manager_state (port 5556, from Pico)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor

Robot config (``script_config`` in info.json) is read from the ``robot_config``
ZMQ topic re-published every ~2 s by the C++ process.  If the config is not
received within the timeout the exporter exits with an error.

Virtual environment setup (run from repo root):
    bash install_scripts/install_data_collection.sh
    source .venv_data_collection/bin/activate

Usage (from repo root):
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cup"
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "walk forward" --dataset-name my_session
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro
import zmq

# Shared virtual environments may be installed against another worktree.
# Direct script execution must resolve the recorder's accompanying modules here.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.camera.depth_preview import camera_images_with_depth_preview
from gear_sonic.data.clock_sync import ClockClient, DEFAULT_CLOCK_PORT
from gear_sonic.data.sender_sync import (
    RecordingInputs, Selection, SenderSynchronizer, selection_inputs, synchronization_features,
)
from gear_sonic.data.episode_finalizer import EpisodeFinalizer
from gear_sonic.data.hub_uploader import EpisodeHubUploader
from gear_sonic.data.exporter import Gr00tDataExporter, RecordingMemoryLimitError
from gear_sonic.utils.data_collection.local_recordings import resolve_recording_destination
from gear_sonic.data.features_sonic_vla import (
    assemble_dataset_configuration,
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
    get_zed_stereo_features,
    get_zed_stereo_modality_config,
)
from gear_sonic.end_effectors.profiles import HandProfile, dataset_robot_type, get_hand_profile, raw_hand_name
from gear_sonic.end_effectors.protocol import (
    HAND_CONFIG_TOPIC,
    HAND_STATE_TOPIC,
    decode_config,
    decode_state,
)
from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.data_collection.hub_config import (
    DATASET_CONFIG_PREFIX,
    DEFAULT_TASK_PROMPT,
    decode_dataset_config,
)
from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity, quat_to_rot6d
from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    ZMQStateSubscriber,
    poll_robot_config_zmq,
)

DEPTH_VIDEO_FEATURE = "observation.images.ego_view_depth"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SonicDataExporterConfig:
    """CLI config for the ROS-free Sonic data exporter."""

    # Dataset
    dataset_name: str | None = None
    """Use the saved local destination by default; override to select another dataset."""

    task_prompt: str = DEFAULT_TASK_PROMPT
    """Language task prompt."""

    root_output_dir: str | None = None
    """Override the saved local output directory (outputs when unconfigured)."""

    data_collection_frequency: int = 50
    """Data collection frequency (Hz)."""

    max_episode_duration_s: float = 240.0
    """Discard a recording at this elapsed duration; 0 disables the limit."""


    sender_time_recording: bool = False
    """Opt in to delayed producer-time alignment; use a dataset with the new schema."""

    synchronization_delay: float = 0.1
    synchronization_wait_timeout: float = 0.25
    hand_clock_port: int = DEFAULT_CLOCK_PORT
    """Read-only clock exchange on a remote hand host (sender-time mode only)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Sonic / SMPL pose (from pico_manager_thread_server)
    sonic_zmq_host: str = "localhost"
    """ZMQ host for Sonic SMPL pose messages."""

    sonic_zmq_port: int = 5556
    """ZMQ port for Sonic SMPL pose messages."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # Robot config
    robot_config_timeout: float = 0
    """Seconds to wait for the ZMQ robot_config message at startup (0 = wait forever)."""

    record_wrist_cameras: bool = True
    """Record wrist camera streams (left_wrist, right_wrist). Requires cameras to be available."""

    record_zed_stereo: bool = True
    """Record both ZED eyes and the same depth visualization video as the browser UI."""

    text_to_speech: bool = True
    """Use text-to-speech voice feedback."""

    hand_profile: str = "auto"
    """Hand profile (auto, dex3.v1, omnihand_o10.v1, or dex1.v1)."""

    hand_state_host: str = "localhost"
    """Host publishing external hand_config/hand_state messages."""

    hand_state_port: int = 5570
    """Port publishing external hand_config/hand_state messages."""

    hand_config_timeout: float = 0
    """Seconds to wait for external hand config (0 waits indefinitely)."""

    hand_state_max_age: float = 0.2
    """External hand-state age warning threshold; available measurements are recorded."""

    proprio_state_max_age: float = 0.1
    """Robot-state age warning threshold (strict admission limit in sender-time mode)."""

    camera_max_age: float = 0.1
    """Camera age warning threshold (strict admission limit in sender-time mode)."""

    teleop_max_age: float = 0.2
    """Planner/SMPL age warning threshold (strict admission limit in sender-time mode)."""

    minimum_recording_rate_hz: float = 45.0
    """Legacy rate reference retained for CLI compatibility; diagnostic only."""

    required_stream_mode: int = 5
    """Stream mode required for recording (1=POSE, 5=VR3PT, 6=IK upper)."""

    require_hand_activity: bool = False
    """Opt in to rejecting an episode when neither hand command changes."""

    minimum_hand_motion_rad: float = 0.02
    """Minimum requested hand-joint range required when hand activity is enforced."""

    recording_status_port: int = 5581
    """ZMQ PUB port for browser-visible recorder status."""

    require_hub_upload: bool = False
    """Legacy option; local recording no longer requires a Hugging Face destination."""

    shutdown_upload_timeout: float = 300.0
    """Seconds to keep uploading queued episodes while shutting down."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RECORDING_STREAM_MODE_NAMES = {
    1: "POSE",
    5: "VR3PT",
    6: "IK upper",
}


def _recording_mode_ready(current_stream_mode: int, required_stream_mode: int) -> bool:
    """Return whether the launch-selected teleop mode is active."""
    return int(current_stream_mode) == int(required_stream_mode)


class TimeDeltaException(Exception):
    def __init__(self, failure_count: int, reset_timeout_sec: float):
        self.failure_count = failure_count
        self.reset_timeout_sec = reset_timeout_sec
        self.message = f"{self.failure_count} failures in {self.reset_timeout_sec} seconds"
        super().__init__(self.message)


class StreamRateTracker:
    """Measure producer and collector rates over a short rolling window."""

    def __init__(self, window_seconds: float = 2.0, stale_after_seconds: float = 1.0):
        self.window_seconds = float(window_seconds)
        self.stale_after_seconds = float(stale_after_seconds)
        self._samples: dict[str, dict[str, deque[tuple[float, float]]]] = {}

    def observe(
        self,
        stream: str,
        *,
        source_timestamp: float | None = None,
        source_sequence: int | None = None,
        received_timestamp: float | None = None,
    ) -> None:
        observed_at = time.monotonic()
        samples = self._samples.setdefault(
            stream,
            {"source": deque(), "source_sequence": deque(), "received": deque()},
        )
        if source_timestamp is not None and np.isfinite(source_timestamp):
            self._append(samples["source"], observed_at, float(source_timestamp))
        if source_sequence is not None:
            self._append(samples["source_sequence"], observed_at, float(source_sequence))
        receiver_event = observed_at if received_timestamp is None else received_timestamp
        if np.isfinite(receiver_event):
            self._append(samples["received"], observed_at, float(receiver_event))
        self._prune(samples, observed_at)

    def snapshot(self, streams: tuple[str, ...]) -> dict[str, dict[str, float | bool | None]]:
        now = time.monotonic()
        result = {}
        for stream in streams:
            samples = self._samples.setdefault(
                stream,
                {"source": deque(), "source_sequence": deque(), "received": deque()},
            )
            self._prune(samples, now)
            latest = samples["received"][-1][0] if samples["received"] else None
            active = latest is not None and now - latest <= self.stale_after_seconds
            result[stream] = {
                "sent_hz": self._source_rate(samples) if active else 0.0,
                "received_hz": self._rate(samples["received"]) if active else 0.0,
                "active": active,
                "age_s": round(now - latest, 3) if latest is not None else None,
            }
        return result

    def _append(
        self,
        samples: deque[tuple[float, float]],
        observed_at: float,
        value: float,
    ) -> None:
        if samples and value == samples[-1][1]:
            return
        if samples and value < samples[-1][1]:
            samples.clear()
        samples.append((observed_at, value))

    def _prune(self, samples_by_kind: dict[str, deque], now: float) -> None:
        cutoff = now - self.window_seconds
        for samples in samples_by_kind.values():
            while samples and samples[0][0] < cutoff:
                samples.popleft()

    @staticmethod
    def _rate(samples: deque[tuple[float, float]]) -> float | None:
        if len(samples) < 2:
            return None
        elapsed = samples[-1][1] - samples[0][1]
        if elapsed <= 0:
            return None
        return round((len(samples) - 1) / elapsed, 2)

    @classmethod
    def _source_rate(cls, samples_by_kind: dict[str, deque]) -> float | None:
        samples = samples_by_kind["source_sequence"]
        if len(samples) >= 2:
            elapsed = samples[-1][0] - samples[0][0]
            sequence_delta = samples[-1][1] - samples[0][1]
            if elapsed > 0 and sequence_delta > 0:
                return round(sequence_delta / elapsed, 2)

        # Legacy publishers may provide timestamps without a sequence. Use
        # the lower-quartile interval to recover their base cadence when a
        # conflating receiver samples across occasional skipped frames.
        samples = samples_by_kind["source"]
        if len(samples) < 2:
            return None
        intervals = np.diff([sample[1] for sample in samples])
        positive = intervals[intervals > 0]
        if positive.size == 0:
            return None
        return round(float(1.0 / np.percentile(positive, 25)), 2)


def unpack_pose_message(packed_data: bytes, topic: str = "pose") -> dict:
    """Unpack a single-frame packed message from pico_manager_thread_server.

    Wire format: [topic_prefix][1280-byte JSON header][concatenated binary fields]
    """
    HEADER_SIZE = 1280

    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic '{topic}'")

    offset = len(topic_bytes)
    if len(packed_data) < offset + HEADER_SIZE:
        raise ValueError(f"Packed data too small: {len(packed_data)} < {offset + HEADER_SIZE}")

    header_bytes = packed_data[offset : offset + HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx > 0:
        header_bytes = header_bytes[:null_idx]

    header = json.loads(header_bytes.decode("utf-8"))
    fields = header.get("fields", [])

    result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
    current_offset = offset + HEADER_SIZE
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": bool,
    }

    for field in fields:
        dtype = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        result[field["name"]] = (
            np.frombuffer(packed_data[current_offset : current_offset + n_bytes], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        current_offset += n_bytes

    return result


def _timestamp_seconds(value, *, nanoseconds: bool = False) -> float | None:
    """Return a scalar timestamp in seconds from a Python or NumPy value."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    timestamp = float(value)
    if not np.isfinite(timestamp) or timestamp <= 0:
        return None
    return timestamp / 1e9 if nanoseconds else timestamp


def _integer_scalar(value, default: int = -1) -> int:
    """Read an optional int64 source identity without rounding or overflowing."""
    try:
        array = np.asarray(value)
        if array.size != 1 or array.dtype.kind not in "iu":
            return default
        result = int(array.item())
        return result if 0 <= result <= np.iinfo(np.int64).max else default
    except (TypeError, ValueError, OverflowError):
        return default


def _required_vector(data: dict, key: str, width: int) -> np.ndarray:
    """Return one finite float32 vector or raise a recording-quality error."""
    if key not in data:
        raise ValueError(f"required robot field '{key}' is missing")
    value = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if value.shape != (width,):
        raise ValueError(f"robot field '{key}' has shape {value.shape}, expected {(width,)}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"robot field '{key}' contains NaN or Inf")
    return value


def _episode_hand_motion_range(episode_buffer: dict) -> float:
    """Return the largest commanded hand-joint range in the buffered episode."""
    ranges = []
    for key in ("teleop.left_hand_joints", "teleop.right_hand_joints"):
        values = episode_buffer.get(key, [])
        if values:
            stacked = np.stack(values).astype(np.float32, copy=False)
            ranges.append(float(np.max(np.ptp(stacked, axis=0))))
    return max(ranges, default=0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_reproducibility_metadata(
    robot_config: dict, dataset_frequency_hz: float, hand_profile: HandProfile | None = None,
    *, hand_state_host: str = "localhost",
) -> dict:
    """Resolve immutable controller artifacts and record their identities."""
    repo_root = Path(__file__).resolve().parents[2]
    deploy_root = repo_root / "gear_sonic_deploy"
    artifacts = {}
    for key in ("model_path", "encoder_file", "planner_path", "obs_config_path"):
        configured = robot_config.get(key)
        if not isinstance(configured, str) or configured in {"", "none"}:
            continue
        candidates = (Path(configured), deploy_root / configured)
        resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        record = {"configured_path": configured}
        if resolved is not None:
            record.update(
                {
                    "path": str(resolved),
                    "size_bytes": resolved.stat().st_size,
                    "sha256": _sha256(resolved),
                }
            )
        else:
            record["missing_at_exporter_start"] = True
        artifacts[key] = record

    parameters = (
        deploy_root
        / "src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
    )
    if parameters.is_file():
        artifacts["policy_parameters"] = {
            "path": str(parameters.resolve()),
            "size_bytes": parameters.stat().st_size,
            "sha256": _sha256(parameters),
        }

    git = {"commit": None, "dirty": None}
    try:
        git["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git["dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass

    return {
        "schema": "sonic.capture.v1",
        "robot_type": dataset_robot_type(hand_profile),
        "gr00t_embodiment_tag": "UNITREE_G1_SONIC",
        "joint_units": "rad",
        "angular_velocity_units": "rad_s",
        "position_units": "m",
        "quaternion_order": "wxyz",
        "capture_timestamp_units": "ns",
        "capture_monotonic_clock_scope": (
            "single_linux_host" if hand_state_host in ("localhost", "127.0.0.1", "::1") else "per_host"
        ),
        "hand_state_host": hand_state_host,
        "hand_capture_clock_domains": {
            "hand_state_source_monotonic_ns": "hand_server",
            "hand_state_publish_monotonic_ns": "hand_server",
            "hand_intent_received_monotonic_ns": "hand_server",
            "hand_intent_source_monotonic_ns": "teleop_host",
            "hand_state_received_monotonic_ns": "recorder_host",
        },
        "hand_freshness_clock": "receiver_local_monotonic_and_server_reported_age",
        "dataset_frequency_hz": float(dataset_frequency_hz),
        "git": git,
        "artifacts": artifacts,
    }


class TimingThresholdMonitor:
    def __init__(self, max_failures=3, reset_timeout_sec=5, time_delta=0.2, raise_exception=False):
        self.max_failures = max_failures
        self.reset_timeout_sec = reset_timeout_sec
        self.failure_count = 0
        self.last_failure_time = 0
        self.time_delta = time_delta
        self.raise_exception = raise_exception

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = 0

    def log_time_delta(self, time_delta_sec: float):
        time_delta = abs(time_delta_sec)
        if time_delta > self.time_delta:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

        if self.is_threshold_exceeded():
            print(
                f"Time delta exception: {self.failure_count} failures in "
                f"{self.reset_timeout_sec} seconds, time delta: {time_delta}"
            )
            if self.raise_exception:
                raise TimeDeltaException(self.failure_count, self.reset_timeout_sec)

    def is_threshold_exceeded(self):
        if self.failure_count >= self.max_failures:
            return True
        if time.monotonic() - self.last_failure_time > self.reset_timeout_sec:
            self.reset()
        return False


def poll_hand_config_zmq(host: str, port: int, timeout_s: float) -> dict:
    """Wait for a complete bilateral controller config before fixing the schema."""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, HAND_CONFIG_TOPIC)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(f"tcp://{host}:{port}")
    deadline = None if timeout_s == 0 else time.monotonic() + timeout_s
    try:
        while deadline is None or time.monotonic() < deadline:
            if socket.poll(200):
                config = decode_config(socket.recv())
                if set(config.get("selected_sides", ())) != {"left", "right"}:
                    raise RuntimeError("data collection requires bilateral hand config")
                return config
    finally:
        socket.close(linger=0)
        context.term()
    raise TimeoutError(f"timed out waiting for hand_config on {host}:{port}")


# ---------------------------------------------------------------------------
# Data Collector
# ---------------------------------------------------------------------------


class GrootDataCollector:
    """Collects data from G1 robot in Sonic CPP + SMPL mode -- no ROS 2.

    Data sources (all ZMQ):
      - ``g1_debug`` topic        -> proprio (body_q, hand_q, actions, base_quat, ...)
      - ``pose`` topic            -> SMPL pose (smpl_joints, body_quat_w, hand_joints, ...)
      - ``planner`` topic         -> planner commands (vr_position, vr_orientation, ...)
      - ``manager_state`` topic   -> current stream mode + toggle flags
      - Camera client             -> ego-view images
    """

    RATE_STREAMS = (
        "camera",
        "left_wrist",
        "right_wrist",
        "robot_state",
        "pico_pose",
        "planner",
        "manager_state",
        "hand_intent",
        "hand_control",
        "hand_state",
    )

    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        data_exporter: Gr00tDataExporter,
        robot_model,
        text_to_speech=None,
        frequency: int = 50,
        sonic_data_zmq_host: str = "localhost",
        sonic_data_zmq_port: int = 5556,
        state_zmq_host: str = "localhost",
        state_zmq_port: int = 5557,
        hand_profile: HandProfile | None = None,
        hand_config: dict | None = None,
        hand_state_host: str = "localhost",
        hand_state_port: int = 5570,
        hand_state_max_age: float = .2,
        proprio_state_max_age: float = 0.1,
        camera_max_age: float = 0.1,
        teleop_max_age: float = 0.2,
        minimum_recording_rate_hz: float = 45.0,
        required_stream_mode: int = 5,
        require_hand_activity: bool = False,
        minimum_hand_motion_rad: float = 0.02,
        recording_status_port: int = 5581,
        require_hub_upload: bool = False,
        shutdown_upload_timeout: float = 300.0,
        sender_time_recording: bool = False,
        synchronization_delay: float = 0.1,
        synchronization_wait_timeout: float = 0.25,
        hand_clock_port: int = DEFAULT_CLOCK_PORT,
        max_episode_duration_s: float = 240.0,
    ):
        if not np.isfinite(max_episode_duration_s) or max_episode_duration_s < 0:
            raise ValueError("max_episode_duration_s must be finite and nonnegative")
        self.max_episode_duration_s = float(max_episode_duration_s)
        self._episode_started_at: float | None = None
        self._episode_stopped_at: float | None = None
        self._sender_sync = None
        if sender_time_recording:
            local_names = {"localhost", "127.0.0.1", "::1"}
            if any(host not in local_names for host in (camera_host, state_zmq_host, sonic_data_zmq_host)):
                raise ValueError("sender-time recording currently requires local camera, robot and teleop publishers")
            camera_names = tuple(key.split(".")[-1] for key, feature in data_exporter.features.items()
                                 if feature.get("dtype") in ("image", "video"))
            required_features = synchronization_features(camera_names)
            if not required_features.keys() <= data_exporter.features.keys():
                raise ValueError("sender-time recording requires a new dataset with synchronization features")
            # Validate before starting the remote client or other background resources.
            self._sender_sync = SenderSynchronizer(
                camera_names=camera_names, frequency=frequency,
                delay_s=synchronization_delay, wait_s=synchronization_wait_timeout,
                allow_stale_hand=True,
            )
            if hand_config is not None and hand_state_host not in local_names:
                if not 1 <= hand_clock_port <= 65535:
                    raise ValueError("hand clock port must be within 1..65535")
                self._sender_sync.hand_clock = ClockClient(f"tcp://{hand_state_host}:{hand_clock_port}")
        self.text_to_speech = text_to_speech
        self.frequency = frequency
        self.loop_period = 1.0 / frequency
        self.data_exporter = data_exporter
        self.robot_model = robot_model
        self.hand_profile = hand_profile
        self.hand_config = hand_config
        self.hand_state_max_age = hand_state_max_age
        self.proprio_state_max_age = proprio_state_max_age
        self.camera_max_age = camera_max_age
        self.teleop_max_age = teleop_max_age
        self.minimum_recording_rate_hz = minimum_recording_rate_hz
        if required_stream_mode not in _RECORDING_STREAM_MODE_NAMES:
            raise ValueError(
                "required_stream_mode must be 1 (POSE), 5 (VR3PT), or 6 (IK upper)"
            )
        self.required_stream_mode = required_stream_mode
        self.require_hand_activity = require_hand_activity
        self.minimum_hand_motion_rad = minimum_hand_motion_rad
        self.require_hub_upload = False
        self.shutdown_upload_timeout = shutdown_upload_timeout
        self.latest_hand_state = None
        self._last_complete_hand_state = None
        self.latest_hand_state_received_at = None
        self.latest_hand_state_received_monotonic_ns = None

        self._episode_state = EpisodeState()
        self._keyboard_listener = ZMQKeyboardSubscriber()
        self.hub_uploader = EpisodeHubUploader(data_exporter)
        self.episode_finalizer = EpisodeFinalizer(data_exporter)
        self._recording_message = "Ready to record locally; upload saved recordings when finished"
        self._recording_audio_event = ""
        self._recording_audio_sequence = time.monotonic_ns()
        self._recording_audio_event_expires_at = 0.0
        self._last_announced_finalizer_error: str | None = None
        self._recording_status_ctx = zmq.Context()
        self._recording_status_socket = self._recording_status_ctx.socket(zmq.PUB)
        self._recording_status_socket.setsockopt(zmq.SNDHWM, 2)
        self._recording_status_socket.bind(f"tcp://*:{recording_status_port}")

        self._image_subscriber = ComposedCameraClientSensor(
            server_ip=camera_host,
            port=camera_port,
            background=True,
            preserve_history=sender_time_recording,
        )

        self.obs_act_buffer = deque(maxlen=100)
        self.latest_image_msg = None
        self.latest_image_received_at = None
        self.latest_image_received_monotonic_ns = None
        self.latest_proprio_msg = None
        self.latest_proprio_received_at = None
        self.latest_proprio_received_monotonic_ns = None
        self.latest_sonic_msg = None
        self.latest_planner_msg = None
        self.latest_manager_msg = None
        self._episode_input_errors: set[str] = set()
        self._episode_hand_diagnostics: dict[str, dict] = {}
        self._episode_frame_diagnostics: dict[str, dict] = {}
        self._episode_input_gaps: list[dict] = []
        self._episode_flagged_frames = 0
        self._last_input_block_log = 0.0
        self.stream_rates = StreamRateTracker()
        self._last_wrist_camera_timestamps: dict[str, float] = {}

        self.current_stream_mode = 0

        self._manager_toggle_dc = False
        self._manager_toggle_da = False
        self._manager_discard_reason: str | None = None
        self._manager_toggle_df = False

        self._state_subscriber = ZMQStateSubscriber(
            host=state_zmq_host,
            port=state_zmq_port,
        )

        self._sonic_zmq_ctx = None
        self._sonic_zmq_socket = None
        try:
            self._sonic_zmq_ctx = zmq.Context()
            self._sonic_zmq_socket = self._sonic_zmq_ctx.socket(zmq.SUB)
            self._sonic_zmq_socket.connect(f"tcp://{sonic_data_zmq_host}:{sonic_data_zmq_port}")
            self._sonic_zmq_socket.setsockopt(zmq.RCVTIMEO, 100)
            self._sonic_zmq_socket.setsockopt(zmq.CONFLATE, 0)
            self._sonic_zmq_socket.setsockopt(zmq.RCVHWM, 20)
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "planner")
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "manager_state")
            time.sleep(0.5)
            print(f"[Sonic] Connected to ZMQ at {sonic_data_zmq_host}:{sonic_data_zmq_port}")
            print("[Sonic] Subscribed to: pose, planner, manager_state")
        except Exception as e:
            print(f"[Sonic] Warning: Failed to initialize ZMQ subscriber: {e}")
            self._sonic_zmq_socket = None

        self._hand_zmq_ctx = None
        self._hand_zmq_socket = None
        if hand_config is not None:
            self._hand_zmq_ctx = zmq.Context()
            self._hand_zmq_socket = self._hand_zmq_ctx.socket(zmq.SUB)
            self._hand_zmq_socket.setsockopt(zmq.SUBSCRIBE, HAND_STATE_TOPIC)
            self._hand_zmq_socket.setsockopt(zmq.RCVHWM, 1)
            self._hand_zmq_socket.setsockopt(zmq.CONFLATE, 1)
            self._hand_zmq_socket.connect(f"tcp://{hand_state_host}:{hand_state_port}")

        self.telemetry = Telemetry(window_size=100)
        self.sonic_timing_monitor = TimingThresholdMonitor(
            max_failures=3, reset_timeout_sec=5, time_delta=0.1
        )

        self._last_latency_log_time = 0.0
        self._initial_yaw = None

        print(f"Recording to {self.data_exporter.meta.root}")

    @property
    def current_episode_index(self):
        return self.data_exporter.episode_buffer["episode_index"]

    def _print_and_say(self, message: str, say: bool = True, blocking: bool = False):
        if self.text_to_speech is not None:
            self.text_to_speech.print_and_say(message, say, blocking=blocking)
        else:
            print(message)

    def _set_recording_audio_event(self, event: str) -> None:
        if event not in {"start", "saved", "discard", "validation_failed", "save_failed"}:
            raise ValueError(f"Unsupported recording audio event: {event}")
        self._recording_audio_sequence += 1
        self._recording_audio_event = event
        # Repeat briefly in status packets so a PUB/SUB slow join cannot lose it.
        self._recording_audio_event_expires_at = time.monotonic() + 2.0

    def _publish_recording_status(self) -> None:
        """Publish authoritative recorder state for the loopback browser UI."""
        state = self._episode_state.get_state()
        hub_status = self.hub_uploader.status()
        finalizer_status = self.episode_finalizer.status()
        local_busy = bool(finalizer_status.get("pending") or finalizer_status.get("finalizing"))
        local_saving = bool(finalizer_status.get("pending_saves", local_busy))
        local_discarding = bool(finalizer_status.get("pending_discards"))
        upload_busy = bool(hub_status.get("pending") or hub_status.get("uploading"))
        finalizer_error = finalizer_status.get("error")
        if finalizer_error and finalizer_error != self._last_announced_finalizer_error:
            self._last_announced_finalizer_error = str(finalizer_error)
            self._set_recording_audio_event("save_failed")
            self._recording_message = (
                "Background save failed; check the recorder error"
            )
        elif not finalizer_error:
            self._last_announced_finalizer_error = None
        hub_status["required"] = False
        hub_status["upload_mode"] = "manual"
        hand_ready = True
        if self.hand_config is not None:
            try:
                self._external_hand_values()
            except (RuntimeError, ValueError, TypeError):
                hand_ready = False
        message = self._recording_message
        if state == self._episode_state.IDLE:
            if local_saving:
                message = f"{message} · finishing previous take"
            elif local_discarding:
                message = f"{message} · removing discarded take"
            elif hub_status["uploading"]:
                message = f"{message} · uploading"
            elif hub_status["retrying"]:
                message = f"{message} · upload retrying automatically"
        payload = {
            "state": state,
            "recording": state == self._episode_state.RECORDING,
            "saving": state == self._episode_state.NEED_TO_SAVE or local_saving,
            "discarding": local_discarding,
            "ready_to_record": (state != self._episode_state.NEED_TO_SAVE
                                and self.episode_finalizer.can_record() and not upload_busy),
            "video_recording_mode": "continuous_h264",
            "buffered_video_bytes": getattr(self.data_exporter, "buffered_video_bytes", 0),
            "max_video_buffer_bytes": getattr(self.data_exporter, "max_video_buffer_bytes", 0),
            "episode_index": self.current_episode_index,
            "recording_audio_event": (
                self._recording_audio_event
                if time.monotonic() <= self._recording_audio_event_expires_at
                else ""
            ),
            "recording_audio_sequence": self._recording_audio_sequence,
            "frame_count": self.data_exporter.episode_buffer.get("size", 0),
            "total_episodes": self.data_exporter.meta.info.get("total_episodes", 0),
            "dataset_root": str(self.data_exporter.meta.root),
            "sources": {
                "proprio": self.latest_proprio_msg is not None,
                "camera": self.latest_image_msg is not None,
                "hands": hand_ready,
            },
            "stream_mode": self.current_stream_mode,
            "required_stream_mode": self.required_stream_mode,
            "required_stream_mode_name": _RECORDING_STREAM_MODE_NAMES[
                self.required_stream_mode
            ],
            "recording_mode_ready": _recording_mode_ready(
                self.current_stream_mode, self.required_stream_mode
            ),
            "message": message,
            "finalizer": finalizer_status,
            "hub": hub_status,
            "stream_rates": self.stream_rates.snapshot(self.RATE_STREAMS),
            "synchronization": self._sender_sync.status() if getattr(self, "_sender_sync", None) else None,
            "hand_freshness_enforced": False,
            "hand_disconnect_snapshot_retention": True,
            "max_episode_duration_s": self.max_episode_duration_s,
            "freshness_policy": "flag_frames",
            "failure_policy": "save_episode",
            "discard_policy": "delete_episode",
            "duration_limit_policy": "discard",
            "flagged_frame_count": getattr(self, "_episode_flagged_frames", 0),
            "episode_elapsed_s": self._episode_elapsed_s(),
            "timestamp": time.time(),
        }
        try:
            self._recording_status_socket.send_json(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _observe_wrist_camera_rates(self, message: dict) -> None:
        """Count fresh captures sampled by the collector, independently per wrist."""
        for name in ("left_wrist", "right_wrist"):
            if name not in message.get("images", {}):
                continue
            timestamp = _timestamp_seconds(message.get("timestamps", {}).get(name))
            if timestamp is None or timestamp == self._last_wrist_camera_timestamps.get(name):
                continue
            self._last_wrist_camera_timestamps[name] = timestamp
            # Source timestamps estimate capture cadence; observation times
            # measure delivery of distinct frames into this collector loop.
            # A shared publisher sequence belongs to the combined message and
            # cannot measure an individual camera's frequency.
            self.stream_rates.observe(name, source_timestamp=timestamp)

    def _poll_state_zmq(self):
        """Poll the ``g1_debug`` ZMQ topic for robot state (non-blocking)."""
        msg = self._state_subscriber.get_msg(clear=True)
        if msg is None:
            return

        ros_timestamp = _timestamp_seconds(msg.get("ros_timestamp"))
        if ros_timestamp is None:
            msg["ros_timestamp"] = time.time()

        self.latest_proprio_msg = msg
        self.latest_proprio_received_at = time.monotonic()
        self.latest_proprio_received_monotonic_ns = time.monotonic_ns()
        if getattr(self, "_sender_sync", None) is not None:
            self._sender_sync.observe("proprio", msg, _integer_scalar(msg.get("sample_monotonic_ns")),
                                      self.latest_proprio_received_monotonic_ns)
        self.stream_rates.observe(
            "robot_state",
            source_timestamp=_timestamp_seconds(
                msg.get("publisher_monotonic_ns"), nanoseconds=True
            ),
            source_sequence=(
                int(msg["index"])
                if isinstance(msg.get("index"), (int, np.integer))
                and not isinstance(msg.get("index"), bool)
                else None
            ),
        )

    def _recordable_hand_state(self, state: dict) -> dict:
        """Retain actual measurements across status-only disconnect messages.

        The recorder never sends these held values to the hand controller. The
        original measurement sequence/time and current disconnected flags make
        held rows distinguishable from new feedback in the dataset.
        """
        fields = ("requested_position_rad", "applied_position_rad", "measured_position_rad")
        sides = state.get("sides", {})
        complete = True
        for side in ("left", "right"):
            for field in fields:
                try:
                    values = np.asarray(sides.get(side, {}).get(field), dtype=np.float64).reshape(-1)
                    complete = complete and values.shape == (self.hand_profile.width,) and np.all(np.isfinite(values))
                except (TypeError, ValueError):
                    complete = False
        if complete:
            self._last_complete_hand_state = deepcopy(state)
            return state

        previous = getattr(self, "_last_complete_hand_state", None)
        # Only the controller's status-only reconnect/fault reports can hold a
        # prior snapshot. Malformed measurement reports still require attention.
        if state.get("mode") not in {"disconnected", "fault"} or previous is None:
            return state
        if any(not state.get(key) or state.get(key) != previous.get(key)
               for key in ("session_id", "profile", "clock_id")):
            return state
        if any(not isinstance(sides.get(side), dict)
               or sides[side].get("valid") is not False
               or sides[side].get("connected") is not False
               or any(field in sides[side] for field in fields)
               for side in ("left", "right")):
            return state
        source_ns = _integer_scalar(previous.get("monotonic_ns"))
        published_ns = _integer_scalar(state.get("published_monotonic_ns"))
        if source_ns <= 0 or published_ns < source_ns:
            return state

        held = deepcopy(state)
        held["monotonic_ns"] = source_ns
        held["sequence"] = previous.get("sequence")
        held["state_age_s"] = (published_ns - source_ns) / 1e9
        held["recording_retained_snapshot"] = True
        for side in ("left", "right"):
            for field in fields:
                held["sides"][side][field] = deepcopy(previous["sides"][side][field])
        return held

    def _poll_hand_zmq(self) -> None:
        if self._hand_zmq_socket is None:
            return
        while self._hand_zmq_socket.poll(0):
            try:
                state = decode_state(self._hand_zmq_socket.recv())
            except Exception as exc:
                print(f"[Hands] rejected state: {exc}")
                continue
            if state.get("session_id") != self.hand_config.get("session_id"):
                print("[Hands] rejected state from a different controller session")
                continue
            if state.get("profile") != self.hand_profile.name:
                print("[Hands] rejected state with a different hand profile")
                continue
            state = self._recordable_hand_state(state)
            self.latest_hand_state = state
            self.latest_hand_state_received_at = time.monotonic()
            self.latest_hand_state_received_monotonic_ns = time.monotonic_ns()
            if getattr(self, "_sender_sync", None) is not None:
                self._sender_sync.observe("hand", state, _integer_scalar(state.get("monotonic_ns")),
                                          self.latest_hand_state_received_monotonic_ns)
            self.stream_rates.observe(
                "hand_state",
                source_timestamp=_timestamp_seconds(
                    state.get("published_monotonic_ns"), nanoseconds=True
                ),
                source_sequence=(
                    int(state["publish_sequence"])
                    if isinstance(state.get("publish_sequence"), int)
                    and not isinstance(state.get("publish_sequence"), bool)
                    else None
                ),
            )
            self.stream_rates.observe(
                "hand_control",
                source_timestamp=_timestamp_seconds(
                    state.get("monotonic_ns"), nanoseconds=True
                ),
                source_sequence=(
                    int(state["sequence"])
                    if isinstance(state.get("sequence"), int)
                    and not isinstance(state.get("sequence"), bool)
                    else None
                ),
            )
            intent_source = _timestamp_seconds(
                state.get("intent_source_monotonic_ns"), nanoseconds=True
            )
            intent_received = _timestamp_seconds(
                state.get("intent_received_monotonic_ns"), nanoseconds=True
            )
            if intent_source is not None and intent_received is not None:
                self.stream_rates.observe(
                    "hand_intent",
                    source_timestamp=intent_source,
                    received_timestamp=intent_received,
                )

    def _latest_recording_inputs(self) -> RecordingInputs:
        """Snapshot the legacy inputs; synchronized rows pass an explicit selection."""
        def received(ns_name: str, seconds_name: str) -> int:
            ns = getattr(self, ns_name, None)
            seconds = getattr(self, seconds_name, None)
            return int(ns) if ns is not None else int(seconds * 1e9) if seconds is not None else -1

        return RecordingInputs(
            getattr(self, "latest_proprio_msg", None), getattr(self, "latest_image_msg", None),
            getattr(self, "latest_hand_state", None), getattr(self, "latest_sonic_msg", None),
            getattr(self, "latest_planner_msg", None), getattr(self, "latest_manager_msg", None),
            getattr(self, "current_stream_mode", 0),
            received("latest_proprio_received_monotonic_ns", "latest_proprio_received_at"),
            received("latest_image_received_monotonic_ns", "latest_image_received_at"),
            received("latest_hand_state_received_monotonic_ns", "latest_hand_state_received_at"),
        )

    def _poll_sender_images(self) -> dict | None:
        messages = self._image_subscriber.read_pending()
        combined = getattr(self, "_sender_latest_images", None)
        for message in messages:
            received_ns = message["receiver_monotonic_ns"]
            for name in message["timestamps"]:
                if name in self._sender_sync.camera_names:
                    self._sender_sync.observe(
                        f"camera.{name}", message,
                        _integer_scalar(message.get("capture_monotonic_ns", {}).get(name)), received_ns,
                    )
            if combined is None:
                combined = {"images": {}, "depths": {}, "timestamps": {}, "camera_received_monotonic_ns": {}}
            combined = {**combined, **{k: v for k, v in message.items()
                                     if k not in ("images", "depths", "timestamps")}}
            for key in ("images", "depths", "timestamps"):
                combined[key] = {**combined[key], **message.get(key, {})}
            for name in message["timestamps"]:
                combined["camera_received_monotonic_ns"][name] = received_ns
        self._image_subscriber.idx += len(messages)
        self._sender_latest_images = combined
        return combined

    def _add_sender_frame(self) -> bool:
        sync = self._sender_sync
        now_ns = time.monotonic_ns()
        if sync.next_target_ns is None:
            sync.trim(now_ns)
            return False
        target = sync.next_target_ns
        if sync.stop_target_ns is not None and target > sync.stop_target_ns:
            self._finish_recording(save=True, discard_reason="operator_discarded")
            return True
        if now_ns < target + sync.delay_ns:
            return False
        selection = sync.select(target, hand=self.hand_config is not None, allow_stale_hand=True, max_ages={
            "proprio": self.proprio_state_max_age, "camera": self.camera_max_age,
            "hand": self.hand_state_max_age, "manager": self.teleop_max_age,
            "sonic": self.teleop_max_age, "planner": self.teleop_max_age,
        })
        if selection.ready:
            inputs = selection_inputs(selection)
            try:
                frame_warnings = self._validate_recording_inputs(inputs)
            except (RuntimeError, ValueError) as exc:
                selection = Selection(target, selection.samples, (str(exc),))
            else:
                # The builder and all feature helpers use only this selection.
                # Writer exceptions propagate to cleanup; they are not input gaps.
                result = self._add_data_frame_sonic(time.monotonic(), inputs, frame_warnings=frame_warnings)
                if sync.next_target_ns is None:
                    return result  # Encoder backpressure finalized and reset the episode.
                sync.advance(target)
                self._recording_message = (
                    "Draining synchronized recording" if sync.stop_target_ns is not None
                    else f"Recording episode {self.current_episode_index}"
                )
                return result
        if now_ns >= target + sync.delay_ns + sync.wait_ns:
            sync.skip(selection, now_ns)
            self._recording_message = f"Recording gap: {'; '.join(selection.problems)}"
        else:
            self._recording_message = f"Waiting for synchronization: {'; '.join(selection.problems)}"
        return False

    def _external_hand_values(
        self, inputs: RecordingInputs | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        inputs = inputs or self._latest_recording_inputs()
        if inputs.hand is None or inputs.hand_received_ns <= 0:
            raise RuntimeError("external hand state is unavailable")
        # Recording captures what the controller actually reported. Its source
        # timestamps distinguish a held snapshot from a new measurement. Age,
        # hold/intent, and health are provenance, not an operator-outcome veto.
        # Never invent measurements when a disconnected report omits them.
        values = []
        for field in ("requested_position_rad", "applied_position_rad", "measured_position_rad"):
            for side in ("left", "right"):
                side_state = inputs.hand.get("sides", {}).get(side)
                if not side_state:
                    raise RuntimeError(f"external {side} hand measurements are unavailable")
                array = np.asarray(side_state.get(field), dtype=np.float64).reshape(-1)
                if array.shape != (self.hand_profile.width,) or not np.all(np.isfinite(array)):
                    raise RuntimeError(f"external {side} {field} has the wrong shape")
                values.append(array)
        return tuple(values)

    def _external_hand_diagnostics(self, inputs: RecordingInputs) -> dict[str, float | None]:
        """Describe retained hand samples without treating them as fresh commands."""
        hand = inputs.hand
        assert hand is not None
        warnings: dict[str, float | None] = {}
        age = inputs.age(hand, inputs.hand_received_ns, time.monotonic())
        if age > self.hand_state_max_age:
            warnings["external hand state is stale"] = age
        controller_age = hand.get("state_age_s")
        if hand.get("recording_retained_snapshot"):
            warnings["external hand snapshot retained during disconnect"] = controller_age
        if isinstance(controller_age, (int, float)) and controller_age > self.hand_state_max_age:
            warnings["external hand controller snapshot is stale"] = controller_age
        if hand.get("mode") in {"fault", "disconnected"}:
            warnings[f"external hand controller is {hand['mode']}"] = None
        if hand.get("input_stale") or hand.get("intent_sequence") is None:
            warnings["external hand target is missing or stale"] = None
        for side in ("left", "right"):
            state = hand["sides"][side]
            if not state.get("valid") or not state.get("connected"):
                warnings[f"external {side} hand is invalid or disconnected"] = None
            if state.get("input_stale"):
                warnings[f"external {side} hand target is stale"] = None
            if state.get("intent_closed") is None:
                warnings[f"external {side} hand has no valid click intent"] = None
        return warnings

    def _record_hand_diagnostics(self, frame_index: int, warnings: dict[str, float | None]) -> None:
        """Keep exact affected row ranges, including unknown auxiliary click labels."""
        if not warnings:
            return
        if not hasattr(self, "_episode_hand_diagnostics"):
            self._episode_hand_diagnostics = {}
        self._record_frame_ranges(self._episode_hand_diagnostics, frame_index, warnings)

    @staticmethod
    def _record_frame_ranges(diagnostics: dict, frame_index: int, warnings: dict[str, float | None]) -> None:
        """Frame ranges are zero-based, inclusive, and refer to saved rows/video frames."""
        for message, age in warnings.items():
            entry = diagnostics.setdefault(message, {"frames": 0, "frame_ranges": []})
            if not entry["frames"]:
                print(f"[Quality] flagging frames: {message}; retaining measurements")
            entry["frames"] += 1
            ranges = entry["frame_ranges"]
            if ranges and ranges[-1][1] == frame_index - 1:
                ranges[-1][1] = frame_index
            else:
                ranges.append([frame_index, frame_index])
            if age is not None:
                entry["max_age_s"] = max(entry.get("max_age_s", 0.0), float(age))

    def _record_input_gap(self, error: str) -> None:
        """Locate omitted malformed/missing samples without inventing replacement rows."""
        if not hasattr(self, "_episode_input_gaps"):
            self._episode_input_gaps = []
        next_index = self.data_exporter.episode_buffer.get("size", 0)
        elapsed = round(self._episode_elapsed_s(), 6)
        gaps = self._episode_input_gaps
        if gaps and gaps[-1]["next_frame_index"] == next_index and gaps[-1]["reason"] == error:
            gaps[-1]["attempts"] += 1
            gaps[-1]["last_elapsed_s"] = elapsed
        else:
            gaps.append({"reason": error, "next_frame_index": next_index, "attempts": 1,
                         "first_elapsed_s": elapsed, "last_elapsed_s": elapsed})

    def _validate_recording_inputs(self, inputs: RecordingInputs | None = None) -> dict[str, float | None]:
        """Reject missing/malformed inputs; retain stale samples with crop diagnostics."""
        inputs = inputs or self._latest_recording_inputs()
        warnings: dict[str, float | None] = {}
        if (
            not _recording_mode_ready(inputs.mode, self.required_stream_mode)
            and inputs.mode != 3
        ):
            required_name = _RECORDING_STREAM_MODE_NAMES[self.required_stream_mode]
            raise RuntimeError(
                f"required teleop mode {required_name} is not active "
                f"(stream mode {inputs.mode})"
            )
        now = time.monotonic()
        if inputs.proprio is None or inputs.proprio_received_ns <= 0:
            raise RuntimeError("robot state is unavailable")
        proprio_age = inputs.age(inputs.proprio, inputs.proprio_received_ns, now)
        if proprio_age > self.proprio_state_max_age:
            warnings["robot state is stale"] = proprio_age
        for key, width in (
            ("body_q", 29),
            ("body_dq", 29),
            ("base_quat", 4),
            ("base_ang_vel", 3),
            ("last_action", 29),
        ):
            _required_vector(inputs.proprio, key, width)
        token = _required_vector(inputs.proprio, "token_state", 64)
        if not np.any(token):
            raise RuntimeError("SONIC motion token is all zeros")

        if inputs.image is None or inputs.image_received_ns <= 0:
            raise RuntimeError("camera frame is unavailable")
        camera_age = inputs.age(inputs.image, inputs.image_received_ns, now)
        if camera_age > self.camera_max_age:
            warnings["camera frame is stale"] = camera_age

        warnings.update(self._validate_camera_inputs(now, inputs.image, inputs.target_ns))

        if self.hand_config is not None:
            self._external_hand_values(inputs)

        if inputs.mode in (5, 6):
            if inputs.planner is None:
                raise RuntimeError("planner command is unavailable in planner mode")
            planner_received = inputs.planner.get("receive_monotonic")
            planner_age = None if planner_received is None else inputs.age(inputs.planner, int(planner_received * 1e9), now)
            if planner_age is None or planner_age > self.teleop_max_age:
                warnings["planner command is stale in planner mode"] = planner_age
            if inputs.planner.get("vr_3pt_position") is None:
                raise RuntimeError("VR 3-point position is missing in planner mode")
            if inputs.planner.get("vr_3pt_orientation") is None:
                raise RuntimeError("VR 3-point orientation is missing in planner mode")
        elif inputs.mode in (1, 4):
            if inputs.sonic is None:
                raise RuntimeError("SMPL pose is unavailable in pose mode")
            pose_received = inputs.sonic.get("receive_monotonic")
            pose_age = None if pose_received is None else inputs.age(inputs.sonic, int(pose_received * 1e9), now)
            if pose_age is None or pose_age > self.teleop_max_age:
                warnings["SMPL pose is stale in pose mode"] = pose_age
            if inputs.sonic.get("smpl_pose") is None:
                raise RuntimeError("SMPL pose is missing in pose mode")
        return warnings

    def _episode_validation(self) -> dict[str, object]:
        """Return saved quality metadata; freshness flags never veto an accepted take."""
        errors = sorted(self._episode_input_errors)
        if getattr(self, "_sender_sync", None) is not None:
            errors.extend(self._sender_sync.errors)
        hand_motion = _episode_hand_motion_range(self.data_exporter.episode_buffer)
        if self.hand_config is not None and self.require_hand_activity:
            if hand_motion < self.minimum_hand_motion_rad:
                errors.append(
                    "hand commands did not move enough "
                    f"({hand_motion:.4f} rad < {self.minimum_hand_motion_rad:.4f} rad)"
                )

        # Continuous rolling rates are diagnostic, not episode pass/fail criteria.
        rates = self.stream_rates.snapshot(self.RATE_STREAMS)
        modes = {
            int(np.asarray(value).reshape(-1)[0])
            for value in self.data_exporter.episode_buffer.get("teleop.stream_mode", [])
        }
        # Base pose (mode 3) is an intentional pause point during a recording:
        # legacy profiles may leave teleop, adjust/reset, and re-enter without
        # closing the episode. It is valid alongside the required
        # teleop mode, but an episode must still contain teleop frames.
        allowed_modes = {self.required_stream_mode, 3}
        unexpected_modes = sorted(mode for mode in modes if mode not in allowed_modes)
        if unexpected_modes:
            required_name = _RECORDING_STREAM_MODE_NAMES[self.required_stream_mode]
            errors.append(
                f"episode contains stream modes {unexpected_modes}; "
                f"required {required_name} ({self.required_stream_mode}) plus base pauses"
            )
        if modes and self.required_stream_mode not in modes:
            required_name = _RECORDING_STREAM_MODE_NAMES[self.required_stream_mode]
            errors.append(f"episode contains no {required_name} teleop frames")
        errors = sorted(set(errors))
        return {
            "passed": not errors,
            "errors": errors,
            "warnings": sorted(set(getattr(self, "_episode_hand_diagnostics", {}))
                               | set(getattr(self, "_episode_frame_diagnostics", {}))),
            "hand_diagnostics": deepcopy(getattr(self, "_episode_hand_diagnostics", {})),
            "frame_diagnostics": deepcopy(getattr(self, "_episode_frame_diagnostics", {})),
            "flagged_frame_count": getattr(self, "_episode_flagged_frames", 0),
            "frame_range_convention": "zero_based_inclusive",
            "input_gaps": deepcopy(getattr(self, "_episode_input_gaps", [])),
            "freshness_policy": "flag_frames",
            "freshness_thresholds_s": {
                "robot_state": getattr(self, "proprio_state_max_age", None),
                "camera": getattr(self, "camera_max_age", None),
                "teleop": getattr(self, "teleop_max_age", None),
                "hand": getattr(self, "hand_state_max_age", None),
            },
            "hand_freshness_enforced": False,
            "hand_disconnect_snapshot_retention": True,
            "episode_duration_s": round(self._episode_elapsed_s(), 3),
            "max_episode_duration_s": getattr(self, "max_episode_duration_s", 240.0),
            "hand_command_range_rad": round(hand_motion, 6),
            "rate_check_enforced": False,
            "synchronization": self._sender_sync.status() if getattr(self, "_sender_sync", None) else None,
            "stream_rates": rates,
        }

    def _episode_elapsed_s(self) -> float:
        started = getattr(self, "_episode_started_at", None)
        if started is None:
            return 0.0
        stopped = getattr(self, "_episode_stopped_at", None)
        return max(0.0, (time.monotonic() if stopped is None else stopped) - started)

    def _episode_duration_limit_reached(self) -> bool:
        limit = getattr(self, "max_episode_duration_s", 240.0)
        return (self._episode_state.get_state() == self._episode_state.RECORDING
                and limit > 0 and self._episode_elapsed_s() >= limit)

    def _finish_recording(self, *, save: bool, discard_reason: str) -> None:
        """Save successes/failures; explicit discard and duration-limit stops delete."""
        episode_index = self.current_episode_index
        if save and self._episode_duration_limit_reached():
            save, discard_reason = False, "episode_duration_limit"
        duration_s = self._episode_elapsed_s()
        buffer_size = self.data_exporter.episode_buffer.get("size", 0)
        if buffer_size <= 0:
            self._set_recording_audio_event("validation_failed")
            self._recording_message = "Nothing saved: no frames collected"
            if getattr(self, "_sender_sync", None) is not None:
                self._sender_sync.reset()
            self._episode_state.reset_state()
            self._episode_started_at = self._episode_stopped_at = None
            self._episode_input_errors = set()
            self._episode_hand_diagnostics = {}
            self._episode_frame_diagnostics = {}
            self._episode_input_gaps = []
            self._episode_flagged_frames = 0
            self._initial_yaw = None
            self._print_and_say("Skipping empty recording", say=False)
            return

        delete = not save and discard_reason in {"operator_discarded", "episode_duration_limit"}
        # Keep frame warnings and input gaps on saved failures too, for later cropping.
        validation = deepcopy(self._episode_validation())
        if not save:
            validation.update(
                passed=False,
                errors=sorted(set(validation.get("errors", [])) | {discard_reason}),
                failure_reason=discard_reason,
                episode_duration_s=round(duration_s, 3),
                max_episode_duration_s=getattr(self, "max_episode_duration_s", 240.0),
            )
        success = bool(validation["passed"])

        episode_buffer, video_writers = self.data_exporter.detach_episode(advance_index=not delete)
        try:
            self.episode_finalizer.enqueue(
                episode_index=episode_index,
                episode_buffer=episode_buffer,
                video_writers=video_writers,
                success=success,
                validation=validation,
                delete=delete,
            )
        except Exception:
            # The finalizer did not take ownership. Preserve the completed take
            # and its writers so the operator can retry saving it.
            self.data_exporter.episode_buffer = episode_buffer
            self.data_exporter.video_writers = video_writers
            raise

        if getattr(self, "_sender_sync", None) is not None:
            self._sender_sync.reset()
        self.sonic_timing_monitor.reset()
        self._episode_input_errors.clear()
        self._episode_hand_diagnostics = {}
        self._episode_frame_diagnostics = {}
        self._episode_input_gaps = []
        self._episode_flagged_frames = 0
        self._initial_yaw = None
        self._episode_state.reset_state()
        self._episode_started_at = self._episode_stopped_at = None
        if delete:
            self._set_recording_audio_event("discard")
            reason = (f" at the {self.max_episode_duration_s:g}s recording limit"
                      if discard_reason == "episode_duration_limit" else "")
            self._recording_message = f"Take discarded{reason}; removing its temporary videos"
            self._print_and_say("Recording discarded", say=False)
        elif success:
            self._set_recording_audio_event("saved")
            self._recording_message = (
                f"Episode {episode_index} accepted"
            )
            if validation.get("flagged_frame_count"):
                self._recording_message += f"; {validation['flagged_frame_count']} frames flagged for review"
            self._print_and_say("Recording accepted; finishing local save", say=False)
        else:
            if save:
                self._set_recording_audio_event("validation_failed")
                reasons = "; ".join(validation["errors"])
                self._recording_message = (
                    f"Episode {episode_index} failed validation: {reasons}. "
                    "Preserved locally as unsuccessful"
                )
                self._print_and_say("Recording failed validation", say=False)
            else:
                self._set_recording_audio_event("validation_failed")
                if discard_reason == "episode_duration_limit":
                    self._recording_message = (
                        f"Episode {episode_index} reached the {self.max_episode_duration_s:g}s recording limit; "
                        "saving as failed"
                    )
                elif discard_reason == "recording_memory_limit":
                    self._recording_message = (
                        f"Episode {episode_index} stopped because an encoder queue filled; "
                        "preserving captured frames as unsuccessful"
                    )
                elif discard_reason == "operator_marked_failure":
                    self._recording_message = (
                        f"Episode {episode_index} marked as failed; finishing local save"
                    )
                else:
                    self._recording_message = (
                        f"Episode {episode_index} stopped: {discard_reason}; preserved as unsuccessful"
                    )
                self._print_and_say("Recording marked as failed; finishing local save", say=False)

    def _request_dataset_upload(self) -> None:
        """Upload a committed snapshot only in response to an explicit command."""
        if self._episode_state.get_state() != self._episode_state.IDLE:
            raise RuntimeError("stop recording and wait for the local save before uploading")
        finalizer = self.episode_finalizer.status()
        if finalizer.get("pending") or finalizer.get("finalizing") or finalizer.get("error"):
            raise RuntimeError("finish local saving before uploading")
        if self.data_exporter.episode_buffer.get("size", 0):
            raise RuntimeError("save the buffered recording before uploading")
        last_episode = int(self.data_exporter.meta.info.get("total_episodes", 0)) - 1
        if last_episode < 0:
            raise RuntimeError("no locally saved recordings to upload")
        hub = self.hub_uploader.status()
        if not hub.get("ready"):
            raise RuntimeError("select an upload destination in the browser first")
        if hub.get("pending") or hub.get("uploading"):
            self._recording_message = "Upload already in progress"
            return
        self.hub_uploader.enqueue(last_episode)
        self._recording_message = f"Uploading all locally saved recordings through episode {last_episode}"

    def _check_recording_commands(self):
        """Check keyboard + ZMQ toggle flags for recording commands."""
        key = self._keyboard_listener.read_msg()
        discard_reason = "operator_discarded"

        if key == "upload":
            try:
                self._request_dataset_upload()
            except Exception as exc:
                self._recording_message = f"Upload not started: {exc}"
                print(self._recording_message)
            return

        if isinstance(key, str) and key.startswith(DATASET_CONFIG_PREFIX):
            try:
                config = decode_dataset_config(key)
                if self._episode_state.get_state() != self._episode_state.IDLE:
                    raise RuntimeError("save the active episode as successful or failed before changing dataset")
                if self.data_exporter.episode_buffer.get("size", 0) > 0:
                    raise RuntimeError("cannot change dataset after collecting frames")
                finalizer_status = self.episode_finalizer.status()
                if finalizer_status["pending"] or finalizer_status["finalizing"]:
                    raise RuntimeError("cannot change dataset while an episode is finalizing")
                if (self.data_exporter.meta.info.get("total_episodes", 0) > 0
                        and config["prompt"] != self.data_exporter.task):
                    raise RuntimeError("cannot change the task prompt after saving an episode")
                self.hub_uploader.configure(
                    repo_id=str(config["repo_id"]),
                    prompt=str(config["prompt"]),
                    private=bool(config["private"]),
                )
                self._recording_message = f"Upload destination selected: {config['repo_id']}; recordings stay local"
                print(f"[Hub] Dataset configured: {config['repo_id']}")
            except Exception as exc:
                self._recording_message = f"Dataset configuration rejected: {exc}"
                print(f"[Hub] {self._recording_message}")
            return

        if self._manager_toggle_da:
            key = "x"
            self._manager_toggle_da = False
            discard_reason = self._manager_discard_reason or discard_reason
            self._manager_discard_reason = None
        elif getattr(self, "_manager_toggle_df", False):
            key = "f"
        elif self._manager_toggle_dc:
            key = "c"
            self._manager_toggle_dc = False

        # A completed gesture has one outcome; never replay a conflicting latched flag.
        self._manager_toggle_da = self._manager_toggle_dc = self._manager_toggle_df = False
        self._manager_discard_reason = None

        if key != "x" and self._episode_duration_limit_reached():
            self._finish_recording(save=False, discard_reason="episode_duration_limit")
            return

        if key == "c":
            if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                return
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                if getattr(self, "_sender_sync", None) is not None:
                    stopped_ns = time.monotonic_ns()
                    self._episode_stopped_at = stopped_ns / 1e9
                    self._sender_sync.stop(stopped_ns)
                    self._episode_state.change_state()
                    self._recording_message = "Draining synchronized recording"
                    return
                self._finish_recording(save=True, discard_reason=discard_reason)
                return
            if (
                self._episode_state.get_state() == self._episode_state.IDLE
                and not _recording_mode_ready(
                    self.current_stream_mode, self.required_stream_mode
                )
            ):
                required_name = _RECORDING_STREAM_MODE_NAMES[self.required_stream_mode]
                message = f"Enter {required_name} teleop before recording"
                self._recording_message = message
                self._print_and_say(message, blocking=False)
                return
            if (
                self._episode_state.get_state() == self._episode_state.IDLE
                and not self.episode_finalizer.can_record()
            ):
                finalizer_status = self.episode_finalizer.status()
                if finalizer_status["error"]:
                    message = (
                        "Recorder finalizer failed; restart after checking the local dataset"
                    )
                else:
                    message = "Local save queue is full; recording resumes when a pending take finishes"
                self._recording_message = message
                self._print_and_say(message, blocking=False)
                return
            hub = getattr(self, "hub_uploader", None)
            hub_status = hub.status() if hub is not None else {}
            if self._episode_state.get_state() == self._episode_state.IDLE and (
                hub_status.get("pending") or hub_status.get("uploading")
            ):
                self._recording_message = "Uploading the previous episode; wait for upload to finish"
                self._print_and_say(self._recording_message, blocking=False)
                return
            self._episode_state.change_state()
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self._initial_yaw = None
                self._episode_started_at = time.monotonic()
                self._episode_stopped_at = None
                if getattr(self, "_sender_sync", None) is not None:
                    self._sender_sync.start(time.monotonic_ns())
                self._episode_input_errors.clear()
                self._episode_hand_diagnostics = {}
                self._episode_frame_diagnostics = {}
                self._episode_input_gaps = []
                self._episode_flagged_frames = 0
                self._set_recording_audio_event("start")
                self._recording_message = f"Recording episode {self.current_episode_index}"
                self._print_and_say(
                    f"Recording started. Episode {self.current_episode_index}", say=False
                )
        elif key in {"x", "f"}:
            if self._episode_state.get_state() in (self._episode_state.RECORDING, self._episode_state.NEED_TO_SAVE):
                self._finish_recording(save=False, discard_reason=(
                    "operator_marked_failure" if key == "f" else discard_reason
                ))

    def _poll_sonic_zmq_messages(self):
        """Poll ZMQ for pose, planner, and manager_state messages (non-blocking)."""
        if self._sonic_zmq_socket is None:
            return

        max_polls = 20
        for _ in range(max_polls):
            try:
                raw = self._sonic_zmq_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            if raw.startswith(b"manager_state"):
                self._handle_manager_state(raw)
            elif raw.startswith(b"planner"):
                self._handle_planner_message(raw)
            elif raw.startswith(b"pose"):
                self._handle_pose_message(raw)

    def _handle_manager_state(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="manager_state")
        except Exception:
            return
        received_monotonic_ns = time.monotonic_ns()

        self.stream_rates.observe(
            "manager_state",
            source_timestamp=_timestamp_seconds(
                data.get("publisher_monotonic_ns"), nanoseconds=True
            ),
        )

        if "stream_mode" in data:
            new_stream_mode = int(data["stream_mode"].flat[0])
            if (
                self._episode_state.get_state() == self._episode_state.RECORDING
                and not _recording_mode_ready(
                    new_stream_mode, self.required_stream_mode
                )
            ):
                required_name = _RECORDING_STREAM_MODE_NAMES[self.required_stream_mode]
                self._recording_message = (
                    f"Recording paused: return to {required_name} mode"
                )
            self.current_stream_mode = new_stream_mode
        self.latest_manager_msg = {
            "stream_mode": self.current_stream_mode,
            "publisher_monotonic_ns": _integer_scalar(data.get("publisher_monotonic_ns")),
            "received_monotonic_ns": received_monotonic_ns,
        }

        if getattr(self, "_sender_sync", None) is not None:
            self._sender_sync.observe("manager", self.latest_manager_msg,
                                      self.latest_manager_msg["publisher_monotonic_ns"], received_monotonic_ns)

        if self._extract_bool(data, "toggle_data_collection"):
            self._manager_toggle_dc = True
        if self._extract_bool(data, "toggle_data_abort"):
            self._manager_toggle_da = True
            if self._manager_discard_reason is None:
                self._manager_discard_reason = "operator_discarded"
        if self._extract_bool(data, "toggle_data_failure"):
            self._manager_toggle_df = True

    def _handle_planner_message(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="planner")
        except Exception:
            return
        received_monotonic_ns = time.monotonic_ns()

        self.stream_rates.observe(
            "planner",
            source_timestamp=_timestamp_seconds(
                data.get("publisher_monotonic_ns"), nanoseconds=True
            ),
        )

        planner_mode = int(data["mode"].flat[0]) if "mode" in data else 0
        planner_movement = (
            data["movement"].flatten().astype(np.float32)
            if "movement" in data and data["movement"].size == 3
            else np.zeros(3, dtype=np.float32)
        )
        planner_facing = (
            data["facing"].flatten().astype(np.float32)
            if "facing" in data and data["facing"].size == 3
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        planner_speed = float(data["speed"].flat[0]) if "speed" in data else -1.0
        planner_height = float(data["height"].flat[0]) if "height" in data else -1.0

        vr_3pt_position = None
        if "vr_position" in data and data["vr_position"].size == 9:
            vr_3pt_position = data["vr_position"].flatten().astype(np.float32)
        vr_3pt_orientation = None
        if "vr_orientation" in data and data["vr_orientation"].size == 12:
            vr_3pt_orientation = data["vr_orientation"].flatten().astype(np.float32)

        self.latest_planner_msg = {
            "planner_mode": planner_mode,
            "planner_movement": planner_movement,
            "planner_facing": planner_facing,
            "planner_speed": planner_speed,
            "planner_height": planner_height,
            "vr_3pt_position": vr_3pt_position,
            "vr_3pt_orientation": vr_3pt_orientation,
            "left_hand_joints": self._extract_hand_joints(data, "left_hand_joints"),
            "right_hand_joints": self._extract_hand_joints(data, "right_hand_joints"),
            "receive_timestamp": time.time(),
            "receive_monotonic": received_monotonic_ns / 1e9,
            "received_monotonic_ns": received_monotonic_ns,
            "publisher_monotonic_ns": int(
                (_timestamp_seconds(data.get("publisher_monotonic_ns"), nanoseconds=True) or 0)
                * 1e9
            ),
        }
        if getattr(self, "_sender_sync", None) is not None:
            self._sender_sync.observe("planner", self.latest_planner_msg,
                                      self.latest_planner_msg["publisher_monotonic_ns"], received_monotonic_ns)


    def _handle_pose_message(self, raw: bytes) -> None:
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27
        G1_R_WRIST_ROLL_IDX = 24
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28

        try:
            pose_data = unpack_pose_message(raw, topic="pose")
        except Exception as e:
            print(f"[Sonic] Error unpacking pose message: {e}")
            return
        received_monotonic_ns = time.monotonic_ns()

        self.stream_rates.observe(
            "pico_pose",
            source_timestamp=_timestamp_seconds(
                pose_data.get("publisher_monotonic_ns"), nanoseconds=True
            ),
        )

        try:
            if "smpl_joints" not in pose_data or len(pose_data["smpl_joints"].shape) != 3:
                return

            left_wrist_joints = None
            right_wrist_joints = None
            if "joint_pos" in pose_data and len(pose_data["joint_pos"].shape) == 2:
                joint_pos = pose_data["joint_pos"][0]
                left_wrist_joints = np.array(
                    [
                        joint_pos[G1_L_WRIST_ROLL_IDX],
                        joint_pos[G1_L_WRIST_PITCH_IDX],
                        joint_pos[G1_L_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )
                right_wrist_joints = np.array(
                    [
                        joint_pos[G1_R_WRIST_ROLL_IDX],
                        joint_pos[G1_R_WRIST_PITCH_IDX],
                        joint_pos[G1_R_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )

            frame_index = None
            if "frame_index" in pose_data:
                frame_index = np.array([pose_data["frame_index"].flat[0]], dtype=np.int64)

            smpl_pose = np.zeros(63, dtype=np.float32)
            if "smpl_pose" in pose_data:
                raw_pose = pose_data["smpl_pose"]
                if raw_pose.ndim == 3:
                    smpl_pose = raw_pose[0].flatten().astype(np.float32)
                elif raw_pose.ndim == 2:
                    smpl_pose = raw_pose.flatten().astype(np.float32)
                elif raw_pose.ndim == 1 and raw_pose.size == 63:
                    smpl_pose = raw_pose.astype(np.float32)

            left_hand_joints = self._extract_hand_joints(pose_data, "left_hand_joints")
            right_hand_joints = self._extract_hand_joints(pose_data, "right_hand_joints")

            vr_3pt_position = None
            if "vr_position" in pose_data and pose_data["vr_position"].size == 9:
                vr_3pt_position = pose_data["vr_position"].flatten().astype(np.float32)
            vr_3pt_orientation = None
            if "vr_orientation" in pose_data and pose_data["vr_orientation"].size == 12:
                vr_3pt_orientation = pose_data["vr_orientation"].flatten().astype(np.float32)

            self.latest_sonic_msg = {
                "smpl_joints": pose_data["smpl_joints"][0],
                "smpl_pose": smpl_pose,
                "body_quat_w": (
                    pose_data["body_quat_w"][0] if "body_quat_w" in pose_data else None
                ),
                "left_hand_joints": left_hand_joints,
                "right_hand_joints": right_hand_joints,
                "left_wrist_joints": left_wrist_joints,
                "right_wrist_joints": right_wrist_joints,
                "vr_3pt_position": vr_3pt_position,
                "vr_3pt_orientation": vr_3pt_orientation,
                "frame_index": frame_index,
                "receive_timestamp": time.time(),
                "receive_monotonic": received_monotonic_ns / 1e9,
                "received_monotonic_ns": received_monotonic_ns,
                "sample_monotonic_ns": int(
                    (
                        _timestamp_seconds(pose_data.get("timestamp_monotonic"))
                        or 0
                    )
                    * 1e9
                ),
                "publisher_monotonic_ns": int(
                    (
                        _timestamp_seconds(
                            pose_data.get("publisher_monotonic_ns"), nanoseconds=True
                        )
                        or 0
                    )
                    * 1e9
                ),
            }
            if getattr(self, "_sender_sync", None) is not None:
                self._sender_sync.observe("sonic", self.latest_sonic_msg,
                                          self.latest_sonic_msg["sample_monotonic_ns"], received_monotonic_ns)
        except Exception as e:
            if not hasattr(self, "_sonic_error_count"):
                self._sonic_error_count = 0
            self._sonic_error_count += 1
            if self._sonic_error_count == 1 or self._sonic_error_count % 100 == 0:
                print(f"[Sonic] Error processing pose message: {e}")

    @staticmethod
    def _extract_hand_joints(pose_data: dict, key: str) -> np.ndarray:
        arr = pose_data.get(key)
        if arr is not None:
            if arr.ndim > 1:
                arr = arr[0]
            return arr.astype(np.float32)
        return np.zeros(7, dtype=np.float32)

    @staticmethod
    def _extract_bool(pose_data: dict, key: str) -> bool:
        val = pose_data.get(key)
        if val is None:
            return False
        if isinstance(val, np.ndarray):
            return bool(val.flat[0])
        return bool(val)

    def _log_latency_periodic(
        self,
        sonic_latency_ms: float | None = None,
    ):
        current_time = time.time()
        if current_time - self._last_latency_log_time >= 1.0:
            self._last_latency_log_time = current_time
            parts = []
            if sonic_latency_ms is not None:
                parts.append(f"Sonic Pose: {sonic_latency_ms:.1f}ms")
            if parts:
                print(f"[Latency] {', '.join(parts)}")

    def _validate_camera_inputs(self, now: float, message: dict | None = None, target_ns: int | None = None) -> dict[str, float | None]:
        """Validate only the streams selected for this dataset."""
        message = (self.latest_image_msg if message is None else message) or {}
        received = message.get("camera_received_monotonic_ns", {})
        warnings: dict[str, float | None] = {}
        for feature_name, feature in self.data_exporter.features.items():
            is_depth = feature_name == DEPTH_VIDEO_FEATURE
            if feature.get("dtype") not in ("image", "video") and not is_depth:
                continue
            name = feature_name.split(".")[-1]
            source_name = "ego_view_depth" if is_depth else name
            sdk_depth_view = is_depth and source_name in message.get("images", {})
            source = message.get("images", {}) if sdk_depth_view or not is_depth else message.get("depths", {})
            image = source.get(source_name)
            if image is None:
                raise RuntimeError(f"required camera {name} is unavailable")
            image_shape = image.shape if sdk_depth_view or not is_depth else (*image.shape, 3)
            if tuple(image_shape) != tuple(feature["shape"]):
                raise RuntimeError(f"camera {name} shape {image.shape} does not match {feature['shape']}")
            timestamp_key = f"capture.{source_name}_source_timestamp_ns"
            if timestamp_key in self.data_exporter.features:
                if _timestamp_seconds(message.get("timestamps", {}).get(source_name)) is None:
                    raise RuntimeError(f"camera {name} capture timestamp is unavailable")
            received_at = received.get(source_name)
            if target_ns is None and received_at is not None and now - received_at / 1e9 > self.camera_max_age:
                warnings[f"camera {name} is stale"] = now - received_at / 1e9
        return warnings

    def _add_images_to_frame_data(self, frame_data: dict, inputs: RecordingInputs | None = None) -> None:
        inputs = inputs or self._latest_recording_inputs()
        if inputs.image is None:
            return
        images = inputs.image["images"]
        if DEPTH_VIDEO_FEATURE in self.data_exporter.features:
            images = camera_images_with_depth_preview(images, inputs.image.get("depths", {}))
        for feature_name, feature_info in self.data_exporter.features.items():
            if feature_name == DEPTH_VIDEO_FEATURE:
                source_name = "ego_view_depth"
                if source_name in images:
                    frame_data[feature_name] = np.ascontiguousarray(images[source_name])
                else:
                    raise ValueError(f"Required depth '{source_name}' not found in camera message")
                timestamp_key = f"capture.{source_name}_source_timestamp_ns"
                if timestamp_key in self.data_exporter.features:
                    frame_data[timestamp_key] = np.asarray(
                        [int(inputs.image["timestamps"][source_name] * 1e9)], dtype=np.int64
                    )
                continue
            if feature_info.get("dtype") in ["image", "video"]:
                image_key = feature_name.split(".")[-1]
                if image_key not in images:
                    raise ValueError(
                        f"Required image '{image_key}' for feature '{feature_name}' "
                        f"not found in image message. Available: {list(images.keys())}"
                    )
                frame_data[feature_name] = images[image_key]
                timestamp_key = f"capture.{image_key}_source_timestamp_ns"
                if timestamp_key in self.data_exporter.features:
                    timestamp = _timestamp_seconds(inputs.image.get("timestamps", {}).get(image_key))
                    if timestamp is None or timestamp >= np.iinfo(np.int64).max / 1e9:
                        raise ValueError(f"Required image '{image_key}' has no valid source timestamp")
                    frame_data[timestamp_key] = np.asarray([int(timestamp * 1e9)], dtype=np.int64)

    def _finalize_frame(self, t_start: float) -> bool:
        t_end = time.monotonic()
        if t_end - t_start > (1 / self.frequency):
            print(f"DataExporter Missed: {t_end - t_start} sec")

        if getattr(self, "_sender_sync", None) is None and self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            self._finish_recording(save=True, discard_reason="operator_discarded")
        return True

    def _add_data_frame(self):
        if getattr(self, "_sender_sync", None) is not None:
            return self._add_sender_frame()
        t_start = time.monotonic()

        if self._episode_state.get_state() != self._episode_state.RECORDING:
            return self._finalize_frame(t_start)

        try:
            inputs = self._latest_recording_inputs()
            frame_warnings = self._validate_recording_inputs(inputs)
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
            self._record_input_gap(error)
            if self.data_exporter.episode_buffer.get("size", 0) > 0:
                self._episode_input_errors.add(error)
            now = time.monotonic()
            if now - self._last_input_block_log > 1.0:
                print(f"[Quality] recording frame blocked: {error}")
                self._recording_message = f"Recording blocked: {error}"
                self._last_input_block_log = now
            return False

        if self._recording_message.startswith("Recording blocked:"):
            self._recording_message = f"Recording episode {self.current_episode_index}"
        return self._add_data_frame_sonic(t_start, inputs, frame_warnings=frame_warnings)

    def _add_data_frame_sonic(self, t_start: float, inputs: RecordingInputs | None = None,
                             *, frame_warnings: dict[str, float | None] | None = None) -> bool:
        """Build one data frame in Sonic CPP + SMPL mode."""
        inputs = inputs or self._latest_recording_inputs()
        if frame_warnings is None:
            frame_warnings = self._validate_recording_inputs(inputs) or {}
        assert inputs.proprio is not None
        proprio = inputs.proprio

        if self.hand_config is not None:
            (
                requested_left,
                requested_right,
                applied_left,
                applied_right,
                measured_left,
                measured_right,
            ) = self._external_hand_values(inputs)
            whole_q = assemble_dataset_configuration(
                self.robot_model,
                proprio["body_q"],
                measured_left,
                measured_right,
                self.hand_profile,
            )
            whole_action_wbc = assemble_dataset_configuration(
                self.robot_model,
                proprio["last_action"],
                requested_left,
                requested_right,
                self.hand_profile,
            )
            neutral = np.zeros(7, dtype=np.float64)
            fk_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=proprio["body_q"],
                left_hand_actuated_joint_values=neutral,
                right_hand_actuated_joint_values=neutral,
            )
        else:
            measured_left = _required_vector(
                proprio, "left_hand_q", self.hand_profile.width
            )
            measured_right = _required_vector(
                proprio, "right_hand_q", self.hand_profile.width
            )
            requested_left = _required_vector(
                proprio, "last_left_hand_action", self.hand_profile.width
            )
            requested_right = _required_vector(
                proprio, "last_right_hand_action", self.hand_profile.width
            )
            applied_left = requested_left
            applied_right = requested_right
            whole_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=proprio["body_q"],
                left_hand_actuated_joint_values=measured_left,
                right_hand_actuated_joint_values=measured_right,
            )
            whole_action_wbc = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=proprio["last_action"],
                left_hand_actuated_joint_values=requested_left,
                right_hand_actuated_joint_values=requested_right,
            )
            fk_q = whole_q

        self.robot_model.cache_forward_kinematics(fk_q)
        eef_parts = []
        for side in ["left", "right"]:
            placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            pos = placement.translation[:3]
            quat = R.from_matrix(placement.rotation).as_quat(scalar_first=True)
            eef_parts.append(np.concatenate([pos, quat]))
        observation_eef_state = np.concatenate(eef_parts)

        frame_data: dict = {
            "observation.state": np.asarray(whole_q, dtype=np.float32),
            "observation.eef_state": np.asarray(observation_eef_state, dtype=np.float32),
            "action.wbc": np.asarray(whole_action_wbc, dtype=np.float32),
            f"observation.{raw_hand_name(self.hand_profile)}_left_raw": np.asarray(measured_left, dtype=np.float32),
            f"observation.{raw_hand_name(self.hand_profile)}_right_raw": np.asarray(measured_right, dtype=np.float32),
            f"action.{raw_hand_name(self.hand_profile)}_left_raw": np.asarray(requested_left, dtype=np.float32),
            f"action.{raw_hand_name(self.hand_profile)}_right_raw": np.asarray(requested_right, dtype=np.float32),
            "observation.left_hand_valid": np.ones(1, dtype=np.uint8),
            "observation.right_hand_valid": np.ones(1, dtype=np.uint8),
            "episode.success": np.ones(1, dtype=np.uint8),
        }

        self._add_cpp_state_features(frame_data, proprio)

        sonic_latency_ms = self._add_sonic_pose_features(frame_data, inputs)

        if self.hand_config is not None:
            side_states = inputs.hand["sides"]
            for side in ("left", "right"):
                frame_data[f"observation.{side}_hand_valid"] = np.asarray(
                    [bool(side_states[side].get("valid") and side_states[side].get("connected"))],
                    dtype=np.uint8,
                )
            frame_data["teleop.left_hand_joints"] = requested_left.astype(np.float32)
            frame_data["teleop.right_hand_joints"] = requested_right.astype(np.float32)
            frame_data["control.hand_applied_position"] = np.concatenate(
                (applied_left, applied_right)
            ).astype(np.float32)
            frame_data["teleop.hand_closed"] = np.asarray(
                # The legacy auxiliary bool column cannot represent unknown.
                # Exact unknown-label rows are identified in hand_diagnostics;
                # training actions remain the reported position commands.
                [bool(side_states[side].get("intent_closed")) for side in ("left", "right")],
                dtype=bool,
            )
        else:
            frame_data["control.hand_applied_position"] = np.concatenate(
                (
                    applied_left,
                    applied_right,
                )
            ).astype(np.float32)
            frame_data["teleop.hand_closed"] = np.zeros(2, dtype=bool)

        self._add_capture_features(frame_data, proprio, inputs)
        self._add_images_to_frame_data(frame_data, inputs)

        self._log_latency_periodic(sonic_latency_ms)

        hand_warnings = self._external_hand_diagnostics(inputs) if self.hand_config is not None else {}
        frame_index = self.data_exporter.episode_buffer.get("size", 0)
        try:
            self.data_exporter.add_frame(frame_data)
        except RecordingMemoryLimitError:
            self._finish_recording(save=False, discard_reason="recording_memory_limit")
            return False
        self._record_hand_diagnostics(frame_index, hand_warnings)
        if not hasattr(self, "_episode_frame_diagnostics"):
            self._episode_frame_diagnostics = {}
        self._record_frame_ranges(self._episode_frame_diagnostics, frame_index, frame_warnings)
        if frame_warnings or hand_warnings:
            self._episode_flagged_frames = getattr(self, "_episode_flagged_frames", 0) + 1
        return self._finalize_frame(t_start)

    def _add_capture_features(self, frame_data: dict, proprio: dict, inputs: RecordingInputs | None = None) -> None:
        """Preserve source identity/timing without exposing it to GR00T."""
        inputs = inputs or self._latest_recording_inputs()
        robot_source_s = _timestamp_seconds(proprio.get("ros_timestamp"))
        frame_data["capture.robot_state_sequence"] = np.asarray(
            [_integer_scalar(proprio.get("index"))], dtype=np.int64
        )
        frame_data["capture.robot_state_source_timestamp_ns"] = np.asarray(
            [-1 if robot_source_s is None else int(robot_source_s * 1e9)], dtype=np.int64
        )
        frame_data["capture.robot_state_sample_monotonic_ns"] = np.asarray(
            [_integer_scalar(proprio.get("sample_monotonic_ns"))], dtype=np.int64
        )
        frame_data["capture.robot_state_publish_monotonic_ns"] = np.asarray(
            [_integer_scalar(proprio.get("publisher_monotonic_ns"))], dtype=np.int64
        )
        frame_data["capture.robot_state_received_monotonic_ns"] = np.asarray(
            [inputs.proprio_received_ns or -1], dtype=np.int64
        )

        image = inputs.image or {}
        frame_data["capture.camera_sequence"] = np.asarray(
            [_integer_scalar(image.get("publisher_sequence"))], dtype=np.int64
        )
        frame_data["capture.camera_source_monotonic_ns"] = np.asarray(
            [_integer_scalar(image.get("publisher_monotonic_ns"))], dtype=np.int64
        )
        frame_data["capture.camera_sample_monotonic_ns"] = np.asarray(
            [_integer_scalar(image.get("sample_monotonic_ns"))], dtype=np.int64
        )
        frame_data["capture.camera_publish_monotonic_ns"] = np.asarray(
            [_integer_scalar(image.get("publisher_monotonic_ns"))], dtype=np.int64
        )
        frame_data["capture.camera_received_monotonic_ns"] = np.asarray(
            [inputs.image_received_ns or -1], dtype=np.int64
        )

        hand = inputs.hand or {}
        frame_data["capture.hand_state_sequence"] = np.asarray(
            [_integer_scalar(hand.get("sequence"))], dtype=np.int64
        )
        frame_data["capture.hand_state_publish_sequence"] = np.asarray(
            [_integer_scalar(hand.get("publish_sequence"))], dtype=np.int64
        )
        frame_data["capture.hand_state_source_monotonic_ns"] = np.asarray(
            [_integer_scalar(hand.get("monotonic_ns"))], dtype=np.int64
        )
        frame_data["capture.hand_state_publish_monotonic_ns"] = np.asarray(
            [_integer_scalar(hand.get("published_monotonic_ns"))], dtype=np.int64
        )
        frame_data["capture.hand_state_received_monotonic_ns"] = np.asarray(
            [inputs.hand_received_ns or -1], dtype=np.int64
        )
        frame_data["capture.hand_intent_sequence"] = np.asarray(
            [_integer_scalar(hand.get("intent_sequence"))], dtype=np.int64
        )
        frame_data["capture.hand_intent_source_monotonic_ns"] = np.asarray(
            [_integer_scalar(hand.get("intent_source_monotonic_ns"))], dtype=np.int64
        )
        frame_data["capture.hand_intent_received_monotonic_ns"] = np.asarray(
            [_integer_scalar(hand.get("intent_received_monotonic_ns"))], dtype=np.int64
        )

        pose = inputs.sonic or {}
        frame_data["capture.pico_pose_sequence"] = np.asarray(
            [_integer_scalar(pose.get("frame_index"))], dtype=np.int64
        )
        for capture_field, message_field in (
            ("sample", "sample"),
            ("publish", "publisher"),
            ("received", "received"),
        ):
            frame_data[f"capture.pico_pose_{capture_field}_monotonic_ns"] = np.asarray(
                [_integer_scalar(pose.get(f"{message_field}_monotonic_ns"))], dtype=np.int64
            )

        planner = inputs.planner or {}
        for capture_field, message_field in (("publish", "publisher"), ("received", "received")):
            frame_data[f"capture.planner_{capture_field}_monotonic_ns"] = np.asarray(
                [_integer_scalar(planner.get(f"{message_field}_monotonic_ns"))], dtype=np.int64
            )

        manager = inputs.manager or {}
        for capture_field, message_field in (("publish", "publisher"), ("received", "received")):
            frame_data[f"capture.manager_{capture_field}_monotonic_ns"] = np.asarray(
                [_integer_scalar(manager.get(f"{message_field}_monotonic_ns"))], dtype=np.int64
            )
        if inputs.target_ns is not None:
            frame_data["capture.sync_target_monotonic_ns"] = np.asarray([inputs.target_ns], dtype=np.int64)
            for key in self.data_exporter.features:
                if not key.startswith("capture.sync."):
                    continue
                stream, field = key[len("capture.sync."):].rsplit(".", 1)
                sample = inputs.samples.get(stream, {})
                frame_data[key] = np.asarray([sample.get(f"_sync_{field}", -1)], dtype=np.int64)


    def _add_cpp_state_features(self, frame_data: dict, proprio: dict) -> None:
        base_quat = _required_vector(proprio, "base_quat", 4)
        frame_data["observation.root_orientation"] = base_quat
        frame_data["observation.projected_gravity"] = compute_projected_gravity(
            base_quat
        ).astype(np.float32)
        frame_data["observation.base_angular_velocity"] = _required_vector(
            proprio, "base_ang_vel", 3
        )
        frame_data["observation.body_joint_velocity"] = _required_vector(
            proprio, "body_dq", 29
        )

        if "init_ref_data_root_rot_array" in proprio:
            frame_data["observation.cpp_rotation_offset"] = np.asarray(
                proprio["init_ref_data_root_rot_array"], dtype=np.float32
            )
        else:
            frame_data["observation.cpp_rotation_offset"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            )

        if "init_base_quat" in proprio:
            frame_data["observation.init_base_quat"] = np.asarray(
                proprio["init_base_quat"], dtype=np.float32
            )
        else:
            frame_data["observation.init_base_quat"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            )

        if "delta_heading" in proprio:
            dh = proprio["delta_heading"]
            if isinstance(dh, np.ndarray):
                dh = dh.item() if dh.size == 1 else dh[0]
            frame_data["teleop.delta_heading"] = np.array([float(dh)], dtype=np.float32)
        else:
            frame_data["teleop.delta_heading"] = np.zeros(1, dtype=np.float32)

        frame_data["action.motion_token"] = _required_vector(proprio, "token_state", 64)
        frame_data["action.motion_token_valid"] = np.ones(1, dtype=np.uint8)

    def _add_sonic_pose_features(self, frame_data: dict, inputs: RecordingInputs | None = None) -> float | None:
        """Add teleop features based on current stream mode."""
        inputs = inputs or self._latest_recording_inputs()
        sonic_latency_ms = None

        frame_data["teleop.stream_mode"] = np.array([inputs.mode], dtype=np.int32)

        smpl_msg = inputs.sonic
        use_smpl = False
        if inputs.mode in (1, 4) and smpl_msg is not None:
            receive_ts = smpl_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = ((inputs.target_ns - smpl_msg["_sync_time_ns"]) / 1e9
                           if inputs.target_ns is not None else time.time() - receive_ts)
                sonic_latency_ms = age_sec * 1000
                self.sonic_timing_monitor.log_time_delta(age_sec)
                if inputs.target_ns is not None or sonic_latency_ms <= 100.0:
                    use_smpl = True
                elif (self.sonic_timing_monitor.failure_count + 1) % 10 == 0:
                    self._print_and_say(
                        f"Sonic pose stale ({sonic_latency_ms:.1f}ms old), using zeros",
                        say=False,
                    )
            else:
                use_smpl = True

        planner_msg = inputs.planner
        use_planner = False
        if inputs.mode in (5, 6) and planner_msg is not None:
            receive_ts = planner_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = ((inputs.target_ns - planner_msg["_sync_time_ns"]) / 1e9
                           if inputs.target_ns is not None else time.time() - receive_ts)
                planner_latency_ms = age_sec * 1000
                if sonic_latency_ms is None:
                    sonic_latency_ms = planner_latency_ms
                if inputs.target_ns is not None or planner_latency_ms <= 200.0:
                    use_planner = True
            else:
                use_planner = True

        # SMPL features
        if use_smpl and smpl_msg.get("smpl_joints") is not None:
            joints = np.asarray(smpl_msg["smpl_joints"], dtype=np.float32)
            if joints.ndim == 2:
                joints = joints.flatten()
            frame_data["teleop.smpl_joints"] = np.ascontiguousarray(joints, dtype=np.float32)
        else:
            frame_data["teleop.smpl_joints"] = np.zeros(72, dtype=np.float32)

        if use_smpl and smpl_msg.get("smpl_pose") is not None:
            pose = np.asarray(smpl_msg["smpl_pose"], dtype=np.float32)
            if pose.ndim > 1:
                pose = pose.flatten()
            frame_data["teleop.smpl_pose"] = np.ascontiguousarray(pose, dtype=np.float32)
        else:
            frame_data["teleop.smpl_pose"] = np.zeros(63, dtype=np.float32)

        if use_smpl and smpl_msg.get("body_quat_w") is not None:
            body_quat_w = smpl_msg["body_quat_w"].astype(np.float32)
            frame_data["teleop.body_quat_w"] = body_quat_w
            frame_data["teleop.target_body_orientation"] = self._compute_target_body_orientation(
                body_quat_w, frame_data
            )
        else:
            frame_data["teleop.body_quat_w"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            frame_data["teleop.target_body_orientation"] = quat_to_rot6d(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            )

        frame_data["teleop.left_wrist_joints"] = (
            smpl_msg["left_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("left_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.right_wrist_joints"] = (
            smpl_msg["right_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("right_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )

        frame_data["teleop.smpl_frame_index"] = (
            smpl_msg["frame_index"].astype(np.int64)
            if use_smpl and smpl_msg is not None and smpl_msg.get("frame_index") is not None
            else np.array([0], dtype=np.int64)
        )
        frame_data["teleop.smpl_valid"] = np.asarray([int(use_smpl)], dtype=np.uint8)

        hand_msg = (
            smpl_msg if inputs.mode in (1, 4) and smpl_msg is not None
            else planner_msg if planner_msg is not None
            else smpl_msg
        )
        frame_data["teleop.left_hand_joints"] = (
            hand_msg["left_hand_joints"].astype(np.float32)
            if hand_msg is not None
            and hand_msg.get("left_hand_joints") is not None
            else np.zeros(7, dtype=np.float32)
        )
        frame_data["teleop.right_hand_joints"] = (
            hand_msg["right_hand_joints"].astype(np.float32)
            if hand_msg is not None
            and hand_msg.get("right_hand_joints") is not None
            else np.zeros(7, dtype=np.float32)
        )

        # Planner command fields
        frame_data["teleop.planner_mode"] = np.array(
            [planner_msg["planner_mode"]] if use_planner else [0],
            dtype=np.int32,
        )
        frame_data["teleop.planner_movement"] = (
            planner_msg["planner_movement"].copy()
            if use_planner and planner_msg.get("planner_movement") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.planner_facing"] = (
            planner_msg["planner_facing"].copy()
            if use_planner and planner_msg.get("planner_facing") is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        frame_data["teleop.planner_speed"] = np.array(
            [planner_msg["planner_speed"]] if use_planner else [-1.0],
            dtype=np.float32,
        )
        frame_data["teleop.planner_height"] = np.array(
            [planner_msg["planner_height"]] if use_planner else [-1.0],
            dtype=np.float32,
        )
        frame_data["teleop.control_mode"] = np.asarray(
            [2 if use_planner else 0 if use_smpl else 255], dtype=np.uint8
        )
        frame_data["teleop.locomotion_speed_m_s"] = np.asarray(
            [planner_msg["planner_speed"] if use_planner else -1.0], dtype=np.float32
        )

        # VR 3-point pose
        frame_data["teleop.vr_3pt_position"] = (
            planner_msg["vr_3pt_position"].astype(np.float32)
            if use_planner and planner_msg.get("vr_3pt_position") is not None
            else np.zeros(9, dtype=np.float32)
        )
        if use_planner and planner_msg.get("vr_3pt_orientation") is not None:
            vr_orientation = planner_msg["vr_3pt_orientation"].astype(np.float32)
            frame_data["teleop.vr_3pt_orientation_wxyz"] = vr_orientation
            frame_data["teleop.vr_3pt_orientation"] = quat_to_rot6d(vr_orientation)
            frame_data["teleop.vr_3pt_valid"] = np.ones(1, dtype=np.uint8)
        else:
            frame_data["teleop.vr_3pt_orientation_wxyz"] = np.zeros(12, dtype=np.float32)
            frame_data["teleop.vr_3pt_orientation"] = np.zeros(18, dtype=np.float32)
            frame_data["teleop.vr_3pt_valid"] = np.zeros(1, dtype=np.uint8)

        return sonic_latency_ms

    def _compute_target_body_orientation(
        self, body_quat_w: np.ndarray, frame_data: dict
    ) -> np.ndarray:
        """Compute yaw-normalised target body orientation as rot6d (6-dim)."""
        delta_heading = float(frame_data.get("teleop.delta_heading", [0.0])[0])

        body_rot = R.from_quat(body_quat_w, scalar_first=True)
        target_rot = R.from_euler("z", delta_heading, degrees=False) * body_rot

        euler = target_rot.as_euler("ZYX", degrees=False)
        current_yaw = euler[0]

        if self._initial_yaw is None:
            self._initial_yaw = current_yaw

        normalised_euler = np.array([current_yaw - self._initial_yaw, euler[1], euler[2]])
        target_quat = (
            R.from_euler("ZYX", normalised_euler, degrees=False)
            .as_quat(scalar_first=True)
            .astype(np.float32)
        )
        return quat_to_rot6d(target_quat)

    def _drain_uploads_on_shutdown(self) -> None:
        """Let queued uploads finish at exit, reporting progress while waiting."""
        deadline = time.monotonic() + self.shutdown_upload_timeout
        while time.monotonic() < deadline:
            status = self.hub_uploader.status()
            if not status["pending"] and not status["uploading"]:
                return
            print(f"[Hub] Finishing uploads ({status['pending']} queued); Ctrl-C to skip")
            try:
                if self.hub_uploader.wait_until_idle(timeout=15.0):
                    return
            except KeyboardInterrupt:
                print("[Hub] Upload skipped by operator; local data is preserved")
                return
        print(
            f"[Hub] Upload still pending after {self.shutdown_upload_timeout:.0f} seconds; "
            "local data is preserved"
        )

    def save_and_cleanup(self):
        try:
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self._finish_recording(
                    save=False,
                    discard_reason="collector_shutdown_before_episode_save",
                )
            self.episode_finalizer.wait_until_idle()
            self._drain_uploads_on_shutdown()
            self._print_and_say(
                f"Recording complete: {self.data_exporter.meta.root}", say=False, blocking=True
            )
        except Exception as e:
            self._print_and_say(f"Error saving episode: {e}", blocking=True)

        try:
            self.episode_finalizer.close(timeout=0.0)
        except Exception:
            pass
        try:
            self.hub_uploader.close(timeout=0.0)
        except Exception:
            pass
        try:
            self._state_subscriber.close()
        except Exception:
            pass
        try:
            self._image_subscriber.close()
        except Exception:
            pass
        for sock in [self._sonic_zmq_socket, self._hand_zmq_socket]:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        try:
            self._recording_status_socket.close(linger=0)
            self._recording_status_ctx.term()
        except Exception:
            pass
        for ctx in [self._sonic_zmq_ctx, self._hand_zmq_ctx]:
            if ctx is not None:
                try:
                    ctx.term()
                except Exception:
                    pass

        if getattr(self, "_sender_sync", None) is not None:
            self._sender_sync.close()
        self._print_and_say("Shutting down data exporter...", say=False)

    def run(self):
        next_tick = time.monotonic()
        try:
            while True:
                t_start = time.monotonic()
                with self.telemetry.timer("total_loop"):
                    with self.telemetry.timer("poll_state"):
                        self._poll_state_zmq()

                    with self.telemetry.timer("poll_sonic"):
                        self._poll_sonic_zmq_messages()

                    with self.telemetry.timer("poll_hands"):
                        self._poll_hand_zmq()

                    with self.telemetry.timer("poll_image"):
                        previous_image_index = self._image_subscriber.idx
                        img_msg = (self._poll_sender_images() if self._sender_sync is not None
                                   else self._image_subscriber.read())
                        if img_msg is not None:
                            self.latest_image_msg = img_msg
                        if self._image_subscriber.idx != previous_image_index and img_msg is not None:
                            receiver_monotonic_ns = _integer_scalar(
                                img_msg.get("receiver_monotonic_ns")
                            )
                            if receiver_monotonic_ns <= 0:
                                receiver_monotonic_ns = time.monotonic_ns()
                            self.latest_image_received_monotonic_ns = receiver_monotonic_ns
                            self.latest_image_received_at = receiver_monotonic_ns / 1e9
                            source_timestamps = [
                                timestamp
                                for value in img_msg.get("timestamps", {}).values()
                                if (timestamp := _timestamp_seconds(value)) is not None
                            ]
                            self.stream_rates.observe(
                                "camera",
                                source_timestamp=(
                                    _timestamp_seconds(
                                        img_msg.get("publisher_monotonic_ns"),
                                        nanoseconds=True,
                                    )
                                    or (max(source_timestamps) if source_timestamps else None)
                                ),
                                source_sequence=(
                                    int(img_msg["publisher_sequence"])
                                    if isinstance(img_msg.get("publisher_sequence"), int)
                                    and not isinstance(img_msg.get("publisher_sequence"), bool)
                                    else None
                                ),
                                received_timestamp=receiver_monotonic_ns / 1e9,
                            )
                            self._observe_wrist_camera_rates(img_msg)

                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                    with self.telemetry.timer("check_recording_commands"):
                        self._check_recording_commands()

                    self._publish_recording_status()

                    end_time = time.monotonic()

                # Pace against an absolute deadline. Sleeping for
                # ``period - work_time`` every iteration accumulates the small
                # wake-up delay from time.sleep(), which made a configured
                # 50 Hz loop settle around 49 Hz. Do not replay a large backlog
                # after genuinely blocking work such as episode finalization.
                next_tick += self.loop_period
                now = time.monotonic()
                if next_tick < now - self.loop_period:
                    skipped_ticks = int((now - next_tick) / self.loop_period) + 1
                    next_tick += skipped_ticks * self.loop_period
                sleep_time = next_tick - now
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if (end_time - t_start) > self.loop_period:
                    self.telemetry.log_timing_info(
                        context="Data Exporter Loop Missed", threshold=0.001
                    )

        except KeyboardInterrupt:
            print("Data exporter terminated by user")

        finally:
            self.save_and_cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_hand_profile(requested: str, hand_config: dict | None) -> HandProfile:
    """Use the external controller's declared profile before creating a schema."""
    name = requested
    if name == "auto":
        name = hand_config["profile"] if hand_config is not None else "dex3.v1"
    profile = get_hand_profile(name)
    if hand_config is not None and hand_config.get("profile") != profile.name:
        raise RuntimeError(f"requested {profile.name}, controller reports {hand_config.get('profile')}")
    return profile


def _validate_recording_dataset_mode(root: Path, expected_features: dict, sender_mode: bool) -> None:
    """Reject incompatible resume before opening any episode video writers."""
    info_path = root / "meta/info.json"
    if not info_path.exists():
        return
    info = json.loads(info_path.read_text())
    def camera_schema(features):
        return {key: (value["dtype"], tuple(value["shape"]))
                for key, value in features.items() if value.get("dtype") in ("image", "video")}

    if camera_schema(info.get("features", {})) != camera_schema(expected_features):
        raise ValueError(
            "Dataset camera schema differs from the requested recording cameras. "
            "Choose a new --dataset-name or use the cameras that created this dataset."
        )
    existing = {key for key in info.get("features", {}) if key.startswith("capture.sync")}
    expected = {key for key in expected_features if key.startswith("capture.sync")}
    if existing != expected or bool(existing) != sender_mode:
        raise ValueError(
            "Dataset synchronization schema differs from this recording mode. "
            "Choose a new --dataset-name or use the mode/cameras that created this dataset."
        )


def main(config: SonicDataExporterConfig):
    if not np.isfinite(config.max_episode_duration_s) or config.max_episode_duration_s < 0:
        raise ValueError("max_episode_duration_s must be finite and nonnegative")
    config.root_output_dir, config.dataset_name = resolve_recording_destination(
        config.dataset_name, config.root_output_dir,
    )

    if config.sender_time_recording:
        if any(host not in {"localhost", "127.0.0.1", "::1"} for host in
               (config.camera_host, config.state_zmq_host, config.sonic_zmq_host)):
            raise ValueError("sender-time recording requires local camera, robot and teleop publishers")
        # Validate before waiting on publishers or opening writers.
        SenderSynchronizer(camera_names=(), frequency=config.data_collection_frequency,
                           delay_s=config.synchronization_delay, wait_s=config.synchronization_wait_timeout)
        if not 1 <= config.hand_clock_port <= 65535:
            raise ValueError("hand clock port must be within 1..65535")
    g1_rm = get_g1_robot_model()

    robot_config = poll_robot_config_zmq(
        config.state_zmq_host, config.state_zmq_port, config.robot_config_timeout
    )
    hand_config = None
    if robot_config.get("hand_control") == "external":
        hand_config = poll_hand_config_zmq(
            config.hand_state_host, config.hand_state_port, config.hand_config_timeout
        )
    hand_profile = resolve_hand_profile(config.hand_profile, hand_config)

    schema_profile = hand_profile if hand_config is not None else None
    dataset_features = get_features_sonic_vla(g1_rm, schema_profile)
    modality_config = get_modality_config_sonic_vla(g1_rm, schema_profile)

    if config.record_wrist_cameras:
        print("[Camera] Wrist cameras enabled — adding to dataset schema")
        dataset_features.update(get_wrist_camera_features())
        wrist_modality = get_wrist_camera_modality_config()
        for key, value in wrist_modality.items():
            if key in modality_config:
                modality_config[key].update(value)
            else:
                modality_config[key] = value
    if config.record_zed_stereo:
        print("[Camera] ZED stereo enabled — adding left RGB and depth to dataset schema")
        dataset_features.update(get_zed_stereo_features())
        for key, value in get_zed_stereo_modality_config().items():
            modality_config.setdefault(key, {}).update(value)

    if config.sender_time_recording:
        camera_names = tuple(key.split(".")[-1] for key, feature in dataset_features.items()
                             if feature.get("dtype") in ("image", "video"))
        dataset_features.update(synchronization_features(camera_names))

    text_to_speech = TextToSpeech() if config.text_to_speech else None

    _validate_recording_dataset_mode(
        Path(config.root_output_dir) / config.dataset_name, dataset_features, config.sender_time_recording,
    )
    data_exporter = Gr00tDataExporter.create(
        save_root=f"{config.root_output_dir}/{config.dataset_name}",
        fps=config.data_collection_frequency,
        features=dataset_features,
        modality_config=modality_config,
        task=config.task_prompt,
        script_config={
            **robot_config,
            "record_wrist_cameras": config.record_wrist_cameras,
            "record_zed_stereo": config.record_zed_stereo,
            "max_episode_duration_s": config.max_episode_duration_s,
            "depth_video_encoding": (
                {"format": "uint8_rgb", "visualization": "browser_ui",
                 "colormap": "opencv_turbo", "normalization_percentiles": [2, 98],
                 "fallback": "camera_depth_video"}
                if config.record_zed_stereo else None
            ),
            "recording_synchronization": {
                "mode": "sender" if config.sender_time_recording else "latest",
                "delay_s": config.synchronization_delay,
                "wait_timeout_s": config.synchronization_wait_timeout,
                "target_clock": "recorder CLOCK_MONOTONIC",
                "remote_hand_mapping": "four-timestamp exchange; 5 ms uncertainty limit; 100 ppm drift budget",
                "time_fields": "measurement/capture for robot, cameras, pose, hands; publication for planner/manager",
            },
            "hand_profile": hand_profile.name,
            "hand_config": hand_config,
            "capture": _capture_reproducibility_metadata(
                robot_config, config.data_collection_frequency, hand_profile,
                hand_state_host=config.hand_state_host,
            ),
        },
        robot_type=dataset_robot_type(hand_profile),
    )

    data_collector = GrootDataCollector(
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=g1_rm,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
        sonic_data_zmq_host=config.sonic_zmq_host,
        sonic_data_zmq_port=config.sonic_zmq_port,
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=config.state_zmq_port,
        hand_profile=hand_profile,
        hand_config=hand_config,
        hand_state_host=config.hand_state_host,
        hand_state_port=config.hand_state_port,
        hand_state_max_age=config.hand_state_max_age,
        proprio_state_max_age=config.proprio_state_max_age,
        camera_max_age=config.camera_max_age,
        teleop_max_age=config.teleop_max_age,
        minimum_recording_rate_hz=config.minimum_recording_rate_hz,
        required_stream_mode=config.required_stream_mode,
        require_hand_activity=config.require_hand_activity,
        minimum_hand_motion_rad=config.minimum_hand_motion_rad,
        recording_status_port=config.recording_status_port,
        require_hub_upload=config.require_hub_upload,
        shutdown_upload_timeout=config.shutdown_upload_timeout,
        sender_time_recording=config.sender_time_recording,
        synchronization_delay=config.synchronization_delay,
        synchronization_wait_timeout=config.synchronization_wait_timeout,
        hand_clock_port=config.hand_clock_port,
        max_episode_duration_s=config.max_episode_duration_s,
    )
    data_collector.run()


if __name__ == "__main__":
    config = tyro.cli(SonicDataExporterConfig)

    main(config)
