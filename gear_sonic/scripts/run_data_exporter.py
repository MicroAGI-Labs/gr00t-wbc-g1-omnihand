"""
Sonic VLA data exporter for G1 -- NO ROS 2 DEPENDENCY.

All data sources use ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (port 5557, from C++ zmq_output_handler)
  2. SMPL pose    -> ZMQ SUB on ``pose`` topic     (port 5556, from pico_manager_thread_server)
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
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    assemble_dataset_configuration,
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.end_effectors.profiles import HandProfile, get_hand_profile
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SonicDataExporterConfig:
    """CLI config for the ROS-free Sonic data exporter."""

    # Dataset
    dataset_name: str | None = None
    """Dataset name (auto-generated if creating new)."""

    task_prompt: str = DEFAULT_TASK_PROMPT
    """Language task prompt."""

    root_output_dir: str = "outputs"
    """Root output directory."""

    data_collection_frequency: int = 50
    """Data collection frequency (Hz)."""


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

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist). Requires cameras to be available."""

    text_to_speech: bool = True
    """Use text-to-speech voice feedback."""

    hand_profile: str = "auto"
    """Hand profile (auto, dex3.v1, or omnihand_o10.v1)."""

    hand_state_host: str = "localhost"
    """Host publishing external hand_config/hand_state messages."""

    hand_state_port: int = 5570
    """Port publishing external hand_config/hand_state messages."""

    hand_config_timeout: float = 0
    """Seconds to wait for external hand config (0 waits indefinitely)."""

    hand_state_max_age: float = 0.2
    """Maximum external hand-state age admitted while recording."""

    proprio_state_max_age: float = 0.1
    """Maximum robot-state age admitted while recording."""

    camera_max_age: float = 0.1
    """Maximum camera-frame age admitted while recording."""

    teleop_max_age: float = 0.2
    """Maximum active planner/SMPL message age admitted while recording."""

    minimum_recording_rate_hz: float = 45.0
    """Minimum source rate required for a successful episode."""

    required_stream_mode: int = 5
    """Stream mode required for recording (1=POSE, 5=VR3PT, 6=IK upper)."""

    require_hand_activity: bool = True
    """Reject a successful episode when neither hand command changes."""

    minimum_hand_motion_rad: float = 0.02
    """Minimum requested hand-joint range required when hand activity is enforced."""

    recording_status_port: int = 5581
    """ZMQ PUB port for browser-visible recorder status."""

    require_hub_upload: bool = False
    """Require browser Hub setup and a completed upload between episodes."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RECORDING_STREAM_MODE_NAMES = {
    1: "POSE",
    5: "VR3PT",
    6: "IK upper",
}


def _recording_mode_ready(current_stream_mode: int, required_stream_mode: int) -> bool:
    """Return whether the launch-selected A+X teleop mode is active."""
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


class EpisodeHubUploader:
    """Upload each finalized episode while keeping recorder state observable."""

    _EPISODE_FILE_RE = re.compile(r"episode_(\d+)\.(?:mp4|parquet)$")

    def __init__(self, data_exporter: Gr00tDataExporter):
        self.data_exporter = data_exporter
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._config: dict[str, object] | None = None
        self._pending = 0
        self._uploading = False
        self._retrying = False
        self._last_uploaded_episode: int | None = None
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="episode-hub-uploader",
            daemon=True,
        )
        self._thread.start()

    def configure(self, repo_id: str, prompt: str, private: bool) -> None:
        config = {"repo_id": repo_id, "prompt": prompt, "private": private}
        with self._condition:
            if self._pending or self._uploading:
                if config != self._config:
                    raise RuntimeError("cannot change dataset while an upload is pending")
                return
            self._config = config
            self._error = None
            self._retrying = False
            self.data_exporter.meta.repo_id = repo_id
            self.data_exporter.task = prompt
            self._condition.notify_all()

    def enqueue(self, episode_index: int) -> None:
        with self._condition:
            if self._config is None:
                raise RuntimeError("choose a Hugging Face dataset before saving an episode")
            self._pending += 1
            self._condition.notify_all()
        self._queue.put(episode_index)

    def can_record(self) -> bool:
        with self._condition:
            return self._config is not None and self._pending == 0 and not self._uploading

    def status(self) -> dict[str, object]:
        with self._condition:
            config = dict(self._config or {})
            return {
                "ready": self._config is not None,
                "repo_id": config.get("repo_id"),
                "prompt": config.get("prompt", self.data_exporter.task),
                "private": config.get("private", True),
                "pending": self._pending,
                "uploading": self._uploading,
                "retrying": self._retrying,
                "last_uploaded_episode": self._last_uploaded_episode,
                "error": self._error,
            }

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._pending or self._uploading:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def close(self, timeout: float = 30.0) -> None:
        self.wait_until_idle(timeout=timeout)
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def _upload_patterns(self, episode_index: int) -> list[str]:
        patterns = ["meta/**"]
        root = Path(self.data_exporter.root)
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            match = self._EPISODE_FILE_RE.search(path.name)
            if match and int(match.group(1)) <= episode_index:
                patterns.append(path.relative_to(root).as_posix())
        return patterns

    def _run(self) -> None:
        while not self._stop.is_set():
            episode_index = self._queue.get()
            if episode_index is None:
                return
            retry_delay = 1.0
            while not self._stop.is_set():
                with self._condition:
                    config = dict(self._config or {})
                    self._uploading = True
                    self._retrying = retry_delay > 1.0
                    self._condition.notify_all()
                try:
                    self.data_exporter.push_to_hub(
                        private=bool(config.get("private", True)),
                        allow_patterns=self._upload_patterns(episode_index),
                        upload_large_folder=True,
                    )
                except Exception as exc:
                    with self._condition:
                        self._uploading = False
                        self._retrying = True
                        self._error = str(exc)[-500:]
                        self._condition.notify_all()
                    if self._stop.wait(retry_delay):
                        return
                    retry_delay = min(retry_delay * 2.0, 30.0)
                    continue

                with self._condition:
                    self._pending -= 1
                    self._uploading = False
                    self._retrying = False
                    self._error = None
                    self._last_uploaded_episode = episode_index
                    self._condition.notify_all()
                break


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
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.flat[0]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return default
    return int(value)


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


def _capture_reproducibility_metadata(robot_config: dict, dataset_frequency_hz: float) -> dict:
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
        "robot_type": "unitree_g1_omnihand_sonic",
        "gr00t_embodiment_tag": "UNITREE_G1_SONIC",
        "joint_units": "rad",
        "angular_velocity_units": "rad_s",
        "position_units": "m",
        "quaternion_order": "wxyz",
        "capture_timestamp_units": "ns",
        "capture_monotonic_clock_scope": "single_linux_host",
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
        require_hand_activity: bool = True,
        minimum_hand_motion_rad: float = 0.02,
        recording_status_port: int = 5581,
        require_hub_upload: bool = False,
    ):
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
        self.require_hub_upload = require_hub_upload
        self.latest_hand_state = None
        self.latest_hand_state_received_at = None
        self.latest_hand_state_received_monotonic_ns = None

        self._episode_state = EpisodeState()
        self._keyboard_listener = ZMQKeyboardSubscriber()
        self.hub_uploader = EpisodeHubUploader(data_exporter)
        self._recording_message = (
            "Choose a Hugging Face dataset to begin"
            if require_hub_upload
            else "Ready to record"
        )
        self._recording_status_ctx = zmq.Context()
        self._recording_status_socket = self._recording_status_ctx.socket(zmq.PUB)
        self._recording_status_socket.setsockopt(zmq.SNDHWM, 2)
        self._recording_status_socket.bind(f"tcp://*:{recording_status_port}")

        self._image_subscriber = ComposedCameraClientSensor(
            server_ip=camera_host,
            port=camera_port,
            background=True,
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
        self._last_input_block_log = 0.0
        self.stream_rates = StreamRateTracker()

        self.current_stream_mode = 0

        self._manager_toggle_dc = False
        self._manager_toggle_da = False
        self._manager_discard_reason: str | None = None

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

    def _publish_recording_status(self) -> None:
        """Publish authoritative recorder state for the loopback browser UI."""
        state = self._episode_state.get_state()
        hub_status = self.hub_uploader.status()
        hub_status["required"] = self.require_hub_upload
        hand_ready = True
        if self.hand_config is not None:
            hand_ready = bool(
                self.latest_hand_state
                and not self.latest_hand_state.get("input_stale", True)
                and self.latest_hand_state.get("intent_sequence") is not None
                and (
                    self.latest_hand_state.get("state_age_s") is None
                    or self.latest_hand_state.get("state_age_s") <= self.hand_state_max_age
                )
                and all(
                    side.get("valid") and side.get("connected")
                    for side in self.latest_hand_state.get("sides", {}).values()
                )
            )
        message = self._recording_message
        if hub_status["uploading"]:
            message = f"Uploading episode {self.current_episode_index - 1} to Hugging Face"
        elif hub_status["retrying"]:
            message = "Upload failed; retrying automatically"
        payload = {
            "state": state,
            "recording": state == self._episode_state.RECORDING,
            "saving": state == self._episode_state.NEED_TO_SAVE,
            "episode_index": self.current_episode_index,
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
            "hub": hub_status,
            "stream_rates": self.stream_rates.snapshot(self.RATE_STREAMS),
            "timestamp": time.time(),
        }
        try:
            self._recording_status_socket.send_json(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

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
            self.latest_hand_state = state
            self.latest_hand_state_received_at = time.monotonic()
            self.latest_hand_state_received_monotonic_ns = time.monotonic_ns()
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

    def _external_hand_values(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if self.latest_hand_state is None or self.latest_hand_state_received_at is None:
            raise RuntimeError("external hand state is unavailable")
        age = time.monotonic() - self.latest_hand_state_received_at
        if age > self.hand_state_max_age:
            raise RuntimeError(f"external hand state is stale ({age:.3f}s)")
        controller_age = self.latest_hand_state.get("state_age_s")
        if isinstance(controller_age, (int, float)) and controller_age > self.hand_state_max_age:
            raise RuntimeError(
                f"external hand controller snapshot is stale ({controller_age:.3f}s)"
            )
        if self.latest_hand_state.get("mode") == "fault":
            raise RuntimeError("external hand controller is faulted")
        if self.latest_hand_state.get("input_stale") or self.latest_hand_state.get("intent_sequence") is None:
            raise RuntimeError("external hand target is missing or stale")
        for side in ("left", "right"):
            if self.latest_hand_state.get("sides", {}).get(side, {}).get("intent_closed") is None:
                raise RuntimeError(f"external {side} hand has no valid click intent")
        values = []
        for field in ("requested_position_rad", "applied_position_rad", "measured_position_rad"):
            for side in ("left", "right"):
                side_state = self.latest_hand_state.get("sides", {}).get(side)
                if not side_state or not side_state.get("valid") or not side_state.get("connected"):
                    raise RuntimeError(f"external {side} hand is invalid or disconnected")
                if side_state.get("input_stale"):
                    raise RuntimeError(f"external {side} hand target is stale")
                array = np.asarray(side_state.get(field), dtype=np.float64).reshape(-1)
                if array.shape != (self.hand_profile.width,) or not np.all(np.isfinite(array)):
                    raise RuntimeError(f"external {side} {field} has the wrong shape")
                values.append(array)
        return tuple(values)

    def _validate_recording_inputs(self) -> None:
        """Reject stale or malformed inputs before they enter an episode."""
        if not _recording_mode_ready(
            self.current_stream_mode, self.required_stream_mode
        ):
            required_name = _RECORDING_STREAM_MODE_NAMES[self.required_stream_mode]
            raise RuntimeError(
                f"required A+X teleop mode {required_name} is not active "
                f"(stream mode {self.current_stream_mode})"
            )
        now = time.monotonic()
        if self.latest_proprio_msg is None or self.latest_proprio_received_at is None:
            raise RuntimeError("robot state is unavailable")
        proprio_age = now - self.latest_proprio_received_at
        if proprio_age > self.proprio_state_max_age:
            raise RuntimeError(f"robot state is stale ({proprio_age:.3f}s)")
        for key, width in (
            ("body_q", 29),
            ("body_dq", 29),
            ("base_quat", 4),
            ("base_ang_vel", 3),
            ("last_action", 29),
        ):
            _required_vector(self.latest_proprio_msg, key, width)
        token = _required_vector(self.latest_proprio_msg, "token_state", 64)
        if not np.any(token):
            raise RuntimeError("SONIC motion token is all zeros")

        if self.latest_image_msg is None or self.latest_image_received_at is None:
            raise RuntimeError("camera frame is unavailable")
        camera_age = now - self.latest_image_received_at
        if camera_age > self.camera_max_age:
            raise RuntimeError(f"camera frame is stale ({camera_age:.3f}s)")

        if self.hand_config is not None:
            self._external_hand_values()

        if self.current_stream_mode in (5, 6):
            if self.latest_planner_msg is None:
                raise RuntimeError("planner command is unavailable in planner mode")
            planner_received = self.latest_planner_msg.get("receive_monotonic")
            if planner_received is None or now - planner_received > self.teleop_max_age:
                raise RuntimeError("planner command is stale in planner mode")
            if self.latest_planner_msg.get("vr_3pt_position") is None:
                raise RuntimeError("VR 3-point position is missing in planner mode")
            if self.latest_planner_msg.get("vr_3pt_orientation") is None:
                raise RuntimeError("VR 3-point orientation is missing in planner mode")
        elif self.current_stream_mode in (1, 4):
            if self.latest_sonic_msg is None:
                raise RuntimeError("SMPL pose is unavailable in pose mode")
            pose_received = self.latest_sonic_msg.get("receive_monotonic")
            if pose_received is None or now - pose_received > self.teleop_max_age:
                raise RuntimeError("SMPL pose is stale in pose mode")
            if self.latest_sonic_msg.get("smpl_pose") is None:
                raise RuntimeError("SMPL pose is missing in pose mode")

    def _episode_validation(self) -> dict[str, object]:
        """Return the quality report used to accept or discard an episode."""
        errors = sorted(self._episode_input_errors)
        hand_motion = _episode_hand_motion_range(self.data_exporter.episode_buffer)
        if self.hand_config is not None and self.require_hand_activity:
            if hand_motion < self.minimum_hand_motion_rad:
                errors.append(
                    "hand commands did not move enough "
                    f"({hand_motion:.4f} rad < {self.minimum_hand_motion_rad:.4f} rad)"
                )

        rates = self.stream_rates.snapshot(self.RATE_STREAMS)
        required_streams = ["robot_state", "camera"]
        if self.hand_config is not None:
            required_streams.append("hand_state")
        modes = {
            int(np.asarray(value).reshape(-1)[0])
            for value in self.data_exporter.episode_buffer.get("teleop.stream_mode", [])
        }
        unexpected_modes = sorted(
            mode for mode in modes if mode != self.required_stream_mode
        )
        if unexpected_modes:
            required_name = _RECORDING_STREAM_MODE_NAMES[self.required_stream_mode]
            errors.append(
                f"episode contains stream modes {unexpected_modes}; "
                f"required {required_name} ({self.required_stream_mode}) only"
            )
        if modes & {5, 6}:
            required_streams.append("planner")
        if modes & {1, 4}:
            required_streams.append("pico_pose")
        for stream in required_streams:
            sent_hz = rates.get(stream, {}).get("sent_hz")
            if sent_hz is None or float(sent_hz) < self.minimum_recording_rate_hz:
                errors.append(
                    f"{stream} source rate is {sent_hz} Hz; "
                    f"minimum is {self.minimum_recording_rate_hz:.1f} Hz"
                )

        errors = sorted(set(errors))
        return {
            "passed": not errors,
            "errors": errors,
            "hand_command_range_rad": round(hand_motion, 6),
            "required_minimum_rate_hz": self.minimum_recording_rate_hz,
            "stream_rates": rates,
        }

    def _check_recording_commands(self):
        """Check keyboard + ZMQ toggle flags for recording commands."""
        key = self._keyboard_listener.read_msg()
        discard_reason = "operator_discarded"

        if isinstance(key, str) and key.startswith(DATASET_CONFIG_PREFIX):
            try:
                config = decode_dataset_config(key)
                if self._episode_state.get_state() != self._episode_state.IDLE:
                    raise RuntimeError("stop or discard the active episode before changing dataset")
                if self.data_exporter.episode_buffer.get("size", 0) > 0:
                    raise RuntimeError("cannot change dataset after collecting frames")
                if self.data_exporter.meta.info.get("total_episodes", 0) > 0:
                    raise RuntimeError("cannot change dataset after saving an episode")
                self.hub_uploader.configure(
                    repo_id=str(config["repo_id"]),
                    prompt=str(config["prompt"]),
                    private=bool(config["private"]),
                )
                self._recording_message = f"Ready to record to {config['repo_id']}"
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
        elif self._manager_toggle_dc:
            key = "c"
            self._manager_toggle_dc = False

        if key == "c":
            if (
                self._episode_state.get_state() == self._episode_state.IDLE
                and not _recording_mode_ready(
                    self.current_stream_mode, self.required_stream_mode
                )
            ):
                required_name = _RECORDING_STREAM_MODE_NAMES[self.required_stream_mode]
                message = f"Enter {required_name} with A+X before recording"
                self._recording_message = message
                self._print_and_say(message, blocking=False)
                return
            if (
                self._episode_state.get_state() == self._episode_state.IDLE
                and self.require_hub_upload
                and not self.hub_uploader.can_record()
            ):
                status = self.hub_uploader.status()
                message = (
                    "Wait for the Hugging Face upload to finish"
                    if status["ready"]
                    else "Choose a Hugging Face dataset in the browser first"
                )
                self._recording_message = message
                self._print_and_say(message, blocking=False)
                return
            self._episode_state.change_state()
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self._initial_yaw = None
                self._episode_input_errors.clear()
                self._recording_message = f"Recording episode {self.current_episode_index}"
                self._print_and_say(
                    f"Started recording {self.current_episode_index}", blocking=False
                )
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                self._recording_message = f"Saving episode {self.current_episode_index}"
                self._print_and_say("Stopping recording, preparing to save", blocking=False)
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                self._print_and_say("Saved episode and back to idle state", blocking=False)
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                buffer_size = self.data_exporter.episode_buffer.get("size", 0)
                if buffer_size > 0:
                    discarded_episode = self.current_episode_index
                    self.data_exporter.save_episode_as_discarded(
                        validation={
                            "passed": False,
                            "errors": [discard_reason],
                        }
                    )
                    if self.hub_uploader.status()["ready"]:
                        self.hub_uploader.enqueue(discarded_episode)
                        message = f"Episode {discarded_episode} discarded and queued for upload"
                    else:
                        message = f"Episode {discarded_episode} discarded"
                else:
                    # A discard can arrive before the first complete frame (for
                    # example while an external hand source is still starting).
                    # LeRobot rejects zero-frame episodes, so just return the
                    # recorder to idle in that case.
                    message = "Nothing discarded: no frames collected"
                self._episode_state.reset_state()
                self._initial_yaw = None
                self._recording_message = message
                self._print_and_say(message, blocking=False)

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
                self._manager_toggle_da = True
                self._manager_discard_reason = (
                    "required_teleop_mode_exited: "
                    f"expected {required_name} ({self.required_stream_mode}), "
                    f"got stream mode {new_stream_mode}"
                )
                self._recording_message = (
                    f"Discarding episode: exited {required_name} mode"
                )
            self.current_stream_mode = new_stream_mode
        self.latest_manager_msg = {
            "publisher_monotonic_ns": _integer_scalar(data.get("publisher_monotonic_ns")),
            "received_monotonic_ns": received_monotonic_ns,
        }

        if self._extract_bool(data, "toggle_data_collection"):
            self._manager_toggle_dc = True
        if self._extract_bool(data, "toggle_data_abort"):
            self._manager_toggle_da = True
            if self._manager_discard_reason is None:
                self._manager_discard_reason = "operator_discarded"

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

    def _add_images_to_frame_data(self, frame_data: dict) -> None:
        if self.latest_image_msg is None:
            return
        images = self.latest_image_msg["images"]
        for feature_name, feature_info in self.data_exporter.features.items():
            if feature_info.get("dtype") in ["image", "video"]:
                image_key = feature_name.split(".")[-1]
                if image_key not in images:
                    raise ValueError(
                        f"Required image '{image_key}' for feature '{feature_name}' "
                        f"not found in image message. Available: {list(images.keys())}"
                    )
                frame_data[feature_name] = images[image_key]

    def _finalize_frame(self, t_start: float) -> bool:
        t_end = time.monotonic()
        if t_end - t_start > (1 / self.frequency):
            print(f"DataExporter Missed: {t_end - t_start} sec")

        if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            saved_episode = self.current_episode_index
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                validation = self._episode_validation()
                if validation["passed"]:
                    self.data_exporter.save_episode(validation=validation)
                else:
                    self.data_exporter.save_episode_as_discarded(validation=validation)
                if self.hub_uploader.status()["ready"]:
                    self.hub_uploader.enqueue(saved_episode)
                self.sonic_timing_monitor.reset()
                self._initial_yaw = None
                if not validation["passed"]:
                    reasons = "; ".join(validation["errors"])
                    self._recording_message = (
                        f"Episode {saved_episode} failed validation and was discarded: {reasons}"
                    )
                    self._print_and_say(
                        f"Episode failed validation and was discarded: {reasons}",
                        blocking=False,
                    )
                elif self.hub_uploader.status()["ready"]:
                    self._recording_message = f"Saved episode {saved_episode}; upload queued"
                    self._print_and_say("Finished saving episode, upload started")
                else:
                    self._recording_message = f"Saved episode {saved_episode}"
                    self._print_and_say("Finished saving episode")
            else:
                self._recording_message = "Nothing saved: no frames collected"
                self._print_and_say("Skipping save: no frames collected", say=False)
            self._episode_state.change_state()
        return True

    def _add_data_frame(self):
        t_start = time.monotonic()

        if self._episode_state.get_state() != self._episode_state.RECORDING:
            return self._finalize_frame(t_start)

        try:
            self._validate_recording_inputs()
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
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
        return self._add_data_frame_sonic(t_start)

    def _add_data_frame_sonic(self, t_start: float) -> bool:
        """Build one data frame in Sonic CPP + SMPL mode."""
        assert self.latest_proprio_msg is not None
        proprio = self.latest_proprio_msg

        if self.hand_config is not None:
            (
                requested_left,
                requested_right,
                applied_left,
                applied_right,
                measured_left,
                measured_right,
            ) = self._external_hand_values()
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
            "observation.omnihand_left_raw": np.asarray(measured_left, dtype=np.float32),
            "observation.omnihand_right_raw": np.asarray(measured_right, dtype=np.float32),
            "action.omnihand_left_raw": np.asarray(requested_left, dtype=np.float32),
            "action.omnihand_right_raw": np.asarray(requested_right, dtype=np.float32),
            "observation.left_hand_valid": np.ones(1, dtype=np.uint8),
            "observation.right_hand_valid": np.ones(1, dtype=np.uint8),
            "episode.success": np.ones(1, dtype=np.uint8),
        }

        self._add_cpp_state_features(frame_data, proprio)

        sonic_latency_ms = self._add_sonic_pose_features(frame_data)

        if self.hand_config is not None:
            side_states = self.latest_hand_state["sides"]
            frame_data["teleop.left_hand_joints"] = requested_left.astype(np.float32)
            frame_data["teleop.right_hand_joints"] = requested_right.astype(np.float32)
            frame_data["control.hand_applied_position"] = np.concatenate(
                (applied_left, applied_right)
            ).astype(np.float32)
            frame_data["teleop.hand_closed"] = np.asarray(
                [side_states[side]["intent_closed"] for side in ("left", "right")],
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

        self._add_capture_features(frame_data, proprio)
        self._add_images_to_frame_data(frame_data)

        self._log_latency_periodic(sonic_latency_ms)

        self.data_exporter.add_frame(frame_data)
        return self._finalize_frame(t_start)

    def _add_capture_features(self, frame_data: dict, proprio: dict) -> None:
        """Preserve source identity/timing without exposing it to GR00T."""
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
            [self.latest_proprio_received_monotonic_ns or -1], dtype=np.int64
        )

        image = self.latest_image_msg or {}
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
            [self.latest_image_received_monotonic_ns or -1], dtype=np.int64
        )

        hand = self.latest_hand_state or {}
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
            [self.latest_hand_state_received_monotonic_ns or -1], dtype=np.int64
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

        pose = self.latest_sonic_msg or {}
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

        planner = self.latest_planner_msg or {}
        for capture_field, message_field in (("publish", "publisher"), ("received", "received")):
            frame_data[f"capture.planner_{capture_field}_monotonic_ns"] = np.asarray(
                [_integer_scalar(planner.get(f"{message_field}_monotonic_ns"))], dtype=np.int64
            )

        manager = self.latest_manager_msg or {}
        for capture_field, message_field in (("publish", "publisher"), ("received", "received")):
            frame_data[f"capture.manager_{capture_field}_monotonic_ns"] = np.asarray(
                [_integer_scalar(manager.get(f"{message_field}_monotonic_ns"))], dtype=np.int64
            )

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

    def _add_sonic_pose_features(self, frame_data: dict) -> float | None:
        """Add teleop features based on current stream mode."""
        sonic_latency_ms = None

        frame_data["teleop.stream_mode"] = np.array([self.current_stream_mode], dtype=np.int32)

        smpl_msg = self.latest_sonic_msg
        use_smpl = False
        if self.current_stream_mode in (1, 4) and smpl_msg is not None:
            receive_ts = smpl_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = time.time() - receive_ts
                sonic_latency_ms = age_sec * 1000
                self.sonic_timing_monitor.log_time_delta(age_sec)
                if sonic_latency_ms <= 100.0:
                    use_smpl = True
                elif (self.sonic_timing_monitor.failure_count + 1) % 10 == 0:
                    self._print_and_say(
                        f"Sonic pose stale ({sonic_latency_ms:.1f}ms old), using zeros",
                        say=False,
                    )
            else:
                use_smpl = True

        planner_msg = self.latest_planner_msg
        use_planner = False
        if self.current_stream_mode in (5, 6) and planner_msg is not None:
            receive_ts = planner_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = time.time() - receive_ts
                planner_latency_ms = age_sec * 1000
                if sonic_latency_ms is None:
                    sonic_latency_ms = planner_latency_ms
                if planner_latency_ms <= 200.0:
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
            smpl_msg if self.current_stream_mode in (1, 4) and smpl_msg is not None
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

    def save_and_cleanup(self):
        try:
            self._print_and_say("saving episode done", blocking=False)
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                saved_episode = self.current_episode_index
                self.data_exporter.save_episode_as_discarded(
                    validation={
                        "passed": False,
                        "errors": ["collector_shutdown_before_episode_save"],
                    }
                )
                if self.hub_uploader.status()["ready"]:
                    self.hub_uploader.enqueue(saved_episode)
            if not self.hub_uploader.wait_until_idle(timeout=30.0):
                print("[Hub] Upload still pending after 30 seconds; local data is preserved")
            self._print_and_say(
                f"Recording complete: {self.data_exporter.meta.root}", say=False, blocking=True
            )
        except Exception as e:
            self._print_and_say(f"Error saving episode: {e}", blocking=True)

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

        self._print_and_say("Shutting down data exporter...", say=False)

    def run(self):
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
                        img_msg = self._image_subscriber.read()
                        if img_msg is not None:
                            self.latest_image_msg = img_msg
                        if self._image_subscriber.idx != previous_image_index and img_msg is not None:
                            self.latest_image_received_at = time.monotonic()
                            self.latest_image_received_monotonic_ns = time.monotonic_ns()
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
                            )

                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                    with self.telemetry.timer("check_recording_commands"):
                        self._check_recording_commands()

                    self._publish_recording_status()

                    end_time = time.monotonic()

                elapsed = time.monotonic() - t_start
                sleep_time = self.loop_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if (end_time - t_start) > self.loop_period:
                    self.telemetry.log_timing_info(
                        context="Data Exporter Loop Missed", threshold=0.001
                    )

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                discarded_episode = self.current_episode_index
                self.data_exporter.save_episode_as_discarded()
                if self.hub_uploader.status()["ready"]:
                    self.hub_uploader.enqueue(discarded_episode)

        finally:
            self.save_and_cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config: SonicDataExporterConfig):
    g1_rm = get_g1_robot_model()

    robot_config = poll_robot_config_zmq(
        config.state_zmq_host, config.state_zmq_port, config.robot_config_timeout
    )
    hand_config = None
    profile_name = config.hand_profile
    if profile_name == "auto":
        profile_name = (
            "omnihand_o10.v1"
            if robot_config.get("hand_control") == "external"
            else "dex3.v1"
        )
    hand_profile = get_hand_profile(profile_name)
    if robot_config.get("hand_control") == "external":
        hand_config = poll_hand_config_zmq(
            config.hand_state_host, config.hand_state_port, config.hand_config_timeout
        )
        if hand_config.get("profile") != hand_profile.name:
            raise RuntimeError(
                f"requested {hand_profile.name}, controller reports {hand_config.get('profile')}"
            )

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

    text_to_speech = TextToSpeech() if config.text_to_speech else None

    data_exporter = Gr00tDataExporter.create(
        save_root=f"{config.root_output_dir}/{config.dataset_name}",
        fps=config.data_collection_frequency,
        features=dataset_features,
        modality_config=modality_config,
        task=config.task_prompt,
        script_config={
            **robot_config,
            "record_wrist_cameras": config.record_wrist_cameras,
            "hand_profile": hand_profile.name,
            "hand_config": hand_config,
            "capture": _capture_reproducibility_metadata(
                robot_config, config.data_collection_frequency
            ),
        },
        robot_type="unitree_g1_omnihand_sonic",
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
    )
    data_collector.run()


if __name__ == "__main__":
    config = tyro.cli(SonicDataExporterConfig)

    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    main(config)
