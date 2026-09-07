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
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro
import zmq

from gear_sonic.camera.composed_camera import (
    ComposedCameraClientSensor,
    estimate_camera_age_s,
)
from gear_sonic.data.causal_sync import CausalSelection, CausalSynchronizer
from gear_sonic.data.episode_finalizer import (
    EpisodeFinalizationResult,
    EpisodeFinalizer,
)
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
from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity, quat_to_rot6d
from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    ZMQStateSubscriber,
    poll_robot_config_zmq,
)

SONIC_STREAM_MODES = frozenset({1})
PLANNER_STREAM_MODES = frozenset({2, 3, 5})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SonicDataExporterConfig:
    """CLI config for the ROS-free Sonic data exporter."""

    # Dataset
    dataset_name: str | None = None
    """Dataset name (auto-generated if creating new)."""

    task_prompt: str = "demo"
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

    camera_max_age: float = 0.25
    """Maximum age of every required camera frame while recording."""

    minimum_camera_rate_hz: float = 25.0
    """Minimum live camera publish rate admitted while recording."""

    finalizer_shutdown_timeout: float = 30.0
    """Maximum shutdown wait for an episode commit."""

    synchronization_delay: float = 0.1
    """Seconds the recorder runs behind Thor time for causal stream alignment."""

    synchronization_wait_timeout: float = 0.25
    """Additional wait for every required stream to advance past a target."""

    proprio_max_age: float = 0.1
    """Maximum age of the selected past robot-state sample."""

    teleop_max_age: float = 0.2
    """Maximum age of selected past manager, pose, and planner samples."""

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

    recording_status_port: int = 5581
    """ZMQ PUB port for browser-visible recorder status."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TimeDeltaException(Exception):
    def __init__(self, failure_count: int, reset_timeout_sec: float):
        self.failure_count = failure_count
        self.reset_timeout_sec = reset_timeout_sec
        self.message = f"{self.failure_count} failures in {self.reset_timeout_sec} seconds"
        super().__init__(self.message)


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

    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        data_exporter: Gr00tDataExporter,
        robot_model,
        text_to_speech=None,
        frequency: int = 20,
        sonic_data_zmq_host: str = "localhost",
        sonic_data_zmq_port: int = 5556,
        state_zmq_host: str = "localhost",
        state_zmq_port: int = 5557,
        hand_profile: HandProfile | None = None,
        hand_config: dict | None = None,
        hand_state_host: str = "localhost",
        hand_state_port: int = 5570,
        hand_state_max_age: float = .2,
        recording_status_port: int = 5581,
        camera_max_age: float = 0.25,
        minimum_camera_rate_hz: float = 25.0,
        finalizer_shutdown_timeout: float = 30.0,
        synchronization_delay: float = 0.1,
        synchronization_wait_timeout: float = 0.25,
        proprio_max_age: float = 0.1,
        teleop_max_age: float = 0.2,
    ):
        self.text_to_speech = text_to_speech
        self.frequency = frequency
        self.loop_period = 1.0 / frequency
        self.loop_period_ns = round(1e9 / frequency)
        self.data_exporter = data_exporter
        self.robot_model = robot_model
        self.hand_profile = hand_profile
        self.hand_config = hand_config
        self.hand_state_max_age = hand_state_max_age
        if camera_max_age <= 0:
            raise ValueError("camera_max_age must be positive")
        if finalizer_shutdown_timeout <= 0:
            raise ValueError("finalizer_shutdown_timeout must be positive")
        if minimum_camera_rate_hz <= 0:
            raise ValueError("minimum_camera_rate_hz must be positive")
        if synchronization_delay <= 0:
            raise ValueError("synchronization_delay must be positive")
        if synchronization_wait_timeout <= 0:
            raise ValueError("synchronization_wait_timeout must be positive")
        if proprio_max_age <= 0:
            raise ValueError("proprio_max_age must be positive")
        if teleop_max_age <= 0:
            raise ValueError("teleop_max_age must be positive")
        self.camera_max_age = camera_max_age
        self.minimum_camera_rate_hz = minimum_camera_rate_hz
        self.finalizer_shutdown_timeout = finalizer_shutdown_timeout
        self.synchronization_delay_ns = int(synchronization_delay * 1e9)
        self.synchronization_wait_timeout_ns = int(
            synchronization_wait_timeout * 1e9
        )
        self.proprio_max_age_ns = int(proprio_max_age * 1e9)
        self.teleop_max_age_ns = int(teleop_max_age * 1e9)
        self._synchronizer = CausalSynchronizer(max_samples_per_stream=32)
        self._recording_start_target_ns: int | None = None
        self._next_target_ns: int | None = None
        self._recording_stop_target_ns: int | None = None
        self._synchronization_errors: list[str] = []
        self._synchronization_skipped_targets = 0
        self.latest_hand_state = None

        self._episode_state = EpisodeState()
        self._keyboard_listener = ZMQKeyboardSubscriber()
        self.episode_finalizer = EpisodeFinalizer(data_exporter, max_pending=1)
        self._last_finalization: EpisodeFinalizationResult | None = None
        self._recording_message = "Ready to record"
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
        self._episode_camera_stats_start: dict[str, object] = {}
        self.latest_proprio_msg = None

        self.current_stream_mode = 0

        self._manager_toggle_dc = False
        self._manager_toggle_da = False

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

    def _required_camera_names(self) -> set[str]:
        return {
            feature_name.rsplit(".", 1)[-1]
            for feature_name, feature in self.data_exporter.features.items()
            if feature.get("dtype") in {"image", "video"}
        }

    def _camera_health(self) -> dict[str, object]:
        now = time.monotonic()
        receiver_age = (
            None
            if self.latest_image_received_at is None
            else max(0.0, now - self.latest_image_received_at)
        )
        images = (self.latest_image_msg or {}).get("images", {})
        camera_ages = {}
        missing = []
        stale = []
        for camera_name in sorted(self._required_camera_names()):
            if images.get(camera_name) is None:
                missing.append(camera_name)
                continue
            age = estimate_camera_age_s(self.latest_image_msg, camera_name)
            if age is None:
                age = receiver_age
            camera_ages[camera_name] = None if age is None else round(age, 4)
            if age is None or age > self.camera_max_age:
                stale.append(camera_name)
        buffer = self._image_subscriber.buffer_stats()
        measured_rate = buffer.get("publisher_hz") or buffer.get("received_hz")
        rate_ready = (
            isinstance(measured_rate, (int, float))
            and measured_rate >= self.minimum_camera_rate_hz
        )
        ready = (
            receiver_age is not None
            and receiver_age <= self.camera_max_age
            and not missing
            and not stale
            and rate_ready
        )
        return {
            "ready": ready,
            "receiver_age_s": (
                None if receiver_age is None else round(receiver_age, 4)
            ),
            "camera_age_s": camera_ages,
            "missing": missing,
            "stale": stale,
            "buffer": buffer,
            "minimum_rate_hz": self.minimum_camera_rate_hz,
            "rate_ready": rate_ready,
        }

    def _consume_finalizer_results(self) -> None:
        for result in self.episode_finalizer.drain_results():
            self._last_finalization = result
            if result.succeeded:
                outcome = "discarded" if result.discarded else "saved"
                message = f"Episode {result.episode_index} {outcome}"
                self._print_and_say(message, blocking=False)
            else:
                message = f"Episode {result.episode_index} save failed: {result.error}"
                if result.recovery_path:
                    message += f"; recovery: {result.recovery_path}"
                self._print_and_say(message, say=False)
            if self._episode_state.get_state() == self._episode_state.IDLE:
                self._recording_message = message

    def _publish_recording_status(self) -> None:
        """Publish authoritative recorder state for the loopback browser UI."""
        self._consume_finalizer_results()
        state = self._episode_state.get_state()
        finalizer = self.episode_finalizer.status()
        camera = self._camera_health()
        hand_ready = True
        if self.hand_config is not None:
            hand_ready = bool(
                self.latest_hand_state
                and time.monotonic_ns() - self.latest_hand_state.get("received_monotonic_ns", 0)
                <= self.hand_state_max_age * 1e9
                and self.latest_hand_state.get("mode") != "fault"
                and not self.latest_hand_state.get("input_stale", True)
                and self.latest_hand_state.get("intent_sequence") is not None
                and all(
                    side.get("valid") and side.get("connected")
                    for side in self.latest_hand_state.get("sides", {}).values()
                )
            )
        payload = {
            "state": state,
            "recording": state == self._episode_state.RECORDING,
            "draining": state == self._episode_state.NEED_TO_SAVE,
            "saving": state == self._episode_state.NEED_TO_SAVE
            or bool(finalizer["pending"]),
            "episode_index": self.current_episode_index,
            "frame_count": self.data_exporter.episode_buffer.get("size", 0),
            "total_episodes": self.data_exporter.meta.info.get("total_episodes", 0),
            "dataset_root": str(self.data_exporter.meta.root),
            "sources": {
                "proprio": self.latest_proprio_msg is not None,
                "camera": camera["ready"],
                "hands": hand_ready,
            },
            "camera": camera,
            "synchronization": {
                "delay_ms": self.synchronization_delay_ns / 1e6,
                "next_target_monotonic_ns": self._next_target_ns,
                "skipped_targets": self._synchronization_skipped_targets,
                "errors": self._synchronization_errors[-5:],
                "buffers": self._synchronizer.status(),
            },
            "finalizer": finalizer,
            "last_finalization": (
                None
                if self._last_finalization is None
                else {
                    "episode_index": self._last_finalization.episode_index,
                    "discarded": self._last_finalization.discarded,
                    "succeeded": self._last_finalization.succeeded,
                    "error": self._last_finalization.error,
                    "recovery_path": self._last_finalization.recovery_path,
                }
            ),
            "message": self._recording_message,
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

        if msg.get("ros_timestamp", 0.0) == 0.0:
            msg["ros_timestamp"] = time.time()

        received_ns = time.monotonic_ns()
        msg["received_monotonic_ns"] = received_ns
        self.latest_proprio_msg = msg
        self._synchronizer.observe("proprio", msg, received_ns)

    def _poll_images(self) -> None:
        for message in self._image_subscriber.read_pending():
            received_ns = message["receiver_monotonic_ns"]
            self.latest_image_msg = message
            self.latest_image_received_at = received_ns / 1e9
            self._synchronizer.observe("camera", message, received_ns)

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
            received_ns = time.monotonic_ns()
            state["received_monotonic_ns"] = received_ns
            previous = self.latest_hand_state
            self.latest_hand_state = state
            if previous is not None and state.get("connection_id") != previous.get("connection_id"):
                if self._next_target_ns is not None and len(self._synchronization_errors) < 100:
                    self._synchronization_errors.append("hand reconnected during recording")
            self._synchronizer.observe("hand", state, received_ns)

    def _external_hand_values(
        self,
        hand_state: dict,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if hand_state.get("mode") == "fault":
            raise RuntimeError("external hand controller is faulted")
        if hand_state.get("input_stale") or hand_state.get("intent_sequence") is None:
            raise RuntimeError("external hand target is missing or stale")
        for side in ("left", "right"):
            if hand_state.get("sides", {}).get(side, {}).get("intent_closed") is None:
                raise RuntimeError(f"external {side} hand has no valid click intent")
        values = []
        for field in ("requested_position_rad", "applied_position_rad", "measured_position_rad"):
            for side in ("left", "right"):
                side_state = hand_state.get("sides", {}).get(side)
                if not side_state or not side_state.get("valid") or not side_state.get("connected"):
                    raise RuntimeError(f"external {side} hand is invalid or disconnected")
                array = np.asarray(side_state.get(field), dtype=np.float64).reshape(-1)
                if array.shape != (self.hand_profile.width,) or not np.all(np.isfinite(array)):
                    raise RuntimeError(f"external {side} {field} has the wrong shape")
                values.append(array)
        return tuple(values)

    def _episode_validation(self, *, discarded: bool, reason: str) -> dict[str, object]:
        current_stats = self._image_subscriber.buffer_stats()
        drop_deltas = {
            key: int(current_stats.get(key, 0))
            - int(self._episode_camera_stats_start.get(key, 0))
            for key in (
                "received",
                "overflow_dropped",
                "latency_dropped",
                "publisher_gap_dropped",
                "publisher_resets",
            )
        }
        errors = list(self._synchronization_errors)
        if discarded and reason:
            errors.append(reason)
        return {
            "passed": not discarded and not errors,
            "errors": errors,
            "synchronization": {
                "delay_ms": self.synchronization_delay_ns / 1e6,
                "wait_timeout_ms": self.synchronization_wait_timeout_ns / 1e6,
                "skipped_targets": self._synchronization_skipped_targets,
                "buffers": self._synchronizer.status(),
            },
            "camera_buffer": {
                **current_stats,
                "episode_deltas": drop_deltas,
            },
            "camera_health_at_stop": self._camera_health(),
        }

    def _finish_recording(self, *, discarded: bool, reason: str) -> None:
        episode_index = self.current_episode_index
        buffer_size = self.data_exporter.episode_buffer.get("size", 0)
        if buffer_size <= 0:
            self._episode_state.reset_state()
            self._initial_yaw = None
            self._recording_start_target_ns = None
            self._next_target_ns = None
            self._recording_stop_target_ns = None
            self._recording_message = "Nothing saved: no frames collected"
            self._print_and_say("Skipping empty recording", say=False)
            return

        validation = self._episode_validation(discarded=discarded, reason=reason)
        effective_discarded = discarded or not validation["passed"]
        episode_buffer, video_writers = self.data_exporter.detach_episode()
        try:
            self.episode_finalizer.enqueue(
                episode_index=episode_index,
                episode_buffer=episode_buffer,
                video_writers=video_writers,
                discarded=effective_discarded,
                validation=validation,
            )
        except Exception:
            # Ownership transfers only after the finalizer accepts the job.
            self.data_exporter.episode_buffer = episode_buffer
            self.data_exporter.video_writers = video_writers
            raise
        self.sonic_timing_monitor.reset()
        self._episode_state.reset_state()
        self._initial_yaw = None
        self._recording_start_target_ns = None
        self._next_target_ns = None
        self._recording_stop_target_ns = None
        outcome = "discard" if effective_discarded else "save"
        self._recording_message = (
            f"Episode {episode_index} queued for background {outcome}"
        )
        self._print_and_say(self._recording_message, say=False)

    def _check_recording_commands(self):
        """Check keyboard + ZMQ toggle flags for recording commands."""
        key = self._keyboard_listener.read_msg()

        if self._manager_toggle_da:
            key = "x"
            self._manager_toggle_da = False
        elif self._manager_toggle_dc:
            key = "c"
            self._manager_toggle_dc = False

        state = self._episode_state.get_state()
        if key == "c":
            if state == self._episode_state.IDLE:
                if not self.episode_finalizer.can_accept():
                    self._recording_message = (
                        "Cannot start: previous episode is still finalizing or failed"
                    )
                    self._print_and_say(self._recording_message, say=False)
                    return
                if not self._camera_health()["ready"]:
                    self._recording_message = "Cannot start: camera stream is not ready"
                    self._print_and_say(self._recording_message, say=False)
                    return
                started_ns = time.monotonic_ns()
                self._episode_state.change_state()
                self._initial_yaw = None
                self._recording_start_target_ns = started_ns
                self._next_target_ns = started_ns
                self._recording_stop_target_ns = None
                self._synchronization_errors = []
                self._synchronization_skipped_targets = 0
                self._episode_camera_stats_start = (
                    self._image_subscriber.buffer_stats()
                )
                self._recording_message = f"Recording episode {self.current_episode_index}"
                self._print_and_say(
                    f"Started recording {self.current_episode_index}", blocking=False
                )
            elif state == self._episode_state.RECORDING:
                self._recording_stop_target_ns = time.monotonic_ns()
                self._episode_state.change_state()
                self._recording_message = (
                    f"Draining synchronized episode {self.current_episode_index}"
                )
        elif key == "x" and state in (
            self._episode_state.RECORDING,
            self._episode_state.NEED_TO_SAVE,
        ):
            self._finish_recording(
                discarded=True,
                reason="operator requested discard",
            )

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

            received_ns = time.monotonic_ns()
            if raw.startswith(b"manager_state"):
                self._handle_manager_state(raw, received_ns)
            elif raw.startswith(b"planner"):
                self._handle_planner_message(raw, received_ns)
            elif raw.startswith(b"pose"):
                self._handle_pose_message(raw, received_ns)

    def _handle_manager_state(self, raw: bytes, received_ns: int | None = None) -> None:
        try:
            data = unpack_pose_message(raw, topic="manager_state")
        except Exception:
            return

        received_ns = time.monotonic_ns() if received_ns is None else received_ns
        stream_mode = self.current_stream_mode
        if "stream_mode" in data:
            stream_mode = int(data["stream_mode"].flat[0])
            self.current_stream_mode = stream_mode
        manager_message = {
            "stream_mode": stream_mode,
            "received_monotonic_ns": received_ns,
        }
        self._synchronizer.observe("manager", manager_message, received_ns)

        if self._extract_bool(data, "toggle_data_collection"):
            self._manager_toggle_dc = True
        if self._extract_bool(data, "toggle_data_abort"):
            self._manager_toggle_da = True

    def _handle_planner_message(self, raw: bytes, received_ns: int | None = None) -> None:
        try:
            data = unpack_pose_message(raw, topic="planner")
        except Exception:
            return

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

        received_ns = time.monotonic_ns() if received_ns is None else received_ns
        planner_message = {
            "planner_mode": planner_mode,
            "planner_movement": planner_movement,
            "planner_facing": planner_facing,
            "planner_speed": planner_speed,
            "planner_height": planner_height,
            "vr_3pt_position": vr_3pt_position,
            "vr_3pt_orientation": vr_3pt_orientation,
            "left_hand_joints": self._extract_hand_joints(data, "left_hand_joints"),
            "right_hand_joints": self._extract_hand_joints(data, "right_hand_joints"),
            "received_monotonic_ns": received_ns,
        }
        self._synchronizer.observe("planner", planner_message, received_ns)

    def _handle_pose_message(self, raw: bytes, received_ns: int | None = None) -> None:
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

            received_ns = time.monotonic_ns() if received_ns is None else received_ns
            sonic_message = {
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
                "received_monotonic_ns": received_ns,
            }
            self._synchronizer.observe("sonic", sonic_message, received_ns)
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

    def _add_images_to_frame_data(
        self,
        frame_data: dict,
        image_message: dict,
    ) -> None:
        images = image_message["images"]
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
        return True

    def _synchronization_limits(self) -> dict[str, int]:
        return {
            "proprio": self.proprio_max_age_ns,
            "camera": int(self.camera_max_age * 1e9),
            "manager": self.teleop_max_age_ns,
            "sonic": self.teleop_max_age_ns,
            "planner": self.teleop_max_age_ns,
            "hand": int(self.hand_state_max_age * 1e9),
        }

    def _selection_for_target(self, target_ns: int) -> CausalSelection:
        required = ["proprio", "camera", "manager"]
        if self.hand_config is not None:
            required.append("hand")
        base = self._synchronizer.select(
            target_ns,
            required_streams=tuple(required),
            max_age_ns=self._synchronization_limits(),
        )
        if not base.ready:
            return base
        stream_mode = int(base.samples["manager"].value["stream_mode"])
        if stream_mode in SONIC_STREAM_MODES:
            required.append("sonic")
        elif stream_mode in PLANNER_STREAM_MODES:
            required.append("planner")
        return self._synchronizer.select(
            target_ns,
            required_streams=tuple(required),
            max_age_ns=self._synchronization_limits(),
        )

    def _selected_camera_errors(self, selection: CausalSelection) -> list[str]:
        camera = selection.samples.get("camera")
        if camera is None:
            return ["camera has no causal sample"]
        message = camera.value
        images = message.get("images", {})
        errors = []
        buffer_stats = self._image_subscriber.buffer_stats()
        measured_rate = buffer_stats.get("publisher_hz") or buffer_stats.get(
            "received_hz"
        )
        if not isinstance(measured_rate, (int, float)) or (
            measured_rate < self.minimum_camera_rate_hz
        ):
            errors.append(
                f"camera rate {measured_rate or 0:.1f} Hz is below "
                f"{self.minimum_camera_rate_hz:.1f} Hz"
            )
        for camera_name in sorted(self._required_camera_names()):
            if images.get(camera_name) is None:
                errors.append(f"camera {camera_name} is missing")
                continue
            age = estimate_camera_age_s(
                message,
                camera_name,
                now_monotonic_ns=selection.target_ns,
            )
            if age is None:
                age = (selection.target_ns - camera.timestamp_ns) / 1e9
            if age > self.camera_max_age:
                errors.append(f"camera {camera_name} is {age * 1000:.1f} ms old")
        return errors

    def _selected_hand_errors(self, selection: CausalSelection) -> list[str]:
        if self.hand_config is None:
            return []
        hand = selection.samples.get("hand")
        if hand is None:
            return ["hand has no causal sample"]
        try:
            self._external_hand_values(hand.value)
        except RuntimeError as exc:
            return [str(exc)]
        return []

    def _record_synchronization_gap(
        self,
        target_ns: int,
        reasons: list[str],
        now_ns: int,
    ) -> None:
        relative_ms = (
            0.0
            if self._recording_start_target_ns is None
            else (target_ns - self._recording_start_target_ns) / 1e6
        )
        detail = f"target {relative_ms:.1f} ms: {'; '.join(reasons)}"
        if len(self._synchronization_errors) < 100:
            self._synchronization_errors.append(detail)
        start_ns = self._recording_start_target_ns or target_ns
        earliest_ns = max(target_ns + self.loop_period_ns, now_ns - self.synchronization_delay_ns)
        tick = max(0, (earliest_ns - start_ns + self.loop_period_ns - 1) // self.loop_period_ns)
        resynchronized_ns = start_ns + tick * self.loop_period_ns
        skipped = max(1, (resynchronized_ns - target_ns) // self.loop_period_ns)
        self._synchronization_skipped_targets += skipped
        self._next_target_ns = resynchronized_ns
        print(f"[Synchronization] {detail}; skipped {skipped} target(s)")

    def _add_data_frame(self):
        t_start = time.monotonic()
        state = self._episode_state.get_state()
        if state not in (
            self._episode_state.RECORDING,
            self._episode_state.NEED_TO_SAVE,
        ):
            # Avoid retaining a full history of decoded images between episodes.
            self._synchronizer.trim_through(time.monotonic_ns())
            return self._finalize_frame(t_start)
        if self._next_target_ns is None:
            return False
        if (
            self._recording_stop_target_ns is not None
            and self._next_target_ns > self._recording_stop_target_ns
        ):
            self._finish_recording(discarded=False, reason="")
            return True

        now_ns = time.monotonic_ns()
        target_ns = self._next_target_ns
        if now_ns < target_ns + self.synchronization_delay_ns:
            return False

        selection = self._selection_for_target(target_ns)
        sample_errors = []
        if selection.ready:
            sample_errors.extend(self._selected_camera_errors(selection))
            sample_errors.extend(self._selected_hand_errors(selection))
        if not selection.ready or sample_errors:
            deadline_ns = (
                target_ns
                + self.synchronization_delay_ns
                + self.synchronization_wait_timeout_ns
            )
            if selection.waiting and not sample_errors and now_ns < deadline_ns:
                return False
            reasons = []
            if selection.waiting:
                reasons.append(f"streams did not advance: {', '.join(selection.waiting)}")
            if selection.missing:
                reasons.append(f"no past sample: {', '.join(selection.missing)}")
            if selection.stale:
                reasons.append(f"past sample too old: {', '.join(selection.stale)}")
            reasons.extend(sample_errors)
            self._record_synchronization_gap(target_ns, reasons, now_ns)
            return False

        self._add_data_frame_sonic(t_start, selection)
        self._synchronizer.trim_through(target_ns)
        self._next_target_ns += self.loop_period_ns
        return True

    def _add_data_frame_sonic(
        self,
        t_start: float,
        selection: CausalSelection,
    ) -> bool:
        """Build one data frame in Sonic CPP + SMPL mode."""
        proprio = selection.samples["proprio"].value
        image_message = selection.samples["camera"].value
        stream_mode = int(selection.samples["manager"].value["stream_mode"])
        hand_state = (
            selection.samples["hand"].value if "hand" in selection.samples else None
        )
        sonic_message = (
            selection.samples["sonic"].value if "sonic" in selection.samples else None
        )
        planner_message = (
            selection.samples["planner"].value if "planner" in selection.samples else None
        )

        if self.hand_config is not None:
            (
                requested_left,
                requested_right,
                applied_left,
                applied_right,
                measured_left,
                measured_right,
            ) = self._external_hand_values(hand_state)
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
            whole_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=proprio["body_q"],
                left_hand_actuated_joint_values=proprio["left_hand_q"],
                right_hand_actuated_joint_values=proprio["right_hand_q"],
            )
            whole_action_wbc = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=proprio["last_action"],
                left_hand_actuated_joint_values=proprio["last_left_hand_action"],
                right_hand_actuated_joint_values=proprio["last_right_hand_action"],
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
            "observation.state": whole_q,
            "observation.eef_state": observation_eef_state,
            "action.wbc": whole_action_wbc,
        }

        self._add_cpp_state_features(frame_data, proprio)

        sonic_latency_ms = self._add_sonic_pose_features(
            frame_data,
            stream_mode=stream_mode,
            smpl_msg=sonic_message,
            planner_msg=planner_message,
            target_ns=selection.target_ns,
        )

        if self.hand_config is not None:
            side_states = hand_state["sides"]
            frame_data["teleop.left_hand_joints"] = requested_left.astype(np.float32)
            frame_data["teleop.right_hand_joints"] = requested_right.astype(np.float32)
            frame_data["control.hand_applied_position"] = np.concatenate(
                (applied_left, applied_right)
            )
            frame_data["teleop.hand_closed"] = np.asarray(
                [side_states[side]["intent_closed"] for side in ("left", "right")],
                dtype=bool,
            )
        else:
            frame_data["control.hand_applied_position"] = np.concatenate(
                (
                    np.asarray(proprio["last_left_hand_action"], dtype=np.float64),
                    np.asarray(proprio["last_right_hand_action"], dtype=np.float64),
                )
            )
            frame_data["teleop.hand_closed"] = np.zeros(2, dtype=bool)

        self._add_images_to_frame_data(frame_data, image_message)
        self._add_synchronization_features(frame_data, selection)

        self._log_latency_periodic(sonic_latency_ms)

        self.data_exporter.add_frame(frame_data)
        return self._finalize_frame(t_start)

    def _add_synchronization_features(
        self,
        frame_data: dict,
        selection: CausalSelection,
    ) -> None:
        def add(name: str, value: np.ndarray) -> None:
            # Existing datasets retain their original immutable schema.
            if name in self.data_exporter.features:
                frame_data[name] = value

        add(
            "capture.sync_target_monotonic_ns",
            np.asarray([selection.target_ns], dtype=np.int64),
        )
        camera_message = selection.samples["camera"].value
        sequence = camera_message.get("publisher_sequence")
        add(
            "capture.camera_sequence",
            np.asarray(
                [
                    sequence
                    if isinstance(sequence, int) and not isinstance(sequence, bool)
                    else -1
                ],
                dtype=np.int64,
            ),
        )
        capture_ages = []
        for camera_name in ("ego_view", "left_wrist", "right_wrist"):
            age = None
            if camera_message.get("images", {}).get(camera_name) is not None:
                age = estimate_camera_age_s(
                    camera_message,
                    camera_name,
                    now_monotonic_ns=selection.target_ns,
                )
            capture_ages.append(-1.0 if age is None else age * 1000)
        add(
            "capture.camera_capture_age_ms",
            np.asarray(capture_ages, dtype=np.float32),
        )
        for stream in ("proprio", "camera", "manager", "sonic", "planner", "hand"):
            sample = selection.samples.get(stream)
            timestamp_ns = -1 if sample is None else sample.timestamp_ns
            age_ms = -1.0 if sample is None else (selection.target_ns - timestamp_ns) / 1e6
            add(
                f"capture.{stream}_received_monotonic_ns",
                np.asarray([timestamp_ns], dtype=np.int64),
            )
            add(
                f"capture.{stream}_age_ms",
                np.asarray([age_ms], dtype=np.float32),
            )

    def _add_cpp_state_features(self, frame_data: dict, proprio: dict) -> None:
        if "base_quat" in proprio:
            base_quat = np.asarray(proprio["base_quat"], dtype=np.float64)
            frame_data["observation.root_orientation"] = base_quat
            frame_data["observation.projected_gravity"] = compute_projected_gravity(
                base_quat
            ).astype(np.float64)

            if "init_ref_data_root_rot_array" in proprio:
                frame_data["observation.cpp_rotation_offset"] = np.asarray(
                    proprio["init_ref_data_root_rot_array"], dtype=np.float64
                )
            else:
                frame_data["observation.cpp_rotation_offset"] = np.array(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
                )
        else:
            frame_data["observation.root_orientation"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            frame_data["observation.projected_gravity"] = np.array(
                [0.0, 0.0, -1.0], dtype=np.float64
            )
            frame_data["observation.cpp_rotation_offset"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        if "init_base_quat" in proprio:
            frame_data["observation.init_base_quat"] = np.asarray(
                proprio["init_base_quat"], dtype=np.float64
            )
        else:
            frame_data["observation.init_base_quat"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        if "delta_heading" in proprio:
            dh = proprio["delta_heading"]
            if isinstance(dh, np.ndarray):
                dh = dh.item() if dh.size == 1 else dh[0]
            frame_data["teleop.delta_heading"] = np.array([float(dh)], dtype=np.float64)
        else:
            frame_data["teleop.delta_heading"] = np.zeros(1, dtype=np.float64)

        if "token_state" in proprio:
            frame_data["action.motion_token"] = np.asarray(proprio["token_state"], dtype=np.float64)
        else:
            frame_data["action.motion_token"] = np.zeros(64, dtype=np.float64)

    def _add_sonic_pose_features(
        self,
        frame_data: dict,
        *,
        stream_mode: int,
        smpl_msg: dict | None,
        planner_msg: dict | None,
        target_ns: int,
    ) -> float | None:
        """Add the causal teleop sample selected for the target time."""
        sonic_latency_ms = None

        frame_data["teleop.stream_mode"] = np.array([stream_mode], dtype=np.int32)

        use_smpl = stream_mode in SONIC_STREAM_MODES and smpl_msg is not None
        if use_smpl:
            received_ns = smpl_msg.get("received_monotonic_ns")
            if isinstance(received_ns, int):
                sonic_latency_ms = max(0.0, (target_ns - received_ns) / 1e6)
                self.sonic_timing_monitor.log_time_delta(sonic_latency_ms / 1000)

        use_planner = stream_mode in PLANNER_STREAM_MODES and planner_msg is not None
        if use_planner and sonic_latency_ms is None:
            received_ns = planner_msg.get("received_monotonic_ns")
            if isinstance(received_ns, int):
                sonic_latency_ms = max(0.0, (target_ns - received_ns) / 1e6)

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

        hand_msg = (
            smpl_msg if stream_mode in SONIC_STREAM_MODES and smpl_msg is not None
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

        # VR 3-point pose
        frame_data["teleop.vr_3pt_position"] = (
            planner_msg["vr_3pt_position"].astype(np.float32)
            if use_planner and planner_msg.get("vr_3pt_position") is not None
            else np.zeros(9, dtype=np.float32)
        )
        if use_planner and planner_msg.get("vr_3pt_orientation") is not None:
            frame_data["teleop.vr_3pt_orientation"] = quat_to_rot6d(
                planner_msg["vr_3pt_orientation"].astype(np.float32)
            )
        else:
            frame_data["teleop.vr_3pt_orientation"] = np.zeros(18, dtype=np.float32)

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
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self._finish_recording(
                    discarded=True,
                    reason="collector shut down before an explicit save",
                )
            self.episode_finalizer.close(
                timeout=self.finalizer_shutdown_timeout
            )
            self._consume_finalizer_results()
            self._print_and_say(
                f"Recording complete: {self.data_exporter.meta.root}", say=False, blocking=True
            )
        except Exception as e:
            self._print_and_say(f"Error finalizing episode: {e}", blocking=True)

        for writer in self.data_exporter.video_writers.values():
            try:
                if self.data_exporter.episode_buffer.get("size", 0):
                    writer.stop(timeout_s=5.0)
                else:
                    writer.cancel(timeout_s=5.0)
            except Exception as exc:
                print(f"[Exporter] Could not close video writer: {exc}")

        try:
            self._image_subscriber.close()
        except Exception as exc:
            print(f"[Camera] Could not close receiver cleanly: {exc}")

        try:
            self._state_subscriber.close()
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
                        self._poll_images()

                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                    with self.telemetry.timer("check_recording_commands"):
                        self._check_recording_commands()

                    self._publish_recording_status()

                    end_time = time.monotonic()

                # Absolute pacing avoids accumulating time.sleep wake-up drift.
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
        },
    )

    data_collector = GrootDataCollector(
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=g1_rm,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        camera_max_age=config.camera_max_age,
        minimum_camera_rate_hz=config.minimum_camera_rate_hz,
        finalizer_shutdown_timeout=config.finalizer_shutdown_timeout,
        synchronization_delay=config.synchronization_delay,
        synchronization_wait_timeout=config.synchronization_wait_timeout,
        proprio_max_age=config.proprio_max_age,
        teleop_max_age=config.teleop_max_age,
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
        recording_status_port=config.recording_status_port,
    )
    data_collector.run()


if __name__ == "__main__":
    config = tyro.cli(SonicDataExporterConfig)

    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    main(config)
