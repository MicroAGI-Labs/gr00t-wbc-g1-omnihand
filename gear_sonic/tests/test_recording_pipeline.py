from __future__ import annotations

from collections import deque
import json
import time
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from gear_sonic.camera.composed_camera import CameraFrameBuffer, ComposedCameraClientSensor
from gear_sonic.data.causal_sync import (
    CausalSelection,
    CausalSynchronizer,
    TimedSample,
)
from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import get_features_sonic_vla
from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import G1SupplementalInfo
from gear_sonic.end_effectors.profiles import OMNIHAND_O10
from gear_sonic.end_effectors.protocol import HAND_STATE_SCHEMA, HAND_STATE_TOPIC, encode
from gear_sonic.scripts.run_camera_web_viewer import CameraWebViewerConfig, RecorderControlHub
from gear_sonic.scripts.run_data_exporter import GrootDataCollector
from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


class FakeImageSubscriber:
    def __init__(self, stats=None):
        self._stats = stats or {
            "capacity": 5,
            "depth": 1,
            "received": 10,
            "overflow_dropped": 2,
            "latency_dropped": 3,
            "publisher_gap_dropped": 0,
            "received_hz": 50.0,
            "publisher_hz": 60.0,
        }

    def buffer_stats(self):
        return dict(self._stats)


class FakeFinalizer:
    def __init__(self):
        self.jobs = []
        self.results = []

    def can_accept(self):
        return True

    def enqueue(self, **job):
        self.jobs.append(job)

    def drain_results(self):
        results = self.results
        self.results = []
        return results


class FakeExporter:
    def __init__(self):
        self.episode_buffer = {"episode_index": 0, "size": 8}
        self.features = {"observation.images.ego_view": {"dtype": "video"}}

    def detach_episode(self):
        completed = self.episode_buffer
        self.episode_buffer = {"episode_index": 1, "size": 0}
        return completed, {"observation.images.ego_view": object()}


def _recording_collector():
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_state = EpisodeState()
    collector._episode_state.change_state()
    collector._manager_toggle_dc = False
    collector._manager_toggle_da = False
    collector._keyboard_listener = type("Keyboard", (), {"read_msg": lambda self: "c"})()
    collector.data_exporter = FakeExporter()
    collector.episode_finalizer = FakeFinalizer()
    collector._image_subscriber = FakeImageSubscriber()
    collector._episode_camera_stats_start = {
        "received": 4,
        "overflow_dropped": 1,
        "latency_dropped": 1,
        "publisher_gap_dropped": 0,
        "publisher_resets": 0,
    }
    collector.camera_max_age = 0.25
    collector.minimum_camera_rate_hz = 25.0
    collector.hand_config = None
    collector.hand_state_max_age = 0.2
    collector.frequency = 50
    collector.loop_period = 0.02
    collector.loop_period_ns = 20_000_000
    collector.synchronization_delay_ns = 100_000_000
    collector.synchronization_wait_timeout_ns = 250_000_000
    collector.proprio_max_age_ns = 100_000_000
    collector.teleop_max_age_ns = 200_000_000
    collector._synchronizer = CausalSynchronizer()
    collector._recording_start_target_ns = time.monotonic_ns()
    collector._next_target_ns = collector._recording_start_target_ns
    collector._recording_stop_target_ns = None
    collector._synchronization_errors = []
    collector._synchronization_warnings = []
    collector._synchronization_skipped_targets = 0
    collector.latest_image_msg = None
    collector.latest_image_received_at = None
    collector.sonic_timing_monitor = SimpleNamespace(reset=lambda: None, log_time_delta=lambda _dt: None)
    collector._initial_yaw = 1.0
    collector._recording_message = "Recording episode 0"
    collector._last_finalization = None
    collector._print_and_say = lambda *args, **kwargs: None
    return collector


@pytest.fixture
def body_state_collector():
    collector = _recording_collector()
    info = G1SupplementalInfo()
    model = SimpleNamespace(joint_names=info.body_actuated_joints, supplemental_info=info)
    collector.data_exporter.features = get_features_sonic_vla(model)
    return collector


@pytest.mark.parametrize("legacy", [False, True])
def test_recorded_body_state_round_trips_and_resumes(tmp_path, body_state_collector, legacy):
    collector = body_state_collector
    features = collector.data_exporter.features
    current_features = dict(features)
    velocity = features["observation.body_joint_velocity"]
    assert velocity["names"][0] == "left_hip_pitch_joint"
    assert velocity["names"][22] == "right_shoulder_pitch_joint"
    assert velocity["names"][-1] == "right_wrist_yaw_joint"
    new_fields = {
        "observation.body_joint_velocity", "observation.base_angular_velocity", "action.motion_token_valid"
    }
    if legacy:
        for key in new_fields:
            del features[key]
    proprio = {} if legacy else {"body_dq": np.arange(29), "base_ang_vel": [0.5, -1.0, 2.0]}
    frames = []
    # Missing key, empty publisher array, null, a valid zero vector, and a nonzero token.
    for token_fields in (
        {}, {"token_state": []}, {"token_state": None},
        {"token_state": np.zeros(64)}, {"token_state": np.arange(64)},
    ):
        proprio.update(token_fields)
        frame = {}
        collector._add_cpp_state_features(frame, proprio)
        frames.append(frame)
    kwargs = dict(
        save_root=tmp_path / "dataset", fps=50,
        features={key: features[key] for key in frames[0]},
        modality_config={"state": {}, "action": {}, "video": {}, "annotation": {}}, task="body state",
    )
    exporter = Gr00tDataExporter.create(**kwargs)
    for frame in frames:
        exporter.add_frame(frame)
    exporter.save_episode()
    resumed = Gr00tDataExporter.create(**(kwargs | {"features": current_features}))
    collector.data_exporter = resumed
    collector._add_cpp_state_features(frame := {}, proprio)
    resumed.add_frame(frame)
    resumed.save_episode()
    assert resumed.meta.total_frames == 6
    table = pq.read_table(resumed.root / resumed.meta.get_data_file_path(0))
    if legacy:
        assert not new_fields.intersection(table.column_names)
    else:
        assert table["action.motion_token_valid"].to_pylist() == [0, 0, 0, 1, 1]
        assert table["action.motion_token_valid"].type.bit_width == 8
        assert table["observation.body_joint_velocity"].type.value_type.bit_width == 32
        np.testing.assert_array_equal(table["observation.body_joint_velocity"][0].as_py(), np.arange(29))
        assert table["observation.base_angular_velocity"][0].as_py() == [0.5, -1.0, 2.0]
    np.testing.assert_array_equal(table["action.motion_token"][0].as_py(), np.zeros(64))
    np.testing.assert_array_equal(table["action.motion_token"][-1].as_py(), np.arange(64))


@pytest.fixture
def pose_messages():
    return {
        "smpl_msg": {"smpl_joints": np.arange(72).reshape(24, 3), "smpl_pose": np.zeros(63),
                     "body_quat_w": np.array([1.0, 0.0, 0.0, 0.0])},
        "planner_msg": {"planner_mode": 0, "planner_speed": 0.5, "planner_height": -1.0,
                        "vr_3pt_position": np.zeros(9),
                        "vr_3pt_orientation": np.array([2, 0, 0, 0, 0, -3, 0, 0, 0.5, 0.5, 0.5, 0.5])},
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_recorded_pose_channels_round_trip_and_resume(tmp_path, body_state_collector, pose_messages, legacy):
    collector = body_state_collector
    features = dict(collector.data_exporter.features)
    new_fields = {"teleop.vr_3pt_orientation_wxyz", "teleop.vr_3pt_valid", "teleop.smpl_valid"}
    if legacy:
        for key in new_fields:
            del collector.data_exporter.features[key]
    frames = []
    for mode in (0, 1, 2, 3, 4, 5):
        collector._add_sonic_pose_features(frame := {}, stream_mode=mode, target_ns=0, **pose_messages)
        frames.append(frame)
    kwargs = dict(
        save_root=tmp_path / "dataset", fps=50, task="poses",
        features={key: features[key] for key in frames[0]},
        modality_config={"state": {}, "action": {}, "video": {}, "annotation": {}},
    )
    exporter = Gr00tDataExporter.create(**kwargs)
    for frame in frames:
        exporter.add_frame(frame)
    exporter.save_episode()
    resumed = Gr00tDataExporter.create(**(kwargs | {"features": features}))
    collector.data_exporter = resumed
    collector._add_sonic_pose_features(frame := {}, stream_mode=5, target_ns=0, **pose_messages)
    resumed.add_frame(frame)
    resumed.save_episode()
    assert resumed.meta.total_frames == 7
    table = pq.read_table(resumed.root / resumed.meta.get_data_file_path(0))
    if legacy:
        assert not new_fields.intersection(table.column_names)
    else:
        assert table["teleop.smpl_valid"].to_pylist() == [0, 1, 0, 0, 0, 0]
        assert table["teleop.vr_3pt_valid"].to_pylist() == [0, 0, 1, 1, 0, 1]
        assert table["teleop.vr_3pt_valid"].type.bit_width == 8
        assert table["teleop.vr_3pt_orientation_wxyz"].type.value_type.bit_width == 32
        np.testing.assert_array_equal(table["teleop.vr_3pt_orientation_wxyz"][0].as_py(), np.zeros(12))
        np.testing.assert_array_equal(
            table["teleop.vr_3pt_orientation_wxyz"][5].as_py(), pose_messages["planner_msg"]["vr_3pt_orientation"]
        )
    np.testing.assert_array_equal(table["teleop.smpl_joints"][1].as_py(), np.arange(72))
    np.testing.assert_array_equal(
        table["teleop.vr_3pt_orientation"][5].as_py(), [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 1]
    )


@pytest.mark.parametrize("message,key,width", [
    ("smpl_msg", "smpl_joints", 72), ("smpl_msg", "smpl_pose", 63), ("smpl_msg", "body_quat_w", 4),
    ("planner_msg", "vr_3pt_position", 9), ("planner_msg", "vr_3pt_orientation", 12),
])
@pytest.mark.parametrize("bad_value", [None, [], [0.0], np.nan, np.inf, "bad"])
def test_incomplete_pose_is_flagged_and_finite(
    body_state_collector, pose_messages, message, key, width, bad_value
):
    pose_messages[message][key] = np.full(width, bad_value) if np.isscalar(bad_value) else bad_value
    is_smpl = message == "smpl_msg"
    frame = {}
    body_state_collector._add_sonic_pose_features(
        frame, stream_mode=1 if is_smpl else 5, target_ns=0, **pose_messages
    )
    assert frame["teleop.smpl_valid" if is_smpl else "teleop.vr_3pt_valid"].item() == 0
    assert all(np.all(np.isfinite(value)) for value in frame.values())


@pytest.mark.parametrize("message,key,width", [("smpl_msg", "body_quat_w", 4),
                                               ("planner_msg", "vr_3pt_orientation", 12)])
def test_zero_quaternion_is_unavailable(body_state_collector, pose_messages, message, key, width):
    pose_messages[message][key][:4] = 0  # One bad quaternion invalidates the VR orientation vector.
    frame = {}
    body_state_collector._add_sonic_pose_features(
        frame, stream_mode=1 if width == 4 else 5, target_ns=0, **pose_messages
    )
    assert frame["teleop.smpl_valid" if width == 4 else "teleop.vr_3pt_valid"].item() == 0
    assert all(np.all(np.isfinite(value)) for value in frame.values())


def test_missing_smpl_pose_on_wire_is_not_a_valid_zero_pose(body_state_collector):
    collector = body_state_collector
    raw = pack_pose_message({"smpl_joints": np.ones((1, 24, 3)), "body_quat_w": np.array([[1., 0., 0., 0.]])})
    collector._handle_pose_message(raw, received_ns=1)
    collector._handle_pose_message(raw, received_ns=3)
    selected = collector._synchronizer.select(2, required_streams=("sonic",), max_age_ns={})
    assert selected.ready
    message = selected.samples["sonic"].value
    assert message["smpl_pose"] is None
    collector._add_sonic_pose_features(frame := {}, stream_mode=1, target_ns=2, smpl_msg=message, planner_msg=None)
    assert frame["teleop.smpl_valid"].item() == 0
    np.testing.assert_array_equal(frame["teleop.smpl_joints"], np.ones(72))
    np.testing.assert_array_equal(frame["teleop.smpl_pose"], np.zeros(63))


@pytest.mark.parametrize("key,bad_value", [
    (key, value)
    for key, width in (("body_dq", 29), ("base_ang_vel", 3), ("token_state", 64))
    for value in ([None, []] if key != "token_state" else []) + [
        [0.0], np.zeros((1, width)), np.full(width, np.nan), np.full(width, np.inf), ["bad"] * width,
    ]
])
def test_malformed_body_state_marks_episode_invalid(body_state_collector, monkeypatch, key, bad_value):
    collector = body_state_collector
    proprio = {"body_dq": np.zeros(29), "base_ang_vel": np.zeros(3), "token_state": np.ones(64)}
    proprio[key] = bad_value
    target = collector._next_target_ns
    for stream, value in (
        ("proprio", proprio), ("manager", {"stream_mode": 0}),
        ("camera", {"images": {"ego_view": np.zeros((2, 2, 3), dtype=np.uint8)}}),
    ):
        collector._synchronizer.observe(stream, value, target)
        collector._synchronizer.observe(stream, value, target + 1)
    monkeypatch.setattr(
        "gear_sonic.scripts.run_data_exporter.time.monotonic_ns",
        lambda: target + collector.synchronization_delay_ns,
    )

    assert collector._add_data_frame() is False

    assert collector._next_target_ns == target + collector.loop_period_ns
    assert "invalid robot state" in collector._synchronization_errors[0]
    collector._finish_recording(discarded=False, reason="")
    [job] = collector.episode_finalizer.jobs
    assert job["discarded"] is True
    assert job["validation"]["passed"] is False


def test_recording_stop_reports_queued_until_background_commit_completes():
    collector = _recording_collector()

    collector._check_recording_commands()

    assert collector._episode_state.get_state() == collector._episode_state.NEED_TO_SAVE
    assert collector.episode_finalizer.jobs == []
    assert "Draining synchronized episode" in collector._recording_message

    collector._next_target_ns = collector._recording_stop_target_ns + 1
    collector._add_data_frame()

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert collector.current_episode_index == 1
    assert "queued for background save" in collector._recording_message
    [job] = collector.episode_finalizer.jobs
    assert job["episode_index"] == 0
    assert job["discarded"] is False
    assert job["validation"]["camera_buffer"]["episode_deltas"] == {
        "received": 6,
        "overflow_dropped": 1,
        "latency_dropped": 2,
        "publisher_gap_dropped": 0,
        "publisher_resets": 0,
    }


@pytest.mark.parametrize(
    ("stream_mode", "extra_stream"),
    [(2, "planner"), (3, "planner"), (4, None), (5, "planner")],
)
def test_collector_uses_streams_required_by_manager_mode(stream_mode, extra_stream):
    collector = _recording_collector()
    target = time.monotonic_ns()
    samples = [
        ("proprio", {}, {}),
        ("camera", {}, {}),
        ("manager", {"stream_mode": stream_mode}, {"stream_mode": stream_mode}),
    ]
    if extra_stream:
        samples.append((extra_stream, {"value": "past"}, {"value": "future"}))
    for stream, past, future in samples:
        collector._synchronizer.observe(stream, past, target - 1_000_000)
        collector._synchronizer.observe(stream, future, target + 1_000_000)

    selection = collector._selection_for_target(target)

    assert selection.ready
    expected = {"proprio", "camera", "manager"}
    if extra_stream:
        expected.add(extra_stream)
        assert selection.samples[extra_stream].value["value"] == "past"
    assert set(selection.samples) == expected


def test_synchronization_metadata_records_nonnegative_selected_ages():
    collector = _recording_collector()
    collector.data_exporter.features.update(
        {
            "capture.sync_target_monotonic_ns": {},
            "capture.camera_sequence": {},
            "capture.camera_capture_age_ms": {},
            "capture.camera_received_monotonic_ns": {},
            "capture.camera_age_ms": {},
            "capture.proprio_received_monotonic_ns": {},
            "capture.proprio_age_ms": {},
            "capture.sonic_received_monotonic_ns": {},
            "capture.sonic_age_ms": {},
        }
    )
    target = time.monotonic_ns()
    camera = {
        "publisher_sequence": 42,
        "images": {"ego_view": np.zeros((2, 2, 3), dtype=np.uint8)},
        "capture_monotonic_ns": {"ego_view": 9_990_000_000},
        "publisher_monotonic_ns": 10_000_000_000,
        "receiver_monotonic_ns": target - 5_000_000,
    }
    selection = CausalSelection(
        target_ns=target,
        samples={
            "camera": TimedSample(target - 5_000_000, camera),
            "proprio": TimedSample(target - 2_000_000, {}),
        },
    )
    frame = {}

    collector._add_synchronization_features(frame, selection)

    assert frame["capture.camera_sequence"].item() == 42
    assert frame["capture.camera_age_ms"].item() == 5.0
    assert frame["capture.proprio_age_ms"].item() == 2.0
    assert frame["capture.camera_capture_age_ms"].tolist() == [15.0, -1.0, -1.0]
    assert frame["capture.sonic_received_monotonic_ns"].item() == -1
    assert frame["capture.sonic_age_ms"].item() == -1.0


def test_collector_emits_target_only_after_future_watermarks_without_using_them(
    monkeypatch,
):
    collector = _recording_collector()
    target = collector._next_target_ns
    camera_past = {
        "images": {"ego_view": np.zeros((2, 2, 3), dtype=np.uint8)},
        "capture_monotonic_ns": {"ego_view": 9_990_000_000},
        "publisher_monotonic_ns": 10_000_000_000,
        "publisher_sequence": 7,
        "receiver_monotonic_ns": target - 5_000_000,
    }
    samples = {
        "proprio": ({"value": "state-past"}, {"value": "state-future"}),
        "camera": (camera_past, {"value": "camera-future"}),
        "manager": ({"stream_mode": 0}, {"stream_mode": 0}),
    }
    for stream, (past, future) in samples.items():
        collector._synchronizer.observe(stream, past, target - 5_000_000)
        collector._synchronizer.observe(stream, future, target + 1_000_000)
    selected = []
    collector._add_data_frame_sonic = lambda _start, selection: selected.append(selection) or True
    monkeypatch.setattr(
        "gear_sonic.scripts.run_data_exporter.time.monotonic_ns",
        lambda: target + collector.synchronization_delay_ns,
    )

    assert collector._add_data_frame() is True

    [selection] = selected
    assert all(sample.timestamp_ns <= target for sample in selection.samples.values())
    assert selection.samples["proprio"].value["value"] == "state-past"
    assert selection.samples["camera"].value is camera_past
    assert collector._next_target_ns == target + collector.loop_period_ns


def test_collector_waits_for_watermarks_then_records_a_bounded_gap(monkeypatch):
    collector = _recording_collector()
    target = collector._next_target_ns
    for stream, value in (
        ("proprio", {}),
        (
            "camera",
            {
                "images": {"ego_view": np.zeros((2, 2, 3), dtype=np.uint8)},
                "receiver_monotonic_ns": target - 1,
            },
        ),
        ("manager", {"stream_mode": 0}),
    ):
        collector._synchronizer.observe(stream, value, target - 1)

    now = target + collector.synchronization_delay_ns
    monkeypatch.setattr(
        "gear_sonic.scripts.run_data_exporter.time.monotonic_ns",
        lambda: now,
    )
    assert collector._add_data_frame() is False
    assert collector._synchronization_errors == []

    now = target + collector.synchronization_delay_ns + collector.synchronization_wait_timeout_ns + 1
    assert collector._add_data_frame() is False
    assert "streams did not advance" in collector._synchronization_errors[0]
    assert collector._synchronization_skipped_targets > 0
    collector._finish_recording(discarded=False, reason="")
    assert collector.episode_finalizer.jobs[0]["discarded"] is True


@pytest.mark.parametrize("mode,dropout,failure", [
    (1, "all", None), (5, "all", None), (1, "pose", None), (5, "pose", None), (5, "hand", None),
    (5, "all", "camera"), (5, "all", "proprio"), (5, "all", "hand"),
    (5, "all", "hand_fault"), (5, "all", "hand_invalid"),
])
def test_pico_dropout_preserves_take_without_hiding_hardware_failures(
    tmp_path, monkeypatch, mode, dropout, failure
):
    collector = _recording_collector()
    exporter = Gr00tDataExporter.create(
        save_root=tmp_path / "dataset", fps=50, task="reconnect",
        features={"capture.sync_target_monotonic_ns": {"dtype": "int64", "shape": (1,), "names": ["target"]}},
        modality_config={"state": {}, "action": {}, "video": {}, "annotation": {}},
    )
    collector.data_exporter = exporter
    collector.hand_config = {"session_id": "session"}
    collector.hand_profile = OMNIHAND_O10
    base = collector._next_target_ns
    now = base
    monkeypatch.setattr("gear_sonic.scripts.run_data_exporter.time.monotonic_ns", lambda: now)
    recorded = []

    def add_frame(_start, selection):
        recorded.append(selection.target_ns)
        exporter.add_frame({"capture.sync_target_monotonic_ns": np.asarray([selection.target_ns], dtype=np.int64)})
        return True

    collector._add_data_frame_sonic = add_frame
    active = "sonic" if mode == 1 else "planner"
    lost = {"all": {"manager", active}, "pose": {active}, "hand": set()}[dropout]
    for tick in range(81):
        now = base + tick * collector.loop_period_ns
        interrupted = 6 <= tick < 56  # PICO stops for exactly one second, then resumes.
        hand = {
            "mode": "fault" if interrupted and failure == "hand_fault" else "tracking",
            "input_stale": interrupted and dropout in {"all", "hand"}, "intent_sequence": tick,
            "sides": {
                side: {"valid": True, "connected": not (interrupted and failure == "hand_invalid"),
                       "intent_closed": False, **{f"{field}_position_rad": np.zeros(10)
                                                   for field in ("requested", "applied", "measured")}}
                for side in ("left", "right")
            },
        }
        for stream, value in (
            ("proprio", {}), ("camera", {}), ("manager", {"stream_mode": mode}),
            (active, {}), ("hand", hand),
        ):
            if not interrupted or stream not in lost | {failure}:
                collector._synchronizer.observe(stream, value, now)
        collector._add_data_frame()

    assert recorded[0] == base
    assert recorded[-1] > base + 1_120_000_000
    assert not any(base + 400_000_000 <= stamp < base + 1_120_000_000 for stamp in recorded)
    assert collector.current_episode_index == 0
    assert collector._synchronization_skipped_targets > 0
    collector._finish_recording(discarded=False, reason="")
    [job] = collector.episode_finalizer.jobs
    assert job["discarded"] is (failure is not None)
    assert job["validation"]["passed"] is (failure is None)
    if failure is None:
        assert job["validation"]["errors"] == []
        assert "before frame" in job["validation"]["warnings"][0]
    else:
        assert job["validation"]["errors"]
    exporter.save_episode(
        job["episode_buffer"], video_writers=job["video_writers"],
        discarded=job["discarded"], validation=job["validation"],
    )
    quality = json.loads((exporter.root / "meta/info.json").read_text())["episode_quality"]["0"]
    assert quality["validation"] == job["validation"]
    assert quality["discarded"] is (failure is not None)
    table = pq.read_table(exporter.root / exporter.meta.get_data_file_path(0))
    assert table["capture.sync_target_monotonic_ns"].to_pylist() == recorded


@pytest.mark.parametrize("camera_hz", [30, 60])
def test_collector_preserves_closest_past_camera_at_50hz(camera_hz):
    collector = _recording_collector()
    client = ComposedCameraClientSensor.__new__(ComposedCameraClientSensor)
    client._background = True
    client._receiver_error = None
    client._background_buffer = CameraFrameBuffer()
    client.idx = 0
    collector._image_subscriber = client
    base = 1_000_000_000
    received = []
    sequence = 0
    for tick in range(31):
        now = base + tick * collector.loop_period_ns
        while base + round(sequence * 1e9 / camera_hz) <= now:
            stamp = base + round(sequence * 1e9 / camera_hz)
            client._background_buffer.put({"receiver_monotonic_ns": stamp})
            received.append(stamp)
            sequence += 1
        collector._poll_images()
        if tick < 5:
            continue
        target = now - collector.synchronization_delay_ns
        selection = collector._synchronizer.select(
            target, required_streams=("camera",), max_age_ns={"camera": 250_000_000}
        )
        assert selection.ready
        assert selection.samples["camera"].timestamp_ns == max(t for t in received if t <= target)
        collector._synchronizer.trim_through(target)
    assert client._background_buffer.stats()["overflow_dropped"] == 0


def test_rejected_finalizer_handoff_restores_completed_episode(monkeypatch):
    collector = _recording_collector()
    completed = collector.data_exporter.episode_buffer

    def reject(**kwargs):
        raise RuntimeError("finalizer unavailable")

    monkeypatch.setattr(collector.episode_finalizer, "enqueue", reject)
    with pytest.raises(RuntimeError, match="finalizer unavailable"):
        collector._finish_recording(discarded=False, reason="")
    assert collector.data_exporter.episode_buffer is completed
    assert "observation.images.ego_view" in collector.data_exporter.video_writers


def test_hand_reconnect_invalidates_recording_but_not_idle_collector():
    collector = _recording_collector()
    collector.hand_config = {"session_id": "session"}
    collector.hand_profile = OMNIHAND_O10
    collector.latest_hand_state = None
    messages = deque()
    collector._hand_zmq_socket = SimpleNamespace(poll=lambda _: bool(messages), recv=messages.popleft)

    def receive(connection, sequence):
        messages.append(
            encode(
                HAND_STATE_TOPIC,
                dict(
                    schema=HAND_STATE_SCHEMA,
                    session_id="session",
                    profile=OMNIHAND_O10.name,
                    connection_id=connection,
                    sequence=sequence,
                    mode="tracking",
                ),
            )
        )
        collector._poll_hand_zmq()

    receive("first", 1)
    receive("first", 2)
    assert collector._synchronization_errors == []
    receive("second", 1)
    assert collector._synchronization_errors == ["hand reconnected during recording"]
    collector._next_target_ns = None
    collector._synchronization_errors.clear()
    receive("third", 1)
    assert collector._synchronization_errors == []


def test_hand_ui_reports_stale_status_without_a_relay(monkeypatch):
    hub = RecorderControlHub(CameraWebViewerConfig(hand_controls=True))
    assert hub.hands_status()["last_status_age_s"] is None
    hub._hand_received_at = 10.0
    hub._hand_status = {"sides": {"left": {"valid": True, "connected": True}}}
    monkeypatch.setattr(time, "monotonic", lambda: 10.1)
    assert hub.hands_status()["connected"]
    monkeypatch.setattr(time, "monotonic", lambda: 10.3)
    assert not hub.hands_status()["connected"]
    hub.config.hand_controls = False
    assert not hub.reconnect_hands()["accepted"]
