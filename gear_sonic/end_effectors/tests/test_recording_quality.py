from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.data.features_sonic_vla import get_g1_robot_model
from gear_sonic.data.sender_sync import RecordingInputs
from gear_sonic.end_effectors.profiles import DEX1
from gear_sonic.scripts import run_data_exporter
from gear_sonic.scripts.run_data_exporter import (
    GrootDataCollector,
    _episode_hand_motion_range,
    _recording_mode_ready,
    _required_vector,
)
from gear_sonic.utils.data_collection.episode_state import EpisodeState


def test_required_vector_accepts_exact_finite_shape_as_float32():
    value = _required_vector({"token": np.arange(64, dtype=np.float64)}, "token", 64)
    assert value.shape == (64,)
    assert value.dtype == np.float32


@pytest.mark.parametrize(
    "payload,error",
    [
        ({}, "missing"),
        ({"token": np.zeros(63)}, "shape"),
        ({"token": np.full(64, np.nan)}, "NaN or Inf"),
    ],
)
def test_required_vector_rejects_untrainable_values(payload, error):
    with pytest.raises(ValueError, match=error):
        _required_vector(payload, "token", 64)


def test_episode_hand_motion_range_detects_a_command_transition():
    still = np.zeros(10, dtype=np.float32)
    closed = still.copy()
    closed[3] = 0.8
    episode = {
        "teleop.left_hand_joints": [still, closed],
        "teleop.right_hand_joints": [still, still],
    }
    assert _episode_hand_motion_range(episode) == pytest.approx(0.8)


def test_recording_mode_requires_exact_launch_selected_mode():
    assert _recording_mode_ready(5, 5)
    assert not _recording_mode_ready(2, 5)
    assert not _recording_mode_ready(1, 5)


def test_recording_start_is_blocked_outside_required_mode():
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector.required_stream_mode = 5
    collector.current_stream_mode = 2
    collector._episode_state = EpisodeState()
    collector._manager_toggle_dc = False
    collector._manager_toggle_da = False
    collector._manager_discard_reason = None
    collector._recording_message = "Ready to record"
    collector._keyboard_listener = type(
        "_Keyboard", (), {"read_msg": lambda self: "c"}
    )()
    collector._print_and_say = lambda *args, **kwargs: None

    collector._check_recording_commands()

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert collector._recording_message == "Enter VR3PT teleop before recording"


@pytest.mark.parametrize("passed", [True, False])
def test_recording_stop_reports_validation_outcome_and_returns_to_idle(passed):
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_state = EpisodeState()
    collector._episode_state.change_state()
    collector._manager_toggle_dc = False
    collector._manager_toggle_da = False
    collector._manager_discard_reason = None
    collector._keyboard_listener = type(
        "_Keyboard", (), {"read_msg": lambda self: "c"}
    )()

    class _Exporter:
        def __init__(self):
            self.episode_buffer = {"episode_index": 0, "size": 10}

        def detach_episode(self, *, advance_index=True):
            completed = self.episode_buffer
            self.episode_buffer = {"episode_index": 1, "size": 0}
            return completed, {"camera": object()}

    class _Finalizer:
        def __init__(self):
            self.jobs = []

        def enqueue(self, **job):
            self.jobs.append(job)

    collector.data_exporter = _Exporter()
    collector.episode_finalizer = _Finalizer()
    collector._episode_validation = lambda: {
        "passed": passed,
        "errors": [] if passed else ["hand commands did not move enough"],
    }
    collector.sonic_timing_monitor = type("_Monitor", (), {"reset": lambda self: None})()
    collector._episode_input_errors = set()
    collector._initial_yaw = 1.0
    collector._print_and_say = lambda *args, **kwargs: None
    audio_events = []
    collector._set_recording_audio_event = audio_events.append

    collector._check_recording_commands()

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert collector.current_episode_index == 1
    assert audio_events == ["saved" if passed else "validation_failed"]
    assert collector.episode_finalizer.jobs[0]["episode_index"] == 0
    assert collector.episode_finalizer.jobs[0]["success"] is passed
    if not passed:
        assert "failed validation: hand commands did not move enough" in collector._recording_message
        assert "Preserved locally as unsuccessful" in collector._recording_message
        assert "upload" not in collector._recording_message
        assert "discarded" not in collector._recording_message


def test_recording_failure_acknowledges_and_saves_in_background():
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_state = EpisodeState()
    collector._episode_state.change_state()
    collector.data_exporter = type(
        "_Exporter",
        (),
        {
            "episode_buffer": {"episode_index": 3, "size": 5},
            "detach_episode": lambda self, **kwargs: (
                self.episode_buffer,
                {"camera": object()},
            ),
        },
    )()
    jobs = []
    collector.episode_finalizer = type(
        "_Finalizer", (), {"enqueue": lambda self, **job: jobs.append(job)}
    )()
    collector.sonic_timing_monitor = type("_Monitor", (), {"reset": lambda self: None})()
    collector._episode_input_errors = set()
    collector._episode_validation = lambda: {"passed": True, "errors": []}
    collector._initial_yaw = 1.0
    collector._print_and_say = lambda *args, **kwargs: None
    audio_events = []
    collector._set_recording_audio_event = audio_events.append

    collector._finish_recording(save=False, discard_reason="operator_marked_failure")

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert audio_events == ["validation_failed"]
    assert jobs[0]["episode_index"] == 3
    assert jobs[0]["success"] is False
    assert jobs[0]["validation"]["errors"] == ["operator_marked_failure"]
    assert jobs[0]["delete"] is False


def test_empty_recording_reports_no_frames_to_save():
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_state = EpisodeState()
    collector._episode_state.state = collector._episode_state.NEED_TO_SAVE
    collector.data_exporter = type(
        "_Exporter", (), {"episode_buffer": {"episode_index": 0, "size": 0}}
    )()
    collector.frequency = 50.0
    collector._print_and_say = lambda *args, **kwargs: None
    audio_events = []
    collector._set_recording_audio_event = audio_events.append

    collector._finalize_frame(run_data_exporter.time.monotonic())

    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert collector._recording_message == "Nothing saved: no frames collected"
    assert audio_events == ["validation_failed"]


def test_manager_mode_exit_does_not_request_automatic_discard(monkeypatch):
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector.required_stream_mode = 5
    collector.current_stream_mode = 5
    collector._episode_state = EpisodeState()
    collector._episode_state.change_state()
    collector._manager_toggle_dc = False
    collector._manager_toggle_da = False
    collector._manager_discard_reason = None
    collector._recording_message = "Recording episode 0"

    class _Rates:
        def observe(self, *args, **kwargs):
            pass

    collector.stream_rates = _Rates()
    monkeypatch.setattr(
        run_data_exporter,
        "unpack_pose_message",
        lambda raw, topic: {"stream_mode": np.asarray([2], dtype=np.int32)},
    )

    collector._handle_manager_state(b"manager_state")

    assert collector.current_stream_mode == 2
    assert not collector._manager_toggle_da
    assert collector._manager_discard_reason is None
    assert collector._recording_message == "Recording paused: return to VR3PT mode"


def test_rolling_stream_rate_is_diagnostic_not_a_discard_criterion():
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_input_errors = set()
    collector.data_exporter = type(
        "_Exporter",
        (),
        {"episode_buffer": {"teleop.stream_mode": [np.asarray([5])]}},
    )()
    collector.hand_config = None
    collector.require_hand_activity = True
    collector.minimum_hand_motion_rad = 0.02
    collector.required_stream_mode = 5
    collector.minimum_recording_rate_hz = 45.0

    class _Rates:
        def snapshot(self, streams):
            return {stream: {"sent_hz": 1.0} for stream in streams}

    collector.stream_rates = _Rates()
    validation = collector._episode_validation()

    assert validation["rate_check_enforced"] is False


@pytest.fixture(scope="module")
def recording_robot_model():
    return get_g1_robot_model()


@pytest.fixture
def hand_recording(monkeypatch, recording_robot_model):
    monkeypatch.setattr(run_data_exporter.time, "monotonic", lambda: 10.0)
    c = GrootDataCollector.__new__(GrootDataCollector)
    c.hand_profile = DEX1
    c.hand_config = {"profile": DEX1.name}
    c.hand_state_max_age = .2
    c.proprio_state_max_age = c.camera_max_age = c.teleop_max_age = .1
    c.required_stream_mode = 5
    c.require_hand_activity = False
    c._episode_input_errors = set()
    c._episode_hand_diagnostics = {}
    c._sender_sync = None
    c._initial_yaw = None
    c.robot_model = recording_robot_model
    c._log_latency_periodic = lambda *args: None
    c._finalize_frame = lambda *args: True
    c.sonic_timing_monitor = run_data_exporter.TimingThresholdMonitor()
    c.stream_rates = SimpleNamespace(snapshot=lambda streams: {})
    frames = []
    def add_frame(frame):
        frames.append(frame)
        c.data_exporter.episode_buffer["size"] += 1
        c.data_exporter.episode_buffer.setdefault("teleop.stream_mode", []).append(frame["teleop.stream_mode"])
    c.data_exporter = SimpleNamespace(features={}, episode_buffer={"size": 0, "episode_index": 0}, add_frame=add_frame)
    hand = {
        "mode": "tracking", "state_age_s": .373, "input_stale": False, "intent_sequence": 12,
        "monotonic_ns": 9_617_000_000, "published_monotonic_ns": 9_990_000_000,
        "sides": {side: {
            "requested_position_rad": [target], "applied_position_rad": [target - .1],
            "measured_position_rad": [target - .2], "valid": True, "connected": True,
            "intent_closed": True, "input_stale": False,
        } for side, target in (("left", 1.0), ("right", 2.0))},
    }
    proprio = {"body_q": np.zeros(29), "body_dq": np.zeros(29), "base_quat": np.array([1., 0, 0, 0]),
               "base_ang_vel": np.zeros(3), "last_action": np.zeros(29), "token_state": np.ones(64)}
    planner = {"receive_monotonic": 9.99, "receive_timestamp": run_data_exporter.time.time(),
               "planner_mode": 1, "planner_speed": .4, "planner_height": .75,
               "vr_3pt_position": np.ones(9), "vr_3pt_orientation": np.tile([1., 0, 0, 0], 3)}
    inputs = RecordingInputs(proprio, {"images": {}, "timestamps": {}}, hand, None, planner, None,
                             5, 9_990_000_000, 9_990_000_000, 9_990_000_000)
    return c, inputs, frames


def test_stale_hand_rows_keep_measurements_timestamps_and_success(hand_recording, tmp_path):
    import pyarrow.parquet as pq
    from gear_sonic.data.exporter import Gr00tDataExporter

    c, inputs, frames = hand_recording
    original = deepcopy(inputs.hand)
    for age in (.204, .373):
        inputs.hand["state_age_s"] = age
        c._validate_recording_inputs(inputs)
        assert c._add_data_frame_sonic(10.0, inputs)
    assert len(frames) == 2
    for frame in frames:
        np.testing.assert_allclose(frame["observation.dex1_left_raw"], [.8])
        np.testing.assert_allclose(frame["action.dex1_left_raw"], [1.0])
        np.testing.assert_allclose(frame["control.hand_applied_position"], [.9, 1.9])
        assert frame["observation.left_hand_valid"].tolist() == [1]
        assert frame["capture.hand_state_source_monotonic_ns"].tolist() == [9_617_000_000]
        assert frame["capture.hand_state_publish_monotonic_ns"].tolist() == [9_990_000_000]
    assert inputs.hand == original
    report = c._episode_validation()
    assert report["passed"] and not report["errors"]
    diagnostic = report["hand_diagnostics"]["external hand controller snapshot is stale"]
    assert diagnostic == {"frames": 2, "frame_ranges": [[0, 1]], "max_age_s": .373}

    # Save/reload the actual rows and outcome, including the original timing.
    fields = ["observation.dex1_left_raw", "action.dex1_left_raw", "control.hand_applied_position",
              "observation.left_hand_valid", "capture.hand_state_source_monotonic_ns",
              "capture.hand_state_publish_monotonic_ns", "episode.success"]
    features = {key: {"dtype": str(frames[0][key].dtype), "shape": frames[0][key].shape,
                      "names": [str(i) for i in range(frames[0][key].size)]} for key in fields}
    writer = Gr00tDataExporter.create(
        save_root=tmp_path / "dataset", fps=50, features=features, task="accepted teleop",
        modality_config={key: {} for key in ("state", "action", "video", "annotation")},
    )
    for frame in frames:
        writer.add_frame({key: frame[key] for key in fields})
    writer.save_episode(success=report["passed"], validation=report)
    table = pq.read_table(writer.root / writer.meta.get_data_file_path(0))
    assert table["episode.success"].to_pylist() == [1, 1]
    assert table["capture.hand_state_source_monotonic_ns"].to_pylist() == [9_617_000_000] * 2
    assert writer.meta.info["discarded_episode_indices"] == []
    assert writer.meta.info["episode_quality"]["0"]["validation"]["hand_diagnostics"] == report["hand_diagnostics"]


def test_missing_click_intent_in_hold_is_recorded_and_identified(hand_recording):
    c, inputs, frames = hand_recording
    inputs.hand.update(mode="hold", intent_sequence=None, input_stale=True)
    inputs.hand["sides"]["left"]["intent_closed"] = None
    c._validate_recording_inputs(inputs)
    c._add_data_frame_sonic(10.0, inputs)
    assert frames[0]["observation.left_hand_valid"].tolist() == [1]
    report = c._episode_validation()
    assert report["passed"]
    assert report["hand_diagnostics"]["external left hand has no valid click intent"]["frame_ranges"] == [[0, 0]]
    assert frames[0]["capture.hand_intent_sequence"].tolist() == [-1]


def disconnected_report(hand, published_ns):
    return {
        **{key: hand[key] for key in ("session_id", "profile", "clock_id")},
        "mode": "disconnected", "monotonic_ns": published_ns - 1_000_000,
        "published_monotonic_ns": published_ns, "sequence": 0,
        "input_stale": True, "intent_sequence": None,
        "sides": {side: {"valid": False, "connected": False, "error": "USB lost"}
                  for side in ("left", "right")},
    }


def test_disconnect_rows_hold_exact_last_values_and_remain_accepted(hand_recording, tmp_path):
    import pyarrow.parquet as pq
    from gear_sonic.data.exporter import Gr00tDataExporter

    c, inputs, frames = hand_recording
    good = deepcopy(inputs.hand)
    good.update(session_id="test-session", profile=DEX1.name, clock_id="same-boot", sequence=123)
    assert c._recordable_hand_state(good) == good
    original = deepcopy(good)
    for published_ns in (9_990_000_000, 9_999_000_000):
        report = disconnected_report(good, published_ns)
        before = deepcopy(report)
        held = c._recordable_hand_state(report)
        assert report == before and good == original
        c._validate_recording_inputs(replace(inputs, hand=held))
        c._add_data_frame_sonic(10.0, replace(inputs, hand=held))
    for frame in frames:
        np.testing.assert_allclose(frame["observation.dex1_left_raw"], [.8])
        np.testing.assert_allclose(frame["observation.dex1_right_raw"], [1.8])
        np.testing.assert_allclose(frame["action.dex1_left_raw"], [1.0])
        np.testing.assert_allclose(frame["control.hand_applied_position"], [.9, 1.9])
        assert frame["capture.hand_state_source_monotonic_ns"].tolist() == [9_617_000_000]
        assert frame["capture.hand_state_sequence"].tolist() == [123]
        assert frame["observation.left_hand_valid"].tolist() == [0]
        assert frame["observation.right_hand_valid"].tolist() == [0]

    recovered = deepcopy(good)
    recovered["monotonic_ns"] = 9_999_500_000
    recovered["sides"]["left"]["measured_position_rad"] = [1.5]
    recovered = c._recordable_hand_state(recovered)
    c._add_data_frame_sonic(10.0, replace(inputs, hand=recovered))
    assert frames[-1]["observation.left_hand_valid"].tolist() == [1]
    np.testing.assert_allclose(frames[-1]["observation.dex1_left_raw"], [1.5])
    report = c._episode_validation()
    assert report["passed"] and not report["errors"]
    assert report["hand_diagnostics"]["external hand snapshot retained during disconnect"]["frame_ranges"] == [[0, 1]]
    fields = ("observation.dex1_left_raw", "observation.left_hand_valid", "episode.success",
              "capture.hand_state_source_monotonic_ns", "capture.hand_state_publish_monotonic_ns")
    features = {key: {"dtype": str(frames[0][key].dtype), "shape": frames[0][key].shape,
                      "names": [str(i) for i in range(frames[0][key].size)]} for key in fields}
    writer = Gr00tDataExporter.create(
        save_root=tmp_path / "dataset", fps=50, features=features, task="accepted freeze",
        modality_config={key: {} for key in ("state", "action", "video", "annotation")},
    )
    for frame in frames:
        writer.add_frame({key: frame[key] for key in fields})
    writer.save_episode(success=report["passed"], validation=report)
    table = pq.read_table(writer.root / writer.meta.get_data_file_path(0))
    assert table["episode.success"].to_pylist() == [1, 1, 1]
    assert table["observation.left_hand_valid"].to_pylist() == [0, 0, 1]
    assert table["capture.hand_state_source_monotonic_ns"].to_pylist() == [9_617_000_000] * 2 + [9_999_500_000]
    assert writer.meta.info["discarded_episode_indices"] == []


@pytest.mark.parametrize("change", ["no_previous", "session_id", "clock_id", "profile", "malformed"])
def test_disconnect_retention_requires_real_compatible_previous_feedback(hand_recording, change):
    c, inputs, frames = hand_recording
    good = deepcopy(inputs.hand)
    good.update(session_id="test-session", profile=DEX1.name, clock_id="same-boot")
    if change != "no_previous":
        c._recordable_hand_state(good)
    report = disconnected_report(good, 9_990_000_000)
    if change in {"session_id", "clock_id", "profile"}:
        report[change] = "other"
    elif change == "malformed":
        report["sides"]["left"]["measured_position_rad"] = [np.nan]
    assert c._recordable_hand_state(report) is report
    with pytest.raises(RuntimeError):
        c._validate_recording_inputs(replace(inputs, hand=report))
    assert frames == []


def test_old_received_snapshot_is_retained_without_refreshing_source_time(hand_recording):
    c, inputs, frames = hand_recording
    inputs = replace(inputs, hand_received_ns=8_000_000_000)
    c._validate_recording_inputs(inputs)
    c._add_data_frame_sonic(10.0, inputs)
    assert frames[0]["capture.hand_state_received_monotonic_ns"].tolist() == [8_000_000_000]
    assert c._episode_validation()["hand_diagnostics"]["external hand state is stale"]["max_age_s"] == 2.0


@pytest.mark.parametrize("bad_values", [None, [], [1, 2], [np.nan], [np.inf]])
def test_missing_or_malformed_hand_measurements_are_never_invented(hand_recording, bad_values):
    c, inputs, frames = hand_recording
    inputs.hand["sides"]["left"]["measured_position_rad"] = bad_values
    with pytest.raises(RuntimeError, match="wrong shape"):
        c._validate_recording_inputs(inputs)
    assert frames == []


@pytest.mark.parametrize("field,error", [("proprio_received_ns", "robot state is stale"),
                                         ("image_received_ns", "camera frame is stale")])
def test_stale_robot_and_camera_rows_are_flagged_and_retained(hand_recording, field, error):
    c, inputs, frames = hand_recording
    inputs = replace(inputs, **{field: 8_000_000_000})
    assert c._validate_recording_inputs(inputs)[error] == 2.0
    assert c._add_data_frame_sonic(10., inputs)
    report = c._episode_validation()
    assert report["passed"]
    assert report["frame_diagnostics"][error] == {
        "frames": 1, "frame_ranges": [[0, 0]], "max_age_s": 2.0,
    }
    assert report["flagged_frame_count"] == 1  # Count once, including simultaneous hand warnings.


def test_camera_recovery_flags_exact_saved_video_frames_without_failing_take(hand_recording, tmp_path):
    import json
    import av
    import pyarrow.parquet as pq
    from gear_sonic.data.episode_finalizer import EpisodeFinalizer
    from gear_sonic.data.exporter import Gr00tDataExporter

    c, inputs, frames = hand_recording
    inputs.hand["state_age_s"] = .01
    cameras = ("ego_view", "left_wrist")
    c.data_exporter.features = {
        f"observation.images.{name}": {"dtype": "video", "shape": (16, 16, 3),
                                     "names": ["height", "width", "channels"]}
        for name in cameras
    }
    c.data_exporter.features.update({
        f"capture.{name}_source_timestamp_ns": {"dtype": "int64", "shape": (1,), "names": ["time"]}
        for name in cameras
    })
    camera_features = deepcopy(c.data_exporter.features)
    c._episode_state = EpisodeState()
    c._episode_state.change_state()
    c._recording_message = "Recording episode 0"
    c._last_input_block_log = 0
    for index, stale in enumerate(("ego_view", "ego_view", None, "left_wrist", "ego_view")):
        image = {
            "images": {name: np.full((16, 16, 3), 40 if name == stale else 160, dtype=np.uint8)
                       for name in cameras},
            "timestamps": {name: 99.0 if name == stale else 100.0 for name in cameras},
            "camera_received_monotonic_ns": {name: 9_800_000_000 if name == stale else 9_990_000_000
                                              for name in cameras},
        }
        planner = {**inputs.planner, "receive_monotonic": 9.8 if index == 4 else 9.99}
        current = replace(inputs, image=image, planner=planner)
        c._latest_recording_inputs = lambda: current
        assert c._add_data_frame()
    report = c._episode_validation()
    assert len(frames) == 5
    assert report["passed"] and not report["errors"]
    assert report["flagged_frame_count"] == 4
    assert report["frame_diagnostics"]["camera ego_view is stale"]["frame_ranges"] == [[0, 1], [4, 4]]
    assert report["frame_diagnostics"]["camera left_wrist is stale"]["frame_ranges"] == [[3, 3]]
    assert report["frame_diagnostics"]["planner command is stale in planner mode"]["frame_ranges"] == [[4, 4]]
    assert report["freshness_thresholds_s"]["camera"] == .1

    features = {**camera_features, "episode.success": {"dtype": "uint8", "shape": (1,), "names": ["success"]},
                "teleop.stream_mode": {"dtype": "int32", "shape": (1,), "names": ["mode"]}}
    writer = Gr00tDataExporter.create(
        save_root=tmp_path / "dataset", fps=50, features=features, task="crop later",
        modality_config={key: {} for key in ("state", "action", "video", "annotation")},
    )
    for frame in frames:
        writer.add_frame({key: frame[key] for key in features})
    c.data_exporter = writer
    c.episode_finalizer = EpisodeFinalizer(writer)
    c._print_and_say = lambda *args, **kwargs: None
    audio = []
    c._set_recording_audio_event = audio.append
    try:
        c._finish_recording(save=True, discard_reason="operator_discarded")
        assert c.episode_finalizer.wait_until_idle(timeout=5)
        assert c.episode_finalizer.status()["error"] is None
        quality = json.loads((writer.root / "meta/episode_quality.jsonl").read_text())
        assert quality["success"] is True
        assert quality["validation"]["frame_diagnostics"] == report["frame_diagnostics"]
        assert quality["validation"]["flagged_frame_count"] == 4
        table = pq.read_table(writer.root / writer.meta.get_data_file_path(0))
        assert table["episode.success"].to_pylist() == [1] * 5
        assert table["capture.ego_view_source_timestamp_ns"].to_pylist() == [
            99_000_000_000, 99_000_000_000, 100_000_000_000, 100_000_000_000, 99_000_000_000,
        ]
        for name in cameras:
            with av.open(str(writer.root / writer.meta.get_video_file_path(0, f"observation.images.{name}"))) as video:
                decoded = list(video.decode(video=0))
                assert len(decoded) == 5
                for index, frame in enumerate(decoded):
                    np.testing.assert_allclose(frame.to_ndarray(format="rgb24"),
                                               frames[index][f"observation.images.{name}"], atol=3)
        assert audio == ["saved"]
        assert "4 frames flagged" in c._recording_message
        assert c._episode_frame_diagnostics == {}
        assert c._episode_input_gaps == []
        assert c._episode_flagged_frames == 0
    finally:
        c.episode_finalizer.close()


def test_missing_measurements_report_gap_location_without_inventing_a_row(hand_recording):
    c, inputs, frames = hand_recording
    c._episode_state = EpisodeState()
    c._episode_state.change_state()
    c._recording_message = "Recording episode 0"
    c._last_input_block_log = 0
    c._latest_recording_inputs = lambda: inputs
    assert c._add_data_frame()
    good = inputs.proprio["body_q"]
    inputs.proprio["body_q"] = np.full(29, np.nan)
    assert not c._add_data_frame()
    assert not c._add_data_frame()
    assert len(frames) == 1
    inputs.proprio["body_q"] = good
    assert c._add_data_frame()
    report = c._episode_validation()
    [gap] = report["input_gaps"]
    assert gap["next_frame_index"] == 1
    assert gap["attempts"] == 2
    assert "NaN or Inf" in gap["reason"]
    assert len(frames) == 2
