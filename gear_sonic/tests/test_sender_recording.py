from collections import deque
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from gear_sonic.camera.composed_camera import CameraFrameBuffer, ComposedCameraClientSensor
from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.data.clock_sync import (
    ClockClient, ClockEstimate, ClockServer, ClockUnavailable, clock_id, estimate_exchange,
)
from gear_sonic.data.sender_sync import SenderSynchronizer, selection_inputs, synchronization_features
from gear_sonic.scripts.run_data_exporter import GrootDataCollector
from gear_sonic.scripts.run_data_exporter import _validate_recording_dataset_mode
from gear_sonic.utils.data_collection.episode_state import EpisodeState

MS = 1_000_000
LIMITS = {s: 0.1 for s in ("proprio", "camera", "manager", "sonic", "planner", "hand")}


def camera(name, source_ms, received_ms):
    return {"images": {name: np.full((2, 2, 3), source_ms % 255, dtype=np.uint8)},
            "depths": {}, "timestamps": {name: source_ms / 1000},
            "capture_monotonic_ns": {name: source_ms * MS},
            "receiver_monotonic_ns": received_ms * MS,
            "publisher_monotonic_ns": (source_ms + 1) * MS}


def populated(*, names=("ego_view",), mode=5):
    sync = SenderSynchronizer(camera_names=names)
    sync.start(1000 * MS)
    for timestamp in (980, 1020):
        for stream, data in (("proprio", {"body_q": np.ones(29)}),
                             ("manager", {"stream_mode": mode}),
                             ("planner", {"planner_mode": 1})):
            sync.observe(stream, data, timestamp * MS, (timestamp + 30) * MS)
        for name in names:
            sync.observe(f"camera.{name}", camera(name, timestamp, timestamp + 40),
                         timestamp * MS, (timestamp + 40) * MS)
    return sync


def test_source_time_selects_measurement_that_arrived_after_target():
    sync = populated()
    selected = sync.select(1000 * MS, hand=False, max_ages=LIMITS)
    assert selected.ready
    assert selected.samples["camera.ego_view"]["_sync_source_ns"] == 980 * MS
    assert selected.samples["camera.ego_view"]["_sync_received_ns"] == 1020 * MS
    assert all(s["_sync_time_ns"] <= selected.target_ns for s in selected.samples.values())
    inputs = selection_inputs(selected)
    assert inputs.mode == 5
    assert inputs.image["timestamps"]["ego_view"] == .980


def test_every_camera_must_advance_and_cached_images_cannot_advance_it():
    sync = populated(names=("left_wrist", "right_wrist"))
    sync.histories["camera.left_wrist"].pop()
    repeated = camera("left_wrist", 980, 1090)
    sync.observe("camera.left_wrist", repeated, 980 * MS, 1090 * MS)
    selection = sync.select(1000 * MS, hand=False, max_ages=LIMITS)
    assert not selection.ready
    assert selection.problems == ("camera.left_wrist: waiting for producer to advance",)
    assert sync.histories["camera.left_wrist"][0]["_sync_received_ns"] == 1020 * MS


def test_selected_manager_mode_controls_required_streams():
    sync = populated(mode=3)
    sync.histories["manager"][-1]["stream_mode"] = 1
    assert sync.select(1000 * MS, hand=False, max_ages=LIMITS).ready
    sync.histories["manager"][0]["stream_mode"] = 1
    assert any("sonic" in reason for reason in sync.select(1000 * MS, hand=False, max_ages=LIMITS).problems)


def test_remote_hand_mapping_and_uncertainty_exclude_ambiguous_future_sample():
    sync = populated()
    estimate = ClockEstimate("orin-boot", 5000 * MS, 2 * MS, 1000 * MS)
    sync.hand_clock = SimpleNamespace(estimate=lambda now: estimate)
    for stamp in (980, 999, 1020):
        sync.observe("hand", {"clock_id": "orin-boot"}, (stamp + 5000) * MS, (stamp + 10) * MS)
    selected = sync.select(1000 * MS, hand=True, max_ages=LIMITS)
    assert selected.ready
    hand = selected.samples["hand"]
    assert hand["_sync_time_ns"] == 980 * MS  # 999 +/- 2 could be after the target.
    assert hand["_sync_source_ns"] == 5980 * MS
    assert hand["_sync_offset_ns"] == 5000 * MS
    sync.observe("hand", {"clock_id": "wrong-host"}, 6030 * MS, 1040 * MS)
    assert not sync.select(1000 * MS, hand=True, max_ages=LIMITS).ready


def test_missing_source_time_never_falls_back_to_receipt_time():
    sync = populated()
    sync.observe("proprio", {}, -1, 1050 * MS)
    selection = sync.select(1000 * MS, hand=False, max_ages=LIMITS)
    assert not selection.ready
    assert "missing positive producer timestamp" in selection.problems[0]


def test_late_sample_and_out_of_order_timestamp_invalidate_episode():
    sync = populated()
    sync.advance(1000 * MS)
    sync.observe("proprio", {}, 990 * MS, 1110 * MS)
    assert any("out-of-order" in error for error in sync.errors)
    sync.observe("hand", {}, 995 * MS, 1110 * MS)
    assert sync.late == 1
    assert any("committed" in error for error in sync.errors)


def test_clock_exchange_accounts_for_processing_and_bounds_asymmetry():
    # True remote-local offset 5000; uplink 10, processing 30, downlink 20.
    estimate = estimate_exchange(1000, 6010, 6040, 1060, "boot")
    assert estimate.offset_ns == 4995
    assert estimate.uncertainty_ns == 15
    assert abs(estimate.to_local(7000) - 2000) <= estimate.uncertainty_ns
    with pytest.raises(ValueError):
        estimate_exchange(1000, 6010, 6040, 1020, "boot")


def test_clock_client_rejects_stale_or_uncertain_estimate():
    client = ClockClient.__new__(ClockClient)
    client._lock = threading.Lock()
    client._error = ""
    client.max_uncertainty_ns = 5 * MS
    client._samples = deque([ClockEstimate("boot", 5, 6 * MS, 1000 * MS)] * 3)
    with pytest.raises(ClockUnavailable, match="uncertainty"):
        client.estimate(1001 * MS)
    with pytest.raises(ClockUnavailable, match="fresh"):
        client.estimate(4000 * MS)


def test_clock_service_exchanges_on_loopback_and_shuts_down():
    server = ClockServer("tcp://127.0.0.1:*")
    client = None
    try:
        server.start()
        client = ClockClient(server.endpoint, max_uncertainty_ns=50 * MS)
        deadline = time.monotonic() + 3
        while True:
            try:
                estimate = client.estimate()
                break
            except ClockUnavailable:
                assert time.monotonic() < deadline
                time.sleep(.02)
        assert estimate.clock_id == clock_id()
        assert abs(estimate.offset_ns) <= estimate.uncertainty_ns
    finally:
        if client:
            client.close()
        server.close()
    assert not server._thread.is_alive()
    assert not client._thread.is_alive()


def receiver():
    client = ComposedCameraClientSensor.__new__(ComposedCameraClientSensor)
    client._background = client._preserve_history = True
    client._background_timestamps = {}
    client._background_frames = {}
    client._background_lock = threading.Lock()
    client._receiver_error = None
    client._history_overflow = 0
    client._background_buffer = CameraFrameBuffer()
    return client


def test_camera_drain_keeps_past_candidates_and_depth_pairing():
    client = receiver()
    first = camera("ego_view", 980, 1020)
    first["depths"]["ego_view"] = np.ones((2, 2), dtype=np.float32)
    decoded = ImageMessageSchema.deserialize(ImageMessageSchema(**{
        k: first[k] for k in ("images", "depths", "timestamps", "capture_monotonic_ns")
    }).serialize()).asdict()
    decoded["receiver_monotonic_ns"] = 1020 * MS
    client._buffer_camera_frames(decoded)
    client._buffer_camera_frames({**decoded, "receiver_monotonic_ns": 1030 * MS})
    client._buffer_camera_frames(camera("ego_view", 1020, 1060))
    pending = client.read_pending()
    assert len(pending) == 2
    assert pending[0]["capture_monotonic_ns"]["ego_view"] == 980 * MS
    assert pending[0]["receiver_monotonic_ns"] == 1020 * MS
    assert np.array_equal(pending[0]["depths"]["ego_view"], first["depths"]["ego_view"])
    assert client.read_pending() == []


def collector(sync):
    c = GrootDataCollector.__new__(GrootDataCollector)
    c._sender_sync = sync
    c.frequency = 50
    c.hand_config = None
    c.proprio_state_max_age = c.camera_max_age = c.hand_state_max_age = c.teleop_max_age = .1
    c._episode_state = EpisodeState()
    c._episode_state.change_state()
    c.data_exporter = SimpleNamespace(episode_buffer={"episode_index": 0})
    c._validate_recording_inputs = lambda inputs: None
    return c


def test_collector_waits_then_uses_explicit_inputs_and_drains_stop(monkeypatch):
    sync = populated()
    c = collector(sync)
    c.latest_proprio_msg = {"value": "future live state"}
    c.current_stream_mode = 1
    rows = []
    c._add_data_frame_sonic = lambda started, inputs: rows.append(inputs) or True
    now = 1099 * MS
    monkeypatch.setattr("gear_sonic.scripts.run_data_exporter.time.monotonic_ns", lambda: now)
    assert not c._add_sender_frame()
    assert rows == []
    now = 1100 * MS
    sync.stop(1000 * MS)
    assert c._add_sender_frame()
    assert len(rows) == 1 and rows[0].mode == 5
    assert rows[0].proprio is not c.latest_proprio_msg
    finished = []
    c._finish_recording = lambda **kwargs: finished.append(kwargs)
    assert c._add_sender_frame()
    assert len(rows) == 1 and finished == [{"save": True, "discard_reason": "operator_discarded"}]


def test_stalled_stream_creates_bounded_gap_without_a_partial_row(monkeypatch):
    sync = populated()
    sync.histories["camera.ego_view"].pop()
    c = collector(sync)
    c._add_data_frame_sonic = lambda *args: pytest.fail("partial row")
    now = 1349 * MS
    monkeypatch.setattr("gear_sonic.scripts.run_data_exporter.time.monotonic_ns", lambda: now)
    assert not c._add_sender_frame()
    assert sync.gaps == 0
    now = 1350 * MS
    assert not c._add_sender_frame()
    assert sync.gaps > 0
    assert sync.errors and "camera.ego_view" in sync.errors[0]


def test_selected_timestamp_metadata_preserves_clock_offset_and_receipt():
    sync = populated()
    selection = sync.select(1000 * MS, hand=False, max_ages=LIMITS)
    inputs = selection_inputs(selection)
    c = collector(sync)
    c.data_exporter.features = synchronization_features(("ego_view",))
    frame = {}
    c._add_capture_features(frame, inputs.proprio, inputs)
    assert frame["capture.sync_target_monotonic_ns"].tolist() == [1000 * MS]
    assert frame["capture.sync.camera.ego_view.source_ns"].tolist() == [980 * MS]
    assert frame["capture.sync.camera.ego_view.received_ns"].tolist() == [1020 * MS]
    assert frame["capture.sync.hand.time_ns"].tolist() == [-1]
    assert all(value.dtype == np.int64 for key, value in frame.items() if key.startswith("capture.sync"))


def test_asynchronous_cameras_select_closest_past_source_at_50hz():
    names = ("ego_view", "left_wrist", "right_wrist")
    sync = SenderSynchronizer(camera_names=names)
    client = receiver()
    sources = {name: [] for name in names}
    base = 1000 * MS
    for tick in range(600):
        now = base + round(tick * 1e9 / 300)
        for phase, name in enumerate(names):
            if tick % 5 == phase:
                source = now - 8 * MS
                frame = camera(name, 1, 1)
                frame["capture_monotonic_ns"][name] = source
                frame["timestamps"][name] = source / 1e9
                frame["receiver_monotonic_ns"] = now
                client._buffer_camera_frames(frame)
                sources[name].append(source)
        if tick % 6 != 5:
            continue
        for message in client.read_pending():
            name = next(iter(message["timestamps"]))
            sync.observe(f"camera.{name}", message, message["capture_monotonic_ns"][name],
                         message["receiver_monotonic_ns"])
        for stream, payload in (("proprio", {}), ("manager", {"stream_mode": 3})):
            sync.observe(stream, payload, now - MS, now)
        if tick < 40:
            continue
        target = now - sync.delay_ns
        selection = sync.select(target, hand=False, max_ages=LIMITS)
        assert selection.ready, selection.problems
        for name in names:
            assert selection.samples[f"camera.{name}"]["_sync_source_ns"] == max(
                stamp for stamp in sources[name] if stamp <= target
            )
        sync.trim(target)
    assert client._history_overflow == sync.dropped == 0


def test_actual_frame_builder_keeps_selected_values_and_timestamp_metadata():
    from gear_sonic.end_effectors.profiles import get_hand_profile
    from gear_sonic.scripts.run_data_exporter import TimingThresholdMonitor

    sync = populated(mode=5)
    proprio = sync.histories["proprio"][0]
    proprio.update({"body_dq": np.ones(29), "base_quat": np.array([1., 0, 0, 0]),
                    "base_ang_vel": np.zeros(3), "last_action": np.full(29, .25),
                    "token_state": np.ones(64), "left_hand_q": np.zeros(7),
                    "right_hand_q": np.zeros(7), "last_left_hand_action": np.zeros(7),
                    "last_right_hand_action": np.zeros(7)})
    sync.histories["planner"][0].update({"receive_monotonic": 1.010,
        "planner_speed": .4, "planner_height": .75,
        "receive_timestamp": time.time(), "vr_3pt_position": np.ones(9),
        "vr_3pt_orientation": np.tile([1., 0, 0, 0], 3)})
    inputs = selection_inputs(sync.select(1000 * MS, hand=False, max_ages=LIMITS))
    c = collector(sync)
    c.hand_profile = get_hand_profile("dex3.v1")
    c.required_stream_mode = 5
    c.sonic_timing_monitor = TimingThresholdMonitor()
    c._initial_yaw = None
    c._log_latency_periodic = lambda *args, **kwargs: None
    c.robot_model = SimpleNamespace(
        get_configuration_from_actuated_joints=lambda **kw: np.concatenate(list(kw.values())),
        cache_forward_kinematics=lambda q: None,
        frame_placement=lambda frame: SimpleNamespace(translation=np.zeros(3), rotation=np.eye(3)),
        supplemental_info=SimpleNamespace(hand_frame_names={"left": "left", "right": "right"}),
    )
    frames = []
    c.data_exporter.features = {"observation.images.ego_view": {"dtype": "video", "shape": (2, 2, 3)},
                                **synchronization_features(("ego_view",))}
    c.data_exporter.add_frame = frames.append
    c.latest_proprio_msg = {"body_q": np.full(29, 999.)}
    c.latest_planner_msg = {"planner_mode": 999}
    c.latest_image_msg = camera("ego_view", 1200, 1200)
    c.current_stream_mode = 1
    # Validate against target time even though wall time is far beyond it.
    GrootDataCollector._validate_recording_inputs(c, inputs)
    assert c._add_data_frame_sonic(time.monotonic(), inputs)
    [frame] = frames
    assert np.all(frame["observation.state"][:29] == 1)
    assert np.all(frame["action.wbc"][:29] == .25)
    assert frame["teleop.stream_mode"].tolist() == [5]
    assert frame["teleop.vr_3pt_valid"].tolist() == [1]
    assert np.array_equal(frame["observation.images.ego_view"], inputs.image["images"]["ego_view"])
    assert frame["capture.sync_target_monotonic_ns"].tolist() == [1000 * MS]


def test_sync_metadata_round_trips_and_legacy_resume_is_rejected(tmp_path):
    from gear_sonic.data.exporter import Gr00tDataExporter
    import pandas as pd

    root = tmp_path / "synchronized"
    features = {"state": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
                **synchronization_features(())}
    writer = Gr00tDataExporter.create(
        save_root=root, fps=50, task="test", robot_type="test", features=features,
        modality_config={name: {} for name in ("state", "action", "video", "annotation")},
    )
    exact_ns = 9_007_199_254_740_993
    for index in range(2):
        frame = {key: np.asarray([exact_ns + index * 20 * MS], dtype=np.int64)
                 for key in features if key.startswith("capture.")}
        writer.add_frame({"state": np.asarray([index], dtype=np.float32), **frame})
    writer.save_episode()
    table = pd.read_parquet(next((root / "data").rglob("*.parquet")))
    assert int(table["capture.sync_target_monotonic_ns"].iloc[0]) == exact_ns
    _validate_recording_dataset_mode(root, features, True)
    before = (root / "meta/info.json").read_bytes()
    with pytest.raises(ValueError, match="new --dataset-name"):
        _validate_recording_dataset_mode(root, {"state": features["state"]}, False)
    assert (root / "meta/info.json").read_bytes() == before
    writer = Gr00tDataExporter.create(
        save_root=root, fps=50, task="test", robot_type="test", features=features,
        modality_config={name: {} for name in ("state", "action", "video", "annotation")},
    )
    assert writer.episode_buffer["episode_index"] == 1


def test_launcher_forwards_sender_mode_and_clock_service():
    import shlex
    from gear_sonic.scripts.launch_data_collection import DataCollectionLaunchConfig, _remote_hand_command
    from gear_sonic.end_effectors.server import build_parser, worker_command

    config = DataCollectionLaunchConfig(hand_backend="dex1", hand_server_host="orin", sender_time_recording=True)
    command = shlex.split(_remote_hand_command(config))[-1]
    assert "--clock-port 5574" in command
    args = build_parser().parse_args(["--clock-port", "5574", "--enable-command"])
    assert "--enable-command" in worker_command(args)
    args.clock_port = args.state_port
    with pytest.raises(ValueError, match="distinct"):
        worker_command(args)
