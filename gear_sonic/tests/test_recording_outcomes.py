"""Success/failure saves and explicit discard reach distinct disk outcomes."""

import json
from types import SimpleNamespace

import av
import numpy as np
import pyarrow.parquet as pq
import pytest

from gear_sonic.data.episode_finalizer import EpisodeFinalizer
from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.scripts.run_data_exporter import GrootDataCollector
from gear_sonic.scripts.process_dataset import process_single_dataset
from gear_sonic.utils.data_collection.episode_state import EpisodeState

CAMERA = "observation.images.ego_view"


@pytest.mark.parametrize("outcome,saved,success", [
    ("operator", False, False),
    ("headset", False, False),
    ("duration_limit", False, False),
    ("operator_failure", True, False),
    ("headset_failure", True, False),
    ("validation_failed", True, False),
    ("accepted", True, True),
    ("collector_shutdown_before_episode_save", True, False),
    ("recording_memory_limit", True, False),
])
def test_collector_outcome_reaches_the_correct_disk_action(tmp_path, outcome, saved, success):
    exporter = Gr00tDataExporter.create(
        save_root=tmp_path / "dataset", fps=50, task="outcome test",
        features={
            CAMERA: {"dtype": "video", "shape": (16, 16, 3),
                     "names": ["height", "width", "channels"]},
            "episode.success": {"dtype": "uint8", "shape": (1,), "names": ["success"]},
        },
        modality_config={key: {} for key in ("state", "action", "video", "annotation")},
    )
    exporter.add_frame({CAMERA: np.full((16, 16, 3), 90, dtype=np.uint8),
                        "episode.success": np.ones(1, dtype=np.uint8)})
    finalizer = EpisodeFinalizer(exporter)
    c = GrootDataCollector.__new__(GrootDataCollector)
    c.data_exporter = exporter
    c.episode_finalizer = finalizer
    c._episode_state = EpisodeState()
    c._episode_state.change_state()
    c.max_episode_duration_s = 240
    c._episode_elapsed_s = lambda: 240 if outcome == "duration_limit" else 1
    c._episode_input_errors = set()
    c.sonic_timing_monitor = SimpleNamespace(reset=lambda: None)
    c._sender_sync = None
    c._manager_toggle_dc = False
    c._manager_toggle_da = outcome == "headset"
    c._manager_toggle_df = outcome == "headset_failure"
    c._manager_discard_reason = None
    key = ("x" if outcome == "operator" else "f" if outcome == "operator_failure"
           else "c" if outcome in {"accepted", "validation_failed"} else None)
    c._keyboard_listener = SimpleNamespace(read_msg=lambda: key)
    c._episode_validation = lambda: {"passed": success,
        "errors": ["missing required camera sample"] if outcome == "validation_failed" else [],
        "frame_diagnostics": {"camera ego_view is stale": {"frames": 1, "frame_ranges": [[0, 0]]}},
        "flagged_frame_count": 1}
    c._print_and_say = lambda *args, **kwargs: None
    events = []
    c._set_recording_audio_event = events.append
    try:
        if outcome in {"collector_shutdown_before_episode_save", "recording_memory_limit"}:
            c._finish_recording(save=False, discard_reason=outcome)
        else:
            c._check_recording_commands()
        assert finalizer.wait_until_idle(timeout=5)
        assert finalizer.status()["error"] is None
        assert exporter.meta.total_episodes == int(saved)
        assert exporter.meta.total_frames == int(saved)
        assert exporter.episode_buffer["episode_index"] == int(saved)
        assert exporter.meta.info["discarded_episode_indices"] == []
        assert exporter.meta.info["failed_episode_indices"] == ([0] if saved and not success else [])
        assert not list((exporter.root / ".recording").rglob("*.mp4"))
        assert not (exporter.root / "recovery").exists()
        if saved:
            table = pq.read_table(exporter.root / exporter.meta.get_data_file_path(0))
            assert table["episode.success"].to_pylist() == [int(success)]
            with av.open(str(exporter.root / exporter.meta.get_video_file_path(0, CAMERA))) as video:
                assert len(list(video.decode(video=0))) == 1
            quality = json.loads((exporter.root / "meta/episode_quality.jsonl").read_text())
            assert quality["success"] is success
            assert quality["validation"]["frame_diagnostics"]["camera ego_view is stale"]["frame_ranges"] == [[0, 0]]
            assert exporter.meta.info["episode_quality"]["0"]["discarded"] is False
            stats, processed, _ = process_single_dataset(
                exporter.root, remove_stale_smpl=False, remove_discarded=True,
            )
            assert stats["episodes_discarded"] == 0
            assert len(processed) == 1
            assert processed[0]["df"]["episode.success"].tolist() == [int(success)]
            if outcome == "validation_failed":
                assert quality["validation"]["errors"] == ["missing required camera sample"]
            if outcome in {"operator_failure", "headset_failure"}:
                assert quality["validation"]["failure_reason"] == "operator_marked_failure"
                assert "marked as failed" in c._recording_message
            if outcome == "duration_limit":
                assert quality["validation"]["failure_reason"] == "episode_duration_limit"
            assert events == ["saved" if success else "validation_failed"]
        else:
            assert not list(exporter.root.rglob("*.mp4"))
            assert not list(exporter.root.rglob("*.parquet"))
            assert not (exporter.root / "meta/episode_quality.jsonl").exists()
            assert events == ["discard"]
    finally:
        finalizer.close()


def test_browser_failure_and_discard_routes_send_distinct_controls():
    from http.server import ThreadingHTTPServer
    import threading
    from urllib.request import Request, urlopen
    from gear_sonic.scripts.run_camera_web_viewer import CameraWebViewerConfig, RecorderControlHub, make_handler

    hub = RecorderControlHub(CameraWebViewerConfig())
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(None, hub, None, None, None))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base, timeout=2) as response:
            assert "Stop &amp; Save Failure" in response.read().decode()
        for path in ("/recording/failure", "/recording/discard"):
            with urlopen(Request(base + path, method="POST"), timeout=2) as response:
                assert json.load(response) == {"accepted": True}
        assert [hub._commands.get_nowait(), hub._commands.get_nowait()] == ["f", "x"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
