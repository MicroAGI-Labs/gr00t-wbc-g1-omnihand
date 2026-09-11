"""Local saving and an explicit, idle-only upload of all committed episodes."""

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gear_sonic.scripts.run_camera_web_viewer import CameraWebViewerConfig, RecorderControlHub
from gear_sonic.scripts.run_data_exporter import GrootDataCollector
from gear_sonic.data.episode_finalizer import EpisodeFinalizer
from gear_sonic.data.hub_uploader import EpisodeHubUploader
from gear_sonic.tests.test_memory_recording import exporter_at, frame
from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.data_collection.hub_config import DATASET_CONFIG_PREFIX


def collector():
    c = GrootDataCollector.__new__(GrootDataCollector)
    c._episode_state = EpisodeState()
    c._keyboard_listener = SimpleNamespace(read_msg=lambda: "upload")
    c.data_exporter = SimpleNamespace(
        episode_buffer={"size": 0}, task="pick up the cup",
        meta=SimpleNamespace(info={"total_episodes": 3}),
    )
    c.episode_finalizer = SimpleNamespace(status=lambda: {"pending": 0, "finalizing": False, "error": None})
    c.hub_uploader = SimpleNamespace(
        status=lambda: {"ready": True, "pending": 0, "uploading": False}, enqueue=Mock(),
    )
    return c


def test_explicit_upload_includes_the_latest_locally_saved_episode():
    c = collector()
    c._check_recording_commands()
    c.hub_uploader.enqueue.assert_called_once_with(2)
    assert "all locally saved recordings" in c._recording_message


@pytest.mark.parametrize("blocked", ["recording", "draining", "saving", "save_error", "buffered", "empty", "no_destination"])
def test_upload_waits_for_committed_local_data(blocked):
    c = collector()
    if blocked in {"recording", "draining"}:
        c._episode_state.change_state()
        if blocked == "draining":
            c._episode_state.change_state()
    if blocked == "saving":
        c.episode_finalizer.status = lambda: {"pending": 1, "finalizing": True}
    if blocked == "save_error":
        c.episode_finalizer.status = lambda: {"error": "disk full"}
    if blocked == "buffered":
        c.data_exporter.episode_buffer["size"] = 10
    if blocked == "empty":
        c.data_exporter.meta.info["total_episodes"] = 0
    if blocked == "no_destination":
        c.hub_uploader.status = lambda: {"ready": False}
    c._check_recording_commands()
    c.hub_uploader.enqueue.assert_not_called()
    assert c._recording_message.startswith("Upload not started:")


def test_repeated_upload_click_does_not_queue_duplicate_work():
    c = collector()
    c.hub_uploader.status = lambda: {"ready": True, "pending": 1, "uploading": False}
    c._check_recording_commands()
    c.hub_uploader.enqueue.assert_not_called()
    assert c._recording_message == "Upload already in progress"


def test_upload_destination_can_be_selected_after_local_saves(monkeypatch):
    import gear_sonic.scripts.run_camera_web_viewer as viewer

    api = Mock()
    api.list_repo_files.return_value = []
    monkeypatch.setattr(viewer, "HfApi", lambda: api)
    browser = RecorderControlHub(CameraWebViewerConfig(hf_namespace="org"))
    browser._status = {"recording": False, "saving": False, "total_episodes": 3,
                       "hub": {"prompt": "pick up the cup"}}
    browser._status_received_at = time.monotonic()
    browser.configure_dataset("dataset", "pick up the cup", True)
    command = browser._commands.get_nowait()
    c = collector()
    c._keyboard_listener.read_msg = lambda: command
    c.hub_uploader.configure = Mock()
    c._check_recording_commands()
    c.hub_uploader.configure.assert_called_once_with(
        repo_id="org/dataset", prompt="pick up the cup", private=True,
    )
    c.hub_uploader.enqueue.assert_not_called()
    assert "recordings stay local" in c._recording_message
    with pytest.raises(RuntimeError, match="prompt is locked"):
        browser.configure_dataset("dataset", "a different task", True)
    assert json.loads(command[len(DATASET_CONFIG_PREFIX):])["prompt"] == c.data_exporter.task


def test_browser_upload_is_a_separate_command():
    browser = RecorderControlHub(CameraWebViewerConfig())
    assert browser._commands.empty()
    browser.send("upload")
    assert browser._commands.get_nowait() == "upload"
    assert browser._commands.empty()


def test_two_local_saves_upload_together_only_after_explicit_request(tmp_path):
    exporter = exporter_at(tmp_path / "dataset")
    snapshots = []
    def upload(snapshot, config):
        snapshots.append((
            len(list((snapshot / "videos").rglob("*.mp4"))),
            len(list((snapshot / "data").rglob("*.parquet"))),
        ))
    hub = EpisodeHubUploader(exporter, upload_runner=upload)
    hub.configure("org/dataset", exporter.task, True)
    finalizer = EpisodeFinalizer(exporter, hub)
    try:
        for index in range(2):
            exporter.add_frame(frame())
            completed, writers = exporter.detach_episode()
            finalizer.enqueue(episode_index=index, episode_buffer=completed, video_writers=writers,
                              success=True, validation={"passed": True})
            assert finalizer.wait_until_idle(timeout=5)
            assert finalizer.status()["error"] is None
            assert finalizer.can_record()
            assert hub.status()["pending"] == 0
            assert snapshots == []
            assert not (exporter.root / ".upload_snapshots").exists()
        c = collector()
        c.data_exporter, c.episode_finalizer, c.hub_uploader = exporter, finalizer, hub
        c._check_recording_commands()
        assert hub.wait_until_idle(timeout=5)
        assert snapshots == [(2, 2)]
        assert hub.status()["last_uploaded_episode"] == 1
    finally:
        finalizer.close()
        hub.close()


def test_local_recording_starts_without_a_hub_destination():
    c = collector()
    c._keyboard_listener.read_msg = lambda: "c"
    c._manager_toggle_dc = c._manager_toggle_da = False
    c._manager_discard_reason = None
    c.current_stream_mode = c.required_stream_mode = 5
    c.require_hub_upload = True  # Even an older launcher must not gate local recording.
    c._sender_sync = None
    c._episode_input_errors = set()
    c._set_recording_audio_event = Mock()
    c._print_and_say = Mock()
    c.data_exporter.episode_buffer["episode_index"] = 3
    c.episode_finalizer.can_record = lambda: True
    c.hub_uploader.status = lambda: {"ready": False, "pending": 0, "uploading": False}
    c._check_recording_commands()
    assert c._episode_state.get_state() == c._episode_state.RECORDING
    c.hub_uploader.enqueue.assert_not_called()
