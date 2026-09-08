"""Offline tests for upload ownership, backpressure, and recorder configuration."""

from http.server import ThreadingHTTPServer
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gear_sonic.data.episode_finalizer import EpisodeFinalizer
from gear_sonic.data.hub_uploader import DATASET_CONFIG_PREFIX, EpisodeHubUploader, upload_snapshot
from gear_sonic.scripts.run_camera_web_viewer import CameraWebViewerConfig, RecorderControlHub, make_handler
from gear_sonic.scripts.run_data_exporter import GrootDataCollector
from gear_sonic.utils.data_collection.episode_state import EpisodeState

CONFIG = {"repo_id": "org/dataset", "prompt": "pick up the cup", "private": True}


class FakeExporter:
    def __init__(self, root):
        self.root = root
        self.task = "demo"
        self.meta = SimpleNamespace(info={"total_episodes": 0}, repo_id="local")
        self.episode_buffer = {"size": 0}
        (root / "meta").mkdir(exist_ok=True)

    def save_episode(self, buffer, **kwargs):
        index = buffer["episode_index"]
        self.meta.info["total_episodes"] = index + 1
        (self.root / "meta/info.json").write_text(json.dumps(self.meta.info))
        for directory, suffix in (("data/chunk-000", "parquet"), ("videos/chunk-000/ego", "mp4")):
            folder = self.root / directory
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"episode_{index:06d}.{suffix}").write_bytes(b"finalized")

    save_episode_as_discarded = save_episode


def finalize(finalizer, index=0, *, discarded=False):
    finalizer.enqueue(
        episode_index=index, episode_buffer={"episode_index": index},
        video_writers={}, discarded=discarded, validation={"passed": not discarded},
    )
    assert finalizer.wait_until_idle(timeout=2)
    assert finalizer.drain_results()[0].succeeded


def test_finalization_continues_during_upload_with_bounded_immutable_snapshots(tmp_path):
    exporter = FakeExporter(tmp_path)
    entered, release = threading.Event(), threading.Event()
    snapshots = []

    def upload(snapshot, config):
        entered.set()
        assert release.wait(timeout=5)
        snapshots.append(json.loads((snapshot / "meta/info.json").read_text())["total_episodes"])
        assert config["repo_id"] == CONFIG["repo_id"]
        assert len(list((snapshot / "data").rglob("*.parquet"))) == snapshots[-1]
        assert not (snapshot / ".hub_upload.json").exists()
        assert not (snapshot / "recovery").exists()

    hub = EpisodeHubUploader(exporter, upload_runner=upload)
    hub.configure(CONFIG)
    finalizer = EpisodeFinalizer(exporter, hub_uploader=hub)
    try:
        finalize(finalizer)
        assert entered.wait(timeout=1)
        for index in (1, 2, 3):
            finalize(finalizer, index, discarded=index == 2)
        assert finalizer.can_accept()
        assert hub.status()["pending"] == 2
        assert len(list((tmp_path / ".upload_snapshots").iterdir())) == 2
        release.set()
        assert hub.wait_until_idle(timeout=3)
        assert snapshots == [1, 4]
        assert hub.status()["last_uploaded_episode"] == 3
    finally:
        release.set()
        finalizer.close()
        hub.close()


def test_staging_failure_preserves_local_save_and_retries_on_next_save(tmp_path, monkeypatch):
    exporter = FakeExporter(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def upload(*_):
        entered.set()
        assert release.wait(timeout=5)

    uploaded = Mock(side_effect=upload)
    hub = EpisodeHubUploader(exporter, upload_runner=uploaded)
    hub.configure(CONFIG)
    finalizer = EpisodeFinalizer(exporter, hub_uploader=hub)
    try:
        finalize(finalizer)
        assert entered.wait(timeout=1)
        with monkeypatch.context() as patch:
            patch.setattr(hub, "_snapshot", Mock(side_effect=OSError("disk full")))
            finalize(finalizer, 1)
        release.set()
        assert hub.wait_until_idle(timeout=2)
        # Completing an older upload must not hide a newer staging failure.
        assert "disk full" in hub.status()["error"]
        assert finalizer.status()["error"] is None
        finalize(finalizer, 2)
        assert hub.wait_until_idle(timeout=2)
        assert uploaded.call_count == 2
        assert hub.status()["error"] is None
    finally:
        release.set()
        finalizer.close()
        hub.close()


def test_failed_upload_retries_and_configuration_survives_restart(tmp_path):
    exporter = FakeExporter(tmp_path)
    upload = Mock(side_effect=[OSError("offline"), None])
    hub = EpisodeHubUploader(exporter, upload_runner=upload)
    hub.configure(CONFIG)
    exporter.save_episode({"episode_index": 0})
    hub.enqueue(0)
    try:
        assert hub.wait_until_idle(timeout=4)
        assert upload.call_count == 2
        with pytest.raises(RuntimeError, match="locked"):
            hub.configure({**CONFIG, "repo_id": "org/different"})
        hub.configure(CONFIG)
    finally:
        hub.close()
    resumed_upload = Mock()
    resumed = EpisodeHubUploader(exporter, upload_runner=resumed_upload)
    try:
        assert resumed.wait_until_idle(timeout=2)
        assert resumed.status()["repo_id"] == CONFIG["repo_id"]
        assert resumed_upload.call_count == 1
    finally:
        resumed.close()


def test_shutdown_stops_upload_child_without_reporting_success(tmp_path, monkeypatch):
    import gear_sonic.data.hub_uploader as module

    entered, terminated = threading.Event(), threading.Event()
    process = Mock(returncode=None)

    def communicate():
        entered.set()
        assert terminated.wait(timeout=3)
        process.returncode = -15
        return None, "interrupted"

    process.communicate.side_effect = communicate
    process.terminate.side_effect = terminated.set
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    exporter = FakeExporter(tmp_path)
    hub = EpisodeHubUploader(exporter)
    hub.configure(CONFIG)
    exporter.save_episode({"episode_index": 0})
    hub.enqueue(0)
    assert entered.wait(timeout=2)
    hub.close(timeout=0)
    assert not hub._thread.is_alive()
    assert hub.status()["last_uploaded_episode"] is None
    assert "--private" in spawn.call_args.args[0]
    assert (tmp_path / "data/chunk-000/episode_000000.parquet").exists()
    assert list((tmp_path / ".upload_snapshots").iterdir()) == []


@pytest.mark.parametrize("remote", ["empty", "ours", "foreign", "populated", "public"])
def test_remote_ownership_and_atomic_upload(tmp_path, monkeypatch, remote):
    import huggingface_hub

    import gear_sonic.data.hub_uploader as module

    monkeypatch.setattr(module.os, "nice", lambda _: None)
    monkeypatch.setattr(module.os, "sched_setaffinity", lambda *_: None)
    (tmp_path / "sonic_dataset.json").write_text('{"source_id":"ours"}')
    (tmp_path / "README.md").write_text("dataset card")
    remote_identity = tmp_path / "remote.json"
    remote_identity.write_text(json.dumps({"source_id": "foreign" if remote == "foreign" else "ours"}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: remote_identity)
    api = Mock(spec=huggingface_hub.HfApi)
    api.repo_info.return_value = SimpleNamespace(private=remote != "public", sha="parent")
    api.list_repo_files.return_value = {
        "ours": ["sonic_dataset.json"], "foreign": ["sonic_dataset.json"],
        "populated": ["meta/info.json"],
    }.get(remote, [])
    api.upload_folder.return_value = SimpleNamespace(oid="uploaded")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    if remote in {"foreign", "populated", "public"}:
        with pytest.raises(ValueError):
            upload_snapshot(tmp_path, "org/dataset", private=True)
        api.upload_folder.assert_not_called()
    else:
        upload_snapshot(tmp_path, "org/dataset", private=True)
        assert api.upload_folder.call_args.kwargs["parent_commit"] == "parent"
        assert api.create_tag.call_args.kwargs["revision"] == "uploaded"


def test_browser_configuration_is_validated_and_queued(monkeypatch):
    import gear_sonic.scripts.run_camera_web_viewer as viewer

    prepare = Mock()
    monkeypatch.setattr(viewer, "prepare_repository", prepare)
    hub = RecorderControlHub(CameraWebViewerConfig())
    hub._status = {"recording": False, "saving": False, "total_episodes": 0}
    hub._status_received_at = time.monotonic()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(None, hub))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/recording/dataset"
        headers = {"Content-Type": "application/json", "X-Sonic-Command": "dataset"}
        with urlopen(Request(url, data=json.dumps(CONFIG).encode(), headers=headers), timeout=2) as response:
            assert json.load(response) == {"accepted": True}
        assert json.loads(hub._commands.get_nowait()[len(DATASET_CONFIG_PREFIX):]) == CONFIG
        prepare.assert_called_once_with(CONFIG)
        for payload in ([], {**CONFIG, "private": "false"}, {**CONFIG, "repo_id": "../dataset"}):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(url, data=json.dumps(payload).encode(), headers=headers), timeout=2)
            assert error.value.code == 400
        assert hub._commands.empty()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("busy", [None, "recording", "finalizing", "saved"])
def test_collector_owns_configuration_lock(tmp_path, busy):
    exporter = FakeExporter(tmp_path)
    hub = EpisodeHubUploader(exporter, upload_runner=Mock())
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector.data_exporter = exporter
    collector.hub_uploader = hub
    collector._episode_state = EpisodeState()
    if busy == "recording":
        collector._episode_state.change_state()
    if busy == "saved":
        exporter.meta.info["total_episodes"] = 1
    collector.episode_finalizer = SimpleNamespace(can_accept=lambda: busy != "finalizing")
    collector._keyboard_listener = SimpleNamespace(read_msg=lambda: DATASET_CONFIG_PREFIX + json.dumps(CONFIG))
    try:
        collector._check_recording_commands()
        assert hub.status()["ready"] == (busy is None)
        assert exporter.task == ("demo" if busy else CONFIG["prompt"])
    finally:
        hub.close()
