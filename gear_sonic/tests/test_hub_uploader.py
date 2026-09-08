"""Offline upload tests: committed snapshots, failure recovery, and configuration."""

from contextlib import closing
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gear_sonic.data.episode_finalizer import EpisodeFinalizer
from gear_sonic.data.hub_uploader import DATASET_CONFIG_PREFIX, EpisodeHubUploader, upload_snapshot
from gear_sonic.scripts.run_camera_web_viewer import CameraWebViewerConfig, RecorderControlHub
from gear_sonic.scripts.run_data_exporter import GrootDataCollector
from gear_sonic.utils.data_collection.episode_state import EpisodeState

CONFIG = {"repo_id": "org/dataset", "prompt": "pick up the cup", "private": True}


class FakeExporter:
    def __init__(self, root):
        self.root, self.task = root, "demo"
        self.meta = SimpleNamespace(info={"total_episodes": 0}, repo_id="local")
        self.episode_buffer = {"size": 0}
        (root / "meta").mkdir()

    def get_episodes_file_paths(self):
        return [
            f"{directory}/episode_{index:06d}.{suffix}"
            for index in range(self.meta.info["total_episodes"])
            for directory, suffix in (("data/chunk-000", "parquet"), ("videos/chunk-000/ego", "mp4"))
        ]

    def save_episode(self, buffer, **kwargs):
        self.meta.info["total_episodes"] = buffer["episode_index"] + 1
        (self.root / "meta/info.json").write_text(json.dumps(self.meta.info))
        for relative in self.get_episodes_file_paths()[-2:]:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"finalized")

    save_episode_as_discarded = save_episode


@pytest.fixture
def session(tmp_path):
    exporter = FakeExporter(tmp_path)
    entered, release = threading.Event(), threading.Event()
    snapshots = []
    release.set()

    def upload(snapshot, config):
        entered.set()
        assert release.wait(timeout=5)
        count = json.loads((snapshot / "meta/info.json").read_text())["total_episodes"]
        assert len(list((snapshot / "data").rglob("*.parquet"))) == count
        assert not (snapshot / ".hub_upload.json").exists() and not (snapshot / "recovery").exists()
        snapshots.append(count)

    hub = EpisodeHubUploader(exporter, upload_runner=upload)
    finalizer = EpisodeFinalizer(exporter, hub_uploader=hub)

    def save(index):
        finalizer.enqueue(episode_index=index, episode_buffer={"episode_index": index},
                          video_writers={}, discarded=False, validation={"passed": True})
        assert finalizer.wait_until_idle(timeout=2)
        assert finalizer.drain_results()[0].succeeded

    try:
        yield SimpleNamespace(hub=hub, exporter=exporter, finalizer=finalizer, save=save,
                              entered=entered, release=release, snapshots=snapshots)
    finally:
        release.set()
        finalizer.close()
        hub.close()


def test_upload_backlog_and_staging_failure_do_not_block_local_saves(session, monkeypatch):
    s = session
    s.hub.configure(CONFIG)
    s.release.clear()
    s.save(0)
    assert s.entered.wait(timeout=1)
    for index in (1, 2):
        s.save(index)
    assert s.hub.status()["pending"] == 2
    assert len(list((s.exporter.root / ".upload_snapshots").iterdir())) == 2
    with monkeypatch.context() as patch:
        patch.setattr(s.hub, "_snapshot", Mock(side_effect=OSError("disk full")))
        s.save(3)
    s.release.set()
    assert s.hub.wait_until_idle(timeout=3)
    assert s.snapshots == [1, 3]  # Each upload keeps its original metadata.
    assert "disk full" in s.hub.status()["error"]  # Older success cannot hide this.
    assert s.finalizer.can_accept()
    s.save(4)
    assert s.hub.wait_until_idle(timeout=2)
    assert s.snapshots == [1, 3, 5]
    assert s.hub.status()["last_uploaded_episode"] == 4
    assert s.hub.status()["error"] is None


def test_retry_configuration_lock_and_restart(session):
    s = session
    s.hub.configure(CONFIG)
    s.hub._upload_runner = Mock(side_effect=[OSError("offline"), None])
    s.save(0)
    assert s.hub.wait_until_idle(timeout=4)
    assert s.hub._upload_runner.call_count == 2
    with pytest.raises(RuntimeError, match="locked"):
        s.hub.configure({**CONFIG, "repo_id": "org/different"})
    s.hub.configure(CONFIG)  # Duplicate delivery is harmless.
    s.hub.close()
    with closing(EpisodeHubUploader(s.exporter, upload_runner=Mock())) as resumed:
        assert resumed.wait_until_idle(timeout=2)
        assert resumed.status()["repo_id"] == CONFIG["repo_id"]
        resumed._upload_runner.assert_called_once()


def test_shutdown_stops_child_without_claiming_success(session, monkeypatch):
    import gear_sonic.data.hub_uploader as module

    s = session
    s.release.clear()
    process = Mock(returncode=None)

    def communicate():
        s.entered.set()
        assert s.release.wait(timeout=3)
        process.returncode = -15
        return None, "interrupted"

    process.communicate.side_effect = communicate
    process.terminate.side_effect = s.release.set
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    s.hub._upload_runner = s.hub._run_subprocess
    s.hub.configure(CONFIG)
    s.save(0)
    assert s.entered.wait(timeout=2)
    s.hub.close(timeout=0)
    assert not s.hub._thread.is_alive()
    assert s.hub.status()["last_uploaded_episode"] is None
    assert "--private" in spawn.call_args.args[0]
    assert (s.exporter.root / "data/chunk-000/episode_000000.parquet").exists()
    assert list((s.exporter.root / ".upload_snapshots").iterdir()) == []


@pytest.mark.parametrize("remote", ["empty", "ours", "foreign", "populated", "public"])
def test_remote_ownership_and_atomic_upload(tmp_path, monkeypatch, remote):
    import huggingface_hub

    import gear_sonic.data.hub_uploader as module

    monkeypatch.setattr(module.os, "nice", lambda _: None)
    monkeypatch.setattr(module.os, "sched_setaffinity", lambda *_: None)
    (tmp_path / "sonic_dataset.json").write_text('{"source_id":"ours"}')
    (tmp_path / "README.md").write_text("dataset card")
    identity = tmp_path / "remote.json"
    identity.write_text(json.dumps({"source_id": "foreign" if remote == "foreign" else "ours"}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: identity)
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


def test_browser_validates_and_queues_configuration(monkeypatch):
    import gear_sonic.scripts.run_camera_web_viewer as viewer

    prepare = Mock()
    monkeypatch.setattr(viewer, "prepare_repository", prepare)
    hub = RecorderControlHub(CameraWebViewerConfig())
    hub._status = {"recording": False, "saving": False, "total_episodes": 0}
    hub._status_received_at = time.monotonic()
    hub.configure_dataset(CONFIG)
    assert json.loads(hub._commands.get_nowait()[len(DATASET_CONFIG_PREFIX):]) == CONFIG
    prepare.assert_called_once_with(CONFIG)
    for payload in ([], {**CONFIG, "private": "false"}, {**CONFIG, "repo_id": "../dataset"}):
        with pytest.raises(ValueError):
            hub.configure_dataset(payload)
    assert hub._commands.empty()


@pytest.mark.parametrize("busy", [None, "recording", "finalizing", "saved"])
def test_collector_owns_configuration_lock(session, busy):
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector.data_exporter, collector.hub_uploader = session.exporter, session.hub
    collector._episode_state = EpisodeState()
    if busy == "recording":
        collector._episode_state.change_state()
    if busy == "saved":
        session.exporter.meta.info["total_episodes"] = 1
    collector.episode_finalizer = SimpleNamespace(can_accept=lambda: busy != "finalizing")
    collector._keyboard_listener = SimpleNamespace(read_msg=lambda: DATASET_CONFIG_PREFIX + json.dumps(CONFIG))
    collector._check_recording_commands()
    assert session.hub.status()["ready"] == (busy is None)
    assert session.exporter.task == ("demo" if busy else CONFIG["prompt"])
