"""Uploads run behind recording: the recorder never waits for the Hub."""

from pathlib import Path
import threading
import time
from unittest.mock import Mock

import pytest

from gear_sonic.data.episode_finalizer import EpisodeFinalizer
from gear_sonic.data.hub_uploader import EpisodeHubUploader


class _FakeMeta:
    def __init__(self):
        self.repo_id = None


class _FakeExporter:
    """Minimal exporter with an injectable blocking snapshot uploader."""

    def __init__(self, root: Path):
        self.root = root
        self.meta = _FakeMeta()
        self.task = ""
        self.calls: list[list[str]] = []
        self.snapshot_info: list[str] = []
        self.first_upload_started = threading.Event()
        self.gate = threading.Event()
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text('{"total_episodes":0}')

    def upload_snapshot(self, snapshot_root: Path, config: dict[str, object]):
        self.calls.append(
            sorted(
                path.relative_to(snapshot_root).as_posix()
                for path in snapshot_root.rglob("*")
                if path.is_file()
            )
        )
        first_upload = not self.first_upload_started.is_set()
        if first_upload:
            self.first_upload_started.set()
            assert self.gate.wait(timeout=10.0), "upload gate never released"
        self.snapshot_info.append(
            (snapshot_root / "meta" / "info.json").read_text()
        )


def _write_episode(root: Path, index: int) -> None:
    data = root / "data" / "chunk-000"
    videos = root / "videos" / "chunk-000" / "observation.images.ego_view"
    data.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    (data / f"episode_{index:06d}.parquet").write_bytes(b"parquet")
    (videos / f"episode_{index:06d}.mp4").write_bytes(b"video")


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached in time")


@pytest.fixture
def uploader(tmp_path):
    exporter = _FakeExporter(tmp_path)
    hub = EpisodeHubUploader(exporter, upload_runner=exporter.upload_snapshot)
    hub.configure(repo_id="ns/dataset", prompt="pick up the cup", private=True)
    try:
        yield hub, exporter
    finally:
        exporter.gate.set()
        hub.close(timeout=5.0)


def test_recording_allowed_while_upload_is_in_flight(uploader):
    hub, exporter = uploader
    _write_episode(exporter.root, 0)

    hub.enqueue(0)
    _wait_for(exporter.first_upload_started.is_set)

    assert hub.status()["uploading"] is True
    assert hub.can_record() is True


def test_recorder_restart_restores_existing_upload_destination(uploader):
    hub, exporter = uploader
    original = hub.status()
    hub.close()
    restored = EpisodeHubUploader(exporter, upload_runner=exporter.upload_snapshot)
    try:
        assert restored.can_record()
        assert restored.status()["repo_id"] == original["repo_id"]
        assert restored.status()["prompt"] == original["prompt"]
        assert restored.status()["private"] == original["private"]
        assert restored.status()["pending"] == 0
        assert exporter.calls == []  # Restoration itself never sends anything.
    finally:
        restored.close()


def test_failed_upload_retries_without_blocking_recording(uploader):
    hub, exporter = uploader
    _write_episode(exporter.root, 0)
    attempted = threading.Event()
    release_retry = threading.Event()
    calls = []

    def upload(snapshot, config):
        calls.append(snapshot)
        attempted.set()
        if len(calls) == 1:
            raise OSError("network unavailable")
        assert release_retry.wait(timeout=5.0)

    hub._upload_runner = upload
    try:
        hub.enqueue(0)
        assert attempted.wait(timeout=1.0)
        _wait_for(lambda: hub.status()["retrying"])
        assert hub.can_record()
        assert hub.status()["last_uploaded_episode"] is None
        assert "network unavailable" in hub.status()["error"]
        release_retry.set()
        assert hub.wait_until_idle(timeout=5.0)
        assert len(calls) == 2
        assert hub.status()["last_uploaded_episode"] == 0
        assert hub.status()["error"] is None
    finally:
        release_retry.set()


def test_shutdown_stops_upload_child_and_preserves_local_episode(uploader, monkeypatch):
    import gear_sonic.data.hub_uploader as module

    hub, exporter = uploader
    _write_episode(exporter.root, 0)
    started, released = threading.Event(), threading.Event()
    process = Mock(returncode=None)

    def communicate():
        started.set()
        assert released.wait(timeout=5.0)
        process.returncode = -15
        return "", "terminated"

    process.communicate.side_effect = communicate
    process.terminate.side_effect = released.set
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    hub._upload_runner = hub._run_upload_subprocess
    try:
        hub.enqueue(0)
        assert started.wait(timeout=2.0)
        hub.close(timeout=0)
        command = spawn.call_args.args[0]
        helper = Path(command[1])
        assert helper == Path(module.__file__).resolve().parents[1] / "scripts/upload_dataset_snapshot.py"
        assert helper.is_file()
        assert "--private" in command
        process.terminate.assert_called_once()
        assert not hub._thread.is_alive()
        assert hub.status()["last_uploaded_episode"] is None
        assert (exporter.root / "data/chunk-000/episode_000000.parquet").is_file()
    finally:
        released.set()


def test_upload_uses_immutable_metadata_snapshot(uploader):
    hub, exporter = uploader
    _write_episode(exporter.root, 0)
    original_info = '{"total_episodes":1}'
    updated_info = '{"total_episodes":2}'
    (exporter.root / "meta" / "info.json").write_text(original_info)

    hub.enqueue(0)
    _wait_for(exporter.first_upload_started.is_set)
    (exporter.root / "meta" / "info.json").write_text(updated_info)
    exporter.gate.set()

    assert hub.wait_until_idle(timeout=10.0)
    assert exporter.snapshot_info == [original_info]
    snapshot_dirs = list((exporter.root / ".upload_snapshots").iterdir())
    assert snapshot_dirs == []


def test_backlog_collapses_into_one_upload_of_the_newest_episode(uploader):
    hub, exporter = uploader
    _write_episode(exporter.root, 0)

    hub.enqueue(0)
    _wait_for(exporter.first_upload_started.is_set)

    for index in (1, 2):
        _write_episode(exporter.root, index)
        hub.enqueue(index)
    assert hub.status()["pending"] == 3

    exporter.gate.set()
    assert hub.wait_until_idle(timeout=10.0)

    assert len(exporter.calls) == 2
    assert "data/chunk-000/episode_000002.parquet" in exporter.calls[-1]
    assert (
        "videos/chunk-000/observation.images.ego_view/episode_000002.mp4"
        in exporter.calls[-1]
    )
    status = hub.status()
    assert status["pending"] == 0
    assert status["last_uploaded_episode"] == 2


def test_recording_is_blocked_only_until_a_dataset_is_chosen(tmp_path):
    exporter = _FakeExporter(tmp_path)
    hub = EpisodeHubUploader(exporter)
    try:
        assert hub.can_record() is False
        hub.configure(repo_id="ns/dataset", prompt="pick up the cup", private=True)
        assert hub.can_record() is True
    finally:
        hub.close(timeout=5.0)


def test_local_finalization_never_enqueues_an_upload():
    class _LocalExporter:
        def __init__(self):
            self.started = threading.Event()
            self.gate = threading.Event()
            self.calls = []

        def save_episode(self, episode_buffer, **kwargs):
            self.calls.append((episode_buffer, kwargs))
            self.started.set()
            assert self.gate.wait(timeout=10.0)

    class _Hub:
        def __init__(self):
            self.episodes = []

        def status(self):
            return {"ready": True}

        def enqueue(self, episode_index):
            self.episodes.append(episode_index)

    exporter = _LocalExporter()
    hub = _Hub()
    finalizer = EpisodeFinalizer(exporter, hub)
    try:
        finalizer.enqueue(
            episode_index=4,
            episode_buffer={"episode_index": 4, "size": 20},
            video_writers={"camera": object()},
            success=True,
            validation={"passed": True, "errors": []},
        )
        assert exporter.started.wait(timeout=2.0)
        assert finalizer.status()["pending"] == 1
        assert hub.episodes == []

        exporter.gate.set()
        assert finalizer.wait_until_idle(timeout=2.0)
        assert hub.episodes == []
        assert finalizer.status()["last_finalized_episode"] == 4
    finally:
        exporter.gate.set()
        finalizer.close(timeout=2.0)


def test_unavailable_uploader_does_not_affect_local_save():
    class _LocalExporter:
        def save_episode(self, episode_buffer, **kwargs):
            return None

    class _Hub:
        def __init__(self):
            self.reported = None

        @staticmethod
        def status():
            raise AssertionError("local saving must not consult the uploader")

        @staticmethod
        def enqueue(episode_index):
            raise OSError("snapshot staging failed")

        def report_enqueue_failure(self, episode_index, error):
            self.reported = (episode_index, str(error))

    hub = _Hub()
    finalizer = EpisodeFinalizer(_LocalExporter(), hub)
    try:
        finalizer.enqueue(
            episode_index=7,
            episode_buffer={"episode_index": 7, "size": 1},
            video_writers={},
            success=True,
            validation={"passed": True, "errors": []},
        )
        assert finalizer.wait_until_idle(timeout=2.0)
        assert finalizer.status()["error"] is None
        assert finalizer.status()["last_finalized_episode"] == 7
        assert hub.reported is None
    finally:
        finalizer.close(timeout=2.0)

def test_finalizer_keeps_ownership_after_an_earlier_job_fails(tmp_path):
    class _FailingExporter:
        root = tmp_path

        def save_episode(self, episode_buffer, **kwargs):
            raise RuntimeError("disk write failed")

    class _Hub:
        @staticmethod
        def status():
            return {"ready": False}

    class _Writer:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    finalizer = EpisodeFinalizer(_FailingExporter(), _Hub())
    first_writer = _Writer()
    second_writer = _Writer()
    try:
        finalizer.enqueue(
            episode_index=0,
            episode_buffer={"episode_index": 0, "size": 1},
            video_writers={"camera": first_writer},
            success=True,
            validation={"passed": True, "errors": []},
        )
        assert finalizer.wait_until_idle(timeout=2.0)
        assert finalizer.status()["error"] is not None

        # An episode already detached by the recorder must still be accepted
        # even though the previous background job failed.
        finalizer.enqueue(
            episode_index=1,
            episode_buffer={"episode_index": 1, "size": 1},
            video_writers={"camera": second_writer},
            success=True,
            validation={"passed": True, "errors": []},
        )
        assert finalizer.wait_until_idle(timeout=2.0)
        assert first_writer.stopped
        assert second_writer.stopped
        assert (tmp_path / "recovery" / "episode_000000.pkl").is_file()
        assert (tmp_path / "recovery" / "episode_000001.pkl").is_file()
    finally:
        finalizer.close(timeout=2.0)
