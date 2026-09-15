from __future__ import annotations

import pickle
import threading
from types import SimpleNamespace

from gear_sonic.data.episode_finalizer import EpisodeFinalizer


class FakeExporter:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.error: Exception | None = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False

    def save_episode(self, episode_buffer, **kwargs):
        self._save(not kwargs.get("success", True), episode_buffer, kwargs)

    def _save(self, discarded, episode_buffer, kwargs):
        self.entered.set()
        if self.block:
            self.release.wait()
        self.calls.append((discarded, episode_buffer, kwargs))
        if self.error is not None:
            raise self.error


class FakeWriter:
    def __init__(self):
        self.stop_timeouts = []

    def stop(self, timeout_s=30.0):
        self.stop_timeouts.append(timeout_s)


def _enqueue(finalizer, *, episode_index=0, discarded=False, writer=None):
    finalizer.enqueue(
        episode_index=episode_index,
        episode_buffer={"episode_index": episode_index, "size": 2},
        video_writers={} if writer is None else {"camera": writer},
        success=not discarded,
        validation={"passed": not discarded, "errors": []},
    )


def test_finalizer_reports_success_only_after_exporter_returns(tmp_path):
    exporter = FakeExporter(tmp_path)
    exporter.block = True
    hub = SimpleNamespace(status=lambda: {"ready": False})
    finalizer = EpisodeFinalizer(exporter, hub)
    _enqueue(finalizer)
    assert exporter.entered.wait(timeout=1.0)

    assert finalizer.status()["last_finalized_episode"] is None
    assert finalizer.status()["finalizing"] is True
    assert finalizer.status()["pending_saves"] == 1
    assert finalizer.status()["pending_discards"] == 0
    assert finalizer.can_record()
    assert not finalizer.status()["at_capacity"]
    exporter.release.set()
    assert finalizer.wait_until_idle(timeout=1.0)

    assert finalizer.status()["error"] is None
    assert finalizer.can_record()
    assert finalizer.status()["last_finalized_episode"] == 0
    finalizer.close()


def test_discard_cleanup_is_reported_separately_from_saving(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def discard(*, video_writers):
        entered.set()
        assert release.wait(timeout=5)
    exporter = SimpleNamespace(root=tmp_path, discard_episode=discard)
    finalizer = EpisodeFinalizer(exporter)
    try:
        finalizer.enqueue(episode_index=0, episode_buffer={"episode_index": 0, "size": 2},
                          video_writers={}, success=False, validation={}, delete=True)
        assert entered.wait(timeout=1)
        assert finalizer.status()["pending_discards"] == 1
        assert finalizer.status()["pending_saves"] == 0
        assert finalizer.can_record()
        release.set()
        assert finalizer.wait_until_idle(timeout=5)
        assert finalizer.status()["pending_discards"] == 0
        assert finalizer.status()["last_finalized_episode"] is None
        assert finalizer.status()["error"] is None
    finally:
        release.set()
        finalizer.close()


def test_finalizer_preserves_owned_buffer_and_blocks_after_failure(tmp_path):
    exporter = FakeExporter(tmp_path)
    exporter.error = OSError("metadata failed")
    writer = FakeWriter()
    hub = SimpleNamespace(status=lambda: {"ready": False})
    finalizer = EpisodeFinalizer(exporter, hub)
    _enqueue(finalizer, discarded=True, writer=writer)
    assert finalizer.wait_until_idle(timeout=1.0)

    assert "metadata failed" in finalizer.status()["error"]
    assert finalizer.status()["last_finalized_episode"] is None
    recovery_path, = (tmp_path / "recovery").glob("episode_000000_*.pkl")
    with recovery_path.open("rb") as recovery_file:
        recovery = pickle.load(recovery_file)
    assert recovery["episode_buffer"]["episode_index"] == 0
    assert recovery["episode_buffer"]["size"] == 2
    assert recovery["recording"] == recovery["episode_buffer"]["_recording"]
    assert recovery["recording"]["id"] in recovery_path.name
    assert not recovery["success"]
    assert writer.stop_timeouts == [30.0]
    assert not finalizer.can_record()
    finalizer.close()


def test_repeated_failures_at_same_index_preserve_every_recovery(tmp_path):
    exporter = FakeExporter(tmp_path)
    exporter.error = OSError("disk write failed")
    for _ in range(2):
        finalizer = EpisodeFinalizer(exporter)
        try:
            _enqueue(finalizer)
            assert finalizer.wait_until_idle(timeout=2)
        finally:
            finalizer.close()
    paths = list((tmp_path / "recovery").glob("episode_000000_*.pkl"))
    assert len(paths) == 2
    payloads = [pickle.loads(path.read_bytes()) for path in paths]
    assert payloads[0]["recording"]["id"] != payloads[1]["recording"]["id"]
