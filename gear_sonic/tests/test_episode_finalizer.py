from __future__ import annotations

import pickle
import threading

import pytest

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
        self._save(False, episode_buffer, kwargs)

    def save_episode_as_discarded(self, episode_buffer, **kwargs):
        self._save(True, episode_buffer, kwargs)

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

    def stop(self, timeout_s):
        self.stop_timeouts.append(timeout_s)


def _enqueue(finalizer, *, episode_index=0, discarded=False, writer=None):
    finalizer.enqueue(
        episode_index=episode_index,
        episode_buffer={"episode_index": episode_index, "size": 2},
        video_writers={} if writer is None else {"camera": writer},
        discarded=discarded,
        validation={"passed": not discarded, "errors": []},
    )


def test_finalizer_reports_success_only_after_exporter_returns(tmp_path):
    exporter = FakeExporter(tmp_path)
    exporter.block = True
    finalizer = EpisodeFinalizer(exporter, max_pending=1)  # type: ignore[arg-type]
    _enqueue(finalizer)
    assert exporter.entered.wait(timeout=1.0)

    assert finalizer.drain_results() == []
    assert finalizer.status()["finalizing"] is True
    assert not finalizer.can_accept()
    with pytest.raises(RuntimeError, match="at capacity"):
        _enqueue(finalizer, episode_index=1)
    exporter.release.set()
    assert finalizer.wait_until_idle(timeout=1.0)

    results = finalizer.drain_results()
    assert len(results) == 1
    assert results[0].succeeded
    assert results[0].episode_index == 0
    assert finalizer.status()["last_finalized_episode"] == 0
    finalizer.close()


def test_finalizer_preserves_owned_buffer_and_blocks_after_failure(tmp_path):
    exporter = FakeExporter(tmp_path)
    exporter.error = OSError("metadata failed")
    writer = FakeWriter()
    finalizer = EpisodeFinalizer(  # type: ignore[arg-type]
        exporter,
        writer_stop_timeout_s=0.25,
    )
    _enqueue(finalizer, discarded=True, writer=writer)
    assert finalizer.wait_until_idle(timeout=1.0)

    [result] = finalizer.drain_results()
    assert not result.succeeded
    assert result.discarded
    assert result.recovery_path is not None
    with open(result.recovery_path, "rb") as recovery_file:
        recovery = pickle.load(recovery_file)
    assert recovery["episode_buffer"] == {"episode_index": 0, "size": 2}
    assert writer.stop_timeouts == [0.25]
    assert not finalizer.can_accept()
    with pytest.raises(RuntimeError, match="previous failure"):
        _enqueue(finalizer, episode_index=1)
    finalizer.close()
