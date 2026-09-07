from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.data import video_writer


class FakeStream:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.codec_context = SimpleNamespace(thread_count=None)
        self.frames = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.error: Exception | None = None

    def encode(self, frame=None):
        if frame is None:
            return []
        self.entered.set()
        if self.block:
            self.release.wait()
        if self.error is not None:
            raise self.error
        self.frames.append(frame)
        return []


class FakeContainer:
    def __init__(self):
        self.stream = FakeStream()
        self.closed = False

    def add_stream(self, codec, rate):
        return self.stream

    def mux(self, packet):
        raise AssertionError("the fake encoder does not produce packets")

    def close(self):
        self.closed = True


@pytest.fixture
def fake_av(monkeypatch):
    container = FakeContainer()
    monkeypatch.setattr(video_writer.av, "open", lambda *args, **kwargs: container)
    monkeypatch.setattr(
        video_writer.av,
        "VideoFrame",
        SimpleNamespace(from_ndarray=lambda frame, format: frame),
    )
    return container


def test_video_writer_drains_and_closes_once(tmp_path, fake_av):
    writer = video_writer.VideoWriter(str(tmp_path / "episode.mp4"), 4, 3, 50)
    writer.add_frame(np.zeros((3, 4, 3), dtype=np.uint8))
    writer.add_frame(np.ones((3, 4, 3), dtype=np.uint8))

    assert writer.stop(timeout_s=1.0).endswith("episode.mp4")
    assert writer.stop(timeout_s=1.0).endswith("episode.mp4")
    assert len(fake_av.stream.frames) == 2
    assert fake_av.closed
    with pytest.raises(RuntimeError, match="after.*stopped"):
        writer.add_frame(np.zeros((3, 4, 3), dtype=np.uint8))


def test_video_writer_reports_encoder_failure_without_deadlock(tmp_path, fake_av):
    fake_av.stream.error = ValueError("encoder broke")
    writer = video_writer.VideoWriter(str(tmp_path / "episode.mp4"), 4, 3, 50)
    writer.add_frame(np.zeros((3, 4, 3), dtype=np.uint8))
    assert fake_av.stream.entered.wait(timeout=1.0)

    with pytest.raises(RuntimeError, match="worker failed") as error:
        writer.stop(timeout_s=1.0)

    assert isinstance(error.value.__cause__, ValueError)
    assert fake_av.closed


@pytest.mark.parametrize("blocked_stage", ["frame", "flush", "close"])
def test_video_writer_stop_has_a_hard_timeout_and_can_be_retried(tmp_path, fake_av, monkeypatch, blocked_stage):
    entered, release = threading.Event(), threading.Event()
    original_encode, original_close = fake_av.stream.encode, fake_av.close

    def encode(frame=None):
        if blocked_stage == ("flush" if frame is None else "frame"):
            entered.set()
            release.wait(timeout=2.0)
        return original_encode(frame)

    def close():
        if blocked_stage == "close":
            entered.set()
            release.wait(timeout=2.0)
        original_close()

    monkeypatch.setattr(fake_av.stream, "encode", encode)
    monkeypatch.setattr(fake_av, "close", close)
    writer = video_writer.VideoWriter(str(tmp_path / "episode.mp4"), 4, 3, 50)
    writer.add_frame(np.zeros((3, 4, 3), dtype=np.uint8))
    try:
        with pytest.raises(TimeoutError, match="did not stop"):
            writer.stop(timeout_s=0.05)
        assert entered.wait(timeout=1.0)
        assert not fake_av.closed
    finally:
        release.set()
        writer.stop(timeout_s=1.0)
    assert fake_av.closed


def test_video_writer_fails_if_backpressure_persists(tmp_path, fake_av):
    fake_av.stream.block = True
    writer = video_writer.VideoWriter(
        str(tmp_path / "episode.mp4"),
        4,
        3,
        50,
        buffer_size=1,
        enqueue_timeout_s=0.01,
    )
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    writer.add_frame(frame)
    assert fake_av.stream.entered.wait(timeout=1.0)
    writer.add_frame(frame)

    with pytest.raises(RuntimeError, match="queue stayed full"):
        writer.add_frame(frame)

    fake_av.stream.release.set()
    writer.stop(timeout_s=1.0)


def test_cancel_removes_incomplete_video_after_worker_closes(tmp_path, fake_av, monkeypatch):
    output = tmp_path / "partial.mp4"
    output.write_bytes(b"incomplete")
    writer = video_writer.VideoWriter(str(output), 4, 3, 50)
    flushed = []
    monkeypatch.setattr(writer, "_flush_stream", lambda: flushed.append(True))
    writer.cancel(timeout_s=1.0)
    assert fake_av.closed and not writer._thread.is_alive()
    assert not output.exists() and not flushed
