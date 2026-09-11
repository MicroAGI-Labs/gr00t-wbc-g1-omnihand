"""Bounded continuous encoding, true discard, and background save ownership."""

import pickle
import threading
from types import SimpleNamespace

import av
import numpy as np
import pytest

from gear_sonic.data.episode_finalizer import EpisodeFinalizer
from gear_sonic.data.exporter import Gr00tDataExporter, RecordingMemoryLimitError
from gear_sonic.data.video_writer import VideoWriter
from gear_sonic.scripts.run_data_exporter import GrootDataCollector
from gear_sonic.utils.data_collection.episode_state import EpisodeState

CAMERA = "observation.images.ego_view"


def exporter_at(root, **kwargs):
    return Gr00tDataExporter.create(
        save_root=root, fps=50, task="Streaming recording",
        features={
            CAMERA: {"dtype": "video", "shape": (16, 16, 3), "names": ["height", "width", "channels"]},
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
        },
        modality_config={key: {} for key in ("state", "action", "video", "annotation")},
        **kwargs,
    )


def frame():
    return {CAMERA: np.full((16, 16, 3), 90, dtype=np.uint8),
            "observation.state": np.asarray([1.0], dtype=np.float32)}


def enqueue(finalizer, exporter, *, delete=False):
    completed, writers = exporter.detach_episode(advance_index=not delete)
    finalizer.enqueue(episode_index=completed["episode_index"], episode_buffer=completed,
                      video_writers=writers, success=not delete,
                      validation={"passed": not delete, "errors": []}, delete=delete)
    return completed, writers


def test_continuous_capture_owns_pixels_without_retaining_them_in_episode(tmp_path):
    exporter = exporter_at(tmp_path / "dataset")
    assert exporter.max_video_buffer_bytes == 256 * 1024**2
    source = frame()
    for _ in range(3):
        exporter.add_frame(source)
    source[CAMERA][:] = 0
    source["observation.state"][:] = -10
    assert exporter.video_writers
    assert all(isinstance(value, str) for value in exporter.episode_buffer[CAMERA])
    np.testing.assert_array_equal(exporter.episode_buffer["observation.state"][0], [1.0])
    writer = exporter.video_writers[CAMERA]
    writer.queue.join()  # Frames are encoded before Stop & Save is called.
    assert exporter.buffered_video_bytes == 0
    exporter.save_episode()
    with av.open(str(exporter.root / exporter.meta.get_video_file_path(0, CAMERA))) as video:
        images = [image.to_ndarray(format="rgb24") for image in video.decode(video=0)]
    assert len(images) == 3
    assert np.abs(images[0].astype(float) - 90).max() <= 3
    assert not list((exporter.root / ".recording").rglob("*.mp4"))


def test_full_queue_rejects_entire_row_before_appending(tmp_path, monkeypatch):
    release = threading.Event()
    original = VideoWriter._encode_frames
    def blocked(writer):
        assert release.wait(timeout=5)
        original(writer)
    monkeypatch.setattr(VideoWriter, "_encode_frames", blocked)
    exporter = exporter_at(tmp_path / "dataset", max_video_buffer_bytes=16 * 16 * 3)
    try:
        exporter.add_frame(frame())
        with pytest.raises(RecordingMemoryLimitError):
            exporter.add_frame(frame())
        assert exporter.episode_buffer["size"] == 1
        assert exporter.episode_buffer["frame_index"] == [0]
        assert len(exporter.episode_buffer[CAMERA]) == 1
        assert exporter.buffered_video_bytes == 16 * 16 * 3
    finally:
        release.set()
        exporter.save_episode()
    assert exporter.meta.total_frames == 1


def test_failed_commit_preserves_measurements_and_compressed_video_paths(tmp_path, monkeypatch):
    exporter = exporter_at(tmp_path / "dataset")
    exporter.add_frame(frame())
    def fail_table(*args, **kwargs):
        raise OSError("table unavailable")
    monkeypatch.setattr(exporter, "_save_episode_table", fail_table)
    hub = SimpleNamespace(status=lambda: {"ready": True},
                          enqueue=lambda *args: pytest.fail("uncommitted data uploaded"))
    finalizer = EpisodeFinalizer(exporter, hub)
    try:
        enqueue(finalizer, exporter)
        assert finalizer.wait_until_idle(timeout=5)
        assert "table unavailable" in finalizer.status()["error"]
        with (exporter.root / "recovery/episode_000000.pkl").open("rb") as recovery_file:
            recovered = pickle.load(recovery_file)
        assert isinstance(recovered["episode_buffer"][CAMERA][0], str)
        np.testing.assert_array_equal(recovered["episode_buffer"]["observation.state"][0], [1.0])
        with av.open(recovered["video_paths"][CAMERA]) as video:
            assert len(list(video.decode(video=0))) == 1
        assert not finalizer.can_record()
    finally:
        finalizer.close()


def test_record_discard_and_reuse_index_while_previous_save_runs(tmp_path, monkeypatch):
    exporter = exporter_at(tmp_path / "dataset")
    entered, release = threading.Event(), threading.Event()
    save = exporter.save_episode
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return save(*args, **kwargs)
    monkeypatch.setattr(exporter, "save_episode", blocked)
    finalizer = EpisodeFinalizer(exporter)
    try:
        exporter.add_frame(frame())
        enqueue(finalizer, exporter)
        assert entered.wait(timeout=1)
        assert finalizer.can_record()
        exporter.add_frame(frame())
        discarded_path = exporter.video_writers[CAMERA].output_path
        enqueue(finalizer, exporter, delete=True)
        assert not finalizer.can_record()
        assert exporter.episode_buffer["episode_index"] == 1
        # Capture can own the same next episode number while discard cleanup waits.
        exporter.add_frame(frame())
        next_writer = exporter.video_writers[CAMERA]
        assert next_writer.output_path != discarded_path
        release.set()
        assert finalizer.wait_until_idle(timeout=5)
        assert finalizer.status()["error"] is None
        assert not discarded_path.exists()
        assert next_writer._accepting_frames
        assert finalizer.status()["last_finalized_episode"] == 0
        enqueue(finalizer, exporter)
        assert finalizer.wait_until_idle(timeout=5)
        assert finalizer.status()["error"] is None
        assert exporter.meta.total_episodes == 2
        assert exporter.meta.total_frames == 2
        assert not list((exporter.root / ".recording").rglob("*.mp4"))
        assert not (exporter.root / "recovery").exists()
        assert exporter.meta.info["discarded_episode_indices"] == []
    finally:
        release.set()
        finalizer.close()


@pytest.mark.parametrize("pending,uploading", [(1, False), (1, True), (0, True)])
def test_recording_cannot_overlap_upload_even_when_upload_not_required(pending, uploading):
    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector._episode_state = EpisodeState()
    collector._keyboard_listener = SimpleNamespace(read_msg=lambda: "c")
    collector._manager_toggle_dc = collector._manager_toggle_da = False
    collector._manager_discard_reason = None
    collector.current_stream_mode = collector.required_stream_mode = 5
    collector.require_hub_upload = False
    collector.episode_finalizer = SimpleNamespace(can_record=lambda: True)
    collector.hub_uploader = SimpleNamespace(status=lambda: {"pending": pending, "uploading": uploading})
    collector._print_and_say = lambda *args, **kwargs: None
    collector._check_recording_commands()
    assert collector._episode_state.get_state() == collector._episode_state.IDLE
    assert "wait for upload" in collector._recording_message
