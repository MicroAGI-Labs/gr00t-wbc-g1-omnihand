"""Regression coverage for repeated episode numbers after interrupted saves."""

from datetime import datetime, timezone
import json
import pickle

import av
import numpy as np
import pytest

from gear_sonic.data import recording_identity as identities
from gear_sonic.data.episode_finalizer import EpisodeFinalizer
from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.utils.data_collection.local_recordings import create_committed_snapshot


CAMERA = "observation.images.ego_view"


def create(root):
    return Gr00tDataExporter.create(
        save_root=root, fps=50, task="resume test",
        features={CAMERA: {"dtype": "video", "shape": (16, 16, 3),
                           "names": ["height", "width", "channels"]}},
        modality_config={key: {} for key in ("state", "action", "video", "annotation")},
    )


def frame(value):
    return {CAMERA: np.full((16, 16, 3), value, dtype=np.uint8)}


def assert_video(path, value):
    with av.open(str(path)) as video:
        images = [image.to_ndarray(format="rgb24") for image in video.decode(video=0)]
    assert len(images) == 1
    assert np.abs(images[0].astype(float) - value).max() <= 3


def test_take_names_remain_unique_with_identical_clock_time(monkeypatch):
    class FixedClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 9, 12, 9, 8, 38, 123456, tzinfo=timezone.utc)

    monkeypatch.setattr(identities, "datetime", FixedClock)
    first = identities.new_recording_identity(4)
    second = identities.new_recording_identity(4)
    assert "episode_000004_20260912T090838.123456Z_" in first["id"]
    assert first["started_at_utc"] == second["started_at_utc"]
    assert first["id"] != second["id"]


@pytest.mark.parametrize("failure_point", ["table", "metadata"])
def test_resume_after_failed_save_keeps_both_takes_and_dataset_compatibility(tmp_path, monkeypatch, failure_point):
    root = tmp_path / "dataset"
    first = create(root)
    first.add_frame(frame(40))
    original_id = first.episode_buffer["_recording"]["id"]
    original_path = first.video_writers[CAMERA].output_path
    assert original_id in str(original_path)

    def fail(*args, **kwargs):
        raise OSError("interrupted save")

    if failure_point == "table":
        monkeypatch.setattr(first, "_save_episode_table", fail)
    else:
        monkeypatch.setattr(first.meta, "save_episode", fail)
    completed, writers = first.detach_episode()
    finalizer = EpisodeFinalizer(first)
    try:
        finalizer.enqueue(episode_index=0, episode_buffer=completed, video_writers=writers,
                          success=True, validation={"passed": True})
        assert finalizer.wait_until_idle(timeout=10)
        assert "interrupted save" in finalizer.status()["error"]
    finally:
        finalizer.close()
    recovery_path, = (root / "recovery").glob("*.pkl")
    recovery_bytes = recovery_path.read_bytes()
    recovered = pickle.loads(recovery_bytes)
    assert recovered["video_paths"][CAMERA] == str(original_path)
    assert_video(original_path, 40)

    resumed = create(root)
    resumed.add_frame(frame(150))
    next_id = resumed.episode_buffer["_recording"]["id"]
    assert next_id != original_id
    resumed.save_episode()

    assert resumed.meta.total_episodes == 1
    assert_video(root / resumed.meta.get_video_file_path(0, CAMERA), 150)
    assert_video(original_path, 40)
    assert recovery_path.read_bytes() == recovery_bytes
    archive, = (root / "recovery").glob("*_uncommitted")
    manifest = json.loads((archive / "manifest.json").read_text())
    relative = str(resumed.meta.get_video_file_path(0, CAMERA))
    assert_video(root / manifest["original_paths"][relative], 40)
    info = json.loads((root / "meta/info.json").read_text())
    assert info["episode_recordings"]["0"]["id"] == next_id
    assert datetime.fromisoformat(info["episode_recordings"]["0"]["started_at_utc"]).tzinfo
    snapshot = create_committed_snapshot(root, snapshot_parent=tmp_path)
    assert_video(snapshot / resumed.meta.get_video_file_path(0, CAMERA), 150)
    assert len(list(snapshot.rglob("*.mp4"))) == 1
    assert create(root).episode_buffer["episode_index"] == 1


def test_stale_recorder_cannot_archive_or_overwrite_a_committed_episode(tmp_path):
    root = tmp_path / "dataset"
    first, stale = create(root), create(root)
    first.add_frame(frame(20))
    stale.add_frame(frame(200))
    first.save_episode()
    before = {str(path.relative_to(root)): path.read_bytes()
              for folder in ("meta", "data", "videos") for path in (root / folder).rglob("*") if path.is_file()}
    try:
        with pytest.raises(RuntimeError, match="another recorder"):
            stale.save_episode()
        assert not (root / "recovery").exists()
        assert before == {str(path.relative_to(root)): path.read_bytes()
                          for folder in ("meta", "data", "videos")
                          for path in (root / folder).rglob("*") if path.is_file()}
    finally:
        stale.skip_and_start_new_episode()


def test_inconsistent_metadata_does_not_move_existing_files(tmp_path):
    root = tmp_path / "dataset"
    exporter = create(root)
    video = root / exporter.meta.get_video_file_path(0, CAMERA)
    video.parent.mkdir(parents=True)
    video.write_bytes(b"retained video")
    (root / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 1}\n')
    exporter.add_frame(frame(90))
    try:
        with pytest.raises(RuntimeError, match="metadata is incomplete"):
            exporter.save_episode()
        assert video.read_bytes() == b"retained video"
        assert not (root / "recovery").exists()
    finally:
        exporter.skip_and_start_new_episode()
