from __future__ import annotations

import json
from types import SimpleNamespace

import av
import numpy as np
import pyarrow.parquet as pq
import pytest

from gear_sonic.data.episode_finalizer import EpisodeFinalizer
from gear_sonic.data.exporter import Gr00tDataExporter


def test_real_exporter_writes_parquet_timing_and_quality_metadata(tmp_path):
    data_exporter = Gr00tDataExporter.create(
        save_root=tmp_path / "dataset",
        fps=50,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["joint"],
            },
            "capture.sync_target_monotonic_ns": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["sync_target_monotonic_ns"],
            },
            "capture.camera_age_ms": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["camera_age_ms"],
            },
        },
        modality_config={
            "state": {},
            "action": {},
            "video": {},
            "annotation": {},
        },
        task="smoke",
    )
    data_exporter.add_frame(
        {
            "observation.state": np.asarray([0.0], dtype=np.float32),
            "capture.sync_target_monotonic_ns": np.asarray([1_000_000_000], dtype=np.int64),
            "capture.camera_age_ms": np.asarray([5.0], dtype=np.float32),
        }
    )
    data_exporter.add_frame(
        {
            "observation.state": np.asarray([1.0], dtype=np.float32),
            "capture.sync_target_monotonic_ns": np.asarray([1_020_000_000], dtype=np.int64),
            "capture.camera_age_ms": np.asarray([8.0], dtype=np.float32),
        }
    )

    data_exporter.save_episode()

    assert data_exporter.meta.total_episodes == 1
    assert data_exporter.meta.total_frames == 2
    parquet_path = tmp_path / "dataset" / data_exporter.meta.get_data_file_path(0)
    assert parquet_path.is_file()
    table = pq.read_table(parquet_path)
    assert table["timestamp"].to_pylist() == pytest.approx([0.0, 0.02])
    assert table["capture.sync_target_monotonic_ns"].to_pylist() == [
        1_000_000_000,
        1_020_000_000,
    ]
    assert table["capture.camera_age_ms"].to_pylist() == [5.0, 8.0]
    quality = json.loads((tmp_path / "dataset/meta/info.json").read_text())["episode_quality"]["0"]
    assert quality == {
        "discarded": False,
        "success": True,
        "validation": {"passed": True, "errors": []},
    }


def test_next_writer_failure_cannot_lose_detached_video(tmp_path, monkeypatch):
    key = "observation.images.ego_view"
    exporter = Gr00tDataExporter.create(
        save_root=tmp_path / "dataset",
        fps=50,
        features={key: {"dtype": "video", "shape": (16, 16, 3), "names": ["height", "width", "channels"]}},
        modality_config={"state": {}, "action": {}, "video": {}, "annotation": {}},
        task="smoke",
    )
    frame = {key: np.zeros((16, 16, 3), dtype=np.uint8)}
    exporter.add_frame(frame)
    exporter.add_frame(frame)

    def fail_next_writer(episode_index=None):
        raise OSError("cannot open next episode video")

    completed, writers = exporter.detach_episode()
    finalizer = EpisodeFinalizer(exporter, SimpleNamespace(status=lambda: {"ready": False}))
    try:
        finalizer.enqueue(
            episode_index=0,
            episode_buffer=completed,
            video_writers=writers,
            success=True,
            validation={"passed": True, "errors": []},
        )
        assert finalizer.wait_until_idle(timeout=5.0)
        assert finalizer.status()["error"] is None
        assert finalizer.status()["last_finalized_episode"] == 0
        monkeypatch.setattr(exporter, "create_video_writer", fail_next_writer)
        with pytest.raises(OSError, match="cannot open next"):
            exporter.add_frame(frame)
        assert exporter.episode_buffer["size"] == 0
        assert exporter.episode_buffer["frame_index"] == []
        assert exporter.video_writers == {}
        assert exporter.meta.total_frames == 2
        with av.open(str(exporter.root / exporter.meta.get_video_file_path(0, key))) as video:
            assert len(list(video.decode(video=0))) == 2
    finally:
        finalizer.close()
