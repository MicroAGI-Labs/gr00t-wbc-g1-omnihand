from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from gear_sonic.data import exporter as exporter_module
from gear_sonic.data.exporter import Gr00tDataExporter


class FakeMeta:
    def __init__(self, root: Path):
        self.root = root
        self.total_episodes = 0
        self.total_frames = 0
        self.fps = 50
        self.features = {"state": {"dtype": "float32"}}
        self.video_keys = []
        self.info = {
            "discarded_episode_indices": [],
            "total_episodes": 0,
        }
        self.episodes = {}
        self.episodes_stats = {}
        self.stats = {}
        self.tasks = {}
        self.task_to_task_index = {}
        self.events = []
        self.raise_on_save = False

    def get_task_index(self, task):
        return self.task_to_task_index.get(task)

    def add_task(self, task):
        index = len(self.tasks)
        self.tasks[index] = task
        self.task_to_task_index[task] = index

    def get_data_file_path(self, episode_index):
        return Path(f"data/episode_{episode_index:06d}.parquet")

    def save_episode(self, *args):
        self.events.append("metadata")
        self.info["total_episodes"] += 1
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "meta/info.json").write_text("new metadata")
        if self.raise_on_save:
            raise OSError("metadata write failed")


def _build_exporter(tmp_path, monkeypatch):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("original metadata")
    data_exporter = object.__new__(Gr00tDataExporter)
    data_exporter.meta = FakeMeta(tmp_path)
    data_exporter.tolerance_s = 1e-4
    data_exporter.image_writer = None
    data_exporter.episode_buffer = {
        "episode_index": 0,
        "size": 2,
        "task": ["demo", "demo"],
        "frame_index": [0, 1],
        "timestamp": [0.0, 0.02],
        "state": [
            np.asarray([0.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        ],
    }
    data_exporter.video_writers = {}

    monkeypatch.setattr(exporter_module, "validate_episode_buffer", lambda *args: None)
    monkeypatch.setattr(
        exporter_module,
        "compute_episode_stats",
        lambda *args: {"state": {}},
    )

    def save_table(_episode_buffer, episode_index):
        data_exporter.meta.events.append("parquet")
        path = tmp_path / data_exporter.meta.get_data_file_path(episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet")

    data_exporter._save_episode_table = save_table
    return data_exporter


def test_timestamp_validation_precedes_files_and_metadata(tmp_path, monkeypatch):
    data_exporter = _build_exporter(tmp_path, monkeypatch)

    def reject_timestamps(*args, **kwargs):
        data_exporter.meta.events.append("validation")
        raise ValueError("bad timestamps")

    monkeypatch.setattr(exporter_module, "check_timestamps_sync", reject_timestamps)

    with pytest.raises(ValueError, match="bad timestamps"):
        data_exporter.save_episode(data_exporter.episode_buffer)

    assert data_exporter.meta.events == ["validation"]
    assert data_exporter.episode_buffer["size"] == 2
    assert not (tmp_path / "data/episode_000000.parquet").exists()


def test_failed_metadata_commit_restores_memory_and_episode_data(
    tmp_path,
    monkeypatch,
):
    data_exporter = _build_exporter(tmp_path, monkeypatch)
    monkeypatch.setattr(exporter_module, "check_timestamps_sync", lambda *args: None)
    data_exporter.meta.raise_on_save = True

    with pytest.raises(OSError, match="metadata write failed"):
        data_exporter.save_episode_as_discarded(data_exporter.episode_buffer)

    assert data_exporter.meta.events == ["parquet", "metadata"]
    assert data_exporter.meta.info == {
        "discarded_episode_indices": [],
        "total_episodes": 0,
    }
    assert data_exporter.meta.tasks == {}
    assert data_exporter.episode_buffer["size"] == 2
    assert (tmp_path / "data/episode_000000.parquet").read_bytes() == b"parquet"


def test_successful_save_keeps_episode_and_quality_metadata(tmp_path, monkeypatch):
    data_exporter = _build_exporter(tmp_path, monkeypatch)
    monkeypatch.setattr(exporter_module, "check_timestamps_sync", lambda *args: None)

    data_exporter.save_episode(data_exporter.episode_buffer)

    assert (tmp_path / "data/episode_000000.parquet").read_bytes() == b"parquet"
    assert data_exporter.meta.info["total_episodes"] == 1
    assert data_exporter.meta.info["episode_quality"]["0"] == {
        "discarded": False,
        "validation": {"passed": True, "errors": []},
    }


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
            "capture.sync_target_monotonic_ns": np.asarray(
                [1_000_000_000], dtype=np.int64
            ),
            "capture.camera_age_ms": np.asarray([5.0], dtype=np.float32),
        }
    )
    data_exporter.add_frame(
        {
            "observation.state": np.asarray([1.0], dtype=np.float32),
            "capture.sync_target_monotonic_ns": np.asarray(
                [1_020_000_000], dtype=np.int64
            ),
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
    quality = json.loads((tmp_path / "dataset/meta/info.json").read_text())[
        "episode_quality"
    ]["0"]
    assert quality == {
        "discarded": False,
        "validation": {"passed": True, "errors": []},
    }
