from pathlib import Path

import numpy as np
import pytest

from gear_sonic.data import exporter as exporter_module
from gear_sonic.data.exporter import Gr00tDataExporter


class _FakeMeta:
    def __init__(self, root: Path):
        self.root = root
        self.total_episodes = 0
        self.total_frames = 0
        self.fps = 50
        self.features = {
            "state": {"dtype": "float32"},
        }
        self.video_keys = []
        self.info = {"discarded_episode_indices": []}
        self._tasks = {}
        self.events = []
        self.raise_on_save = False

    def get_task_index(self, task):
        return self._tasks.get(task)

    def add_task(self, task):
        self._tasks.setdefault(task, len(self._tasks))

    def get_data_file_path(self, episode_index):
        return f"data/episode_{episode_index:06d}.parquet"

    def save_episode(self, *args):
        self.events.append("metadata")
        if self.raise_on_save:
            raise OSError("metadata write failed")


def _build_exporter(tmp_path, monkeypatch):
    data_exporter = object.__new__(Gr00tDataExporter)
    data_exporter.meta = _FakeMeta(tmp_path)
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


def test_discard_metadata_rolls_back_when_commit_fails(tmp_path, monkeypatch):
    data_exporter = _build_exporter(tmp_path, monkeypatch)
    monkeypatch.setattr(
        exporter_module,
        "check_timestamps_sync",
        lambda *args, **kwargs: data_exporter.meta.events.append("validation"),
    )
    data_exporter.meta.raise_on_save = True

    with pytest.raises(OSError, match="metadata write failed"):
        data_exporter.save_episode_as_discarded(data_exporter.episode_buffer)

    assert data_exporter.meta.events == ["validation", "parquet", "metadata"]
    assert data_exporter.meta.info["discarded_episode_indices"] == []
    assert data_exporter.episode_buffer["size"] == 2
