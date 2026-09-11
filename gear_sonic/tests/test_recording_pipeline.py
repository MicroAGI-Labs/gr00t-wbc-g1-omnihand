"""Integration regressions for the current sender-time recording pipeline.

Selection/watermarks and full frame construction live in test_sender_recording;
pose validation lives in end_effectors/tests/test_recording_quality. These tests
cover the split PR's metadata precision, round-trip, and finalizer handoff cases
against the current RecordingInputs API.
"""
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.sender_sync import selection_inputs
from gear_sonic.scripts.run_data_exporter import GrootDataCollector, _integer_scalar
from gear_sonic.tests.test_sender_recording import LIMITS, MS, collector, populated
from gear_sonic.utils.data_collection.episode_state import EpisodeState


@pytest.mark.parametrize("value,expected", [
    (None, -1), ([], -1), ([1, 2], -1), ([[1], [2, 3]], -1),
    (True, -1), ("123", -1), (-1, -1), (1.5, -1), (np.nan, -1),
    (np.inf, -1), (2**63, -1), (np.uint64(2**64 - 1), -1),
    (0, 0), (np.asarray([17]), 17), (2**53 + 17, 2**53 + 17),
    (2**63 - 1, 2**63 - 1),
])
def test_source_identity_is_int64_or_unknown_without_rounding(value, expected):
    assert _integer_scalar(value) == expected


def test_selected_source_metadata_and_failure_label_round_trip_and_resume(tmp_path):
    sync = populated()
    sync.histories["proprio"][0]["index"] = 2**53 + 17
    inputs = selection_inputs(sync.select(1000 * MS, hand=False, max_ages=LIMITS))
    c = collector(sync)
    c.data_exporter.features = {}
    captured = {}
    c._add_capture_features(captured, inputs.proprio, inputs)
    fields = ("capture.robot_state_sequence", "capture.sync_target_monotonic_ns")
    features = {name: {"dtype": "int64", "shape": (1,), "names": [name]} for name in fields}
    features["episode.success"] = {"dtype": "uint8", "shape": (1,), "names": ["success"]}
    options = dict(save_root=tmp_path / "dataset", fps=50, features=features,
                   modality_config={key: {} for key in ("state", "action", "video", "annotation")},
                   task="recording metadata")
    exporter = Gr00tDataExporter.create(**options)
    source = {**{name: captured[name] for name in fields}, "episode.success": np.ones(1, dtype=np.uint8)}
    exporter.add_frame(source)
    exporter.add_frame(source)
    exporter.save_episode(success=False, validation={"passed": False, "errors": ["operator_marked_failure"]})
    table = pq.read_table(exporter.root / exporter.meta.get_data_file_path(0))
    assert table[fields[0]].to_pylist() == [2**53 + 17] * 2
    assert table[fields[0]].type.bit_width == 64
    assert table[fields[1]].to_pylist() == [1000 * MS] * 2
    assert table["episode.success"].to_pylist() == [0, 0]
    assert exporter.meta.info["discarded_episode_indices"] == []
    assert exporter.meta.info["failed_episode_indices"] == [0]
    assert exporter.meta.info["episode_quality"]["0"]["discarded"] is False
    assert exporter.meta.info["episode_quality"]["0"]["success"] is False
    np.testing.assert_array_equal(source["episode.success"], [1])
    resumed = Gr00tDataExporter.create(**options)
    resumed.add_frame(source)
    resumed.save_episode()
    assert resumed.meta.total_frames == 3
    assert resumed.meta.total_episodes == 2
    assert resumed.meta.info["failed_episode_indices"] == [0]


def test_rejected_finalizer_handoff_preserves_completed_episode_and_writers():
    c = GrootDataCollector.__new__(GrootDataCollector)
    c._episode_state = EpisodeState()
    c._episode_state.change_state()
    completed = {"episode_index": 0, "size": 2, "frame_index": [0, 1]}
    writers = {"camera": object()}
    exporter = SimpleNamespace(episode_buffer=completed, video_writers=writers)
    def detach(*, advance_index=True):
        exporter.episode_buffer = {"episode_index": 1, "size": 0}
        exporter.video_writers = {}
        return completed, writers
    exporter.detach_episode = detach
    c.data_exporter = exporter
    def reject(**job):
        raise RuntimeError("finalizer queue unavailable")
    c.episode_finalizer = SimpleNamespace(enqueue=reject)
    c._episode_validation = lambda: {"passed": True, "errors": []}
    with pytest.raises(RuntimeError, match="queue unavailable"):
        c._finish_recording(save=True, discard_reason="operator_discarded")
    assert exporter.episode_buffer is completed
    assert exporter.video_writers is writers
    assert c._episode_state.get_state() == c._episode_state.RECORDING
