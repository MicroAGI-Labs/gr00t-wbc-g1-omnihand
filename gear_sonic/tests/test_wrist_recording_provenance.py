import numpy as np
import pytest

from gear_sonic.data.features_sonic_vla import get_wrist_camera_features
from gear_sonic.scripts.run_data_exporter import GrootDataCollector


def test_wrist_schema_declares_source_timestamps():
    features = get_wrist_camera_features()
    assert features["capture.left_wrist_source_timestamp_ns"]["dtype"] == "int64"
    assert features["capture.right_wrist_source_timestamp_ns"]["shape"] == (1,)


def test_wrist_frame_keeps_source_timestamp():
    collector = object.__new__(GrootDataCollector)
    collector.data_exporter = type(
        "Exporter", (), {"features": get_wrist_camera_features()}
    )()
    frame = {}
    collector._add_images_to_frame_data(
        frame,
        {
            "images": {
                "left_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                "right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            },
            "timestamps": {"left_wrist": 100.25, "right_wrist": 100.50},
        },
    )
    assert frame["capture.left_wrist_source_timestamp_ns"] == 100250000000
    assert frame["capture.right_wrist_source_timestamp_ns"] == 100500000000


def test_wrist_frame_rejects_missing_source_timestamp():
    collector = object.__new__(GrootDataCollector)
    collector.data_exporter = type("Exporter", (), {"features": get_wrist_camera_features()})()
    with pytest.raises(ValueError, match="source timestamp"):
        collector._add_images_to_frame_data(
            {},
            {
                "images": {
                    "left_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                    "right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                },
                "timestamps": {},
            },
        )
