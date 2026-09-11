import json
import av
import numpy as np
import pytest

from gear_sonic.camera.depth_preview import camera_images_with_depth_preview
from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import get_wrist_camera_features, get_zed_stereo_features
from gear_sonic.data.sender_sync import RecordingInputs
from gear_sonic.scripts.run_data_exporter import (
    GrootDataCollector, SonicDataExporterConfig, _validate_recording_dataset_mode,
)


def camera_features():
    return {
        "observation.images.ego_view": {"dtype": "video", "shape": (480, 640, 3),
                                        "names": ["height", "width", "channel"]},
        **get_zed_stereo_features(), **get_wrist_camera_features(),
    }


def test_all_five_videos_save_with_ui_depth_colors_and_capture_times(tmp_path):
    config = SonicDataExporterConfig()
    assert config.record_wrist_cameras and config.record_zed_stereo
    features = camera_features()
    root = tmp_path / "dataset"
    writer = Gr00tDataExporter.create(
        save_root=root, fps=50, task="camera test", robot_type="test", features=features,
        modality_config={name: {} for name in ("state", "action", "video", "annotation")},
    )
    collector = object.__new__(GrootDataCollector)
    collector.data_exporter = writer
    collector.camera_max_age = 0.1
    depth = np.broadcast_to(np.linspace(0.1, 8, 640, dtype=np.float32), (480, 640))
    images = {key.split(".")[-1]: np.full((480, 640, 3), 20 + i * 30, dtype=np.uint8)
              for i, (key, feature) in enumerate(features.items()) if feature["dtype"] == "video"}
    image = {"images": images, "depths": {"ego_view_depth": depth},
             "timestamps": {name: 100.25 for name in images}}
    expected = camera_images_with_depth_preview(images, image["depths"])["ego_view_depth"]
    assert not np.array_equal(expected, images["ego_view_depth"])
    collector._validate_camera_inputs(0, image)
    for _ in range(3):
        frame = {}
        collector._add_images_to_frame_data(frame, RecordingInputs(None, image, None, None, None, None, 5))
        np.testing.assert_array_equal(frame["observation.images.ego_view_depth"], expected)
        for key in features:
            if key.startswith("capture."):
                assert frame[key][0] == 100250000000
        writer.add_frame(frame)
    writer.save_episode()
    videos = list(root.glob("videos/**/*.mp4"))
    assert len(videos) == 5
    for path in videos:
        with av.open(str(path)) as video:
            decoded = list(video.decode(video=0))
        assert len(decoded) == 3
        if "ego_view_depth" in str(path):
            actual = decoded[0].to_ndarray(format="rgb24")
            assert np.abs(actual.astype(float) - expected).mean() < 5
    _validate_recording_dataset_mode(root, features, False)


@pytest.mark.parametrize("depth", [None, np.zeros((480, 640)), np.full((480, 640), np.nan)])
def test_depth_video_uses_same_camera_fallback_as_ui(depth):
    sdk = np.full((480, 640, 3), 42, dtype=np.uint8)
    depths = {} if depth is None else {"ego_view_depth": depth}
    result = camera_images_with_depth_preview({"ego_view_depth": sdk}, depths)
    np.testing.assert_array_equal(result["ego_view_depth"], sdk)


def test_resume_cannot_silently_keep_old_three_camera_schema(tmp_path):
    features = camera_features()
    old = {key: feature for key, feature in features.items() if key not in get_zed_stereo_features()}
    (tmp_path / "meta").mkdir()
    path = tmp_path / "meta/info.json"
    path.write_text(json.dumps({"features": old}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="camera schema.*new --dataset-name"):
        _validate_recording_dataset_mode(tmp_path, features, False)
    assert path.read_bytes() == before
