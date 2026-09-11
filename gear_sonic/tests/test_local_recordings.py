import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gear_sonic.utils.data_collection import local_recordings as recordings
from gear_sonic.scripts.upload_local_recordings import check_remote_prefix


def test_saved_recording_folder_is_reused_and_cli_can_override(tmp_path, monkeypatch):
    config = tmp_path / "recording.json"
    config.write_text(json.dumps({"root_output_dir": str(tmp_path / "recordings"), "dataset_name": "g1_teleop"}))
    monkeypatch.setattr(recordings, "recording_config_path", lambda: config)
    assert recordings.resolve_recording_destination() == (str(tmp_path / "recordings"), "g1_teleop")
    assert recordings.resolve_recording_destination("other", "/tmp/elsewhere") == ("/tmp/elsewhere", "other")


def make_dataset(root):
    (root / "meta").mkdir(parents=True)
    info = {"total_episodes": 1, "total_frames": 2, "chunks_size": 1000,
            "features": {"camera": {"dtype": "video"}},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 2}) + "\n")
    (root / "meta/episodes_stats.jsonl").write_text(json.dumps({"episode_index": 0, "stats": {}}) + "\n")
    (root / "meta/episode_quality.jsonl").write_text(json.dumps({"episode_index": 0, "success": False}) + "\n")
    for path in ("data/chunk-000/episode_000000.parquet", "videos/chunk-000/camera/episode_000000.mp4",
                 ".recording/take/camera.mp4", "recovery/episode_000001.pkl"):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"preserved")
    return info


def test_snapshot_preserves_failed_validation_but_excludes_active_and_recovery_files(tmp_path):
    root = tmp_path / "dataset"
    make_dataset(root)
    snapshot = recordings.create_committed_snapshot(root, snapshot_parent=tmp_path)
    assert len(list(snapshot.rglob("*.mp4"))) == 1
    assert len(list(snapshot.rglob("*.parquet"))) == 1
    assert not (snapshot / ".recording").exists()
    assert not (snapshot / "recovery").exists()
    assert json.loads((snapshot / "meta/episode_quality.jsonl").read_text())["success"] is False
    before = (snapshot / "meta/info.json").read_bytes()
    (root / "meta/info.json").write_text("changed")
    assert (snapshot / "meta/info.json").read_bytes() == before


@pytest.mark.parametrize("problem", ["metadata", "missing_video"])
def test_incomplete_episode_cannot_be_uploaded(tmp_path, problem):
    root = tmp_path / "dataset"
    info = make_dataset(root)
    if problem == "metadata":
        info["total_frames"] = 3
        (root / "meta/info.json").write_text(json.dumps(info))
    else:
        (root / "videos/chunk-000/camera/episode_000000.mp4").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        recordings.create_committed_snapshot(root, snapshot_parent=tmp_path)
    assert not list(tmp_path.glob("sonic-upload-*"))


def test_remote_prefix_rejects_other_data_and_public_repositories(tmp_path):
    path = tmp_path / "data/episode_000000.parquet"
    path.parent.mkdir()
    path.write_bytes(b"abc")
    file = SimpleNamespace(rfilename="data/episode_000000.parquet", size=3,
                           lfs=SimpleNamespace(sha256=hashlib.sha256(b"abc").hexdigest()))
    remote = SimpleNamespace(private=True, siblings=[file])
    api = SimpleNamespace(repo_exists=lambda *a, **kw: True, repo_info=lambda *a, **kw: remote)
    check_remote_prefix(api, "org/data", tmp_path, public=False)
    path.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="different data"):
        check_remote_prefix(api, "org/data", tmp_path, public=False)
    remote.private = False
    with pytest.raises(ValueError, match="public"):
        check_remote_prefix(api, "org/data", tmp_path, public=False)
