"""Timestamped take identities and preservation of interrupted local saves."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import uuid


def new_recording_identity(episode_index: int) -> dict[str, str]:
    started = datetime.now(timezone.utc)
    return {
        "id": f"episode_{episode_index:06d}_{started:%Y%m%dT%H%M%S.%fZ}_{uuid.uuid4().hex}",
        "started_at_utc": started.isoformat(),
    }


def recording_identity(episode_buffer: dict) -> dict[str, str]:
    if "_recording" not in episode_buffer:
        episode_buffer["_recording"] = new_recording_identity(int(episode_buffer["episode_index"]))
    return episode_buffer["_recording"]


@contextmanager
def recording_write_lock(root: Path):
    """Serialize publication and orphan preservation across recorder processes."""
    with (root / ".recording-write.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def preserve_uncommitted_files(root: Path, episode_index: int) -> Path | None:
    """Archive only the next uncommitted index, while holding the write lock.

    Timestamped staging/recovery paths remain untouched. The manifest maps
    legacy numbered paths to their preserved locations for manual recovery.
    """
    info_path = root / "meta/info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    if info["total_episodes"] != episode_index:
        raise RuntimeError("Dataset changed in another recorder; restart before saving")
    expected = list(range(episode_index))
    for name in ("episodes.jsonl", "episodes_stats.jsonl"):
        rows = [json.loads(line) for line in (root / "meta" / name).read_text().splitlines() if line.strip()]
        if [row["episode_index"] for row in rows] != expected:
            raise RuntimeError("Episode metadata is incomplete; preserve the dataset before recovery")

    args = {"episode_chunk": episode_index // info["chunks_size"], "episode_index": episode_index}
    paths = [Path(info["data_path"].format(**args))]
    paths.extend(Path(info["video_path"].format(**args, video_key=key))
                 for key, feature in info["features"].items() if feature["dtype"] == "video")
    for relative in paths:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Dataset path escapes its root: {relative}")
    existing = [path for path in paths if (root / path).exists()]
    if not existing:
        return None

    archive = root / "recovery" / (new_recording_identity(episode_index)["id"] + "_uncommitted")
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / "meta", archive / "meta")
    manifest = {
        "episode_index": episode_index,
        "reason": "Files from an interrupted save; episode absent from committed metadata",
        "original_paths": {str(path): str((archive / path).relative_to(root)) for path in existing},
    }
    # Write the path mapping before moving anything, including if recovery is interrupted.
    with (archive / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for path in existing:
        destination = archive / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(root / path, destination)
    print(f"[Exporter] Preserved unfinished episode {episode_index} in {archive}")
    return archive
