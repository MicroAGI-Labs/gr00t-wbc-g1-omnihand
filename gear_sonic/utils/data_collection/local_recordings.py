"""Machine-local recording destination and immutable, committed upload snapshots."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile


def recording_config_path() -> Path:
    return Path.home() / ".config" / "sonic" / "recording.json"


def resolve_recording_destination(
    dataset_name: str | None = None, root_output_dir: str | None = None,
) -> tuple[str, str]:
    path = recording_config_path()
    saved = json.loads(path.read_text()) if path.is_file() else {}
    root = root_output_dir or saved.get("root_output_dir") or "outputs"
    name = dataset_name or saved.get("dataset_name") or datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    if not isinstance(root, str) or not isinstance(name, str):
        raise ValueError(f"Invalid recording destination in {path}")
    return str(Path(root).expanduser()), name


def create_committed_snapshot(root: Path, *, snapshot_parent: Path | None = None) -> Path:
    """Copy stable metadata and only the episode files it references.

    Open encoders live in .recording and never enter the snapshot. Metadata
    consistency checks reject a save in progress; the caller can retry once it
    finishes. Accepted episodes that failed validation are retained.
    """
    root = root.resolve()
    before = {p.name: p.read_bytes() for p in (root / "meta").iterdir() if p.is_file()}
    info = json.loads(before["info.json"])
    episodes = [json.loads(line) for line in before["episodes.jsonl"].splitlines() if line.strip()]
    stats = [json.loads(line) for line in before["episodes_stats.jsonl"].splitlines() if line.strip()]
    count = info["total_episodes"]
    if not count:
        raise ValueError("No saved episodes to upload")
    expected = list(range(count))
    if ([ep["episode_index"] for ep in episodes] != expected
            or [ep["episode_index"] for ep in stats] != expected
            or sum(ep["length"] for ep in episodes) != info["total_frames"]):
        raise RuntimeError("Episode metadata is incomplete; wait for saving to finish")
    snapshot = Path(tempfile.mkdtemp(prefix="sonic-upload-", dir=snapshot_parent))
    try:
        (snapshot / "meta").mkdir()
        for name, content in before.items():
            (snapshot / "meta" / name).write_bytes(content)
        keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
        for ep in episodes:
            index = ep["episode_index"]
            args = {"episode_chunk": index // info["chunks_size"], "episode_index": index}
            paths = [info["data_path"].format(**args)]
            paths.extend(info["video_path"].format(**args, video_key=key) for key in keys)
            for relative in paths:
                source = (root / relative).resolve()
                if not source.is_relative_to(root):
                    raise ValueError(f"Dataset path escapes its root: {relative}")
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
        for name in ("README.md", "LICENSE"):
            if (root / name).is_file():
                shutil.copy2(root / name, snapshot / name)
        after = {p.name: p.read_bytes() for p in (root / "meta").iterdir() if p.is_file()}
        if before != after:
            raise RuntimeError("Dataset changed while snapshotting; retry after saving finishes")
        return snapshot
    except BaseException:
        shutil.rmtree(snapshot)
        raise
