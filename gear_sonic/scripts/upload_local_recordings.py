"""Manually upload a consistent snapshot of the persistent local dataset."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.scripts.upload_dataset_snapshot import _limit_background_resources, upload_snapshot
from gear_sonic.utils.data_collection.local_recordings import create_committed_snapshot, resolve_recording_destination


def check_remote_prefix(api, repo_id: str, snapshot: Path, *, public: bool) -> None:
    """Do not overwrite different episodes already present in the destination."""
    if not api.repo_exists(repo_id, repo_type="dataset"):
        return
    remote = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    if not remote.private and not public:
        raise ValueError("Destination is public; use --public explicitly or choose a private dataset")
    for remote_file in remote.siblings:
        name = remote_file.rfilename
        if not name.startswith(("data/", "videos/")):
            continue
        local = snapshot / name
        if not local.is_file() or local.stat().st_size != remote_file.size:
            raise ValueError(f"Destination contains different data at {name}; choose another dataset")
        digest = hashlib.sha256() if remote_file.lfs else hashlib.sha1()
        if not remote_file.lfs:
            digest.update(f"blob {local.stat().st_size}\0".encode())
        with local.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        expected = remote_file.lfs.sha256 if remote_file.lfs else remote_file.blob_id
        if digest.hexdigest() != expected:
            raise ValueError(f"Destination contains different data at {name}; choose another dataset")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Hugging Face dataset, e.g. MicroAGI-Labs/g1-teleop")
    parser.add_argument("--dataset-dir", type=Path, help="Override the configured local recording folder")
    parser.add_argument("--public", action="store_true", help="Allow a public dataset (default: private)")
    parser.add_argument("--dry-run", action="store_true", help="Check the local snapshot without contacting Hugging Face")
    args = parser.parse_args(argv)
    parent, name = resolve_recording_destination()
    root = (args.dataset_dir or Path(parent) / name).expanduser().resolve()
    if not (root / "meta/info.json").is_file():
        parser.error(f"No saved dataset found at {root}")
    _limit_background_resources()
    with (root / ".upload.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("An upload for this local folder is already running")
        snapshot = create_committed_snapshot(root, snapshot_parent=root.parent)
        try:
            info = json.loads((snapshot / "meta/info.json").read_text())
            total_bytes = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
            print(f"Local folder: {root}")
            print(f"Saved episodes: {info['total_episodes']} | rows: {info['total_frames']} | {total_bytes / 1024**2:.1f} MiB")
            print(f"Destination: https://huggingface.co/datasets/{args.repo_id}", flush=True)
            if args.dry_run:
                print("Local snapshot checked. Nothing uploaded.")
                return
            from huggingface_hub import HfApi
            from lerobot.common.datasets.lerobot_dataset import create_lerobot_dataset_card

            api = HfApi()
            check_remote_prefix(api, args.repo_id, snapshot, public=args.public)
            if not (snapshot / "README.md").exists():
                card = create_lerobot_dataset_card(dataset_info=info)
                card.save(snapshot / "README.md")
            upload_snapshot(snapshot, args.repo_id, private=not args.public)
            print(f"Uploaded {info['total_episodes']} saved episodes to https://huggingface.co/datasets/{args.repo_id}")
        finally:
            shutil.rmtree(snapshot)


if __name__ == "__main__":
    main()
