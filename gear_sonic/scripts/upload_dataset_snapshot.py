"""Upload one immutable LeRobot dataset snapshot in an isolated process."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path


def _limit_background_resources(max_cpus: int = 2) -> None:
    """Keep Hub hashing/network work away from the recorder's hot path."""
    with contextlib.suppress(OSError):
        os.nice(10)
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        with contextlib.suppress(OSError):
            available = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, set(available[-max_cpus:]))


def upload_snapshot(
    snapshot_root: Path,
    repo_id: str,
    *,
    private: bool,
    max_cpus: int = 2,
) -> None:
    _limit_background_resources(max_cpus=max_cpus)

    # Import after applying affinity so native worker pools inherit the limit.
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError
    from lerobot.common.datasets.lerobot_dataset import (
        CODEBASE_VERSION,
        REPOCARD_NAME,
        create_lerobot_dataset_card,
    )

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=snapshot_root,
        repo_type="dataset",
    )

    if not api.file_exists(repo_id, REPOCARD_NAME, repo_type="dataset"):
        dataset_info = json.loads((snapshot_root / "meta" / "info.json").read_text())
        card = create_lerobot_dataset_card(
            tags=None,
            dataset_info=dataset_info,
            license="apache-2.0",
        )
        card.push_to_hub(repo_id=repo_id, repo_type="dataset")

    with contextlib.suppress(RevisionNotFoundError):
        api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    api.create_tag(
        repo_id,
        tag=CODEBASE_VERSION,
        revision=None,
        repo_type="dataset",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_root", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--max-cpus", type=int, default=2)
    args = parser.parse_args()
    upload_snapshot(
        args.snapshot_root,
        args.repo_id,
        private=args.private,
        max_cpus=max(1, args.max_cpus),
    )


if __name__ == "__main__":
    main()
