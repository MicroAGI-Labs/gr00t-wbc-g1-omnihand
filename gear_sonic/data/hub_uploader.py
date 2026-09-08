"""Upload committed LeRobot episodes without putting network work on the recorder."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

DATASET_CONFIG_PREFIX = "dataset_config:"
_IDENTITY_FILE = "sonic_dataset.json"


def validate_dataset_config(payload: object) -> dict:
    from huggingface_hub.utils import validate_repo_id

    if not isinstance(payload, dict):
        raise ValueError("dataset configuration must be an object")
    repo_id, prompt = payload.get("repo_id"), payload.get("prompt")
    private = payload.get("private", True)
    if not isinstance(repo_id, str) or repo_id.count("/") != 1:
        raise ValueError("dataset must be namespace/name")
    repo_id = repo_id.strip()
    validate_repo_id(repo_id)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1000:
        raise ValueError("task prompt must contain 1 to 1000 characters")
    if not isinstance(private, bool):
        raise ValueError("private must be a boolean")
    return {"repo_id": repo_id, "prompt": prompt.strip(), "private": private}


class EpisodeHubUploader:
    """One active upload and one replaceable, cumulative pending snapshot.

    ``enqueue`` runs in the local finalizer, after its metadata commit and before
    the next one. Metadata is copied; immutable episode files are hard-linked.
    """

    def __init__(self, data_exporter, *, upload_runner=None):
        self.data_exporter = data_exporter
        self.root = Path(data_exporter.root)
        self._config_path = self.root / ".hub_upload.json"
        self._config = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._queued = None
        self._active = False
        self._process = None
        self._error = None
        self._staging_error = None
        self._last_uploaded = None
        self._upload_runner = upload_runner or self._run_subprocess
        if self._config_path.exists():
            saved = json.loads(self._config_path.read_text())
            self._config = {**validate_dataset_config(saved), "source_id": saved["source_id"]}
            self.data_exporter.task = self._config["prompt"]
            self.data_exporter.meta.repo_id = self._config["repo_id"]
        self._thread = threading.Thread(target=self._run, name="episode-hub-uploader", daemon=True)
        self._thread.start()
        if self._config and self.data_exporter.meta.info.get("total_episodes", 0):
            self.enqueue(self.data_exporter.meta.info["total_episodes"] - 1)

    def configure(self, payload: object) -> None:
        config = validate_dataset_config(payload)
        with self._condition:
            if self._stop.is_set():
                raise RuntimeError("uploader is closed")
            if self._config and all(self._config[k] == v for k, v in config.items()):
                return  # Repeated command delivery is idempotent.
            if self._active or self._queued or self.data_exporter.meta.info.get("total_episodes", 0):
                raise RuntimeError("dataset and task are locked after the first saved episode")
            config["source_id"] = uuid.uuid4().hex
            temporary = self._config_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(config))
            temporary.replace(self._config_path)
            self._config = config
            self.data_exporter.task = config["prompt"]
            self.data_exporter.meta.repo_id = config["repo_id"]

    def status(self) -> dict:
        with self._condition:
            config = self._config or {}
            return {
                "ready": bool(config),
                "repo_id": config.get("repo_id"),
                "prompt": config.get("prompt", self.data_exporter.task),
                "private": config.get("private", True),
                "pending": int(self._active) + int(self._queued is not None),
                "uploading": self._active,
                "last_uploaded_episode": self._last_uploaded,
                "error": self._staging_error or self._error,
            }

    def enqueue(self, episode_index: int) -> None:
        if not self._config or self._stop.is_set():
            return
        try:
            snapshot = self._snapshot(episode_index)
        except Exception as exc:
            with self._condition:
                self._staging_error = f"Episode {episode_index} upload staging failed: {exc}"[-500:]
            return  # Local save already succeeded; do not poison the finalizer.
        with self._condition:
            if self._stop.is_set():
                shutil.rmtree(snapshot, ignore_errors=True)
                return
            old = self._queued
            self._queued = (episode_index, snapshot)
            self._staging_error = None
            self._condition.notify_all()
        if old:
            shutil.rmtree(old[1], ignore_errors=True)

    def _snapshot(self, episode_index: int) -> Path:
        staging = self.root / ".upload_snapshots"
        staging.mkdir(exist_ok=True)
        snapshot = Path(tempfile.mkdtemp(prefix=f"episode_{episode_index:06d}_", dir=staging))
        try:
            shutil.copytree(self.root / "meta", snapshot / "meta")
            for directory, suffix in (("data", "parquet"), ("videos", "mp4")):
                for source in (self.root / directory).rglob(f"episode_*.{suffix}"):
                    match = re.fullmatch(r"episode_(\d+)\.(?:parquet|mp4)", source.name)
                    if match and int(match[1]) <= episode_index:
                        target = snapshot / source.relative_to(self.root)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.link(source, target)
            if len(list((snapshot / "data").rglob("*.parquet"))) != episode_index + 1:
                raise FileNotFoundError("finalized episode data is incomplete")
            for name in ("README.md", "LICENSE"):
                if (self.root / name).is_file():
                    shutil.copy2(self.root / name, snapshot / name)
            (snapshot / _IDENTITY_FILE).write_text(json.dumps({"source_id": self._config["source_id"]}))
            return snapshot
        except Exception:
            shutil.rmtree(snapshot, ignore_errors=True)
            raise

    def wait_until_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._active or self._queued:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout: float = 5.0) -> None:
        self.wait_until_idle(timeout)
        with self._condition:
            self._stop.set()
            process = self._process
            self._condition.notify_all()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            raise RuntimeError("upload worker did not stop")
        if self._queued:
            shutil.rmtree(self._queued[1], ignore_errors=True)
            self._queued = None

    def _run_subprocess(self, snapshot: Path, config: dict) -> None:
        command = [sys.executable, "-m", __name__, str(snapshot), config["repo_id"]]
        if config["private"]:
            command.append("--private")
        with self._condition:
            if self._stop.is_set():
                return
            self._process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            process = self._process
        try:
            _, stderr = process.communicate()
            if process.returncode:
                raise RuntimeError((stderr or f"upload exited {process.returncode}")[-500:])
        finally:
            with self._condition:
                self._process = None

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._queued is not None or self._stop.is_set())
                if self._stop.is_set():
                    return
                job, self._queued = self._queued, None
                self._active = True
            delay = 1.0
            try:
                while not self._stop.is_set():
                    try:
                        self._upload_runner(job[1], dict(self._config))
                    except Exception as exc:
                        with self._condition:
                            self._error = str(exc)[-500:]
                        if self._stop.wait(delay):
                            break
                        delay = min(delay * 2, 30.0)
                        old = None
                        with self._condition:
                            if self._queued:
                                old = job
                                job, self._queued = self._queued, None
                        if old:
                            shutil.rmtree(old[1], ignore_errors=True)
                        continue
                    if self._stop.is_set():
                        break
                    with self._condition:
                        self._last_uploaded = job[0]
                        self._error = None
                    break
            finally:
                shutil.rmtree(job[1], ignore_errors=True)
                with self._condition:
                    self._active = False
                    self._condition.notify_all()


def _check_repository(api, repo_id: str, private: bool, source_id: str | None = None):
    from huggingface_hub import hf_hub_download

    api.create_repo(repo_id, private=private, repo_type="dataset", exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if info.private != private:
        raise ValueError("repository visibility differs from the selected setting")
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=info.sha)
    if _IDENTITY_FILE in files:
        identity = Path(hf_hub_download(repo_id, _IDENTITY_FILE, repo_type="dataset", revision=info.sha))
        if not source_id or json.loads(identity.read_text()).get("source_id") != source_id:
            raise ValueError("repository belongs to another local dataset; choose an empty repository")
    elif any(path.startswith(("meta/", "data/", "videos/")) for path in files):
        raise ValueError("repository already contains a dataset; choose an empty repository")
    return info, files


def prepare_repository(config: dict) -> None:
    """Check/create an empty destination in the HTTP thread before recording."""
    from huggingface_hub import HfApi

    _check_repository(HfApi(), config["repo_id"], config["private"])


def upload_snapshot(snapshot: Path, repo_id: str, *, private: bool) -> None:
    """Run only in the upload child process, using the host's existing HF login."""
    with contextlib.suppress(OSError):
        os.nice(10)
        if hasattr(os, "sched_getaffinity"):
            os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, create_lerobot_dataset_card

    api = HfApi()
    source_id = json.loads((snapshot / _IDENTITY_FILE).read_text())["source_id"]
    info, files = _check_repository(api, repo_id, private, source_id)
    if "README.md" not in files and not (snapshot / "README.md").exists():
        create_lerobot_dataset_card(
            tags=None, dataset_info=json.loads((snapshot / "meta/info.json").read_text()), license="apache-2.0"
        ).save(snapshot / "README.md")
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=snapshot, parent_commit=info.sha
    )
    with contextlib.suppress(RevisionNotFoundError):
        api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=commit.oid, repo_type="dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    upload_snapshot(args.snapshot, args.repo_id, private=args.private)
