"""Upload immutable snapshots of committed episodes outside the recorder loop."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from gear_sonic.data.exporter import Gr00tDataExporter


@dataclass(frozen=True)
class _SnapshotUploadJob:
    episode_index: int
    snapshot_root: Path


class EpisodeHubUploader:
    """Upload each finalized episode while keeping recorder state observable."""

    _EPISODE_FILE_RE = re.compile(r"episode_(\d+)\.(?:mp4|parquet)$")

    def __init__(self, data_exporter: Gr00tDataExporter, upload_runner=None):
        self.data_exporter = data_exporter
        self._queue: queue.Queue[_SnapshotUploadJob | None] = queue.Queue()
        self._upload_runner = upload_runner or self._run_upload_subprocess
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._config: dict[str, object] | None = None
        self._pending = 0
        self._uploading = False
        self._retrying = False
        self._last_uploaded_episode: int | None = None
        self._error: str | None = None
        self._process: subprocess.Popen | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="episode-hub-uploader",
            daemon=True,
        )
        self._thread.start()

    def configure(self, repo_id: str, prompt: str, private: bool) -> None:
        config = {"repo_id": repo_id, "prompt": prompt, "private": private}
        with self._condition:
            if self._pending or self._uploading:
                if config != self._config:
                    raise RuntimeError("cannot change dataset while an upload is pending")
                return
            self._config = config
            self._error = None
            self._retrying = False
            self.data_exporter.meta.repo_id = repo_id
            self.data_exporter.task = prompt
            self._condition.notify_all()

    def enqueue(self, episode_index: int) -> None:
        snapshot_root = self._create_snapshot(episode_index)
        with self._condition:
            if self._config is None:
                shutil.rmtree(snapshot_root, ignore_errors=True)
                raise RuntimeError("choose a Hugging Face dataset before saving an episode")
            self._pending += 1
            self._condition.notify_all()
        self._queue.put(
            _SnapshotUploadJob(
                episode_index=episode_index,
                snapshot_root=snapshot_root,
            )
        )

    def can_record(self) -> bool:
        """Recording only needs a configured dataset; uploads run in the background."""
        with self._condition:
            return self._config is not None

    def report_enqueue_failure(self, episode_index: int, error: Exception) -> None:
        """Expose upload staging failures without poisoning local finalization."""
        detail = f"episode {episode_index} upload staging failed: {error}"
        print(f"[Hub] {detail}")
        with self._condition:
            self._retrying = False
            self._error = detail[-500:]
            self._condition.notify_all()

    def status(self) -> dict[str, object]:
        with self._condition:
            config = dict(self._config or {})
            return {
                "ready": self._config is not None,
                "repo_id": config.get("repo_id"),
                "prompt": config.get("prompt", self.data_exporter.task),
                "private": config.get("private", True),
                "pending": self._pending,
                "uploading": self._uploading,
                "retrying": self._retrying,
                "last_uploaded_episode": self._last_uploaded_episode,
                "error": self._error,
            }

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._pending or self._uploading:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def close(self, timeout: float = 30.0) -> None:
        self.wait_until_idle(timeout=timeout)
        self._stop.set()
        with self._condition:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def _create_snapshot(self, episode_index: int) -> Path:
        """Freeze metadata and hard-link finalized episode files for upload."""
        root = Path(self.data_exporter.root)
        snapshots_root = root / ".upload_snapshots"
        snapshots_root.mkdir(parents=True, exist_ok=True)
        snapshot_root = Path(
            tempfile.mkdtemp(
                prefix=f"episode_{episode_index:06d}_",
                dir=snapshots_root,
            )
        )
        try:
            source_meta = root / "meta"
            if not source_meta.is_dir():
                raise FileNotFoundError(
                    f"dataset metadata directory is missing: {source_meta}"
                )
            shutil.copytree(
                source_meta,
                snapshot_root / "meta",
                copy_function=shutil.copy2,
            )

            for root_file in ("README.md", "LICENSE"):
                source = root / root_file
                if source.is_file():
                    shutil.copy2(source, snapshot_root / root_file)

            selected = 0
            for dirpath, _, filenames in os.walk(root, onerror=lambda _error: None):
                path = Path(dirpath)
                if path == snapshots_root or snapshots_root in path.parents:
                    continue
                for filename in filenames:
                    match = self._EPISODE_FILE_RE.search(filename)
                    if not match or int(match.group(1)) > episode_index:
                        continue
                    source = path / filename
                    target = snapshot_root / source.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
                    selected += 1
            if selected == 0:
                raise FileNotFoundError(
                    f"no finalized files found through episode {episode_index}"
                )
            return snapshot_root
        except Exception:
            shutil.rmtree(snapshot_root, ignore_errors=True)
            raise

    @staticmethod
    def _discard_snapshot(job: _SnapshotUploadJob) -> None:
        shutil.rmtree(job.snapshot_root, ignore_errors=True)

    def _run_upload_subprocess(
        self,
        snapshot_root: Path,
        config: dict[str, object],
    ) -> None:
        helper = Path(__file__).resolve().parents[1] / "scripts" / "upload_dataset_snapshot.py"
        command = [
            sys.executable,
            str(helper),
            str(snapshot_root),
            str(config["repo_id"]),
            "--max-cpus",
            "2",
        ]
        if bool(config.get("private", True)):
            command.append("--private")
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._condition:
            self._process = process
        try:
            _, stderr = process.communicate()
        finally:
            with self._condition:
                if self._process is process:
                    self._process = None
        if process.returncode != 0:
            detail = (stderr or "no error output").strip()[-2000:]
            raise RuntimeError(
                f"snapshot upload process exited with {process.returncode}: {detail}"
            )

    def _coalesce_queued(self, job: _SnapshotUploadJob) -> _SnapshotUploadJob:
        """Fold queued jobs into the newest complete dataset snapshot."""
        newest = job
        collapsed_jobs = []
        while True:
            try:
                queued = self._queue.get_nowait()
            except queue.Empty:
                break
            if queued is None:
                self._queue.put(None)
                break
            if queued.episode_index >= newest.episode_index:
                collapsed_jobs.append(newest)
                newest = queued
            else:
                collapsed_jobs.append(queued)
        for collapsed in collapsed_jobs:
            self._discard_snapshot(collapsed)
        if collapsed_jobs:
            with self._condition:
                self._pending -= len(collapsed_jobs)
                self._condition.notify_all()
        return newest

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._queue.get()
            if job is None:
                return
            retry_delay = 1.0
            while not self._stop.is_set():
                job = self._coalesce_queued(job)
                with self._condition:
                    config = dict(self._config or {})
                    self._uploading = True
                    self._retrying = retry_delay > 1.0
                    self._condition.notify_all()
                try:
                    self._upload_runner(job.snapshot_root, config)
                except Exception as exc:
                    with self._condition:
                        self._uploading = False
                        self._retrying = True
                        self._error = str(exc)[-500:]
                        self._condition.notify_all()
                    if self._stop.wait(retry_delay):
                        return
                    retry_delay = min(retry_delay * 2.0, 30.0)
                    continue

                self._discard_snapshot(job)
                with self._condition:
                    self._pending -= 1
                    self._uploading = False
                    self._retrying = False
                    self._error = None
                    self._last_uploaded_episode = job.episode_index
                    self._condition.notify_all()
                break
