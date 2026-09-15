"""Finalize locally owned episode buffers without initiating uploads."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import queue
import tempfile
import threading
import time

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.hub_uploader import EpisodeHubUploader
from gear_sonic.data.recording_identity import recording_identity


@dataclass
class _EpisodeFinalizationJob:
    episode_index: int
    episode_buffer: dict
    video_writers: dict
    success: bool
    validation: dict
    delete: bool = False


class EpisodeFinalizer:
    """Finalize local episode files without blocking recorder control/status."""

    def __init__(
        self,
        data_exporter: Gr00tDataExporter,
        hub_uploader: EpisodeHubUploader | None = None,
        max_pending: int = 2,
    ):
        self.data_exporter = data_exporter
        self.max_pending = max_pending
        self._queue: queue.Queue[_EpisodeFinalizationJob | None] = queue.Queue()
        self._condition = threading.Condition()
        self._pending = 0
        self._pending_saves = 0
        self._pending_discards = 0
        self._finalizing = False
        self._last_finalized_episode: int | None = None
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="episode-finalizer",
            daemon=True,
        )
        self._thread.start()

    def enqueue(
        self,
        *,
        episode_index: int,
        episode_buffer: dict,
        video_writers: dict,
        success: bool,
        validation: dict,
        delete: bool = False,
    ) -> None:
        with self._condition:
            self._pending += 1
            if delete:
                self._pending_discards += 1
            else:
                self._pending_saves += 1
            self._condition.notify_all()
        self._queue.put(
            _EpisodeFinalizationJob(
                episode_index=episode_index,
                episode_buffer=episode_buffer,
                video_writers=video_writers,
                success=success,
                validation=validation,
                delete=delete,
            )
        )

    def can_record(self) -> bool:
        with self._condition:
            return self._error is None and self._pending < self.max_pending

    def status(self) -> dict[str, object]:
        with self._condition:
            return {
                "pending": self._pending,
                "pending_saves": self._pending_saves,
                "pending_discards": self._pending_discards,
                "finalizing": self._finalizing,
                "last_finalized_episode": self._last_finalized_episode,
                "error": self._error,
                "at_capacity": self._pending >= self.max_pending,
            }

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._pending or self._finalizing:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def close(self, timeout: float | None = None) -> None:
        self.wait_until_idle(timeout=timeout)
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def _persist_failed_job(self, job: _EpisodeFinalizationJob) -> Path:
        """Preserve an owned episode buffer if normal finalization fails."""
        recovery_root = Path(self.data_exporter.root) / "recovery"
        recovery_root.mkdir(parents=True, exist_ok=True)
        identity = recording_identity(job.episode_buffer)
        payload = {
            "episode_index": job.episode_index,
            "recording": dict(identity),
            "episode_buffer": job.episode_buffer,
            "success": job.success,
            "validation": job.validation,
            "video_paths": {key: str(writer.output_path) for key, writer in job.video_writers.items()
                            if hasattr(writer, "output_path")},
        }
        with tempfile.NamedTemporaryFile(
            dir=recovery_root, prefix=identity["id"] + "_", suffix=".pkl.tmp", delete=False,
        ) as recovery_file:
            temporary_path = Path(recovery_file.name)
            pickle.dump(payload, recovery_file, protocol=pickle.HIGHEST_PROTOCOL)
            recovery_file.flush()
            os.fsync(recovery_file.fileno())
        recovery_path = temporary_path.with_suffix("")
        os.link(temporary_path, recovery_path)
        temporary_path.unlink()
        return recovery_path

    @staticmethod
    def _stop_job_writers(job: _EpisodeFinalizationJob) -> None:
        for writer in job.video_writers.values():
            try:
                writer.stop()
            except Exception:
                pass

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            with self._condition:
                self._finalizing = True
                self._condition.notify_all()
            try:
                if job.delete:
                    self.data_exporter.discard_episode(video_writers=job.video_writers)
                else:
                    self.data_exporter.save_episode(
                        job.episode_buffer,
                        video_writers=job.video_writers,
                        success=job.success,
                        validation=job.validation,
                    )
                if not job.delete:
                    with self._condition:
                        self._last_finalized_episode = job.episode_index
            except Exception as exc:
                if job.delete:
                    recovery_detail = "; discarded take cleanup failed; check .recording"
                else:
                    self._stop_job_writers(job)
                    try:
                        recovery_path = self._persist_failed_job(job)
                        recovery_detail = f"; recovery saved to {recovery_path}"
                    except Exception as recovery_exc:
                        recovery_detail = f"; recovery also failed: {recovery_exc}"
                error = f"{exc}{recovery_detail}"
                print(f"[Finalizer] Episode {job.episode_index} failed: {error}")
                with self._condition:
                    self._error = error[-500:]
            finally:
                # Release completed buffers and encoders before waiting again.
                deleted = job.delete
                del job
                with self._condition:
                    self._pending -= 1
                    if deleted:
                        self._pending_discards -= 1
                    else:
                        self._pending_saves -= 1
                    self._finalizing = False
                    self._condition.notify_all()
