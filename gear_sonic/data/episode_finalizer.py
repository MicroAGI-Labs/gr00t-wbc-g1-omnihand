"""Background finalization for detached data-collection episodes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import queue
import threading
import time
from typing import Any

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.video_writer import VideoWriter


@dataclass(frozen=True)
class EpisodeFinalizationJob:
    episode_index: int
    episode_buffer: dict[str, Any]
    video_writers: dict[str, VideoWriter]
    discarded: bool
    validation: dict[str, Any]


@dataclass(frozen=True)
class EpisodeFinalizationResult:
    episode_index: int
    discarded: bool
    error: str | None = None
    recovery_path: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class EpisodeFinalizer:
    """Serialize local episode commits without blocking the capture loop."""

    def __init__(
        self,
        data_exporter: Gr00tDataExporter,
        *,
        max_pending: int = 2,
        writer_stop_timeout_s: float = 5.0,
    ):
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if writer_stop_timeout_s <= 0:
            raise ValueError("writer_stop_timeout_s must be positive")
        self.data_exporter = data_exporter
        self.max_pending = max_pending
        self.writer_stop_timeout_s = writer_stop_timeout_s
        self._queue: queue.Queue[EpisodeFinalizationJob | None] = queue.Queue()
        self._condition = threading.Condition()
        self._outstanding = 0
        self._finalizing = False
        self._last_finalized_episode: int | None = None
        self._error: str | None = None
        self._results: deque[EpisodeFinalizationResult] = deque()
        self._closed = False
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
        episode_buffer: dict[str, Any],
        video_writers: dict[str, VideoWriter],
        discarded: bool,
        validation: dict[str, Any],
    ) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("episode finalizer is closed")
            if self._error is not None:
                raise RuntimeError("episode finalizer has a previous failure that requires attention")
            if self._outstanding >= self.max_pending:
                raise RuntimeError("episode finalizer queue is at capacity")
            self._outstanding += 1
            self._condition.notify_all()
        self._queue.put(
            EpisodeFinalizationJob(
                episode_index=episode_index,
                episode_buffer=episode_buffer,
                video_writers=video_writers,
                discarded=discarded,
                validation=validation,
            )
        )

    def can_accept(self) -> bool:
        with self._condition:
            return not self._closed and self._error is None and self._outstanding < self.max_pending

    def status(self) -> dict[str, object]:
        with self._condition:
            return {
                "pending": self._outstanding,
                "finalizing": self._finalizing,
                "last_finalized_episode": self._last_finalized_episode,
                "error": self._error,
                "at_capacity": self._outstanding >= self.max_pending,
            }

    def drain_results(self) -> list[EpisodeFinalizationResult]:
        with self._condition:
            results = list(self._results)
            self._results.clear()
            return results

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._outstanding:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def close(self, timeout: float = 30.0) -> None:
        if timeout <= 0:
            raise ValueError("finalizer close timeout must be positive")
        if not self.wait_until_idle(timeout=timeout):
            raise TimeoutError(f"episode finalizer did not become idle within {timeout:.1f}s")
        with self._condition:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("episode finalizer thread did not stop")

    def _persist_failed_job(self, job: EpisodeFinalizationJob) -> Path:
        recovery_root = Path(self.data_exporter.root) / "recovery" / f"episode_{job.episode_index:06d}"
        recovery_root.mkdir(parents=True, exist_ok=True)
        recovery_path = recovery_root / "episode_buffer.pkl"
        temporary_path = recovery_path.with_suffix(".pkl.tmp")
        payload = {
            "episode_index": job.episode_index,
            "episode_buffer": job.episode_buffer,
            "discarded": job.discarded,
            "validation": job.validation,
        }
        with open(temporary_path, "wb") as recovery_file:
            pickle.dump(payload, recovery_file, protocol=pickle.HIGHEST_PROTOCOL)
            recovery_file.flush()
            os.fsync(recovery_file.fileno())
        os.replace(temporary_path, recovery_path)
        return recovery_path

    def _stop_job_writers(self, job: EpisodeFinalizationJob) -> list[str]:
        errors = []
        for key, writer in job.video_writers.items():
            try:
                writer.stop(timeout_s=self.writer_stop_timeout_s)
            except Exception as exc:
                errors.append(f"{key}: {exc}")
        return errors

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            with self._condition:
                self._finalizing = True
                self._condition.notify_all()
            result: EpisodeFinalizationResult | None = None
            try:
                if job.discarded:
                    self.data_exporter.save_episode_as_discarded(
                        job.episode_buffer,
                        video_writers=job.video_writers,
                        validation=job.validation,
                    )
                else:
                    self.data_exporter.save_episode(
                        job.episode_buffer,
                        video_writers=job.video_writers,
                        validation=job.validation,
                    )
                result = EpisodeFinalizationResult(
                    episode_index=job.episode_index,
                    discarded=job.discarded,
                )
                with self._condition:
                    self._last_finalized_episode = job.episode_index
            except Exception as exc:
                writer_errors = self._stop_job_writers(job)
                try:
                    recovery_path = self._persist_failed_job(job)
                    recovery_value = str(recovery_path)
                except Exception as recovery_exc:
                    recovery_value = None
                    writer_errors.append(f"recovery: {recovery_exc}")
                detail = str(exc)
                if writer_errors:
                    detail = f"{detail}; cleanup errors: {'; '.join(writer_errors)}"
                result = EpisodeFinalizationResult(
                    episode_index=job.episode_index,
                    discarded=job.discarded,
                    error=detail[-500:],
                    recovery_path=recovery_value,
                )
                print(
                    f"[Finalizer] Episode {job.episode_index} failed: {result.error}",
                    flush=True,
                )
                with self._condition:
                    self._error = result.error
            finally:
                with self._condition:
                    if result is not None:
                        self._results.append(result)
                    self._outstanding -= 1
                    self._finalizing = False
                    self._condition.notify_all()
