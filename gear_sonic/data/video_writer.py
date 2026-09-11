import os
import queue
import threading
import time

import av
import numpy as np


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        buffer_size: int = 50,
        enqueue_timeout_s: float = 0.25,
    ):
        if buffer_size <= 0:
            raise ValueError("video buffer size must be positive")
        if enqueue_timeout_s <= 0:
            raise ValueError("video enqueue timeout must be positive")
        self.output_path = output_path
        self._first_frame = True
        self._enqueue_timeout_s = enqueue_timeout_s

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=buffer_size)
        self.container = av.open(output_path, mode="w")
        try:
            self.stream = self.container.add_stream(codec, rate=fps)
            self.stream.width = width
            self.stream.height = height
            self.stream.codec_context.thread_count = min(2, os.cpu_count() or 1)
            if codec in ("h264", "libx264"):
                self.stream.options = {"preset": "veryfast", "tune": "zerolatency", "crf": "23"}
        except Exception:
            self.container.close()
            raise
        self._writer_error: Exception | None = None
        self._accepting_frames = True
        self._cancelled = False
        self._stop_enqueued = False
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._writer_worker,
            name=f"video-writer-{os.path.basename(output_path)}",
            daemon=True,
        )
        self._thread.start()

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        assert (
            frame.shape[1] == self.stream.width and frame.shape[0] == self.stream.height
        ), (
            f"Incorrect frame dimensions. Input dimensions: {frame.shape[1]}x{frame.shape[0]}. "
            f"Expected dimensions: {self.stream.width}x{self.stream.height}"
        )

    def add_frame(self, frame: np.ndarray) -> None:
        if not self._accepting_frames:
            raise RuntimeError("cannot add a frame after the video writer has stopped")
        if self._writer_error is not None:
            raise RuntimeError("video writer worker failed") from self._writer_error
        self._assert_dimensions(frame)
        try:
            self.queue.put(frame, timeout=self._enqueue_timeout_s)
        except queue.Full as exc:
            raise RuntimeError(
                f"video writer queue stayed full for {self._enqueue_timeout_s:.2f}s"
            ) from exc

    def check_ready(self, frame: np.ndarray) -> bool:
        """Preflight a synchronized row before its single producer queues any camera."""
        if not self._accepting_frames:
            raise RuntimeError("cannot add a frame after the video writer has stopped")
        if self._writer_error is not None:
            raise RuntimeError("video writer worker failed") from self._writer_error
        self._assert_dimensions(frame)
        return not self.queue.full()

    def _writer_worker(self) -> None:
        try:
            self._encode_frames()
            if self._writer_error is None and not self._cancelled:
                self._flush_stream()
        except Exception as exc:
            self._writer_error = exc
        finally:
            try:
                self.container.close()
            except Exception as exc:
                if self._writer_error is None:
                    self._writer_error = exc
            finally:
                self._closed = True

    def _encode_frames(self) -> None:
        while True:
            frame = self.queue.get()
            try:
                if frame is None:
                    return
                # Once encoding fails, drain queued frames so shutdown remains
                # bounded; stop() reports the original exception.
                if self._writer_error is not None or self._cancelled:
                    continue
                self._assert_dimensions(frame)
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

                if self._first_frame:
                    # Each camera has its own encoder thread. Capture only its
                    # startup logs; redirecting process-wide stderr races when
                    # several videos start together.
                    with av.logging.Capture(local=True):
                        packets = self.stream.encode(frame)
                        for packet in packets:
                            self.container.mux(packet)
                    self._first_frame = False
                else:
                    packets = self.stream.encode(frame)
                    for packet in packets:
                        self.container.mux(packet)
            except Exception as exc:
                self._writer_error = exc
            finally:
                self.queue.task_done()

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def _join_worker(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        if not self._stop_enqueued:
            remaining = deadline - time.monotonic()
            try:
                self.queue.put(None, timeout=max(0.0, remaining))
            except queue.Full as exc:
                raise TimeoutError(
                    f"video writer did not accept its stop request within {timeout_s:.1f}s"
                ) from exc
            self._stop_enqueued = True
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError(f"video writer did not stop within {timeout_s:.1f}s")

    def stop(self, timeout_s: float = 30.0) -> str:
        """Wait up to timeout_s for draining, encoder flush, and container close.

        A timeout leaves the worker owning its container; stop can be retried.
        """
        self._finish(timeout_s, cancel=False)
        return self.output_path

    def cancel(self, timeout_s: float = 5.0) -> None:
        """Stop safely and remove the incomplete output file."""
        self._finish(timeout_s, cancel=True)

    def _finish(self, timeout_s: float, *, cancel: bool) -> None:
        if timeout_s <= 0:
            raise ValueError("video writer shutdown timeout must be positive")
        deadline = time.monotonic() + timeout_s
        if not self._close_lock.acquire(timeout=timeout_s):
            raise TimeoutError("video writer shutdown is already in progress")
        try:
            if cancel and self._closed and not self._cancelled:
                return
            self._accepting_frames = False
            self._cancelled = self._cancelled or cancel
            self._join_worker(max(0.0, deadline - time.monotonic()))
            if cancel and os.path.exists(self.output_path):
                os.remove(self.output_path)
            if not cancel and self._writer_error is not None:
                raise RuntimeError("video writer worker failed") from self._writer_error
        finally:
            self._close_lock.release()
