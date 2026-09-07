import os
import queue
import sys
import threading

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
    ):
        self.output_path = output_path
        self._first_frame = True

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=buffer_size)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        # FFmpeg otherwise creates roughly two threads per CPU for every open
        # episode. Keep encoding predictable so finalization and Hub activity
        # cannot starve the 50 Hz recorder/control subscribers.
        self.stream.codec_context.thread_count = min(2, os.cpu_count() or 1)
        self._writer_error: BaseException | None = None
        self._accepting_frames = True
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(target=self._writer_worker, daemon=True)
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
        self.queue.put(frame)

    def _writer_worker(self) -> None:
        while True:
            frame = self.queue.get()
            try:
                if frame is None:
                    return
                # After an encoder failure, keep draining the queue so stop()
                # cannot deadlock. The original exception is raised by stop().
                if self._writer_error is not None:
                    continue
                self._assert_dimensions(frame)
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

                if self._first_frame:
                    stderr_fd = sys.stderr.fileno()
                    old_stderr = os.dup(stderr_fd)
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, stderr_fd)
                    try:
                        packets = self.stream.encode(frame)
                        for packet in packets:
                            self.container.mux(packet)
                    finally:
                        os.dup2(old_stderr, stderr_fd)
                        os.close(old_stderr)
                        os.close(devnull)
                        self._first_frame = False
                else:
                    packets = self.stream.encode(frame)
                    for packet in packets:
                        self.container.mux(packet)
            except BaseException as exc:
                self._writer_error = exc
            finally:
                self.queue.task_done()

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def stop(self) -> str:
        """Drain queued frames, stop the encoder thread, and close the container."""
        with self._close_lock:
            if self._closed:
                return self.output_path
            self._accepting_frames = False
            self.queue.put(None)
            self._thread.join()
            if self._writer_error is not None:
                self.container.close()
                self._closed = True
                raise RuntimeError("video writer worker failed") from self._writer_error
            self._flush_stream()
            self.container.close()
            self._closed = True
            return self.output_path

    def cancel(self) -> None:
        """Immediately stops writing and deletes the output file."""
        with self._close_lock:
            if self._closed:
                return
            self._accepting_frames = False
            self.queue.put(None)
            self._thread.join()
            self.container.close()
            self._closed = True
            if os.path.exists(self.output_path):
                os.remove(self.output_path)

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            self.container.close()
