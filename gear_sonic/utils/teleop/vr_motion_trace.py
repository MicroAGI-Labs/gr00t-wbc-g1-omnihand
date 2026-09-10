"""Bounded, asynchronous before/after command logging for limiter trials."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import time

import numpy as np
import msgpack
import zmq


class VRMotionTrace:
    def __init__(self, directory, limits, *, capacity=2048, feedback_endpoint=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = directory / f"vr-motion-{stamp}.jsonl"
        self._file = self.path.open("x")
        self._file.write(json.dumps({"type": "metadata", "version": 1,
                                    "wall_time_ns": time.time_ns(),
                                    "monotonic_ns": time.monotonic_ns(),
                                    "limits": None if limits is None else asdict(limits)}) + "\n")
        self._file.flush()
        self._queue = Queue(maxsize=capacity)
        self._feedback_endpoint = feedback_endpoint
        self._stop = Event()
        self.dropped = 0
        self.error = ""
        self._worker = Thread(target=self._run, name="vr-motion-trace", daemon=True)
        self._worker.start()

    def record(self, target, output, conditioner, *, now, stream_mode, generated, packet_timestamp,
               source_fresh=True, source_timestamp_ns=None):
        if self.error or self._stop.is_set():
            return
        sample = {"type": "sample", "time": now, "stream_mode": int(stream_mode),
                  "generated": generated, "packet_timestamp": packet_timestamp, "source_fresh": source_fresh,
                  "source_timestamp_ns": source_timestamp_ns,
                  "input": np.asarray(target).copy(), "output": np.asarray(output).copy(),
                  "dropped": self.dropped}
        if conditioner is not None:
            sample.update({name: getattr(conditioner, name).copy() for name in (
                "velocity", "acceleration", "angular_velocity", "angular_acceleration",
                "limited", "reserve_active")})
            sample.update(fault=conditioner.fault, rejected=conditioner.rejected,
                          seed_generation=conditioner.seed_generation)
        try:
            self._queue.put_nowait(sample)
        except Full:
            self.dropped += 1

    def _run(self):
        context = feedback = None
        try:
            if self._feedback_endpoint:
                context = zmq.Context()
                feedback = context.socket(zmq.SUB)
                feedback.setsockopt(zmq.LINGER, 0)
                feedback.setsockopt(zmq.CONFLATE, 1)
                feedback.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
                feedback.connect(self._feedback_endpoint)
            last_flush = time.monotonic()
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    sample = self._queue.get(timeout=0.1)
                except Empty:
                    sample = None
                if sample is not None:
                    self._file.write(json.dumps(sample, default=lambda value: value.tolist(),
                                                separators=(",", ":")) + "\n")
                if feedback is not None and feedback.poll(0):
                    packet = feedback.recv()
                    try:
                        data = msgpack.unpackb(packet[len(b"g1_debug"):], raw=False)
                        values = {key: data[key] for key in (
                            "body_q_measured", "body_dq_measured", "body_q_target", "last_action",
                            "vr_3point_position", "vr_3point_orientation", "publisher_monotonic_ns",
                        ) if key in data}
                        self._file.write(json.dumps({"type": "feedback", "time": time.monotonic(),
                                                    "data": values}, separators=(",", ":")) + "\n")
                    except (ValueError, TypeError, msgpack.ExtraData):
                        pass
                if time.monotonic() - last_flush >= 1:
                    self._file.flush()
                    last_flush = time.monotonic()
        except Exception as exc:
            self.error = str(exc)
            print(f"[VRMotionTrace] Logging stopped: {exc}")
        finally:
            if feedback is not None:
                feedback.close()
            if context is not None:
                context.term()
            if not self.error:
                self._file.write(json.dumps({"type": "end", "dropped": self.dropped}) + "\n")
            self._file.close()

    def close(self):
        self._stop.set()
        self._worker.join(timeout=3)
