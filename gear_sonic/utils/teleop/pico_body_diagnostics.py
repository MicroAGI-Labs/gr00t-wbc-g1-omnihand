"""Read-only raw PICO diagnostics; the existing SDK reader supplies every sample."""

from collections import deque
import threading
import time

import numpy as np
import zmq

DEFAULT_PICO_BODY_PORT = 5574
STALE_SECONDS = 2.0


class PicoBodySnapshot:
    def __init__(self):
        self._lock = threading.Lock()
        self._poses = None
        self._available = False
        self._error = "Waiting for PICO input"
        self._stamps = {"packet": None, "body": None}
        self._advanced = {"packet": None, "body": None}
        self._sample_at = None
        self._rate_samples = {"body": deque(), "body_read": deque()}

    def _rate(self, kind, now):
        samples = self._rate_samples[kind]
        while samples and now - samples[0] > STALE_SECONDS:
            samples.popleft()
        if not samples:
            return 0.0
        if len(samples) < 2:
            return None
        return (len(samples) - 1) / max(now - samples[0], 1e-9)

    def record(self, poses, packet_timestamp_ns, body_timestamp_ns, *, available=True, error="", now=None):
        now = time.monotonic() if now is None else now
        array = None if poses is None else np.asarray(poses, dtype=np.float64)
        if array is not None and (array.shape != (24, 7) or not np.isfinite(array).all()):
            array, available, error = None, False, "Invalid body pose received"
        with self._lock:
            self._available, self._error, self._sample_at = available, error, now
            valid_body = available and array is not None
            # Keep the last valid frame visible through an outage, labelled stale.
            if array is not None:
                self._poses = array.tolist()
            for kind, stamp in (("packet", packet_timestamp_ns), ("body", body_timestamp_ns)):
                stamp = int(stamp) if stamp is not None and int(stamp) > 0 else None
                if stamp is not None and stamp != self._stamps[kind]:
                    self._advanced[kind] = now
                    if valid_body:
                        rate_kind = "body_read" if kind == "packet" else "body"
                        self._rate_samples[rate_kind].append(now)
                self._stamps[kind] = stamp
            for kind in self._rate_samples:
                self._rate(kind, now)

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            ages = {k: None if t is None else max(0.0, now - t) for k, t in self._advanced.items()}
            sampled = self._sample_at is not None and now - self._sample_at <= STALE_SECONDS
            packet_live = sampled and ages["packet"] is not None and ages["packet"] <= STALE_SECONDS
            body_live = packet_live and self._available and self._stamps["body"] is not None and ages["body"] is not None and ages["body"] <= STALE_SECONDS
            state = ("waiting" if self._sample_at is None else "disconnected" if not packet_live else
                     "unavailable" if not self._available else "timestamp_unavailable" if self._stamps["body"] is None else
                     "live" if body_live else "stale")
            return {
                "state": state, "packet_live": packet_live, "body_live": body_live,
                "body_available": self._available, "poses": self._poses,
                "packet_timestamp_ns": str(self._stamps["packet"]) if self._stamps["packet"] is not None else None,
                "body_timestamp_ns": str(self._stamps["body"]) if self._stamps["body"] is not None else None,
                "packet_age_s": ages["packet"], "body_age_s": ages["body"], "error": self._error,
                "body_hz": self._rate("body", now) if self._stamps["body"] is not None else None,
                "body_read_hz": self._rate("body_read", now),
            }


class PicoBodyPublisher(PicoBodySnapshot):
    """Publish a latest-only snapshot at 10 Hz; never access XRT from this thread."""
    def __init__(self, port=DEFAULT_PICO_BODY_PORT):
        super().__init__()
        self.port = port
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 1)
        try:
            socket.bind(f"tcp://127.0.0.1:{self.port}")
            while not self._stop.is_set():
                socket.send_json(self.snapshot())
                self._stop.wait(0.1)
        except Exception as exc:
            print(f"[PICO diagnostics] Publisher unavailable: {exc}")
        finally:
            socket.close(linger=0)
            context.term()


class PicoBodySubscriber:
    def __init__(self, host="localhost", port=DEFAULT_PICO_BODY_PORT):
        self.host, self.port = host, port
        self._lock = threading.Lock()
        self._payload, self._received_at = None, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def status(self):
        with self._lock:
            payload = dict(self._payload or {"state": "waiting", "poses": None})
            age = None if self._received_at is None else max(0.0, time.monotonic() - self._received_at)
        payload["connected"] = age is not None and age <= STALE_SECONDS
        for key in ("packet_age_s", "body_age_s"):
            if payload.get(key) is not None:
                payload[key] += age or 0.0
        if not payload["connected"]:
            payload.update(state="disconnected" if age is not None else "waiting", packet_live=False, body_live=False)
            payload.update(body_hz=0.0 if payload.get("body_hz") is not None else None, body_read_hz=0.0)
        return payload

    def _run(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect(f"tcp://{self.host}:{self.port}")
        try:
            while not self._stop.is_set():
                if socket.poll(100):
                    try:
                        payload = socket.recv_json()
                        if not isinstance(payload, dict):
                            continue
                        with self._lock:
                            self._payload, self._received_at = payload, time.monotonic()
                    except (ValueError, UnicodeError):
                        continue
        finally:
            socket.close(linger=0)
            context.term()
