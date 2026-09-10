"""Read-only monotonic-clock exchange for recording across Linux hosts.

Run on a remote producer with ``python -m gear_sonic.data.clock_sync --port 5574``.
The midpoint estimate assumes symmetric transit; half the round-trip time bounds
the offset error for nonnegative transit times. This does not synchronize clocks.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json
import threading
import time

import zmq

DEFAULT_CLOCK_PORT = 5574
SCHEMA = "sonic.clock.v1"


def clock_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


class ClockUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ClockEstimate:
    clock_id: str
    offset_ns: int  # remote minus recorder
    uncertainty_ns: int
    measured_ns: int

    def to_local(self, source_ns: int) -> int:
        return source_ns - self.offset_ns


def estimate_exchange(t1: int, t2: int, t3: int, t4: int, identity: str) -> ClockEstimate:
    """t1/t4 are client send/receive; t2/t3 are server receive/send."""
    if any(type(t) is not int or t <= 0 for t in (t1, t2, t3, t4)):
        raise ValueError("clock timestamps must be positive integer nanoseconds")
    if not isinstance(identity, str) or not identity or t4 < t1 or t3 < t2:
        raise ValueError("invalid clock exchange")
    transit = (t4 - t1) - (t3 - t2)
    if transit < 0:
        raise ValueError("negative clock transit time")
    return ClockEstimate(identity, ((t2 - t1) + (t3 - t4)) // 2, (transit + 1) // 2, t4)


class ClockServer:
    """A dedicated thread owns its REP socket; no robot or motor I/O."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="recording-clock-server", daemon=True)

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(2):
            raise RuntimeError("clock server startup timed out")
        if self._error:
            raise RuntimeError(f"clock server failed: {self._error}") from self._error

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.MAXMSGSIZE, 4096)
        try:
            identity = clock_id()
            socket.bind(self.endpoint)
            self.endpoint = socket.getsockopt(zmq.LAST_ENDPOINT).decode()
            self._ready.set()
            while not self._stop.is_set():
                if not socket.poll(100):
                    continue
                raw = socket.recv()
                received = time.monotonic_ns()
                try:
                    request = json.loads(raw)
                    if request.get("schema") != SCHEMA or type(request.get("t1")) is not int:
                        raise ValueError("invalid request")
                    reply = {"schema": SCHEMA, "t1": request["t1"], "t2": received,
                             "clock_id": identity, "t3": time.monotonic_ns()}
                except (ValueError, AttributeError):
                    reply = {"error": "invalid clock request"}
                socket.send_json(reply)
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            socket.close()
            context.term()

    def close(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=2)


class ClockClient:
    """Continuously estimate offset with a short window of low-RTT exchanges."""

    def __init__(self, endpoint: str, *, max_uncertainty_ns: int = 5_000_000):
        self.endpoint = endpoint
        self.max_uncertainty_ns = max_uncertainty_ns
        self._samples: deque[ClockEstimate] = deque(maxlen=20)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error = "waiting for clock exchanges"
        self._thread = threading.Thread(target=self._run, name="recording-clock-client", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        context = zmq.Context()
        try:
            while not self._stop.is_set():
                # A new socket after each exchange also recovers cleanly from timeout.
                socket = context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.MAXMSGSIZE, 4096)
                try:
                    socket.connect(self.endpoint)
                    t1 = time.monotonic_ns()
                    socket.send_json({"schema": SCHEMA, "t1": t1})
                    if not socket.poll(200):
                        raise ClockUnavailable("clock exchange timed out")
                    reply = socket.recv_json()
                    t4 = time.monotonic_ns()
                    if reply.get("schema") != SCHEMA or reply.get("t1") != t1:
                        raise ValueError("invalid clock reply")
                    sample = estimate_exchange(t1, reply["t2"], reply["t3"], t4, reply["clock_id"])
                    with self._lock:
                        if self._samples and self._samples[-1].clock_id != sample.clock_id:
                            self._samples.clear()
                        self._samples.append(sample)
                        self._error = ""
                except (zmq.ZMQError, ClockUnavailable, ValueError, KeyError, TypeError, AttributeError) as exc:
                    with self._lock:
                        self._error = str(exc)
                finally:
                    socket.close()
                self._stop.wait(0.1)
        finally:
            context.term()

    def estimate(self, now_ns: int | None = None) -> ClockEstimate:
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            samples = [s for s in self._samples if 0 <= now - s.measured_ns <= 2_000_000_000]
            if len(samples) < 3:
                raise ClockUnavailable(self._error or "insufficient fresh clock exchanges")
            best = min(samples, key=lambda s: s.uncertainty_ns)
        # Budget 100 ppm relative drift since the selected exchange.
        uncertainty = best.uncertainty_ns + (now - best.measured_ns) // 10_000
        if uncertainty > self.max_uncertainty_ns:
            raise ClockUnavailable(f"clock uncertainty {uncertainty / 1e6:.2f} ms exceeds limit")
        return ClockEstimate(best.clock_id, best.offset_ns, uncertainty, best.measured_ns)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_CLOCK_PORT)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be within 1..65535")
    server = ClockServer(f"tcp://*:{args.port}")
    try:
        server.start()
        print(f"Recording clock service: {server.endpoint}", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
