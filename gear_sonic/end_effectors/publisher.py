"""Publish hand snapshots independently of blocking native SDK calls."""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
from queue import Empty, Full
import time

import zmq

from .protocol import HAND_CONFIG_TOPIC, HAND_STATE_TOPIC, encode


def age_state(state: dict, now_ns: int, sequence: int, frequency: float, target_timeout: float) -> dict:
    """Advance publication diagnostics without inventing a new feedback sample."""
    payload = copy.deepcopy(state)
    age = max(0.0, (now_ns - state["monotonic_ns"]) / 1e9)
    payload.update(
        publish_sequence=sequence,
        published_monotonic_ns=now_ns,
        state_age_s=age,
        publish_target_hz=frequency,
    )
    for item in (payload, *payload["sides"].values()):
        if item.get("input_age_s") is not None:
            item["input_age_s"] += age
            item["input_stale"] = bool(item.get("input_stale") or item["input_age_s"] > target_timeout)
    for side in payload["sides"].values():
        side["feedback_age_s"] = side.get("feedback_age_s", 0.0) + age
    payload["input_stale"] = bool(
        payload.get("input_stale") or any(side.get("input_stale", True) for side in payload["sides"].values())
    )
    return payload


def _publish_loop(endpoint, frequency, target_timeout, updates, stop, ready, parent_pid):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(endpoint)
        ready.send(None)
        ready.close()
        state = config = None
        sequence = 0
        last_config = -math.inf
        period = 1.0 / frequency
        deadline = time.monotonic()
        while not stop.is_set() and os.getppid() == parent_pid:
            try:
                new_state, new_config = updates.get_nowait()
            except Empty:
                pass
            else:
                if config != new_config:
                    last_config = -math.inf
                state, config = new_state, new_config
            now = time.monotonic()
            if state is not None:
                sequence += 1
                payload = age_state(state, time.monotonic_ns(), sequence, frequency, target_timeout)
                socket.send(encode(HAND_STATE_TOPIC, payload), flags=zmq.NOBLOCK)
                if config is not None and now - last_config >= 2.0:
                    socket.send(encode(HAND_CONFIG_TOPIC, config), flags=zmq.NOBLOCK)
                    last_config = now
            deadline += period
            if deadline < now:
                deadline = now + period
            stop.wait(max(0.0, deadline - time.monotonic()))
    except Exception as exc:
        if not ready.closed:
            ready.send(str(exc))
        raise
    finally:
        ready.close()
        socket.close()
        context.term()


class HandStatePublisher:
    """A bounded mailbox and one publisher process; SDK recovery stays in the caller."""

    def __init__(self, endpoint: str, frequency: float, target_timeout: float) -> None:
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("publish frequency must be finite and positive")
        if not math.isfinite(target_timeout) or target_timeout <= 0:
            raise ValueError("target timeout must be finite and positive")
        context = mp.get_context("spawn")  # Never fork a live SDK or ZeroMQ context.
        self.updates = context.Queue(maxsize=1)
        self.stop = context.Event()
        reader, writer = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_publish_loop,
            args=(endpoint, frequency, target_timeout, self.updates, self.stop, writer, os.getpid()),
            daemon=True,
        )
        try:
            self.process.start()
            writer.close()
            if not reader.poll(5.0):
                raise RuntimeError("hand state publisher did not start")
            error = reader.recv()
            if error is not None:
                raise RuntimeError(f"hand state publisher failed: {error}")
        except BaseException:
            self.close()
            raise
        finally:
            reader.close()
            writer.close()

    def publish(self, state: dict, config: dict | None = None) -> None:
        if not self.process.is_alive():
            raise RuntimeError("hand state publisher stopped")
        # Drop a queued snapshot in favour of the latest. A feeder may briefly
        # own the slot; in that case drop this update, never block hardware I/O.
        try:
            self.updates.get_nowait()
        except Empty:
            pass
        try:
            self.updates.put_nowait(copy.deepcopy((state, config)))
        except Full:
            pass

    def close(self) -> None:
        self.stop.set()
        if self.process.pid is not None:
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)
        self.updates.cancel_join_thread()
        self.updates.close()
