"""Bounded recording histories indexed by producer time, not arrival time."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

from gear_sonic.data.clock_sync import ClockClient, ClockUnavailable


@dataclass(frozen=True)
class RecordingInputs:
    proprio: dict | None
    image: dict | None
    hand: dict | None
    sonic: dict | None
    planner: dict | None
    manager: dict | None
    mode: int
    proprio_received_ns: int = -1
    image_received_ns: int = -1
    hand_received_ns: int = -1
    target_ns: int | None = None
    samples: dict[str, dict] | None = None

    def age(self, message: dict | None, received_ns: int, now: float) -> float:
        if self.target_ns is not None and message is not None:
            # Use the oldest plausible sample time for conservative age admission.
            return (self.target_ns - message["_sync_time_ns"] + message["_sync_uncertainty_ns"]) / 1e9
        return now - received_ns / 1e9


@dataclass(frozen=True)
class Selection:
    target_ns: int
    samples: dict[str, dict]
    problems: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.problems


class SenderSynchronizer:
    """Independent source histories, mapped into the recorder monotonic clock.

    Local producers must run on the recorder host. A remote hand producer must
    expose a clock exchange endpoint and include the Linux boot ID in its state.
    Source timestamps are required: this mode never falls back to receipt time.
    """

    def __init__(self, *, camera_names: tuple[str, ...], frequency: int = 50,
                 delay_s: float = 0.1, wait_s: float = 0.25,
                 hand_clock: ClockClient | None = None, capacity: int = 32,
                 allow_stale_hand: bool = False):
        if not all(math.isfinite(v) and v > 0 for v in (frequency, delay_s, wait_s)):
            raise ValueError("frequency, synchronization delay and wait must be positive")
        if capacity < 2:
            raise ValueError("history capacity must be at least two")
        self.camera_names = camera_names
        self.period_ns = round(1e9 / frequency)
        self.delay_ns = round(delay_s * 1e9)
        self.wait_ns = round(wait_s * 1e9)
        self.capacity = capacity
        self.hand_clock = hand_clock
        self.allow_stale_hand = allow_stale_hand
        self.histories: dict[str, deque[dict]] = {}
        self.input_errors: dict[str, str] = {}
        self.dropped = 0
        self.late = 0
        self.gaps = 0
        self.errors: list[str] = []
        self.next_target_ns: int | None = None
        self.stop_target_ns: int | None = None
        self.start_ns: int | None = None
        self._committed_ns = 0
        self._hand_identity: str | None = None

    def observe(self, stream: str, value: dict, source_ns: int, received_ns: int) -> None:
        try:
            if type(source_ns) is not int or source_ns <= 0:
                raise ValueError("missing positive producer timestamp")
            offset = uncertainty = 0
            identity = "local"
            if stream == "hand" and self.hand_clock is not None:
                estimate = self.hand_clock.estimate(received_ns)
                identity = estimate.clock_id
                if value.get("clock_id") != identity:
                    raise ClockUnavailable("hand timestamp clock identity differs from clock service")
                if self._hand_identity not in (None, identity):
                    self.histories.pop("hand", None)
                    self._episode_error("hand clock restarted")
                self._hand_identity = identity
                offset, uncertainty = estimate.offset_ns, estimate.uncertainty_ns
                # Also budget drift between measurement and the offset exchange.
                uncertainty += abs(source_ns - offset - estimate.measured_ns) // 10_000
            timestamp = source_ns - offset
            if timestamp <= 0:
                raise ValueError("producer timestamp maps outside the recorder clock")
            if timestamp - uncertainty > received_ns:
                raise ValueError("producer timestamp is in the future; check clock domain")
            history = self.histories.setdefault(stream, deque(maxlen=self.capacity))
            if history and source_ns <= history[-1]["_sync_source_ns"]:
                if source_ns < history[-1]["_sync_source_ns"]:
                    self._episode_error(f"{stream}: out-of-order producer timestamp")
                return  # Held snapshots/cached cameras do not advance watermarks.
            if timestamp <= self._committed_ns:
                self.late += 1
                if stream != "hand" or not self.allow_stale_hand:
                    self._episode_error(f"{stream}: sample arrived after its target was committed")
                    return
                # Keep late hand feedback for subsequent rows with its actual
                # timestamp. Already-written rows are never rewritten.
            if history and timestamp <= history[-1]["_sync_time_ns"]:
                raise ClockUnavailable("clock mapping moved backwards")
            if len(history) == self.capacity:
                self.dropped += 1
            history.append({**value, "_sync_time_ns": timestamp, "_sync_source_ns": source_ns,
                            "_sync_received_ns": received_ns, "_sync_offset_ns": offset,
                            "_sync_uncertainty_ns": uncertainty, "_sync_clock_id": identity})
            self.input_errors.pop(stream, None)
        except (ValueError, ClockUnavailable) as exc:
            self.input_errors[stream] = str(exc)

    def _episode_error(self, error: str) -> None:
        if self.next_target_ns is not None and error not in self.errors and len(self.errors) < 100:
            self.errors.append(error)

    def start(self, now_ns: int) -> None:
        self.start_ns = self.next_target_ns = now_ns
        self.stop_target_ns = None
        self._committed_ns = 0
        self.errors = []
        self.gaps = 0

    def stop(self, now_ns: int) -> None:
        self.stop_target_ns = now_ns

    def reset(self) -> None:
        self.next_target_ns = self.stop_target_ns = self.start_ns = None
        self._committed_ns = 0

    def select(self, target_ns: int, *, hand: bool, max_ages: dict[str, float],
               allow_stale_hand: bool | None = None) -> Selection:
        if allow_stale_hand is None:
            allow_stale_hand = self.allow_stale_hand
        required = ["proprio", "manager", *(f"camera.{n}" for n in self.camera_names)]
        if hand:
            required.append("hand")
        samples: dict[str, dict] = {}
        problems = []

        def select_stream(stream: str) -> None:
            history = self.histories.get(stream, ())
            retain_hand = stream == "hand" and allow_stale_hand
            if stream in self.input_errors:
                problems.append(f"{stream}: {self.input_errors[stream]}")
                return
            if not history or (not retain_hand and
                    history[-1]["_sync_time_ns"] - history[-1]["_sync_uncertainty_ns"] <= target_ns):
                problems.append(f"{stream}: waiting for producer to advance")
                return
            sample = next((s for s in reversed(history)
                           if s["_sync_time_ns"] + s["_sync_uncertainty_ns"] <= target_ns), None)
            if sample is None:
                problems.append(f"{stream}: no past sample")
                return
            age = (target_ns - sample["_sync_time_ns"] + sample["_sync_uncertainty_ns"]) / 1e9
            limit = max_ages["camera" if stream.startswith("camera.") else stream]
            if age > limit and not retain_hand:
                problems.append(f"{stream}: past sample is stale ({age:.3f}s)")
                return
            samples[stream] = sample

        for stream in required:
            select_stream(stream)
        if "manager" in samples:
            mode = samples["manager"]["stream_mode"]
            if mode in (1, 4):
                select_stream("sonic")
            elif mode in (5, 6):
                select_stream("planner")
        return Selection(target_ns, samples, tuple(problems))

    def advance(self, target_ns: int) -> None:
        self._committed_ns = target_ns
        self.next_target_ns = target_ns + self.period_ns
        self.trim(target_ns)

    def skip(self, selection: Selection, now_ns: int) -> None:
        self._episode_error(f"target {selection.target_ns}: {'; '.join(selection.problems)}")
        earliest = max(selection.target_ns + self.period_ns, now_ns - self.delay_ns)
        start = self.start_ns if self.start_ns is not None else selection.target_ns
        next_target = start + ((earliest - start + self.period_ns - 1) // self.period_ns) * self.period_ns
        self.gaps += (next_target - selection.target_ns) // self.period_ns
        self.next_target_ns = next_target
        self.trim(selection.target_ns)

    def trim(self, target_ns: int) -> None:
        for history in self.histories.values():
            while len(history) > 1 and history[1]["_sync_time_ns"] + history[1]["_sync_uncertainty_ns"] <= target_ns:
                history.popleft()

    def status(self) -> dict[str, Any]:
        return {"mode": "sender", "delay_ms": self.delay_ns / 1e6,
                "next_target_ns": self.next_target_ns, "skipped_targets": self.gaps,
                "input_errors": dict(self.input_errors), "errors": self.errors[-5:],
                "hand_clock_id": self._hand_identity,
                "history_depths": {k: len(v) for k, v in self.histories.items()},
                "history_overflow": self.dropped, "late_samples": self.late}

    def close(self) -> None:
        if self.hand_clock is not None:
            self.hand_clock.close()


def selection_inputs(selection: Selection) -> RecordingInputs:
    samples = selection.samples
    cameras = [sample for name, sample in samples.items() if name.startswith("camera.")]
    newest = max(cameras, key=lambda s: s["_sync_time_ns"])
    image = {**newest, "images": {}, "depths": {}, "timestamps": {},
             "capture_monotonic_ns": {}, "camera_received_monotonic_ns": {}}
    for sample in cameras:
        for field in ("images", "depths", "timestamps", "capture_monotonic_ns"):
            image[field].update(sample.get(field, {}))
        for name in sample["timestamps"]:
            image["camera_received_monotonic_ns"][name] = sample["_sync_received_ns"]
    return RecordingInputs(
        samples["proprio"], image, samples.get("hand"), samples.get("sonic"),
        samples.get("planner"), samples["manager"], int(samples["manager"]["stream_mode"]),
        samples["proprio"]["_sync_received_ns"], newest["_sync_received_ns"],
        samples.get("hand", {}).get("_sync_received_ns", -1), selection.target_ns, samples,
    )


def synchronization_features(camera_names: tuple[str, ...]) -> dict:
    features = {"capture.sync_target_monotonic_ns": {"dtype": "int64", "shape": (1,), "names": ["ns"]}}
    for stream in ("proprio", "manager", "sonic", "planner", "hand", *(f"camera.{n}" for n in camera_names)):
        for field in ("source_ns", "time_ns", "received_ns", "offset_ns", "uncertainty_ns"):
            features[f"capture.sync.{stream}.{field}"] = {"dtype": "int64", "shape": (1,), "names": ["ns"]}
    return features
