"""Causal, collector-clock sample selection for fixed-rate recording."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimedSample:
    timestamp_ns: int
    value: Any


@dataclass(frozen=True)
class CausalSelection:
    target_ns: int
    samples: dict[str, TimedSample]
    waiting: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not (self.waiting or self.missing or self.stale)

    def ages_ms(self) -> dict[str, float]:
        return {name: (self.target_ns - sample.timestamp_ns) / 1e6 for name, sample in self.samples.items()}


class CausalSampleBuffer:
    """Short ordered history supporting latest-at-or-before lookup."""

    def __init__(self, max_samples: int = 32):
        if max_samples < 2:
            raise ValueError("causal sample buffer requires at least two samples")
        self.max_samples = max_samples
        self._samples: deque[TimedSample] = deque(maxlen=max_samples)
        self._out_of_order_dropped = 0

    @property
    def watermark_ns(self) -> int | None:
        return self._samples[-1].timestamp_ns if self._samples else None

    def add(self, value: Any, timestamp_ns: int) -> None:
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
            raise TypeError("sample timestamp must be an integer number of nanoseconds")
        if timestamp_ns <= 0:
            raise ValueError("sample timestamp must be positive")
        if self._samples and timestamp_ns <= self._samples[-1].timestamp_ns:
            self._out_of_order_dropped += 1
            return
        self._samples.append(TimedSample(timestamp_ns=timestamp_ns, value=value))

    def latest_at_or_before(self, target_ns: int) -> TimedSample | None:
        for sample in reversed(self._samples):
            if sample.timestamp_ns <= target_ns:
                return sample
        return None

    def trim_through(self, target_ns: int) -> None:
        """Retain only the last selected past sample and all future samples."""
        while len(self._samples) > 1 and self._samples[1].timestamp_ns <= target_ns:
            self._samples.popleft()

    def stats(self) -> dict[str, int | None]:
        return {
            "depth": len(self._samples),
            "capacity": self.max_samples,
            "watermark_ns": self.watermark_ns,
            "out_of_order_dropped": self._out_of_order_dropped,
        }


class CausalSynchronizer:
    """Select only samples in the causal past after every stream advances."""

    def __init__(self, max_samples_per_stream: int = 32):
        if max_samples_per_stream < 2:
            raise ValueError("max_samples_per_stream must be at least two")
        self.max_samples_per_stream = max_samples_per_stream
        self._streams: dict[str, CausalSampleBuffer] = {}

    def observe(self, stream: str, value: Any, timestamp_ns: int) -> None:
        self._streams.setdefault(
            stream,
            CausalSampleBuffer(self.max_samples_per_stream),
        ).add(value, timestamp_ns)

    def select(
        self,
        target_ns: int,
        *,
        required_streams: tuple[str, ...],
        max_age_ns: dict[str, int],
    ) -> CausalSelection:
        waiting = []
        missing = []
        stale = []
        selected = {}
        for stream in required_streams:
            buffer = self._streams.get(stream)
            watermark = None if buffer is None else buffer.watermark_ns
            if watermark is None or watermark <= target_ns:
                waiting.append(stream)
                continue
            sample = buffer.latest_at_or_before(target_ns)
            if sample is None:
                missing.append(stream)
                continue
            selected[stream] = sample
            maximum = max_age_ns.get(stream)
            if maximum is not None and target_ns - sample.timestamp_ns > maximum:
                stale.append(stream)
        return CausalSelection(
            target_ns=target_ns,
            samples=selected,
            waiting=tuple(waiting),
            missing=tuple(missing),
            stale=tuple(stale),
        )

    def trim_through(self, target_ns: int) -> None:
        for buffer in self._streams.values():
            buffer.trim_through(target_ns)

    def status(self) -> dict[str, dict[str, int | None]]:
        return {name: buffer.stats() for name, buffer in self._streams.items()}
