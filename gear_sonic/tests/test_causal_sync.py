from __future__ import annotations

import pytest

from gear_sonic.data.causal_sync import CausalSampleBuffer, CausalSynchronizer


def test_buffer_selects_closest_sample_from_past_never_future():
    buffer = CausalSampleBuffer()
    buffer.add("past", 983)
    buffer.add("future", 1016)

    selected = buffer.latest_at_or_before(1000)

    assert selected is not None
    assert selected.value == "past"
    assert selected.timestamp_ns == 983


def test_synchronizer_waits_until_every_stream_advances_past_target():
    synchronizer = CausalSynchronizer()
    synchronizer.observe("camera", "camera-past", 980)
    synchronizer.observe("state", "state-past", 990)
    synchronizer.observe("camera", "camera-future", 1020)

    selection = synchronizer.select(
        1000,
        required_streams=("camera", "state"),
        max_age_ns={"camera": 100, "state": 100},
    )

    assert not selection.ready
    assert selection.waiting == ("state",)
    synchronizer.observe("state", "state-future", 1001)

    selection = synchronizer.select(
        1000,
        required_streams=("camera", "state"),
        max_age_ns={"camera": 100, "state": 100},
    )

    assert selection.ready
    assert selection.samples["camera"].value == "camera-past"
    assert selection.samples["state"].value == "state-past"
    assert all(age >= 0 for age in selection.ages_ms().values())


def test_synchronizer_rejects_a_past_sample_that_is_too_old():
    synchronizer = CausalSynchronizer()
    synchronizer.observe("camera", "old", 800)
    synchronizer.observe("camera", "future", 1001)

    selection = synchronizer.select(
        1000,
        required_streams=("camera",),
        max_age_ns={"camera": 100},
    )

    assert not selection.ready
    assert selection.stale == ("camera",)
    assert selection.samples["camera"].value == "old"


def test_buffer_drops_non_monotonic_arrivals_and_bounds_memory():
    buffer = CausalSampleBuffer(max_samples=3)
    buffer.add("one", 1)
    buffer.add("duplicate", 1)
    buffer.add("three", 3)
    buffer.add("two", 2)
    buffer.add("four", 4)
    buffer.add("five", 5)

    assert buffer.latest_at_or_before(3).value == "three"
    assert buffer.stats() == {
        "depth": 3,
        "capacity": 3,
        "watermark_ns": 5,
        "out_of_order_dropped": 2,
    }


def test_buffer_rejects_invalid_configuration_and_timestamps():
    with pytest.raises(ValueError):
        CausalSampleBuffer(max_samples=1)
    buffer = CausalSampleBuffer()
    with pytest.raises(TypeError):
        buffer.add("bad", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        buffer.add("bad", 0)


def test_trim_keeps_last_past_sample_and_future_history():
    buffer = CausalSampleBuffer()
    for timestamp in (10, 20, 30, 40):
        buffer.add(str(timestamp), timestamp)

    buffer.trim_through(25)

    assert buffer.stats()["depth"] == 3
    assert buffer.latest_at_or_before(25).value == "20"
    assert buffer.latest_at_or_before(35).value == "30"
