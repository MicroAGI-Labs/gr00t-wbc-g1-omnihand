from __future__ import annotations

from gear_sonic.data.causal_sync import CausalSampleBuffer, CausalSynchronizer


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


def test_buffer_bounds_memory_drops_non_monotonic_samples_and_trims_history():
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
    buffer.trim_through(4)
    assert buffer.stats()["depth"] == 2
    assert buffer.latest_at_or_before(4).value == "four"
