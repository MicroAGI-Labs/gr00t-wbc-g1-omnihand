"""Exercise source-age semantics and the actual spawned status publisher."""

import copy
import time

import pytest
import zmq

from gear_sonic.end_effectors.protocol import HAND_STATE_SCHEMA, HAND_STATE_TOPIC, decode_state, feedback_age_s
from gear_sonic.end_effectors.publisher import HandStatePublisher, age_state


def snapshot():
    return {
        "schema": HAND_STATE_SCHEMA, "sequence": 42, "monotonic_ns": 1_000_000_000,
        "input_stale": False, "input_age_s": 0.01,
        "sides": {
            "left": {"valid": True, "connected": True, "input_age_s": 0.01, "input_stale": False},
            "right": {"valid": True, "connected": True, "input_age_s": 0.49, "input_stale": False},
        },
    }


def test_republishing_ages_feedback_and_each_side_without_changing_source_identity():
    source = snapshot()
    saved = copy.deepcopy(source)
    state = age_state(source, 1_100_000_000, 7, 50, 0.5)
    assert state["sequence"] == 42 and state["monotonic_ns"] == 1_000_000_000
    assert state["publish_sequence"] == 7 and state["published_monotonic_ns"] == 1_100_000_000
    assert feedback_age_s(state) == pytest.approx(0.1)
    assert not state["sides"]["left"]["input_stale"]
    assert state["sides"]["right"]["input_stale"] and state["input_stale"]
    assert source == saved


@pytest.mark.parametrize("age", [-1, float("nan"), float("inf"), None, True, "0"])
def test_invalid_source_or_feedback_ages_cannot_be_admitted(age):
    assert feedback_age_s({"state_age_s": age}) == float("inf")
    assert feedback_age_s({"sides": {"left": {"feedback_age_s": age}}}) == float("inf")


def test_spawned_publisher_repeats_samples_with_new_publication_timestamps_and_closes():
    with zmq.Context() as context, context.socket(zmq.SUB) as subscriber:
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.SUBSCRIBE, HAND_STATE_TOPIC)
        # Bind the subscriber on a random port; the process PUB connects through
        # an explicitly reserved endpoint after this temporary bind is released.
        port = subscriber.bind_to_random_port("tcp://127.0.0.1")
        endpoint = f"tcp://127.0.0.1:{port}"
        subscriber.unbind(endpoint)
        subscriber.connect(endpoint)
        publisher = HandStatePublisher(endpoint, 50, 0.5)
        try:
            source = snapshot()
            source["monotonic_ns"] = time.monotonic_ns()
            publisher.publish(source)
            source["sequence"] = -1  # The mailbox owns its snapshot.
            messages = []
            deadline = time.monotonic() + 3
            while len(messages) < 5 and time.monotonic() < deadline:
                if subscriber.poll(100):
                    messages.append(decode_state(subscriber.recv()))
            assert len(messages) == 5
            assert {message["sequence"] for message in messages} == {42}
            assert messages[-1]["publish_sequence"] > messages[0]["publish_sequence"]
            assert messages[-1]["state_age_s"] > messages[0]["state_age_s"]
            # A slow consumer cannot block the control process or grow a queue.
            started = time.monotonic()
            for _ in range(500):
                publisher.publish(source)
            assert time.monotonic() - started < 2
        finally:
            publisher.close()
        assert not publisher.process.is_alive()


def test_publisher_bind_failure_is_reported_without_leaving_a_child():
    with zmq.Context() as context, context.socket(zmq.PUB) as occupied:
        occupied.setsockopt(zmq.LINGER, 0)
        port = occupied.bind_to_random_port("tcp://127.0.0.1")
        with pytest.raises(RuntimeError, match="publisher failed"):
            HandStatePublisher(f"tcp://127.0.0.1:{port}", 50, 0.5)
