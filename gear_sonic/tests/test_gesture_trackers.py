import pytest

from gear_sonic.utils.teleop.gesture_trackers import (
    DoublePressTracker,
    recording_face_action,
)


def test_double_press_confirms_second_press_within_window():
    tracker = DoublePressTracker(window_seconds=2.0)

    assert tracker.register(now=10.0) is False
    assert tracker.register(now=12.0) is True


def test_press_after_window_starts_a_new_pair():
    tracker = DoublePressTracker(window_seconds=2.0)

    assert tracker.register(now=10.0) is False
    assert tracker.register(now=12.01) is False
    assert tracker.register(now=13.0) is True


def test_confirmed_pair_does_not_leak_into_next_pair():
    tracker = DoublePressTracker(window_seconds=2.0)

    assert tracker.register(now=10.0) is False
    assert tracker.register(now=11.0) is True
    assert tracker.register(now=11.5) is False


def test_reset_discards_pending_press():
    tracker = DoublePressTracker(window_seconds=2.0)

    assert tracker.register(now=10.0) is False
    tracker.reset()
    assert tracker.register(now=11.0) is False


def test_window_must_be_positive():
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        DoublePressTracker(window_seconds=0.0)


@pytest.mark.parametrize("face_command", ["ax", "xb"])
def test_active_recording_can_be_saved_with_ax_or_xb(face_command):
    assert (
        recording_face_action(
            face_command,
            recorder_is_recording=True,
            recording_mode_ready=True,
        )
        == "save"
    )


def test_only_ya_discards_an_active_recording():
    assert (
        recording_face_action(
            "ya",
            recorder_is_recording=True,
            recording_mode_ready=True,
        )
        == "discard"
    )
    for face_command in (None, "by", "ab", "xy"):
        assert (
            recording_face_action(
                face_command,
                recorder_is_recording=True,
                recording_mode_ready=True,
            )
            is None
        )


def test_xb_starts_only_when_recording_mode_is_ready():
    assert recording_face_action(
        "xb", recorder_is_recording=False, recording_mode_ready=True
    ) == "start"
    assert recording_face_action(
        "xb", recorder_is_recording=False, recording_mode_ready=False
    ) is None
