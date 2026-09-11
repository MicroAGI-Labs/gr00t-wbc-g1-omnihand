"""Exercise every press/release order, including rolling and cancelled chords."""

from itertools import combinations, permutations

import pytest

from gear_sonic.scripts.pico_manager_thread_server import FaceChordTracker


def update(tracker, down):
    return tracker.update(*(key in down for key in "abxy"))


@pytest.mark.parametrize("singles", [False, True])
@pytest.mark.parametrize("buttons,expected", [("xb", "xb"), ("yb", "by"), ("ya", "ya")])
def test_recording_chords_emit_once_only_after_full_release(singles, buttons, expected):
    for press_order in permutations(buttons):
        for release_order in permutations(buttons):
            tracker = FaceChordTracker(singles=singles)
            down = set()
            assert update(tracker, down) is None
            for button in press_order:
                down.add(button)
                assert update(tracker, down) is None
            for _ in range(4):
                assert update(tracker, down) is None
            for button in release_order:
                down.remove(button)
                assert update(tracker, down) == (None if down else expected)
            assert update(tracker, down) is None


@pytest.mark.parametrize("singles", [False, True])
@pytest.mark.parametrize("buttons", ["".join(keys) for n in (3, 4) for keys in combinations("abxy", n)])
def test_three_and_four_button_sequences_never_emit_recording_or_pose_subsets(singles, buttons):
    for press_order in permutations(buttons):
        for release_order in permutations(buttons):
            tracker = FaceChordTracker(singles=singles)
            down = set()
            update(tracker, down)
            for button in press_order:
                down.add(button)
                assert update(tracker, down) is None
            for button in release_order:
                down.remove(button)
                assert update(tracker, down) is None
            # A cancelled gesture does not poison the next distinct gesture.
            assert update(tracker, "yb") is None
            assert update(tracker, ()) == "by"


@pytest.mark.parametrize("singles", [False, True])
@pytest.mark.parametrize("sequence", [("y", "ya", "y", "yb", "b", ""),
                                     ("b", "yb", "b", "xb", "x", "")])
def test_rolling_between_two_chords_without_full_release_cancels_both(singles, sequence):
    tracker = FaceChordTracker(singles=singles)
    update(tracker, ())
    for down in sequence:
        assert update(tracker, down) is None
