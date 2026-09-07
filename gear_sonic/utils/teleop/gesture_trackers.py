"""Stateful helpers for recognizing controller gestures."""

import time


class DoublePressTracker:
    """Confirm an event only when it occurs twice within a time window."""

    def __init__(self, window_seconds: float = 2.0) -> None:
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self._first_press_at: float | None = None

    def reset(self) -> None:
        self._first_press_at = None

    def register(self, now: float | None = None) -> bool:
        """Register one completed press and return whether it confirms the pair.

        A press after the current window expires becomes the first press of a
        new pair. ``time.monotonic`` keeps the gesture independent of wall-clock
        adjustments.
        """
        pressed_at = time.monotonic() if now is None else now
        if (
            self._first_press_at is None
            or pressed_at - self._first_press_at > self.window_seconds
        ):
            self._first_press_at = pressed_at
            return False

        self.reset()
        return True


def recording_face_action(
    face_command: str | None,
    *,
    recorder_is_recording: bool,
    recording_mode_ready: bool,
) -> str | None:
    """Resolve face chords without letting mode controls discard a take.

    A+X is a mode gesture only while idle. During an active recording it is
    consumed as save, with the same outcome as X+B. Y+A is the sole explicit
    discard gesture.
    """
    if recorder_is_recording:
        if face_command in {"ax", "xb"}:
            return "save"
        if face_command == "ya":
            return "discard"
        return None

    if face_command == "xb" and recording_mode_ready:
        return "start"
    return None
