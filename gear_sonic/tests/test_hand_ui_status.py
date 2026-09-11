"""A running status publisher must not disguise stopped or stale motor feedback."""

from copy import deepcopy

import pytest

from gear_sonic.scripts import run_camera_web_viewer as viewer


@pytest.mark.parametrize(
    "mode,receipt_age,state_age,side_valid,expected",
    [
        ("hold", .01, .04, True, "connected"),
        ("tracking", .01, None, True, "connected"),
        ("fault", .01, 20, False, "fault"),
        ("disconnected", .01, 20, False, "reconnecting"),
        ("hold", .01, 2, True, "stale"),
        ("hold", .6, .6, True, "stale"),
        ("hold", .01, -1, True, "stale"),
        ("hold", .01, float("nan"), True, "stale"),
        ("hold", .01, True, True, "stale"),
        ("hold", 2, .01, True, "offline"),
        ("fault", 2, 20, False, "offline"),
        ("hold", .01, .01, False, "fault"),
    ],
)
def test_hardware_connection_is_distinct_from_publisher(
    monkeypatch, mode, receipt_age, state_age, side_valid, expected
):
    monkeypatch.setattr(viewer.time, "monotonic", lambda: 100.0)
    hub = viewer.HandControlHub(viewer.CameraWebViewerConfig(enable_hand_controls=True))
    hub._status_received_at = 100 - receipt_age
    hub._status = {
        "mode": mode,
        "state_age_s": state_age,
        "sides": {
            "left": {"connected": side_valid, "valid": side_valid, "error": None},
            "right": {"connected": True, "valid": True, "error": None},
        },
    }
    original = deepcopy(hub._status)
    status = hub.status()
    assert status["connection_state"] == expected
    assert status["connected"] == (expected == "connected")
    assert status["publisher_connected"] == (receipt_age < 1)
    assert ("retrying automatically" in status["status_message"]) == (expected == "reconnecting")
    assert "connected" not in hub._status
    assert hub._status["sides"] == original["sides"]


def test_absent_and_disabled_services_do_not_report_connected():
    assert viewer.HandControlHub(viewer.CameraWebViewerConfig()).status() == {
        "enabled": False, "connected": False
    }
    status = viewer.HandControlHub(viewer.CameraWebViewerConfig(enable_hand_controls=True)).status()
    assert status["connection_state"] == "offline"
    assert status["connected"] is False


@pytest.mark.parametrize("sides", [{}, {"right": {"connected": True, "valid": True, "error": "motor fault"}}])
def test_missing_or_faulted_motor_feedback_is_not_connected(monkeypatch, sides):
    monkeypatch.setattr(viewer.time, "monotonic", lambda: 100.0)
    hub = viewer.HandControlHub(viewer.CameraWebViewerConfig(enable_hand_controls=True))
    hub._status_received_at = 100
    hub._status = {"mode": "hold", "state_age_s": .01, "sides": sides}
    status = hub.status()
    assert status["connection_state"] == "fault"
    assert status["connected"] is False
