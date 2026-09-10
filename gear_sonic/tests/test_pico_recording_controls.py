"""Exercise real manager gestures and wire messages without headset or robot hardware."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.scripts.run_data_exporter import unpack_pose_message


@pytest.fixture
def run_manager(monkeypatch):
    def run(steps, *, calibration_results=None, transition_results=None, teleop_mode="pose", planner_sent=None):
        steps = iter(steps)
        statuses = deque()
        frame = {}
        pub = Mock()
        sub = Mock(poll=lambda _: bool(statuses), recv_json=statuses.popleft)
        context = Mock(socket=Mock(side_effect=[pub, Mock(), sub, Mock(poll=lambda _: False)]))
        reader = Mock(spec=manager.PicoReader, disconnected=False, get_latest=lambda: None)
        reader.reconnect.side_effect = lambda: setattr(reader, "disconnected", False)
        monkeypatch.setattr(manager.zmq, "Context", lambda: context)
        monkeypatch.setattr(manager, "time", SimpleNamespace(
            time=lambda: frame.get("wall_time", 100.0),
            monotonic=lambda: frame.get("now", 10.0), monotonic_ns=lambda: int(frame.get("now", 10.0)*1e9), sleep=lambda _: None,
        ))
        monkeypatch.setattr("gear_sonic.utils.teleop.gesture_trackers.time", manager.time)
        monkeypatch.setattr(manager, "_init_input_source", lambda *_: reader)
        for name in ("ThreePointPose", "PoseStreamer", "PlannerStreamer", "HandIntentStream"):
            monkeypatch.setattr(manager, name, Mock())
        if calibration_results is not None:
            manager.PlannerStreamer.return_value.recalibrate_for_vr3pt.side_effect = calibration_results
        planner = manager.PlannerStreamer.return_value
        planner.run_once.return_value = True
        planner.vr_fault = None
        planner.controller_input_lost = False
        planner.feedback_reader.upper_body_position_target = np.zeros(17)
        planner.feedback_reader.upper_body_planner_target = np.full(17, 0.1)
        if transition_results is not None:
            planner.poll_fresh_feedback.side_effect = transition_results
        if planner_sent is not None:
            planner.run_once.side_effect = planner_sent
        monkeypatch.setattr(manager, "get_controller_inputs", lambda _: (frame.get("pause", False), 0, 0, 0, 0))
        monkeypatch.setattr(manager, "get_axis_clicks", lambda _: (frame.get("stick", False), False))

        def buttons(_):
            try:
                frame.clear()
                frame.update(next(steps))
            except StopIteration:
                raise KeyboardInterrupt from None
            if "recording" in frame:
                statuses.append({"recording": frame["recording"], "timestamp": frame.get("timestamp", 101.0)})
            if frame.get("disconnect"):
                reader.disconnected = True
            return tuple(key in frame.get("buttons", "") for key in "abxy")

        monkeypatch.setattr(manager, "get_abxy_buttons", buttons)
        manager.run_pico_manager(input_source="isaac", teleop_mode=teleop_mode, legacy_vr_controls=True)
        payloads = [call.args[0] for call in pub.send.call_args_list]
        states = [unpack_pose_message(raw, topic="manager_state")
                  for raw in payloads if raw.startswith(b"manager_state")]
        holds = [call.kwargs["hold"] for call in manager.HandIntentStream.return_value.publish.call_args_list]
        if teleop_mode == "pose":
            assert holds == [state["stream_mode"].item() in {0, 4} for state in states]
        commands = [raw for raw in payloads if raw.startswith(b"command")]
        return states, commands
    return run


def enter_mode(mode):
    steps = [{"buttons": "abxy"}, {}]  # OFF -> PLANNER
    if mode in (1, 3):
        steps += [{"buttons": "ax"}, {}] * 2  # PLANNER -> POSE
    if mode == 3:
        steps += [{"buttons": "by"}, {}]  # POSE -> FROZEN_UPPER_BODY
    elif mode == 5:
        steps += [{"stick": True}, {}]  # PLANNER -> VR_3PT
    return steps


@pytest.mark.parametrize("mode", [1, 2, 5])
@pytest.mark.parametrize("gesture", ["ax", "xb", "ya"])
def test_recording_gestures_emit_once_on_release_without_mode_switch(run_manager, mode, gesture):
    steps = enter_mode(mode)
    states, commands = run_manager(steps + [
        {"buttons": gesture, "recording": True}, {"buttons": gesture},
        {"buttons": gesture[0]}, {}, {},
    ])
    tail = states[len(steps):]
    assert [s["stream_mode"].item() for s in tail] == [mode] * 5
    assert [bool(s["toggle_data_collection"].item()) for s in tail] == [
        False, False, False, gesture != "ya", False,
    ]
    assert [bool(s["toggle_data_abort"].item()) for s in tail] == [False, False, False, gesture == "ya", False]
    assert len(commands) == (1 if mode == 2 else 2)


@pytest.mark.parametrize("mode,destination", [(1, 2), (2, 1), (5, 1)])
def test_idle_ax_switches_mode_only_after_two_completed_gestures(run_manager, mode, destination):
    states, _ = run_manager(enter_mode(mode) + [{"buttons": "ax", "recording": False}, {}] * 2)
    assert states[-3]["stream_mode"].item() == mode
    assert states[-1]["stream_mode"].item() == destination
    assert not any(s["toggle_data_collection"].item() or s["toggle_data_abort"].item() for s in states)


@pytest.mark.parametrize("elapsed,destination", [(0.0, 1), (2.0, 1), (2.001, 2)])
def test_ax_confirmation_uses_monotonic_release_times(run_manager, elapsed, destination):
    states, _ = run_manager(enter_mode(2) + [
        {"buttons": "ax"}, {"now": 10.0, "wall_time": 100.0},
        {"buttons": "ax"}, {"now": 10.0 + elapsed, "wall_time": 1.0},
    ])
    assert states[-1]["stream_mode"].item() == destination


def test_expired_ax_becomes_first_gesture_of_new_pair(run_manager):
    states, _ = run_manager(enter_mode(2) + [
        {"buttons": "ax"}, {"now": 10.0},
        {"buttons": "ax"}, {"now": 12.1},
        {"buttons": "ax"}, {"now": 13.0},
    ])
    assert states[-3]["stream_mode"].item() == 2
    assert states[-1]["stream_mode"].item() == 1


def test_holding_or_partially_releasing_ax_does_not_confirm_twice(run_manager):
    states, commands = run_manager(enter_mode(2) + [
        {"buttons": "ax"}, {"buttons": "ax"}, {"buttons": "a"},
        {"buttons": "ax"}, {}, {},
    ])
    assert all(s["stream_mode"].item() == 2 for s in states)
    assert len(commands) == 1


def test_confirmed_ax_pair_is_consumed(run_manager):
    states, _ = run_manager(enter_mode(2) + [{"buttons": "ax"}, {}] * 4)
    assert [s["stream_mode"].item() for s in states[-8:]] == [2, 2, 2, 1, 1, 1, 1, 2]


@pytest.mark.parametrize("kwargs", [
    {"upper_body_mask": [True]*17},
    {"upper_body_position": [0.0]*16},
    {"upper_body_velocity": [0.0]*16},
    {"upper_body_position": [0.0]*17, "upper_body_mask": [True]*16},
])
def test_rejects_incomplete_arm_override_messages(kwargs):
    with pytest.raises(ValueError):
        manager.build_planner_message(0, [0, 0, 0], [0, 1, 0], **kwargs)
