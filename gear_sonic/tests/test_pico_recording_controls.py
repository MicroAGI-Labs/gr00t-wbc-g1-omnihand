"""Exercise real manager gestures and wire messages without headset or robot hardware."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import numpy as np
import pytest

from gear_sonic.end_effectors.protocol import decode_intent
from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.scripts.run_data_exporter import unpack_pose_message


@pytest.fixture
def run_manager(monkeypatch):
    def run(steps, *, calibration_results=None):
        steps = iter(steps)
        statuses = deque()
        frame = {}
        pub, hand_pub = Mock(), Mock()
        sub = Mock(poll=lambda _: bool(statuses), recv_json=statuses.popleft)
        context = Mock(socket=Mock(side_effect=[pub, hand_pub, sub]))
        reader = Mock(spec=manager.PicoReader, disconnected=False, get_latest=lambda: None)
        reader.reconnect.side_effect = lambda: setattr(reader, "disconnected", False)
        monkeypatch.setattr(manager.zmq, "Context", lambda: context)
        monkeypatch.setattr(manager, "time", SimpleNamespace(
            time=lambda: frame.get("wall_time", 100.0),
            monotonic=lambda: frame.get("now", 10.0), sleep=lambda _: None,
        ))
        monkeypatch.setattr(manager, "_init_input_source", lambda *_: reader)
        for name in ("ThreePointPose", "PoseStreamer", "PlannerStreamer", "HandIntentStream"):
            monkeypatch.setattr(manager, name, Mock())
        if calibration_results is not None:
            manager.PlannerStreamer.return_value.recalibrate_for_vr3pt.side_effect = calibration_results
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
        manager.run_pico_manager(input_source="isaac")
        payloads = [call.args[0] for call in pub.send.call_args_list]
        states = [unpack_pose_message(raw, topic="manager_state")
                  for raw in payloads if raw.startswith(b"manager_state")]
        assert not any(raw.startswith(b"hand_intent") for raw in payloads)
        pub.bind.assert_called_once_with("tcp://*:5556")
        hand_pub.bind.assert_called_once_with(f"tcp://*:{manager.HAND_INTENT_PORT}")
        hand_calls = manager.HandIntentStream.return_value.publish.call_args_list
        assert len(hand_calls) == len(states)
        for call, state in zip(hand_calls, states):
            assert call.args == (hand_pub, reader)
            assert call.kwargs["hold"] == (state["stream_mode"].item() in {0, 4})
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


@pytest.mark.parametrize("buttons", ["ax", "xa"])
def test_policy_stop_does_not_leak_save_or_discard_on_partial_release(run_manager, buttons):
    states, commands = run_manager(enter_mode(1) + [
        {"buttons": buttons, "recording": True}, {"buttons": "abxy"},
        {"buttons": buttons}, {"buttons": buttons[0]}, {},
    ])
    assert states[-1]["stream_mode"].item() == 0
    assert commands[-1] == manager.build_command_message(start=False, stop=True, planner=True)
    assert not any(s["toggle_data_collection"].item() or s["toggle_data_abort"].item() for s in states)


def test_pause_and_policy_off_hold_hands_until_resume(run_manager):
    states, _ = run_manager(enter_mode(1) + [
        {"pause": True}, {"pause": True}, {}, {"buttons": "abxy"}, {},
    ])
    assert [s["stream_mode"].item() for s in states[-5:]] == [4, 4, 1, 0, 0]


def test_hand_intent_keeps_stale_headset_samples_invalid_during_hold(monkeypatch):
    socket = Mock()
    reader = Mock(get_timestamp_ns=Mock(side_effect=[10, 10, 11]))
    monkeypatch.setattr(manager, "get_controller_inputs", lambda _: (False, 0, 0.8, 0.9, 0))
    stream = manager.HandIntentStream()
    for hold in (False, True, False):
        stream.publish(socket, reader, hold=hold)
    first, held, resumed = [decode_intent(call.args[0]) for call in socket.send.call_args_list]
    assert first["left"]["closed"] and first["right"]["closed"]
    assert held["hold"] and not held["left"]["valid"] and not held["right"]["valid"]
    assert not resumed["hold"] and resumed["left"]["valid"] and resumed["right"]["valid"]
    assert first["sequence"] < held["sequence"] < resumed["sequence"]


def test_standalone_pose_streamer_publishes_hands_on_its_own_port(monkeypatch):
    body, hands = Mock(), Mock()
    context = Mock(socket=Mock(side_effect=[body, hands]))
    reader = Mock(get_timestamp_ns=lambda: 10)
    monkeypatch.setattr(manager.zmq, "Context", lambda: context)
    monkeypatch.setattr(manager, "_init_input_source", lambda *_: reader)
    monkeypatch.setattr(manager, "ThreePointPose", Mock())
    streamer = Mock(run_once=Mock(side_effect=[None, KeyboardInterrupt]))
    monkeypatch.setattr(manager, "PoseStreamer", Mock(return_value=streamer))
    monkeypatch.setattr(manager, "get_controller_inputs", lambda _: (False, 0, 0, 0, 0))
    monkeypatch.setattr(manager.time, "sleep", lambda _: None)
    manager.run_pico(input_source="isaac", hand_intent_port=15669)
    hands.bind.assert_called_once_with("tcp://*:15669")
    body.bind.assert_called_once_with("tcp://*:5556")
    assert not decode_intent(hands.send.call_args.args[0])["hold"]
    assert not any(call.args[0].startswith(b"hand_intent") for call in body.send.call_args_list)
    hands.close.assert_called_once()
    context.term.assert_called_once()


@pytest.mark.parametrize("entrypoint", [manager.run_pico, manager.run_pico_manager])
@pytest.mark.parametrize("port", [5556, 0, 65536])
def test_invalid_hand_ports_fail_before_starting_the_headset(monkeypatch, entrypoint, port):
    monkeypatch.setattr(manager, "_init_input_source", lambda *_: pytest.fail("headset started"))
    with pytest.raises(ValueError, match="hand intent port"):
        entrypoint(hand_intent_port=port)


@pytest.mark.parametrize("pico_manager", [True, False])
def test_launcher_wires_matching_custom_hand_ports_without_launching_processes(monkeypatch, pico_manager):
    from gear_sonic.scripts import launch_data_collection as launcher

    sent = {}
    monkeypatch.setattr(launcher, "_check_prerequisites", lambda _: None)
    monkeypatch.setattr(launcher, "_get_local_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(launcher, "_kill_existing_session", lambda: None)
    monkeypatch.setattr(launcher, "_switch_camera_source", lambda _: None)
    monkeypatch.setattr(launcher, "_create_tmux_session", lambda _: None)
    monkeypatch.setattr(launcher, "_send_to_pane", lambda pane, command, **_: sent.update({pane: command}))
    monkeypatch.setattr(launcher.subprocess, "run", Mock())
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    config = launcher.DataCollectionLaunchConfig(
        sim=True, hand_backend="omnihand", hand_intent_port=15669, pico_manager=pico_manager,
    )
    launcher.main(config)
    assert "--hand-intent-port 15669" in sent[1]
    assert (" --manager" in sent[1]) == pico_manager
    assert "--intent-endpoint tcp://localhost:15669" in sent[launcher._hand_pane(config)]


@pytest.mark.parametrize("status", [None, False, True])
def test_ax_uses_authoritative_recording_status_after_xb_start(run_manager, status):
    # With no reply, optimistic start stays active; a rejected start restores the two-gesture guard.
    reply = {} if status is None else {"recording": status}
    states, _ = run_manager(enter_mode(2) + [{"buttons": "xb"}, {}, {"buttons": "ax", **reply}, {}])
    assert states[-1]["stream_mode"].item() == 2
    assert bool(states[-1]["toggle_data_collection"].item()) == (status is not False)


def test_stale_idle_status_cannot_turn_save_into_mode_switch(run_manager):
    states, _ = run_manager(enter_mode(2) + [
        {"buttons": "xb"}, {}, {"buttons": "ax", "recording": False, "timestamp": 99.0}, {},
    ])
    assert states[-1]["stream_mode"].item() == 2
    assert states[-1]["toggle_data_collection"].item()


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


@pytest.mark.parametrize("interruption,destination", [
    ([{"buttons": "xb"}, {}, {"buttons": "ax"}, {}], 2),  # start then save
    ([{"recording": True}, {"recording": False}], 2),  # browser recording
    ([{"buttons": "ya"}, {}], 2),  # discard
    ([{"buttons": "abxy"}, {}, {"buttons": "abxy"}, {}], 2),  # stop then start
    ([{"stick": True}, {}], 5),  # enter VR_3PT
])
def test_recording_and_mode_changes_cancel_pending_ax(run_manager, interruption, destination):
    states, _ = run_manager(enter_mode(2) + [{"buttons": "ax"}, {}]
                           + interruption + [{"buttons": "ax"}, {}])
    assert states[-1]["stream_mode"].item() == destination
    assert not states[-1]["toggle_data_collection"].item()


def test_reconnect_requires_a_new_ax_pair_after_policy_restart(run_manager):
    states, _ = run_manager(enter_mode(2) + [
        {"buttons": "ax"}, {}, {"disconnect": True},
        {"buttons": "abxy"}, {}, {"buttons": "ax"}, {},
    ])
    assert any(s["stream_mode"].item() == 0 for s in states)
    assert states[-1]["stream_mode"].item() == 2
    assert not any(s["toggle_data_collection"].item() or s["toggle_data_abort"].item() for s in states)


def feedback_packet(body):
    return msgpack.packb({
        "body_q_measured": body, "left_hand_q_measured": [0.1] * 7,
        "right_hand_q_measured": [0.2] * 7,
    }, use_bin_type=True)


@pytest.fixture
def calibration(monkeypatch):
    packets = deque()
    clock = SimpleNamespace(now=0.0)

    def get_data():
        return packets.popleft()[1] if packets and packets[0][0] <= clock.now else None

    monkeypatch.setattr(manager, "ZMQPoller", lambda **_: SimpleNamespace(get_data=get_data))
    monkeypatch.setattr(manager, "time", SimpleNamespace(
        monotonic=lambda: clock.now, sleep=lambda dt: setattr(clock, "now", clock.now + dt),
    ))
    streamer = manager.PlannerStreamer.__new__(manager.PlannerStreamer)
    streamer.feedback_reader = manager.FeedbackReader()
    streamer.three_point = Mock()
    return streamer, packets, clock


@pytest.mark.parametrize("body", [[0.0] * 29, np.linspace(-0.2, 0.2, 29).tolist()])
def test_vr3pt_calibrates_from_new_measured_feedback_not_queued_packet(calibration, body):
    streamer, packets, clock = calibration
    packets.extend([(0.0, feedback_packet([1.0] * 29)), (0.02, feedback_packet(body))])
    assert streamer.recalibrate_for_vr3pt()
    streamer.three_point.reset_with_measured_q.assert_called_once()
    np.testing.assert_array_equal(streamer.three_point.reset_with_measured_q.call_args.args[0], body)
    np.testing.assert_array_equal(
        streamer.feedback_reader.upper_body_position_target,
        np.asarray(body)[streamer.feedback_reader.upper_body_joint_indices],
    )
    assert clock.now == pytest.approx(0.02)


@pytest.mark.parametrize("packet", [
    None, b"\xc1", msgpack.packb(None), msgpack.packb([]), msgpack.packb({}),
    feedback_packet([0.0] * 28), feedback_packet([[0.0] * 29]),
    feedback_packet([float("nan")] * 29), feedback_packet([float("inf")] * 29),
    feedback_packet(["invalid"] * 29),
])
def test_missing_or_invalid_vr3pt_feedback_preserves_held_targets_and_allows_retry(calibration, packet):
    streamer, packets, clock = calibration
    feedback = streamer.feedback_reader
    packets.append((0.0, feedback_packet([0.1] * 29)))
    assert feedback.poll_feedback()
    held = (feedback.upper_body_position_target, feedback.left_hand_position_target,
            feedback.right_hand_position_target, feedback.full_body_q_measured)
    packets.extend([(0.0, feedback_packet([0.5] * 29)), (0.01, packet)])
    assert not streamer.recalibrate_for_vr3pt()
    streamer.three_point.reset_with_measured_q.assert_not_called()
    for actual, expected in zip((feedback.upper_body_position_target, feedback.left_hand_position_target,
                                feedback.right_hand_position_target, feedback.full_body_q_measured), held):
        assert actual is expected
    assert clock.now == pytest.approx(0.1)
    packets.append((clock.now + 0.02, feedback_packet([0.2] * 29)))
    assert streamer.recalibrate_for_vr3pt()
    np.testing.assert_array_equal(streamer.three_point.reset_with_measured_q.call_args.args[0], [0.2] * 29)


@pytest.mark.parametrize("parent", [2, 3])
def test_rejected_vr3pt_switch_keeps_current_mode_until_explicit_retry(run_manager, parent):
    steps = enter_mode(parent)
    states, commands = run_manager(steps + [
        {"stick": True, "recording": True}, {}, {"stick": True}, {}, {"stick": True}, {},
    ], calibration_results=[False, True])
    assert [s["stream_mode"].item() for s in states[-6:]] == [parent, parent, 5, 5, parent, parent]
    assert len(commands) == (3 if parent == 2 else 5)
    assert not any(s["toggle_data_collection"].item() or s["toggle_data_abort"].item() for s in states)


def test_policy_stop_still_works_after_rejected_vr3pt_switch(run_manager):
    states, commands = run_manager(enter_mode(2) + [
        {"stick": True}, {}, {"buttons": "abxy"}, {},
    ], calibration_results=[False])
    assert states[-1]["stream_mode"].item() == 0
    assert commands[-1] == manager.build_command_message(start=False, stop=True, planner=True)
