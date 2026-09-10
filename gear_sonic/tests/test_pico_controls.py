"""Offline operator controls, direct tracking, and independent hand holds."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.scripts.run_data_exporter import unpack_pose_message
from gear_sonic.tests.test_commanded_pose_transition import streamer, vr_pose, decode_vr
from gear_sonic.utils.teleop.pico_controls import PicoHandGate, PicoLocomotion, controller_poses, fresh_controller_sample
from gear_sonic.utils.teleop.vr_arm_clutch import VRArmClutch


def test_direct_pose_basis_and_invalid_tracking():
    raw = [1, 2, 3, 0, 0, 0, 1]
    poses = controller_poses(raw, raw, raw)
    np.testing.assert_allclose(poses[:, :3], np.tile([-3, -1, 2], (3, 1)))
    np.testing.assert_allclose(poses[:, 3:], np.tile([1, 0, 0, 0], (3, 1)))
    with pytest.raises(ValueError):
        controller_poses(raw, [0] * 7, raw)


def test_each_arm_refreshes_headset_heading_only_on_its_own_press():
    clutch = VRArmClutch(controller_frame=True)
    pose, held = vr_pose(), vr_pose()
    pose[2, 3:] = [1, 0, 0, 0]
    options = dict(source_fresh=True)
    # Held-at-start buttons cannot engage until released.
    np.testing.assert_array_equal(clutch.update(pose, held, (1, 1), **options), held)
    assert not clutch.tracking.any()
    clutch.update(pose, held, (0, 0), **options)
    clutch.update(pose, held, (1, 0), **options)
    pose[2, 3:] = Rotation.from_euler("z", np.pi / 2).as_quat(scalar_first=True)
    clutch.update(pose, held, (1, 1), **options)
    pose[:2, 0] += 0.03
    result = clutch.update(pose, held, (1, 1), **options)
    np.testing.assert_allclose(result[0, :3] - held[0, :3], [0.03, 0, 0], atol=1e-12)
    np.testing.assert_allclose(result[1, :3] - held[1, :3], [0, -0.03, 0], atol=1e-12)
    np.testing.assert_array_equal(result[2], held[2])


@pytest.mark.parametrize("gesture, action", [(("a",), "a"), (("b",), "b"), (("x",), "x"),
    (("x", "b"), "xb"), (("y", "a"), "ya"), (("a", "x"), None),
    (("a", "x", "b", "y"), None)])
def test_face_gestures_do_not_leak_partial_actions(gesture, action):
    tracker = manager.FaceChordTracker(singles=True)
    def update(buttons):
        return tracker.update(*(key in buttons for key in ("a", "b", "x", "y")))
    update(())
    for i in range(len(gesture)):
        assert update(gesture[:i + 1]) is None
    for i in range(1, len(gesture)):
        assert update(gesture[i:]) is None
    assert update(()) == action
    assert update(()) is None
    tracker.reset()
    update(("a",))
    assert update(()) is None  # Fresh release required after loss.


def test_locomotion_is_exclusive_and_requires_neutral_after_mode_and_speed_changes():
    controls = PicoLocomotion()
    def step(forward=0., turn=0., click=False, action=None, fresh=True):
        return controls.update((0, forward, turn, 0), click, action, fresh=fresh)
    assert step(forward=1) == (0, 0)
    step()
    assert step(forward=1, click=True) == (0, 0)
    assert step(forward=1) == (0, 0)
    step()
    assert step(forward=1) == (1, 0)
    assert step(forward=-1) == (-1, 0)
    assert step(turn=1) == (0, 1)
    assert step(turn=-1) == (0, -1)
    assert step(forward=1, turn=1) == (0, 0)
    assert step(forward=1, action="x") == (0, 0)
    assert controls.slow
    step()
    assert step(forward=1) == (1, 0)
    assert step(fresh=False) == (0, 0)
    assert step(forward=1) == (0, 0)
    step()
    assert step(forward=1) == (1, 0)
    step(click=True)
    step()
    assert step(turn=1) == (0, 0)


def test_hands_require_fresh_trigger_press_and_hold_independently():
    gate = PicoHandGate()
    options = dict(valid=True)
    assert gate.update((1, 1), (True, True), **options)[0].all()
    gate.update((0, 0), (True, True), **options)
    assert not gate.update((1, 1), (True, True), **options)[0].any()
    holds, _ = gate.update((0, 0), (False, True), **options)
    np.testing.assert_array_equal(holds, [True, False])
    assert gate.update((1, 0), (True, True), **options)[0][0]
    gate.update((0, 0), (True, True), **options)
    assert not gate.update((1, 0), (True, True), **options)[0][0]
    holds, opening = gate.update((1, 1), (False, False), valid=True, force_open=True)
    assert opening.all() and not holds.any()
    assert gate.update((0, 0), (False, False), valid=True)[1].all()
    assert gate.update((0, 0), (True, True), valid=False)[0].all()
    assert gate.update((1, 1), (True, True), valid=True)[0].all()


def test_xrt_controller_sample_does_not_require_body_tracking(monkeypatch):
    stamp = [100]
    monkeypatch.setattr(manager, "xrt", SimpleNamespace(
        get_time_stamp_ns=lambda: stamp[0],
        get_left_controller_pose=lambda: [1, 2, 3, 0, 0, 0, 1],
        get_right_controller_pose=lambda: [1, 2, 3, 0, 0, 0, 1],
        get_headset_pose=lambda: [1, 2, 3, 0, 0, 0, 1],
    ))
    monkeypatch.setattr(manager, "get_controller_inputs", lambda: (False, 0, 0, 0, 0))
    monkeypatch.setattr(manager, "get_controller_axes", lambda: (0,) * 4)
    monkeypatch.setattr(manager, "get_abxy_buttons", lambda: (False,) * 4)
    monkeypatch.setattr(manager, "get_axis_clicks", lambda: (False,) * 2)
    reader = manager.PicoReader(controller_tracking=True)
    reader._latest = reader._read_sample()
    assert "body_poses_np" not in reader._latest
    assert reader._read_sample() is reader._latest
    stamp[0] += 1
    assert reader._read_sample()["source_timestamp_ns"] == 101


@pytest.fixture
def direct_streamer(streamer, monkeypatch):
    state = SimpleNamespace(now=100., stamp=1, pose=vr_pose(), grips=[0, 0], triggers=[0, 0], axes=[0.] * 4,
                            click=False, source_time=100.)
    streamer.controller_tracking = True
    streamer.locomotion = PicoLocomotion()
    streamer.vr_arm_clutch = VRArmClutch(controller_frame=True)
    streamer.last_vr_pose = vr_pose()
    streamer.left_hand_ik_solver = streamer.right_hand_ik_solver = None
    streamer.three_point = SimpleNamespace()
    streamer.reader.controller_tracking = True
    streamer.reader.get_latest = lambda: dict(controller_poses=state.pose.copy(), timestamp_ns=state.stamp,
        timestamp_monotonic=state.source_time, controller_axes=tuple(state.axes),
        controller_inputs=(False, *state.triggers, *state.grips), axis_clicks=(state.click, False))
    streamer.reader.get_timestamp_ns = lambda: state.stamp
    monkeypatch.setattr(manager, "get_controller_inputs", lambda reader: (False, 0, 0, *state.grips))
    monkeypatch.setattr(manager, "get_controller_axes", lambda reader: tuple(state.axes))
    monkeypatch.setattr(manager, "get_axis_clicks", lambda reader: (state.click, False))
    monkeypatch.setattr(manager, "compute_hand_joints_from_inputs", lambda *args: (np.zeros(7), np.zeros(7)))
    monkeypatch.setattr(manager, "time", SimpleNamespace(
        monotonic=lambda: state.now, monotonic_ns=lambda: int(state.now * 1e9), time=lambda: state.now,
        sleep=lambda seconds: setattr(state, "now", state.now + seconds),
    ))
    def advance(fresh=True):
        state.now += 0.02
        if fresh:
            state.stamp += 1
            state.source_time = state.now
    def tick(fresh=True, action=None):
        advance(fresh)
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT, face_command=action)
        return decode_vr(streamer.packets[-1])
    state.advance = advance
    return streamer, state, tick


def test_direct_streamer_holds_head_and_recovers_with_release_not_ax(direct_streamer):
    streamer, state, tick = direct_streamer
    tick()
    state.grips[:] = [1, 1]
    tick()
    for _ in range(30):
        state.pose[:2, 0] += 0.001
        state.pose[2, 3:] = Rotation.from_euler("z", 1).as_quat(scalar_first=True)
        tick()
    assert streamer.last_vr_pose[0, 0] > vr_pose()[0, 0] + 0.005
    np.testing.assert_array_equal(streamer.last_vr_pose[2], vr_pose()[2])
    for _ in range(15):
        tick(fresh=False)
    assert not streamer.vr_arm_clutch.tracking.any()
    state.pose[:2, 0] += 1
    for _ in range(100):
        tick()
    assert streamer.controller_input_lost
    assert not streamer.vr_arm_clutch.tracking.any()
    held = streamer.last_vr_pose.copy()
    state.grips[:] = [0, 0]
    streamer.feedback_reader.last_body_feedback_monotonic = state.now
    assert streamer.recalibrate_for_vr3pt()
    tick()
    state.grips[:] = [1, 1]
    np.testing.assert_allclose(tick(), held, atol=1e-7)


def test_home_return_preserves_live_locomotion_and_returns_waist(direct_streamer):
    streamer, state, tick = direct_streamer
    tick()
    state.click = True
    tick()
    state.click = False
    tick()
    state.axes[1] = 1
    tick()
    home = vr_pose()
    home[:2, 2] += 0.1
    streamer.last_vr_pose[2, 3:] = Rotation.from_euler("z", 0.3).as_quat(scalar_first=True)
    transition = streamer.begin_vr_return(to_base=True, duration_s=0.2, goal_override=home,
                                          return_head_home=True)
    for _ in range(15):
        state.advance()
        sent, complete = streamer.send_vr_return_sample(transition, state.now, allow_locomotion=True)
        assert sent
        packet = unpack_pose_message(streamer.packets[-1], topic="planner")
        assert np.linalg.norm(packet["movement"]) > 0
    assert complete
    np.testing.assert_allclose(streamer.last_vr_pose, home, atol=1e-7)


def test_rest_return_uses_saved_idle_arms_and_preserves_waist(direct_streamer):
    streamer, state, _ = direct_streamer
    rest = streamer.disconnect_idle_pose.copy()
    streamer.last_vr_pose[:2, 2] += 0.2
    streamer.last_vr_pose[2, 3:] = Rotation.from_euler("z", 0.3).as_quat(scalar_first=True)
    held = streamer.last_vr_pose.copy()
    # Fresh planner feedback during walking must not change the cached resting pose.
    streamer.feedback_reader.upper_body_planner_target += 0.7
    transition = streamer.begin_vr_rest_return(duration_s=0.2)
    np.testing.assert_allclose(transition.start, held, atol=1e-12)
    np.testing.assert_allclose(transition.goal[:2], rest[:2], atol=1e-12)
    np.testing.assert_allclose(transition.goal[2], held[2], atol=1e-12)
    np.testing.assert_array_equal(streamer.disconnect_idle_pose, rest)
    streamer.disconnect_idle_pose = None
    assert streamer.begin_vr_rest_return(duration_s=0.2) is None


@pytest.mark.parametrize("forward", [-1., 1.])
def test_slow_walk_commands_06_mps_at_full_stick_and_stops_at_neutral(direct_streamer, forward):
    streamer, state, tick = direct_streamer
    streamer.locomotion.enabled = True
    tick(action="x")
    state.axes[1] = forward
    tick()
    packet = unpack_pose_message(streamer.packets[-1], "planner")
    assert packet["mode"][0] == manager.LocomotionMode.SLOW_WALK.value
    assert packet["speed"][0] == pytest.approx(0.6)
    assert packet["movement"][0] == pytest.approx(forward)
    state.axes[1] = 0.
    tick()
    packet = unpack_pose_message(streamer.packets[-1], "planner")
    assert packet["mode"][0] == manager.LocomotionMode.IDLE.value
    np.testing.assert_array_equal(packet["movement"], [0, 0, 0])


def test_slow_turn_halves_yaw_without_translation(direct_streamer):
    streamer, state, tick = direct_streamer
    streamer.yaw_accumulator = manager.YawAccumulator()
    streamer.locomotion.enabled = True
    tick()
    state.axes[2] = 1.
    tick()
    normal_yaw = streamer.yaw_accumulator.yaw_angle_change()
    tick(action="x")
    state.axes[2] = 0.
    tick()
    state.axes[2] = 1.
    tick()
    assert streamer.yaw_accumulator.yaw_angle_change() == normal_yaw / 2
    np.testing.assert_array_equal(unpack_pose_message(streamer.packets[-1], "planner")["movement"], [0, 0, 0])


def test_latched_tracking_loss_reanchors_after_stopping_without_ax(direct_streamer):
    streamer, state, tick = direct_streamer
    tick()
    state.grips[:] = [1, 1]
    tick()
    for _ in range(20):
        state.pose[:2, 0] += 0.001
        tick()
    for _ in range(80):
        tick(fresh=False)
    assert streamer.controller_input_lost
    held = streamer.last_vr_pose.copy()
    state.advance()
    state.pose[:2, 0] += 1.
    streamer.feedback_reader.last_body_feedback_monotonic = state.now
    assert not streamer.recalibrate_for_vr3pt()  # Held grips cannot clear the fault.
    state.grips[:] = [0, 0]
    assert streamer.recalibrate_for_vr3pt()
    np.testing.assert_allclose(tick(), held, atol=1e-7)
    assert not streamer.controller_input_lost and not streamer.vr_arm_clutch.tracking.any()
    state.grips[:] = [0, 0]
    tick()
    state.grips[:] = [1, 1]
    np.testing.assert_allclose(tick(), held, atol=1e-7)
    assert streamer.vr_arm_clutch.tracking.all()


def test_tracking_timeout_holds_at_100ms_and_requires_recalibration(direct_streamer):
    streamer, state, tick = direct_streamer
    state.pose[2, 3:] = [1, 0, 0, 0]
    streamer.locomotion.enabled = True
    tick()
    state.grips[:] = [1, 1]
    tick()
    state.pose[:2, 0] += 0.03
    state.axes[1] = 1
    tick()
    held = streamer.last_vr_pose.copy()
    np.testing.assert_allclose(held[:2, 0], vr_pose()[:2, 0] + 0.03)  # No smoothing.
    sample = streamer.reader.get_latest()
    state.now = state.source_time + 0.099
    assert fresh_controller_sample(sample, state.now)
    assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert not streamer.vr_fault
    assert np.linalg.norm(unpack_pose_message(streamer.packets[-1], "planner")["movement"]) > 0
    state.now = state.source_time + 0.100
    assert not fresh_controller_sample(sample, state.now)
    assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert streamer.controller_input_lost
    np.testing.assert_allclose(decode_vr(streamer.packets[-1]), held, atol=1e-7)
    for fresh in (False, True):
        state.pose[:2, 0] += 1.
        for _ in range(10):
            np.testing.assert_allclose(tick(fresh=fresh), held, atol=1e-7)
            packet = unpack_pose_message(streamer.packets[-1], "planner")
            np.testing.assert_array_equal(packet["movement"], [0, 0, 0])
            assert packet["mode"][0] == manager.LocomotionMode.IDLE.value
            # The existing SONIC timeout fallback holds this same target even
            # during a long SDK reconnect or a publisher process failure.
            np.testing.assert_allclose(packet["vr_base_pose"].reshape(2, 7), held[:2], atol=1e-7)
        assert not streamer.vr_arm_clutch.tracking.any()
    streamer.feedback_reader.last_body_feedback_monotonic = state.now
    assert not streamer.recalibrate_for_vr3pt()
    state.grips[:] = [0, 0]
    assert not streamer.recalibrate_for_vr3pt()  # Sticks must be centered too.
    state.axes[:] = [0] * 4
    assert streamer.recalibrate_for_vr3pt()
    np.testing.assert_allclose(tick(), held, atol=1e-7)
    state.grips[:] = [1, 0]
    np.testing.assert_allclose(tick(), held, atol=1e-7)
    state.pose[:2, 0] += 0.02
    moved = tick()
    assert moved[0, 0] == pytest.approx(held[0, 0] + 0.02, abs=1e-7)
    np.testing.assert_allclose(moved[1:], held[1:], atol=1e-7)


@pytest.mark.parametrize("return_button", ["a", "b"])
def test_timeout_interrupts_generated_return_even_after_fresh_input(direct_streamer, return_button):
    streamer, state, tick = direct_streamer
    tick()
    if return_button == "a":
        home = vr_pose()
        home[:, 2] += 0.2
        transition = streamer.begin_vr_return(to_base=True, duration_s=0.2, goal_override=home)
    else:
        streamer.last_vr_pose[:2, 2] += 0.2
        transition = streamer.begin_vr_rest_return(duration_s=0.2)
    state.advance()
    assert streamer.send_vr_return_sample(transition, state.now)[0]
    held = streamer.last_vr_pose.copy()
    state.now = state.source_time + 0.100
    for fresh in (False, True):
        for _ in range(20):
            state.advance(fresh)
            sent, complete = streamer.send_vr_return_sample(transition, state.now)
            assert sent and not complete
            np.testing.assert_allclose(streamer.last_vr_pose, held, atol=1e-12)
            np.testing.assert_allclose(streamer.held_vr_pose, held, atol=1e-12)


def test_manager_start_home_rest_record_and_ui_stop(direct_streamer, monkeypatch):
    from gear_sonic.end_effectors.protocol import decode_intent
    from gear_sonic.utils.teleop.pose_transition import IDLE_BASE_UPPER_BODY_RAD

    streamer, state, _ = direct_streamer
    streamer.last_vr_pose = None  # Exercise initial FK seeding, without A+X.
    home = vr_pose()
    home[:2, 2] += 0.05
    monkeypatch.setattr(streamer, "vr_pose_from_upper_body", lambda joints:
                        home.copy() if np.array_equal(joints, IDLE_BASE_UPPER_BODY_RAD) else vr_pose())
    streamer.feedback_reader.poll_feedback = lambda **kwargs: (
        setattr(streamer.feedback_reader, "last_body_feedback_monotonic", state.now) or True)
    streamer.reset_yaw = lambda: None
    streamer.reader.disconnected = False
    streamer.reader.stop = lambda: None
    streamer.three_point.close = lambda: None
    monkeypatch.setattr(manager, "_init_input_source", lambda *args: streamer.reader)
    monkeypatch.setattr(manager, "PlannerStreamer", lambda **kwargs: streamer)
    monkeypatch.setattr(manager, "PoseStreamer", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(manager, "ThreePointPose", lambda **kwargs: streamer.three_point)
    emitted, frame = [], [-1]
    socket = SimpleNamespace(
        bind=lambda *args: None, connect=lambda *args: None,
        setsockopt=lambda *args: None, setsockopt_string=lambda *args: None,
        poll=lambda *args: False, close=lambda: None,
        send=lambda data: emitted.append((frame[0], data)),
    )
    control = SimpleNamespace(**socket.__dict__)
    control_pending = [False]
    control.poll = lambda *args: control_pending[0]
    def receive_stop():
        control_pending[0] = False
        return dict(action="safe_idle", sequence=1)
    control.recv_json = receive_stop
    sockets = iter((socket, socket, socket, control))
    streamer.socket = socket
    monkeypatch.setattr(manager.zmq, "Context", lambda: SimpleNamespace(
        socket=lambda *args: next(sockets), term=lambda: None))
    rest_requests = []
    begin_rest = streamer.begin_vr_rest_return
    def return_to_legs(**kwargs):
        rest_requests.append(frame[0])
        return begin_rest(**kwargs)
    monkeypatch.setattr(streamer, "begin_vr_rest_return", return_to_legs)
    gestures = {1: "abxy", 9: "xb", 15: "a", 29: "b", 42: "xb", 45: "xb", 48: "ya", 115: "abxy"}
    def buttons(reader):
        frame[0] += 1
        i = frame[0]
        if i == 120:
            raise KeyboardInterrupt
        state.advance()
        state.click = i == 4
        state.axes[1] = 1 if 6 <= i < 100 else 0
        state.grips[:] = [1, 1] if 7 <= i < 55 else [0, 0]
        state.triggers[:] = [1, 1] if i >= 8 else [0, 0]
        if i == 55:
            control_pending[0] = True
        return tuple(key in gestures.get(i, "") for key in "abxy")
    monkeypatch.setattr(manager, "get_abxy_buttons", buttons)
    manager.run_pico_manager(teleop_mode="vr3pt", input_source="isaac", target_fps=50,
                             idle_base_transition_duration=0.2)
    states = {i: unpack_pose_message(data, "manager_state") for i, data in emitted
              if data.startswith(b"manager_state")}
    assert states[2]["stream_mode"][0] == manager.StreamMode.PLANNER_VR_3PT.value
    assert all(states[i]["stream_mode"][0] == manager.StreamMode.PLANNER_VR_3PT.value
               for i in range(10, 55))  # A and B returns remain recordable.
    assert rest_requests == [30]  # Neither AXBY nor XB leaks a B return.
    assert [i for i, value in states.items() if value["toggle_data_collection"][0]] == [10, 43, 46]
    assert [i for i, value in states.items() if value["toggle_data_abort"][0]] == [49]
    hands = {i: decode_intent(data) for i, data in emitted if data.startswith(b"hand_intent ")}
    assert hands[14]["left"]["closed"] and not hands[14]["left"]["hold"]
    for i in range(16, 26):
        assert not hands[i]["left"]["closed"] and not hands[i]["left"]["hold"]
    for i in range(30, 40):
        assert hands[i]["hold"]  # B freezes the hand instead of issuing open/close.
    assert not hands[42]["hold"] and hands[42]["left"]["hold"]
    planner = {i: unpack_pose_message(data, "planner") for i, data in emitted if data.startswith(b"planner")}
    for i in range(16, 40):
        assert np.linalg.norm(planner[i]["movement"]) > 0  # A and B keep walking.
    np.testing.assert_allclose(planner[28]["vr_position"].reshape(3, 3), home[:, :3], atol=1e-7)
    np.testing.assert_allclose(planner[42]["vr_position"].reshape(3, 3), vr_pose()[:, :3], atol=1e-7)
    assert not streamer.locomotion.slow  # XB never leaks X.
    assert all(states[i]["stream_mode"][0] == manager.StreamMode.PLANNER.value for i in range(90, 115))
    assert all(np.linalg.norm(planner[i]["movement"]) == 0 for i in range(55, 115))
    assert states[115]["stream_mode"][0] == manager.StreamMode.PLANNER_VR_3PT.value


@pytest.mark.parametrize("return_button", [None, "a", "b"])
@pytest.mark.parametrize("outage_seconds", [0, 20])
def test_manager_timeout_cancels_returns_and_waits_for_grip_recalibration(
    direct_streamer, monkeypatch, return_button, outage_seconds,
):
    streamer, state, _ = direct_streamer
    streamer.last_vr_pose = None
    home = vr_pose()
    home[:2, 2] += 0.2
    monkeypatch.setattr(streamer, "vr_pose_from_upper_body", lambda joints: home.copy())
    streamer.feedback_reader.poll_feedback = lambda **kwargs: (
        setattr(streamer.feedback_reader, "last_body_feedback_monotonic", state.now) or True)
    streamer.reset_yaw = lambda: None
    streamer.reader.disconnected = False
    streamer.reader.stop = streamer.three_point.close = lambda: None
    reconnects = []
    def reconnect():
        reconnects.append(state.now)
        state.now += outage_seconds
        streamer.reader.disconnected = False
        streamer.feedback_reader.vr_pose = streamer.last_vr_pose.copy()
    streamer.reader.reconnect = reconnect
    monkeypatch.setattr(manager, "PicoReader", type(streamer.reader))
    monkeypatch.setattr(manager, "_init_input_source", lambda *args: streamer.reader)
    monkeypatch.setattr(manager, "PlannerStreamer", lambda **kwargs: streamer)
    monkeypatch.setattr(manager, "PoseStreamer", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(manager, "ThreePointPose", lambda **kwargs: streamer.three_point)
    emitted, frame = [], [-1]
    socket = SimpleNamespace(
        bind=lambda *args: None, connect=lambda *args: None,
        setsockopt=lambda *args: None, setsockopt_string=lambda *args: None,
        poll=lambda *args: False, close=lambda: None,
        send=lambda data: emitted.append((frame[0], data)),
    )
    streamer.socket = socket
    monkeypatch.setattr(manager.zmq, "Context", lambda: SimpleNamespace(
        socket=lambda *args: socket, term=lambda: None))
    def buttons(reader):
        frame[0] += 1
        i = frame[0]
        if i == 42:
            raise KeyboardInterrupt
        state.advance(fresh=not 11 <= i < 25)
        state.click = i == 4
        state.axes[1] = 1 if 6 <= i < 32 else 0
        state.grips[:] = [1, 1] if 6 <= i < 35 else ([1, 0] if i >= 37 else [0, 0])
        state.pose[:2, 0] += 0.001 if i < 25 or i >= 38 else 0.1
        if outage_seconds and i == 17:
            streamer.reader.disconnected = True
        gesture = "abxy" if i == 1 else (return_button if i == 8 else "")
        return tuple(key in (gesture or "") for key in "abxy")
    monkeypatch.setattr(manager, "get_abxy_buttons", buttons)
    manager.run_pico_manager(teleop_mode="vr3pt", input_source="xrt", target_fps=50,
                             idle_base_transition_duration=0.2)
    states = {i: unpack_pose_message(data, "manager_state")["stream_mode"][0]
              for i, data in emitted if data.startswith(b"manager_state")}
    planner = {i: data for i, data in emitted if data.startswith(b"planner")}
    assert bool(reconnects) == bool(outage_seconds)
    assert all(states[i] == manager.StreamMode.PLANNER_IDLE_BASE_POSE.value for i in range(20, 35))
    held = decode_vr(planner[20])
    for i in range(20, 38):
        np.testing.assert_allclose(decode_vr(planner[i]), held, atol=1e-7)
        np.testing.assert_array_equal(unpack_pose_message(planner[i], "planner")["movement"], [0, 0, 0])
    assert states[35] == manager.StreamMode.PLANNER_VR_3PT.value
    assert decode_vr(planner[38])[0, 0] > held[0, 0]
    np.testing.assert_allclose(decode_vr(planner[38])[1:], held[1:], atol=1e-7)


def test_isaac_direct_sample_rejects_invalid_tracking_without_body_data():
    from gear_sonic.utils.teleop.input_readers import isaac_controller_sample

    tracked = dict(is_valid=True, pose=dict(position=[1, 2, 3], orientation=[0, 0, 0, 1]))
    raw = dict(head=tracked, left_controller=dict(aim_pose=tracked, inputs=dict(squeeze_value=1)),
               right_controller=dict(grip_pose=tracked, inputs=dict(trigger_value=1, primary_click=True)))
    sample = isaac_controller_sample(raw)
    assert sample["controller_inputs"] == (False, 0, 1, 1, 0)
    assert sample["face_buttons"] == (True, False, False, False)
    np.testing.assert_array_equal(sample["controller_poses"][:, :3], np.tile([-3, -1, 2], (3, 1)))
    raw["head"] = dict(is_valid=False)
    with pytest.raises(ValueError, match="head"):
        isaac_controller_sample(raw)
