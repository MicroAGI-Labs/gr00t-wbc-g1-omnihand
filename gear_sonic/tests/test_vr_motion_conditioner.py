"""Offline motion, glitch, and send-transaction tests; no robot connections."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.teleop.vr_motion_conditioner import VRMotionConditioner, VRMotionLimits, rotation_error
from gear_sonic.tests.test_commanded_pose_transition import streamer, vr_pose, decode_vr
from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.scripts.run_data_exporter import unpack_pose_message


def initial():
    pose = vr_pose()
    limiter = VRMotionConditioner()
    limiter.seed(pose, 0.0)
    return limiter, pose


def test_diagonal_reversals_obey_vector_speed_and_acceleration():
    limiter, origin = initial()
    previous = origin.copy()
    previous_v = previous_w = np.zeros((3, 3))
    for i in range(1, 301):
        target = origin.copy()
        target[:, :3] += 0.09 * np.sin(i * 0.045) * np.ones((3, 3))
        target[:, 3:] = (Rotation.from_rotvec(np.tile([0, 0, 0.5 * np.sin(i * 0.07)], (3, 1))) *
                        Rotation.from_quat(origin[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        output = limiter.update(target, i * 0.02)
        v = (output[:, :3] - previous[:, :3]) / 0.02
        w = rotation_error(output[:, 3:], previous[:, 3:]) / 0.02
        assert np.max(np.linalg.norm(v, axis=1)) <= limiter.limits.speed + 1e-9
        assert np.max(np.linalg.norm(w, axis=1)) <= limiter.limits.angular_speed + 1e-9
        assert np.max(np.linalg.norm(v - previous_v, axis=1)) / 0.02 <= limiter.limits.acceleration + 1e-8
        assert np.max(np.linalg.norm(w - previous_w, axis=1)) / 0.02 <= limiter.limits.angular_acceleration + 1e-8
        previous, previous_v, previous_w = output, v, w
    assert not limiter.fault


def test_normal_motion_has_small_lag_and_quiet_jitter_is_reduced():
    limiter, origin = initial()
    for i in range(1, 101):
        target = origin.copy()
        target[:, 0] += 0.1 * i * 0.02
        output = limiter.update(target, i * 0.02)
    assert not limiter.fault
    assert np.max(np.abs(target[:, 0] - output[:, 0])) < 0.02
    limiter, origin = initial()
    samples = []
    for i in range(1, 101):
        target = origin.copy()
        target[:, 0] += (-1) ** i * 0.001
        samples.append(limiter.update(target, i * 0.02)[0, 0])
    assert np.std(samples[20:]) < 0.0006


def test_single_spike_is_discarded_and_persistent_jump_latches():
    limiter, origin = initial()
    spike = origin.copy()
    spike[0, 0] += 1
    np.testing.assert_allclose(limiter.update(spike, 0.02), origin)
    assert not limiter.fault
    limiter.update(origin, 0.04)
    for i in range(3):
        limiter.update(spike, 0.06 + i * 0.02)
    assert limiter.fault
    for i in range(30):
        np.testing.assert_allclose(limiter.update(spike, 0.12 + i * 0.02), origin)


def test_fast_hand_motion_keeps_following_and_reverses_without_rearming():
    limiter, origin = initial()
    outputs = []
    for i in range(1, 251):
        target = origin.copy()
        # A 60 cm, 115-degree reach outruns both limits and the old lag
        # thresholds, then reverses before the robot catches the first goal.
        phase = max(0, min(i, 40 - i))
        target[:, 0] += 0.03 * phase
        target[:, 3:] = (Rotation.from_rotvec(np.tile([0, 0, 0.1 * phase], (3, 1))) *
                        Rotation.from_quat(origin[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        outputs.append(limiter.update(target, i * 0.02))
        assert not limiter.fault
        assert np.max(np.linalg.norm(limiter.filtered[:, :3] - limiter.pose[:, :3], axis=1)) < 0.26
    assert limiter.stopped
    assert outputs[19][0, 0] > origin[0, 0] + 0.02
    assert max(p[0, 0] for p in outputs) < origin[0, 0] + 0.4
    np.testing.assert_allclose(limiter.pose, origin, atol=1e-4)


@pytest.mark.parametrize("failure", ["nan", "timing"])
def test_invalid_input_and_clock_gaps_do_not_allow_catchup(failure):
    limiter, origin = initial()
    target = origin.copy()
    target[0, 0] += 0.02
    limiter.update(target, 0.02)
    before = limiter.pose.copy()
    if failure == "nan":
        target[0, 0] = np.nan
    output = limiter.update(target, 5.0 if failure == "timing" else 0.04)
    assert limiter.fault
    assert np.max(np.linalg.norm(output[:, :3] - before[:, :3], axis=1)) <= 0.0021


def test_quaternion_sign_flip_is_not_a_tracking_jump():
    limiter, origin = initial()
    target = origin.copy()
    target[:, 3:] *= -1
    output = limiter.update(target, 0.02)
    assert not limiter.fault
    np.testing.assert_allclose(rotation_error(output[:, 3:], origin[:, 3:]), 0, atol=1e-12)


def test_send_failure_does_not_advance_filter_state(streamer, monkeypatch):
    streamer.vr_conditioner, origin = initial()
    streamer.last_vr_pose = origin.copy()
    before = deepcopy(streamer.vr_conditioner)
    monkeypatch.setattr(manager.time, "monotonic", lambda: 0.02)
    monkeypatch.setattr(streamer.socket, "send", lambda packet: (_ for _ in ()).throw(RuntimeError("send failed")))
    goal = origin.copy()
    goal[0, 0] += 0.2
    with pytest.raises(RuntimeError, match="send failed"):
        streamer.run_once(manager.StreamMode.PLANNER, vr_pose_override=goal, force_locomotion_idle=True)
    np.testing.assert_array_equal(streamer.vr_conditioner.pose, before.pose)
    assert streamer.vr_conditioner.last_time == before.last_time


def test_generated_return_completes_only_after_limited_target_arrives(streamer, monkeypatch):
    streamer.vr_conditioner, origin = initial()
    streamer.last_vr_pose = origin.copy()
    goal = origin.copy()
    goal[:2, 0] += 0.8
    monkeypatch.setattr(streamer, "vr_pose_from_upper_body", lambda joints: goal)
    now = [0.0]
    monkeypatch.setattr(manager.time, "monotonic", lambda: now[0])
    transition = streamer.begin_vr_return(to_base=True, duration_s=0.1)
    complete = False
    for i in range(1, 401):
        now[0] = i * 0.02
        sent, complete = streamer.send_vr_return_sample(transition, now[0])
        assert sent
        if i == 5:
            assert not complete
        if complete:
            break
    assert complete
    np.testing.assert_allclose(decode_vr(streamer.packets[-1]), goal, atol=1e-4)


def test_live_fault_brakes_holds_and_reanchors_without_a_jump(streamer, monkeypatch):
    monkeypatch.setattr(manager, "_process_3pt_pose", lambda sample: sample.copy())
    monkeypatch.setattr(manager, "get_controller_inputs", lambda reader: (False, 0, 0, 0, 0))
    monkeypatch.setattr(manager, "compute_hand_joints_from_inputs", lambda *args: (np.zeros(7), np.zeros(7)))
    monkeypatch.setattr(manager, "get_controller_axes", lambda reader: (0, 1, 0, 0))
    streamer.three_point = manager.ThreePointPose(robot_model=object())
    streamer.left_hand_ik_solver = streamer.right_hand_ik_solver = None
    streamer.vr_conditioner, origin = initial()
    streamer.last_vr_pose = origin.copy()
    streamer.mode = manager.LocomotionMode.WALK
    sample = origin.copy()
    now = [0.0]
    monkeypatch.setattr(manager.time, "monotonic", lambda: now[0])
    streamer.reader.get_latest = lambda: {"body_poses_np": sample}
    streamer.reader.get_timestamp_ns = lambda: int(now[0] * 1e9)
    assert streamer.recalibrate_for_vr3pt()
    assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    for i in range(1, 6):
        now[0] = i * 0.02
        sample[:, 0] += 0.008
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    sample[0, 0] += 1.0
    for i in range(6, 9):
        now[0] = i * 0.02
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert streamer.vr_conditioner.fault
    assert not streamer.recalibrate_for_vr3pt()
    for i in range(9, 30):
        now[0] = i * 0.02
        assert streamer.run_once(manager.StreamMode.PLANNER_IDLE_BASE_POSE)
        packet = unpack_pose_message(streamer.packets[-1], topic="planner")
        assert packet["mode"][0] == manager.LocomotionMode.IDLE.value
        np.testing.assert_array_equal(packet["movement"], 0)
    assert streamer.vr_conditioner.stopped
    held = streamer.last_vr_pose.copy()
    assert streamer.recalibrate_for_vr3pt()
    now[0] += 0.02
    assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert not streamer.vr_conditioner.fault
    np.testing.assert_allclose(decode_vr(streamer.packets[-1]), held, atol=1e-7)
