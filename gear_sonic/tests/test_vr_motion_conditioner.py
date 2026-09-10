"""Offline motion, glitch, and send-transaction tests; no robot connections."""
from copy import deepcopy
from types import SimpleNamespace

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


@pytest.mark.parametrize("disabled", [False, True])
def test_limiter_is_optional_and_enabled_by_default(monkeypatch, disabled):
    monkeypatch.setattr(manager, "FeedbackReader", lambda **kwargs: None)
    monkeypatch.setattr(manager, "init_hand_ik_solvers", lambda: (None, None))
    options = {"vr_motion_limits": None} if disabled else {}
    instance = manager.PlannerStreamer(None, None, None, **options)
    assert (instance.vr_conditioner is None) == disabled


def test_disabled_limiter_sends_live_pose_without_lag(streamer, monkeypatch):
    monkeypatch.setattr(manager, "get_controller_inputs", lambda reader: (False, 0, 0, 0, 0))
    monkeypatch.setattr(manager, "compute_hand_joints_from_inputs", lambda *args: (np.zeros(7), np.zeros(7)))
    streamer.left_hand_ik_solver = streamer.right_hand_ik_solver = None
    target = vr_pose()
    streamer.reader.get_latest = lambda: {"body_poses_np": target}
    streamer.three_point = SimpleNamespace(process_smpl_pose=lambda pose: pose.copy())
    for stamp, offset in enumerate((0.0, 0.3, -0.2), start=1):
        target[0, 0] = offset
        target[0, 3:] = Rotation.from_euler("z", offset).as_quat(scalar_first=True)
        streamer.reader.get_timestamp_ns = lambda: stamp
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
        np.testing.assert_allclose(decode_vr(streamer.packets[-1]), target, atol=1e-7)
        assert "vr_motion_limits" not in unpack_pose_message(streamer.packets[-1], topic="planner")


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


def test_feasible_motion_passes_through_without_lag_or_smoothing():
    limiter, origin = initial()
    for i in range(1, 401):
        now = i * 0.02
        target = origin.copy()
        # Starts at rest, with translation/rotation derivatives below the caps.
        target[:, 0] += 0.04 * (1 - np.cos(now))
        target[:, 3:] = (Rotation.from_rotvec(np.tile([0, 0, 0.3 * (1 - np.cos(now))], (3, 1))) *
                        Rotation.from_quat(origin[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        output = limiter.update(target, now)
        np.testing.assert_allclose(output, target, atol=1e-14)
        assert not limiter.limited.any()
        assert not limiter.fault


def test_small_feasible_jitter_is_not_filtered():
    limiter, origin = initial()
    for i in range(1, 101):
        target = origin.copy()
        # At 50 Hz this stays below the jerk ceiling as well as speed/acceleration.
        target[:, 0] += (-1) ** i * 0.000001
        np.testing.assert_array_equal(limiter.update(target, i * 0.02), target)
        assert not limiter.limited.any()


def test_cornering_and_braking_have_bounded_speed_and_acceleration():
    limiter, target = initial()
    position = target[:, :3].copy()
    previous_velocity = np.zeros((3, 3))
    now = 0.0
    peak_speed = 0.0
    rounded_corner = False
    for i in range(300):
        dt = (0.012, 0.024, 0.016, 0.028)[i % 4]
        now += dt
        direction = ([0.4, 0, 0] if now < 1 else
                     [0, 0.4, 0] if now < 2 else
                     [-0.4, 0, 0] if now < 3 else [0, 0, 0])
        target[:, :3] += np.asarray(direction) * dt
        # Also exercise a tracking fault while still moving: braking must
        # preserve the same continuity and limits as an ordinary reversal.
        if now > 3.2:
            target[0, 0] = np.nan
        output = limiter.update(target, now)
        step = output[:, :3] - position
        acceleration = (limiter.velocity - previous_velocity) / dt
        assert np.max(np.linalg.norm(step, axis=1)) <= limiter.limits.speed * dt + 1e-10
        assert np.max(np.linalg.norm(acceleration, axis=1)) <= limiter.limits.acceleration + 1e-9
        peak_speed = max(peak_speed, np.linalg.norm(limiter.velocity[0]))
        rounded_corner |= limiter.velocity[0, 0] > 0.02 and limiter.velocity[0, 1] > 0.02
        position = output[:, :3].copy()
        previous_velocity = limiter.velocity.copy()
    assert peak_speed > 0.14
    assert rounded_corner
    assert limiter.fault and limiter.stopped


def test_over_limit_jitter_stays_near_input_and_settles_without_drift():
    limiter, origin = initial()
    samples = []
    for i in range(1, 2001):
        target = origin.copy()
        target[:, 0] += (-1) ** i * 0.001
        samples.append(limiter.update(target, i * 0.02)[0, 0] - origin[0, 0])
    # Unlike the old low-pass filter, a rate limiter need not average the
    # jitter to zero. It must stay in its vicinity, not integrate a lasting bias.
    assert np.max(np.abs(samples)) < 0.0015
    for i in range(2001, 2101):
        limiter.update(origin, i * 0.02)
    np.testing.assert_allclose(limiter.pose, origin, atol=1e-12)
    assert limiter.stopped and not limiter.limited.any()


def test_recovery_rejoins_moving_target_without_a_velocity_jump():
    limiter, target = initial()
    previous = target.copy()
    previous_v = np.zeros((3, 3))
    was_limited = False
    rejoined = False
    for i in range(1, 501):
        target[:, 0] += (0.4 if i <= 25 else 0.04 if i < 350 else 0) * 0.02
        output = limiter.update(target, i * 0.02)
        v = (output[:, :3] - previous[:, :3]) / 0.02
        assert np.max(np.linalg.norm(v, axis=1)) <= limiter.limits.speed + 1e-10
        assert np.max(np.linalg.norm(v - previous_v, axis=1)) <= limiter.limits.acceleration * 0.02 + 1e-10
        was_limited |= limiter.limited.any()
        if 250 < i < 350:
            np.testing.assert_allclose(output, target, atol=1e-12)
            assert not limiter.limited.any()
            rejoined = True
        previous, previous_v = output, v
    assert was_limited and rejoined and limiter.stopped
    np.testing.assert_allclose(limiter.pose, target, atol=1e-12)


def test_limiting_one_wrist_does_not_filter_other_targets():
    limiter, origin = initial()
    for i in range(1, 101):
        target = origin.copy()
        target[0, 0] += 0.4 * i * 0.02
        target[1:, 1] += 0.02 * (1 - np.cos(i * 0.02))
        output = limiter.update(target, i * 0.02)
        np.testing.assert_allclose(output[1:], target[1:], atol=1e-14)
        assert limiter.limited[0, 0]
        assert not limiter.limited[:, 1:].any()


def test_newest_pose_changes_acceleration_on_the_very_next_tick():
    limiter, origin = initial()
    for i in range(1, 21):
        target = origin.copy()
        target[:, 0] += 0.004 * i
        target[:, 3:] = (Rotation.from_rotvec(np.tile([0, 0, 0.04 * i], (3, 1))) *
                        Rotation.from_quat(origin[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        limiter.update(target, i * 0.02)
    previous_a = limiter.acceleration.copy()
    previous_alpha = limiter.angular_acceleration.copy()
    newest = limiter.pose.copy()
    newest[:, 0] -= 0.005
    newest[:, 3:] = (Rotation.from_rotvec(np.tile([0, 0, -0.05], (3, 1))) *
                    Rotation.from_quat(newest[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
    limiter.update(newest, 0.42)
    assert not limiter.fault
    # Acceleration starts changing on this tick, at the fixed jerk ceiling;
    # it cannot jump to full reverse acceleration.
    np.testing.assert_allclose(limiter.acceleration[:, 0], previous_a[:, 0] - limiter.limits.jerk * 0.02)
    np.testing.assert_allclose(limiter.angular_acceleration[:, 2], previous_alpha[:, 2] - limiter.limits.angular_jerk * 0.02)


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


def test_delayed_rotation_uses_time_since_last_accepted_sample():
    limiter, origin = initial()
    # The recorded failure: one rejected 35-degree turn followed by a
    # continuous turn, formerly compared against a frozen reference at 20 ms.
    for now, degrees, rejects in [(0.12043, 35.35, 1), (0.22702, 49.085, 0), (0.24181, 54.66, 0)]:
        target = origin.copy()
        target[0, 3:] = (Rotation.from_euler("z", degrees, degrees=True) *
                        Rotation.from_quat(origin[0, 3:], scalar_first=True)).as_quat(scalar_first=True)
        limiter.update(target, now)
        assert limiter.rejected == rejects
        assert not limiter.fault
    assert limiter.seed_generation == 1


def test_repeated_control_ticks_do_not_count_as_new_tracking_jumps():
    limiter, origin = initial()
    spike = origin.copy()
    spike[0, 0] += 1
    limiter.update(spike, 0.02)
    for now in [0.04, 0.06, 0.08]:
        limiter.update(spike, now, source_fresh=False)
        assert limiter.rejected == 1
        assert not limiter.fault
    limiter.update(spike, 0.10)
    limiter.update(spike, 0.12)
    assert limiter.fault == "repeated 3PT tracking jumps"


def test_generated_return_can_follow_an_isolated_rejected_sample():
    limiter, origin = initial()
    spike = origin.copy()
    spike[0, 0] += 1
    limiter.update(spike, 0.02)
    target = origin.copy()
    target[0, 0] += 0.05
    for i in range(2, 401):
        limiter.update(target, i * 0.02, live=False)
    assert not limiter.fault
    assert limiter.rejected == 0
    assert limiter.reached(target)


def test_continuous_output_does_not_hide_a_stale_source():
    limiter, origin = initial()
    target = origin.copy()
    target[0, 0] += 0.05
    limiter.update(target, 0.02)
    for i in range(2, 51):
        limiter.update(target, i * 0.02, source_fresh=False)
        assert not limiter.fault
    limiter.update(target, 1.021, source_fresh=False)
    assert limiter.fault == "stale 3PT input"
    limiter.update(target, 1.04)
    assert limiter.fault == "stale 3PT input"


def test_short_source_pause_brakes_and_resumes_without_reanchoring():
    limiter, origin = initial()
    target = origin.copy()
    for i in range(1, 401):
        fresh = i < 10 or i >= 25
        if fresh:
            target[0, 0] = origin[0, 0] + 0.05
        limiter.update(target, i * 0.02, source_fresh=fresh)
        assert not limiter.fault
    assert limiter.reached(target)
    assert limiter.seed_generation == 1


def test_streamer_publishes_between_source_samples_and_stops_on_source_timeout(streamer, monkeypatch):
    monkeypatch.setattr(manager, "get_controller_inputs", lambda reader: (False, 0, 0, 0, 0))
    monkeypatch.setattr(manager, "compute_hand_joints_from_inputs", lambda *args: (np.zeros(7), np.zeros(7)))
    streamer.left_hand_ik_solver = streamer.right_hand_ik_solver = None
    streamer.vr_conditioner, origin = initial()
    target = origin.copy()
    target[0, 0] += 0.05
    streamer.reader.get_latest = lambda: {"body_poses_np": target}
    streamer.three_point = SimpleNamespace(process_smpl_pose=lambda sample: sample.copy())
    now = [0.0]
    stamp = [1]
    monkeypatch.setattr(manager.time, "monotonic", lambda: now[0])
    streamer.reader.get_timestamp_ns = lambda: stamp[0]
    for i in range(1, 101):
        now[0] = i * 0.02
        if i % 4 == 1:
            stamp[0] += 1
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
        assert not streamer.vr_conditioner.fault
    assert len(streamer.packets) == 100
    assert decode_vr(streamer.packets[2])[0, 0] > decode_vr(streamer.packets[1])[0, 0]
    for i in range(101, 149):
        now[0] = i * 0.02
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert streamer.vr_conditioner.fault == "stale 3PT input"
    packet = unpack_pose_message(streamer.packets[-1], topic="planner")
    assert packet["mode"][0] == manager.LocomotionMode.IDLE.value


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


@pytest.mark.parametrize("gap", [0.109, 0.199, 0.339, 0.999, 1.001, 5.0])
@pytest.mark.parametrize("moving_ticks", [3, 25, 100])
@pytest.mark.parametrize("jerk", [3.0, 6.0, 12.0])
@pytest.mark.parametrize("speed,acceleration", [(0.15, 0.6), (0.35, 0.9)])
def test_publisher_derivatives_match_real_timestamps_across_gap_and_braking(gap, moving_ticks, jerk, speed, acceleration):
    limits = VRMotionLimits(speed=speed, acceleration=acceleration, jerk=jerk, angular_jerk=np.deg2rad(jerk * 600))
    limiter = VRMotionConditioner(limits)
    target = vr_pose()
    limiter.seed(target, 0)
    previous = target.copy()
    previous_v = np.zeros((2, 3, 3))
    previous_a = np.zeros_like(previous_v)
    now = 0.0
    for i in range(moving_ticks + 200):
        dt = gap if i == moving_ticks else 0.02
        now += dt
        if i < moving_ticks:
            target[:, 0] += 0.008
            target[:, 3:] = (Rotation.from_rotvec(np.tile([0, 0, 0.04], (3, 1))) *
                            Rotation.from_quat(target[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        output = limiter.update(target, now)
        v = np.stack([(output[:, :3] - previous[:, :3]) / dt,
                      rotation_error(output[:, 3:], previous[:, 3:]) / dt])
        a = (v - previous_v) / dt
        j = (a - previous_a) / dt
        # Independent finite differences catch a fictitious integration dt.
        np.testing.assert_allclose(v, [limiter.velocity, limiter.angular_velocity], atol=1e-10)
        np.testing.assert_allclose(a, [limiter.acceleration, limiter.angular_acceleration], atol=1e-8)
        for channel, (vmax, amax, jmax) in enumerate((
            (limits.speed, limits.acceleration, limits.jerk),
            (limits.angular_speed, limits.angular_acceleration, limits.angular_jerk),
        )):
            assert np.max(np.linalg.norm(v[channel], axis=1)) <= 1.5 * vmax + 1e-9
            assert np.max(np.linalg.norm(a[channel], axis=1)) <= 1.5 * amax + 1e-8
            assert np.max(np.linalg.norm(j[channel], axis=1)) <= jmax + 1e-6
        if i >= moving_ticks:
            assert limiter.fault == ("3PT control gap" if gap >= limits.control_gap_timeout_s else "")
        previous, previous_v, previous_a = output, v, a
    if gap >= limits.control_gap_timeout_s:
        assert limiter.stopped


@pytest.mark.parametrize("gap", [0.109, 0.199, 0.339, 0.999])
def test_brief_gap_recovers_without_recalibration_or_catching_up_in_one_step(gap):
    limiter, origin = initial()
    target = origin.copy()
    target[:, 0] += 0.05
    output = limiter.update(target, gap)
    assert not limiter.fault
    np.testing.assert_allclose(output, origin, atol=1e-12)
    for i in range(1, 401):
        limiter.update(target, gap + i * 0.02)
        assert not limiter.fault
    assert limiter.seed_generation == 1
    assert limiter.reached(target)


@pytest.mark.parametrize("gap", [1.0, 1.001, 2.0])
def test_one_second_gap_latches_until_recalibration(gap):
    limiter, origin = initial()
    limiter.update(origin, gap)
    assert limiter.fault == "3PT control gap"
    limiter.update(origin, gap + 0.02)
    assert limiter.fault == "3PT control gap"


@pytest.mark.parametrize("invalid_now", [float("nan"), float("inf"), -1.0])
def test_invalid_clock_does_not_poison_last_valid_timestamp(invalid_now):
    limiter, origin = initial()
    target = origin.copy()
    target[:, 0] += 0.01
    limiter.update(target, 0.02)
    previous = deepcopy(limiter)
    np.testing.assert_array_equal(limiter.update(target, invalid_now), previous.pose)
    assert limiter.last_time == previous.last_time
    assert limiter.fault == "invalid control clock"
    limiter.update(target, 0.04)
    assert np.isfinite(limiter.pose).all()
    assert limiter.last_time == 0.04


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


@pytest.mark.parametrize("duration_s", [0.1, 5.0])
def test_generated_return_completes_only_after_limited_target_arrives(streamer, monkeypatch, duration_s):
    streamer.vr_conditioner, origin = initial()
    streamer.last_vr_pose = origin.copy()
    goal = origin.copy()
    goal[:2, 0] += 0.8
    goal[:2, 3:] = Rotation.from_euler("xyz", [0.3, 0.6, -0.4]).as_quat(scalar_first=True)
    monkeypatch.setattr(streamer, "vr_pose_from_upper_body", lambda joints: goal)
    now = [0.0]
    monkeypatch.setattr(manager.time, "monotonic", lambda: now[0])
    transition = streamer.begin_vr_return(to_base=True, duration_s=duration_s)
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
    held = streamer.held_vr_pose.copy()
    for i in range(100):
        now[0] += (0.016, 0.024, 0.018, 0.022)[i % 4]
        assert streamer.run_once(manager.StreamMode.PLANNER_IDLE_BASE_POSE)
        assert streamer.vr_conditioner.stopped
    np.testing.assert_array_equal(streamer.held_vr_pose, held)
    # A completed return must remain eligible for the next A+X entry.
    monkeypatch.setattr(streamer, "poll_fresh_feedback", lambda **kwargs: True)
    streamer.reader.get_latest = lambda: {"body_poses_np": origin.copy()}
    monkeypatch.setattr(manager, "_process_3pt_pose", lambda sample: sample.copy())
    streamer.three_point = manager.ThreePointPose(robot_model=object())
    assert streamer.recalibrate_for_vr3pt()


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
    for i in range(9, 110):
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


@pytest.mark.parametrize("jerk", [3.0, 6.0, 12.0])
def test_sampled_vector_jerk_and_hard_reserve_during_random_motion_and_fault(jerk):
    limits = VRMotionLimits(jerk=jerk, angular_jerk=np.deg2rad(jerk * 600))
    limiter = VRMotionConditioner(limits)
    target = vr_pose()
    limiter.seed(target, 0)
    previous = target.copy()
    velocities = np.zeros((2, 3, 3))
    accelerations = np.zeros_like(velocities)
    rng = np.random.default_rng(459)
    now = 0.0
    peaks = np.zeros((2, 3))
    for i in range(1500):
        dt = (0.008, 0.012, 0.02, 0.03, 0.05)[i % 5]
        now += dt
        if i < 1300:
            target[:, :3] += rng.uniform(-0.5, 0.5, (3, 3)) * dt
            target[:, 3:] = (Rotation.from_rotvec(rng.uniform(-3, 3, (3, 3)) * dt) *
                            Rotation.from_quat(target[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        else:
            target[0, 0] = np.nan
        output = limiter.update(target, now)
        new_velocities = np.stack([(output[:, :3] - previous[:, :3]) / dt,
                                  rotation_error(output[:, 3:], previous[:, 3:]) / dt])
        new_accelerations = (new_velocities - velocities) / dt
        jerks = (new_accelerations - accelerations) / dt
        for axis, (vmax, amax, jmax) in enumerate((
            (limits.speed, limits.acceleration, limits.jerk),
            (limits.angular_speed, limits.angular_acceleration, limits.angular_jerk),
        )):
            measured = [np.max(np.linalg.norm(values[axis], axis=1))
                        for values in (new_velocities, new_accelerations, jerks)]
            peaks[axis] = np.maximum(peaks[axis], measured)
            assert measured[0] <= vmax * 1.5 + 1e-8
            assert measured[1] <= amax * 1.5 + 1e-7
            assert measured[2] <= jmax + 1e-5
        previous, velocities, accelerations = output, new_velocities, new_accelerations
    assert limiter.fault and limiter.stopped
    # The test actually exercises jerk saturation in both position and rotation.
    np.testing.assert_allclose(peaks[:, 2], [limits.jerk, limits.angular_jerk], rtol=1e-6)


def test_speed_reserve_brakes_before_hard_ceiling_without_relaxing_jerk():
    from gear_sonic.utils.teleop.vr_motion_conditioner import limit_acceleration

    # Already accelerating near nominal speed; simply clamping velocity here
    # would require an acceleration/jerk discontinuity.
    velocity = np.array([[0.15, 0.0, 0.0]])
    acceleration = np.array([[0.6, 0.0, 0.0]])
    reserve_used = False
    peak = 0.0
    for _ in range(200):
        previous = acceleration.copy()
        acceleration, constrained = limit_acceleration(
            velocity, acceleration, np.array([[0.9, 0.0, 0.0]]), 0.02, 0.225, 0.9, 3.0,
        )
        velocity += acceleration * 0.02
        assert np.linalg.norm(acceleration - previous) <= 3 * 0.02 + 1e-10
        assert np.linalg.norm(velocity) <= 0.225 + 1e-10
        reserve_used |= constrained.any()
        peak = max(peak, np.linalg.norm(velocity))
    assert reserve_used and 0.15 < peak <= 0.225 + 1e-10


def test_trace_records_successful_packets_only_and_copies_arrays(streamer, monkeypatch, tmp_path):
    import json
    from gear_sonic.utils.teleop.vr_motion_trace import VRMotionTrace

    streamer.vr_conditioner, origin = initial()
    trace = VRMotionTrace(tmp_path, streamer.vr_conditioner.limits)
    streamer.vr_motion_trace = trace
    monkeypatch.setattr(manager.time, "monotonic", lambda: 0.02)
    goal = origin.copy()
    goal[0, 0] += 0.02
    assert streamer.run_once(manager.StreamMode.PLANNER, vr_pose_override=goal)
    goal[0, 0] = 999
    monkeypatch.setattr(streamer.socket, "send", lambda packet: (_ for _ in ()).throw(RuntimeError("send failed")))
    with pytest.raises(RuntimeError):
        streamer.run_once(manager.StreamMode.PLANNER, vr_pose_override=origin)
    trace.close()
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert len(rows) == 3 and not trace.error
    assert rows[0]["limits"]["jerk"] == 6
    assert rows[1]["input"][0][0] == origin[0, 0] + 0.02
    assert rows[1]["output"][0][0] < rows[1]["input"][0][0]
    assert rows[1]["dropped"] == 0
    packet = unpack_pose_message(streamer.packets[0], topic="planner")
    np.testing.assert_allclose(packet["vr_jerk_limits"], [6, 20 * np.pi])


def test_trace_analysis_excludes_generated_returns_and_detects_live_derivatives(tmp_path):
    from gear_sonic.utils.teleop.vr_motion_trace import VRMotionTrace
    from gear_sonic.scripts.analyze_vr_motion_trace import summarize

    limiter, origin = initial()
    trace = VRMotionTrace(tmp_path, limiter.limits)
    for i in range(1, 51):
        target = origin.copy()
        target[:, 0] += 0.04 * (1 - np.cos(i * 0.02))
        output = limiter.update(target, i * 0.02)
        trace.record(target, output, limiter, now=i * 0.02, stream_mode=5,
                     generated=i > 40, packet_timestamp=i)
    trace.close()
    report = summarize(trace.path)
    assert report["live_samples"] == 40
    assert report["samples"] == 50
    assert report["position_error_mm_p95"] == 0
    assert 0 < report["output"]["jerk_m_s3"]["peak"] < 0.04
    assert report["input"] == report["output"]


def test_trace_captures_robot_feedback_without_a_second_pico_client(tmp_path):
    import json
    import time
    import msgpack
    import zmq
    from gear_sonic.utils.teleop.vr_motion_trace import VRMotionTrace

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    trace = VRMotionTrace(tmp_path, VRMotionLimits(), feedback_endpoint=f"tcp://127.0.0.1:{port}")
    try:
        for _ in range(20):
            publisher.send(b"g1_debug" + msgpack.packb({"body_q_measured": [0.1] * 29}))
            time.sleep(0.02)
    finally:
        trace.close()
        publisher.close()
        context.term()
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    feedback = [r for r in rows if r["type"] == "feedback"]
    assert feedback and not trace.error
    assert feedback[-1]["data"]["body_q_measured"] == [0.1] * 29


def test_trace_audit_does_not_hide_timing_fault_spikes(tmp_path):
    import json
    from gear_sonic.scripts.analyze_vr_motion_trace import summarize

    pose = vr_pose()
    rows = []
    now = 0.0
    # Reproduce the historical output sequence independently of the fixed
    # implementation: a 220 ms gap followed by the inconsistent velocity step.
    for i in range(12):
        dt = 0.22 if i == 10 else 0.02
        now += dt
        pose[:, 0] += 0.1476 * 0.02 if i == 10 else 0.1428 * 0.02 if i == 11 else 0.15 * dt
        rows.append(dict(type="sample", time=now, output=pose.tolist(), input=pose.tolist(),
                         stream_mode=5, generated=False, seed_generation=1,
                         fault="3PT control gap" if i >= 10 else "", dropped=0))
    path = tmp_path / "fault.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    report = summarize(path)
    assert report["output"]["jerk_m_s3"]["peak"] < 1e-6
    assert report["output_sequence"]["acceleration_m_s2"]["peak"] > 6
    assert report["output_sequence"]["jerk_m_s3"]["peak"] > 350
    assert report["output_sequence_gaps_over_100ms"] == 1
    assert report["fault_samples"] == 2
