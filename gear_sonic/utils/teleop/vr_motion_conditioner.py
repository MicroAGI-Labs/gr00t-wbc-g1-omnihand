"""Jerk-limited 3PT commands in calibrated Cartesian coordinates.

Feasible sampled motion passes through without filtering. During recovery the
latest target replaces the previous goal; no backlog of operator poses is played.
Speed/acceleration have operating limits and 50% hard reserve. Jerk is never
relaxed within the publisher's sampled sequence. This does not bound the
receiver's held-target sequence during packet loss or measured motor motion.
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from .pose_transition import validate_vr_pose


@dataclass(frozen=True)
class VRMotionLimits:
    speed: float = 0.15
    acceleration: float = 0.6
    angular_speed: float = np.pi / 2
    angular_acceleration: float = 2 * np.pi
    jerk: float = 6.0
    angular_jerk: float = 20 * np.pi
    hard_limit_factor: float = 1.5
    jump: float = 0.10
    angular_jump: float = np.pi / 6
    control_gap_timeout_s: float = 1.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("3PT limits must be finite and positive")
        if self.hard_limit_factor < 1:
            raise ValueError("hard limit factor must be at least one")


def cap(vector, maximum):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector * np.minimum(1.0, maximum / np.maximum(norm, 1e-12))


def rotation_error(target, current):
    return (Rotation.from_quat(target, scalar_first=True) *
            Rotation.from_quat(current, scalar_first=True).inv()).as_rotvec()


def limit_acceleration(velocity, acceleration, desired, dt, hard_speed, hard_acceleration, jerk):
    """Keep a feasible braking reserve without ever clipping velocity or jerk.

    The invariant ||v|| + ||a||²/(2J) <= V_hard reserves enough speed to
    reduce acceleration to zero at jerk J, even in the worst direction. The
    zero-acceleration braking candidate preserves this invariant under our
    semi-implicit discrete integration. Its feasible set is convex, so a line
    search toward the requested candidate can never require relaxing jerk.
    """
    desired = cap(desired, hard_acceleration)
    candidate = acceleration + cap(desired - acceleration, jerk * dt)

    def reserve(value):
        return np.linalg.norm(velocity + value * dt, axis=1) + np.sum(value * value, axis=1) / (2 * jerk)

    constrained = reserve(candidate) > hard_speed
    if np.any(constrained):
        braking = acceleration - cap(acceleration, jerk * dt)
        low, high = np.zeros(len(velocity)), np.ones(len(velocity))
        for _ in range(32):
            middle = (low + high) / 2
            trial = braking + middle[:, None] * (candidate - braking)
            feasible = reserve(trial) <= hard_speed
            low = np.where(feasible, middle, low)
            high = np.where(feasible, high, middle)
        candidate[constrained] = (braking + low[:, None] * (candidate - braking))[constrained]
    return candidate, constrained


class VRMotionConditioner:
    def __init__(self, limits=VRMotionLimits()):
        self.limits = limits
        self.pose = None
        self.last_time = None
        self.velocity = np.zeros((3, 3))
        self.angular_velocity = np.zeros((3, 3))
        self.acceleration = np.zeros((3, 3))
        self.angular_acceleration = np.zeros((3, 3))
        self.raw_velocity = np.zeros((2, 3, 3))
        self.target_velocity = np.zeros((2, 3, 3))
        self.velocity_qualified = np.zeros((2, 3), dtype=bool)
        self.limited = np.zeros((2, 3), dtype=bool)
        self.reserve_active = np.zeros((2, 3), dtype=bool)
        self.fault = ""
        self.rejected = 0
        self.seed_generation = 0

    @property
    def stopped(self):
        return bool(np.all(self.stopped_points))

    @property
    def stopped_points(self):
        return np.maximum.reduce([
            np.linalg.norm(state, axis=1) for state in
            (self.velocity, self.angular_velocity, self.acceleration, self.angular_acceleration)
        ]) < 1e-4

    def seed(self, pose, now):
        """Use a known stationary held target, never an uncalibrated Pico frame."""
        self.pose = validate_vr_pose(pose)
        self.seed_generation += 1
        self.raw = self.pose.copy()
        self.raw_time = now
        self.raw_source_timestamp_ns = None
        self.last_source_timestamp_ns = None
        self.last_source_time = now
        self.velocity[:] = self.angular_velocity[:] = 0
        self.acceleration[:] = self.angular_acceleration[:] = 0
        self.raw_velocity[:] = 0
        self.target_velocity[:] = 0
        self.velocity_qualified[:] = False
        self.limited[:] = False
        self.reserve_active[:] = False
        self.last_time = now
        self.fault = ""
        self.rejected = 0

    def update(self, target, now, *, live=True, source_fresh=True, source_timestamp_ns=None,
               brake_mask=(False, False, False), reanchor_mask=(False, False, False)):
        if self.pose is None:
            raise RuntimeError("3PT conditioner must be seeded from the held target")
        elapsed = now - self.last_time
        if not np.isfinite(elapsed) or elapsed < 0:
            self.fault = "invalid control clock"
            # There is no valid interval over which to integrate. Preserve the
            # last valid clock/state so a later valid update can brake from it.
            return self.pose.copy()
        if elapsed == 0:
            return self.pose.copy()
        brief_gap = elapsed > 0.1
        if elapsed >= self.limits.control_gap_timeout_s:
            self.fault = "3PT control gap"
        if live and source_timestamp_ns is not None:
            if source_timestamp_ns <= 0:
                source_fresh = False
            elif self.last_source_timestamp_ns is not None:
                if source_timestamp_ns < self.last_source_timestamp_ns:
                    self.fault = "3PT source clock moved backwards"
                source_fresh &= source_timestamp_ns > self.last_source_timestamp_ns
            if source_fresh:
                self.last_source_timestamp_ns = source_timestamp_ns
        if live:
            if source_fresh:
                self.last_source_time = now
            elif now - self.last_source_time >= self.limits.control_gap_timeout_s:
                self.fault = "stale 3PT input"
        # The derivative state must describe the actual emitted pose/time
        # sequence. Substituting 20 ms after a pause made it inconsistent and
        # caused a large acceleration spike on the following ordinary tick.
        # This is a publisher-sequence bound, not a receiver hold guarantee.
        dt = elapsed
        self.last_time = now
        limits = self.limits
        rejected = False
        previous_raw = self.raw.copy()
        raw_dt = now - self.raw_time
        # Device-clock differences measure fresh-sample motion; control-clock
        # differences still govern output integration and stale-input braking.
        sample_dt = raw_dt
        if live and source_timestamp_ns is not None:
            sample_dt = (None if self.raw_source_timestamp_ns is None else
                         (source_timestamp_ns - self.raw_source_timestamp_ns) * 1e-9)
        accepted = False
        try:
            target = validate_vr_pose(target)
        except (ValueError, TypeError):
            self.fault = "invalid 3PT pose"
            target = self.pose.copy()
        brake_mask = np.asarray(brake_mask, dtype=bool)
        reset_reference = brake_mask | np.asarray(reanchor_mask, dtype=bool)
        # A clutch release discards the old pursuit goal. A new anchor starts
        # at the emitted pose and must not look like a tracking jump or velocity.
        # Preserve actual velocity/acceleration so braking remains continuous.
        self.raw[reset_reference] = self.pose[reset_reference]
        previous_raw[reset_reference] = self.pose[reset_reference]
        self.raw_velocity[:, reset_reference] = 0
        self.target_velocity[:, reset_reference] = 0
        self.velocity_qualified[:, reset_reference] = False
        if live and source_fresh and not self.fault:
            delta = target[:, :3] - self.raw[:, :3]
            angle = rotation_error(target[:, 3:], self.raw[:, 3:])
            # The reference can predate the previous control tick after a
            # rejected frame or a repeated SDK sample. Match its actual age.
            rejected = (np.any(np.linalg.norm(delta, axis=1) > max(limits.jump, 3 * limits.speed * raw_dt)) or
                        np.any(np.linalg.norm(angle, axis=1) > max(limits.angular_jump, 3 * limits.angular_speed * raw_dt)))
            self.rejected = self.rejected + 1 if rejected else 0
            if self.rejected >= 3:
                self.fault = "repeated 3PT tracking jumps"
            if not rejected:
                self.raw = target.copy()
                self.raw_time = now
                accepted = True
        elif not live and not self.fault:
            self.raw = target.copy()
            self.raw_time = now
            self.rejected = 0
            accepted = True

        if accepted:
            self.raw_source_timestamp_ns = source_timestamp_ns if live else None

        # Brief interruptions brake for this update, then follow fresh input
        # again without discarding calibration. Only a full timeout latches.
        braking = brake_mask | (bool(self.fault) or self.rejected > 0 or brief_gap
                               or (live and now - self.last_source_time > 0.1))
        for angular, velocity, acceleration, vmax, amax, jmax in (
            (False, self.velocity, self.acceleration, limits.speed, limits.acceleration, limits.jerk),
            (True, self.angular_velocity, self.angular_acceleration, limits.angular_speed, limits.angular_acceleration, limits.angular_jerk),
        ):
            error = (rotation_error(self.raw[:, 3:], self.pose[:, 3:]) if angular
                     else self.raw[:, :3] - self.pose[:, :3])
            exact_velocity = error / dt
            exact_acceleration = (exact_velocity - velocity) / dt
            if accepted:
                if sample_dt is None or sample_dt <= 0:
                    # A seeded held pose has no device timestamp. Establish a
                    # reference first rather than mixing unrelated clocks.
                    self.raw_velocity[int(angular)] = 0
                    self.target_velocity[int(angular)] = 0
                    self.velocity_qualified[int(angular)] = False
                else:
                    measured_velocity = (rotation_error(self.raw[:, 3:], previous_raw[:, 3:]) if angular
                                         else self.raw[:, :3] - previous_raw[:, :3]) / sample_dt
                    target_acceleration = (measured_velocity - self.raw_velocity[int(angular)]) / sample_dt
                    self.raw_velocity[int(angular)] = measured_velocity
                    self.target_velocity[int(angular)] = measured_velocity
                    # Require two consecutive plausible velocity changes before
                    # retaining feedforward: one quiet interval in noisy tracking
                    # must not drive the following repeated control ticks.
                    qualified = np.linalg.norm(target_acceleration, axis=1) <= amax
                    self.target_velocity[int(angular), ~(qualified & self.velocity_qualified[int(angular)])] = 0
                    self.velocity_qualified[int(angular)] = qualified
            self.target_velocity[int(angular), braking] = 0
            self.velocity_qualified[int(angular), braking] = False
            target_velocity = self.raw_velocity[int(angular)]
            hard_speed = vmax * limits.hard_limit_factor
            hard_acceleration = amax * limits.hard_limit_factor
            direct = ((np.linalg.norm(exact_velocity, axis=1) <= vmax) &
                      (np.linalg.norm(exact_acceleration, axis=1) <= amax) &
                      (np.linalg.norm(exact_acceleration - acceleration, axis=1) <= jmax * dt) &
                      (np.linalg.norm(exact_velocity, axis=1) + np.sum(exact_acceleration ** 2, axis=1) / (2 * jmax) <= hard_speed) &
                      ~braking)
            # Rejoining must allow acceleration to settle as well as matching
            # one pose. Otherwise repeated one-tick snaps create a limit cycle.
            joining_acceleration = (target_velocity - exact_velocity) / dt
            can_join = ((np.linalg.norm(joining_acceleration - exact_acceleration, axis=1) <= jmax * dt) &
                        (np.linalg.norm(joining_acceleration, axis=1) <= min(amax, jmax * dt)))
            direct &= ~self.limited[int(angular)] | can_join
            # Match the latest measured target velocity while closing the
            # existing position error. This avoids a permanent following gap
            # after limiting. No future target is extrapolated.
            # Between fresh samples, retain only the last qualified velocity.
            # Repeated poses are not measurements of zero operator velocity.
            target_velocity = self.target_velocity[int(angular)]
            response = min(12.0, jmax / amax)
            desired_velocity = cap(target_velocity + response / 3 * (error - target_velocity * dt), vmax)
            # Recovery pursues the current pose. Target velocity must not pull
            # the command away from that pose or invent a sideways goal.
            direction = error / np.maximum(np.linalg.norm(error, axis=1, keepdims=True), 1e-12)
            desired_velocity = direction * np.maximum(0, np.sum(desired_velocity * direction, axis=1, keepdims=True))
            desired_velocity[braking] = 0
            desired_jerk = 3 * response ** 2 * (desired_velocity - velocity) - 3 * response * acceleration
            desired_acceleration = cap(acceleration + desired_jerk * dt, amax)
            if brief_gap:
                # After a gap, aim for zero interval velocity, not the missed
                # operator target. The same jerk/acceleration/reserve checks
                # below decide how much braking is feasible; never replay
                # missing ticks or integrate the pursuit controller over them.
                desired_acceleration = cap(-velocity / dt, amax)
            desired_acceleration[direct] = exact_acceleration[direct]
            acceleration[:], self.reserve_active[int(angular)] = limit_acceleration(
                velocity, acceleration, desired_acceleration, dt, hard_speed, hard_acceleration, jmax,
            )
            velocity += acceleration * dt
            self.limited[int(angular)] = ~direct
            if angular:
                self.pose[:, 3:] = (Rotation.from_rotvec(velocity * dt) *
                    Rotation.from_quat(self.pose[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
                self.pose[direct, 3:] = self.raw[direct, 3:]
            else:
                self.pose[:, :3] += velocity * dt
                self.pose[direct, :3] = self.raw[direct, :3]
        return self.pose.copy()

    def reached(self, target):
        return (self.stopped and np.max(np.linalg.norm(target[:, :3] - self.pose[:, :3], axis=1)) < 1e-4
                and np.max(np.linalg.norm(rotation_error(target[:, 3:], self.pose[:, 3:]), axis=1)) < 1e-4)
