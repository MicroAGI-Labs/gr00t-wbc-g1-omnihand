"""Latest-target 3PT conditioning in calibrated Cartesian coordinates."""

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
    max_lag: float = 0.25
    max_angular_lag: float = np.pi / 3
    jump: float = 0.10
    angular_jump: float = np.pi / 6
    cutoff_hz: float = 8.0
    fast_cutoff_hz: float = 32.0
    translation_response_s: float = 0.06

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("3PT limits must be finite and positive")


def cap(vector, maximum):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector * np.minimum(1.0, maximum / np.maximum(norm, 1e-12))


def rotation_error(target, current):
    return (Rotation.from_quat(target, scalar_first=True) *
            Rotation.from_quat(current, scalar_first=True).inv()).as_rotvec()


class VRMotionConditioner:
    def __init__(self, limits=VRMotionLimits()):
        self.limits = limits
        self.pose = None
        self.last_time = None
        self.velocity = np.zeros((3, 3))
        self.drive_velocity = np.zeros((3, 3))
        self.target_velocity = np.zeros((3, 3))
        self.angular_velocity = np.zeros((3, 3))
        self.fault = ""
        self.rejected = 0

    @property
    def stopped(self):
        return max(np.max(np.linalg.norm(self.velocity, axis=1)),
                   np.max(np.linalg.norm(self.drive_velocity, axis=1)),
                   np.max(np.linalg.norm(self.angular_velocity, axis=1))) < 1e-4

    def seed(self, pose, now):
        """Use a known stationary held target, never an uncalibrated Pico frame."""
        self.pose = validate_vr_pose(pose)
        self.filtered = self.pose.copy()
        self.raw = self.pose.copy()
        self.velocity[:] = self.angular_velocity[:] = 0
        self.drive_velocity[:] = 0
        self.target_velocity[:] = 0
        self.last_time = now
        self.fault = ""
        self.rejected = 0

    def update(self, target, now, *, live=True):
        if self.pose is None:
            raise RuntimeError("3PT conditioner must be seeded from the held target")
        elapsed = now - self.last_time
        if not np.isfinite(elapsed) or elapsed < 0:
            self.fault = "invalid control clock"
            elapsed = 0.02
        if elapsed == 0:
            return self.pose.copy()
        if elapsed > 0.1:
            self.fault = "3PT control gap"
        dt = min(elapsed, 0.02 if elapsed > 0.1 else elapsed)
        self.last_time = now
        limits = self.limits
        previous_filtered = self.filtered.copy()
        rejected = False
        try:
            target = validate_vr_pose(target)
        except (ValueError, TypeError):
            self.fault = "invalid 3PT pose"
            target = self.pose.copy()
        if live and not self.fault:
            delta = target[:, :3] - self.raw[:, :3]
            angle = rotation_error(target[:, 3:], self.raw[:, 3:])
            rejected = (np.any(np.linalg.norm(delta, axis=1) > max(limits.jump, 3 * limits.speed * dt)) or
                        np.any(np.linalg.norm(angle, axis=1) > max(limits.angular_jump, 3 * limits.angular_speed * dt)))
            self.rejected = self.rejected + 1 if rejected else 0
            if self.rejected >= 3:
                self.fault = "repeated 3PT tracking jumps"
            if not rejected:
                # Adaptive low pass: quiet tracking is smoothed more than fast motion.
                for linear, change, maximum in ((True, delta, limits.speed), (False, angle, limits.angular_speed)):
                    activity = np.clip(np.linalg.norm(change, axis=1) / (dt * maximum), 0, 1)
                    cutoff = limits.cutoff_hz + activity * (limits.fast_cutoff_hz - limits.cutoff_hz)
                    alpha = (dt / (dt + 1 / (2 * np.pi * cutoff)))[:, None]
                    if linear:
                        self.filtered[:, :3] += alpha * (target[:, :3] - self.filtered[:, :3])
                    else:
                        error = rotation_error(target[:, 3:], self.filtered[:, 3:])
                        self.filtered[:, 3:] = (Rotation.from_rotvec(alpha * error) *
                            Rotation.from_quat(self.filtered[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
                self.raw = target.copy()
                # A slow robot can legitimately lag behind the operator. Bound
                # the pursuit goal without faulting or changing calibration;
                # each sample replaces the old goal, including on reversals.
                self.filtered[:, :3] = self.pose[:, :3] + cap(
                    self.filtered[:, :3] - self.pose[:, :3], limits.max_lag
                )
                lookahead = cap(rotation_error(self.filtered[:, 3:], self.pose[:, 3:]), limits.max_angular_lag)
                self.filtered[:, 3:] = (Rotation.from_rotvec(lookahead) *
                    Rotation.from_quat(self.pose[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        elif not live and not self.fault:
            self.filtered = target.copy()
            self.raw = target.copy()

        braking = bool(self.fault) or rejected
        for angular, velocity, vmax, amax in (
            (False, self.velocity, limits.speed, limits.acceleration),
            (True, self.angular_velocity, limits.angular_speed, limits.angular_acceleration),
        ):
            error = (rotation_error(self.filtered[:, 3:], self.pose[:, 3:]) if angular
                     else self.filtered[:, :3] - self.pose[:, :3])
            distance = np.linalg.norm(error, axis=1, keepdims=True)
            # Leave enough distance to brake on subsequent discrete ticks.
            braking_speed = np.sqrt((amax * dt) ** 2 + 2 * amax * distance) - amax * dt
            feedforward = (rotation_error(self.filtered[:, 3:], previous_filtered[:, 3:]) / dt if angular
                           else (self.filtered[:, :3] - previous_filtered[:, :3]) / dt)
            desired = cap(feedforward + cap(error / dt, braking_speed), vmax)
            if not angular:
                # Differentiating noisy samples directly into a saturated
                # velocity drive can cause chatter and directional drift.
                self.target_velocity += -np.expm1(-dt / limits.translation_response_s) * (
                    feedforward - self.target_velocity
                )
                # Damped position/velocity tracking instead of switching between
                # full acceleration and braking near the translation endpoint.
                desired = cap(self.target_velocity + 4.0 * error, vmax)
            if braking:
                desired[:] = 0
            if angular:
                velocity += cap(desired - velocity, amax * dt)
                self.pose[:, 3:] = (Rotation.from_rotvec(velocity * dt) *
                    Rotation.from_quat(self.pose[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
            else:
                # Slew a bounded drive velocity, then integrate its continuous
                # first-order response exactly over this tick. Unlike clipping
                # each position step, this keeps translation acceleration
                # continuous through corners, reversals, and speed saturation.
                tau = limits.translation_response_s
                drive_start = self.drive_velocity.copy()
                # Ease into the slew limit as well: alternating tracking noise
                # must not repeatedly saturate it and accumulate a position bias.
                drive_response = 1.0 if braking else -np.expm1(-2 * dt / tau)
                self.drive_velocity += cap(drive_response * (desired - drive_start), amax * dt)
                drive_acceleration = (self.drive_velocity - drive_start) / dt
                transient = velocity - drive_start + tau * drive_acceleration
                decay = np.exp(-dt / tau)
                self.pose[:, :3] += (drive_start * dt +
                    drive_acceleration * (0.5 * dt * dt - tau * dt) +
                    transient * tau * (-np.expm1(-dt / tau)))
                velocity[:] = self.drive_velocity - tau * drive_acceleration + transient * decay
        return self.pose.copy()

    def reached(self, target):
        return (self.stopped and np.max(np.linalg.norm(target[:, :3] - self.pose[:, :3], axis=1)) < 1e-4
                and np.max(np.linalg.norm(rotation_error(target[:, 3:], self.pose[:, 3:]), axis=1)) < 1e-4)
