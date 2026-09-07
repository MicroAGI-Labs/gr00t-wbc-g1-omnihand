# Pico SMPL stream server for body tracking visualization

"""

# Recommended Command Line Arguments:
    # With VR3 PT visualization (by --vis_vr3pt) and optional SMPL body visualization (by --vis_smpl)
    # If you want to enable waist tracking in the VR3 PT visualization, please add --waist_tracking
    python pico_manager_thread_server.py --manager \
        --vis_vr3pt --vis_smpl \
        --waist_tracking

    # VR3 PT visualization only (without SMPL body) — lower latency
    python pico_manager_thread_server.py --manager --vis_vr3pt

# DEBUG VR3 PT VISUALIZATION:
    # A standalone test mode that captures one live frame and visualizes it.
    python pico_manager_thread_server.py --vr3pt_live

# TIMING COMPARISON:
    # The visualizer automatically reports timing every 5 seconds when running:
    #   [Vis Timing] vr3pt: X.XXms | smpl: X.XXms | render: X.XXms | vr3pt_only: X.XXms | both(vr3pt+smpl): X.XXms

"""

from collections import defaultdict, deque
from enum import Enum, IntEnum
import os
import socket
import subprocess
import threading
import time

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation as sRot
import torch
import zmq

from gear_sonic.end_effectors.controller import TriggerHysteresis
from gear_sonic.end_effectors.protocol import (
    DEFAULT_HAND_INTENT_PORT,
    HAND_INTENT_SCHEMA,
    HAND_INTENT_TOPIC,
    encode as encode_hand_message,
)
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)
from gear_sonic.utils.teleop import input_readers
from gear_sonic.utils.teleop.gesture_trackers import (
    DoublePressTracker,
    recording_face_action,
)
from gear_sonic.utils.teleop.pose_transition import (
    IDLE_BASE_UPPER_BODY_MASK,
    IDLE_BASE_UPPER_BODY_RAD,
    UPPER_BODY_WIDTH,
    JointPoseTransition,
)
from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller

try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        build_command_message,
        build_planner_message,
        pack_pose_message,
    )
except ImportError:

    def build_command_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_command_message unavailable")

    def build_planner_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_planner_message unavailable")

    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")


try:
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    print("Warning: gear_sonic.isaac_utils.rotations not available.")
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
        G1GripperInverseKinematicsSolver,
    )
except ImportError:
    print("Warning: G1GripperInverseKinematicsSolver not available.")
    G1GripperInverseKinematicsSolver = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer
except ImportError:
    print("Warning: VR3PtPoseVisualizer not available (pyvista may not be installed).")
    VR3PtPoseVisualizer = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses
except ImportError:
    print("Warning: get_g1_key_frame_poses not available (pyvista may not be installed).")
    get_g1_key_frame_poses = None


class LocomotionMode(IntEnum):
    """Locomotion mode enum for robot movement."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19


class StreamMode(Enum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_IDLE_BASE_POSE = 3
    # Wire-compatible alias for older tools and recorded metadata.
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5
    PLANNER_IK_UPPER = 6


PLANNER_STREAM_MODES = frozenset(
    {
        StreamMode.PLANNER,
        StreamMode.PLANNER_IDLE_BASE_POSE,
        StreamMode.PLANNER_VR_3PT,
        StreamMode.PLANNER_IK_UPPER,
    }
)

DEFAULT_TELEOP_CONTROL_PORT = 5573


class FaceChordTracker:
    """Confirm a two-button face chord only after the whole gesture is released.

    Any third face button cancels the candidate. This lets the four-button
    policy chord be pressed and released in any order without leaking one of
    its two-button subsets into mode selection or data collection.
    """

    CHORDS = {
        frozenset(("a", "x")): "ax",
        frozenset(("b", "y")): "by",
        frozenset(("x", "b")): "xb",
        frozenset(("y", "a")): "ya",
        frozenset(("a", "b")): "ab",
        frozenset(("x", "y")): "xy",
    }

    def __init__(self) -> None:
        self._active = False
        self._candidate: str | None = None
        self._cancelled = False

    def reset(self) -> None:
        self._active = False
        self._candidate = None
        self._cancelled = False

    def update(self, a: bool, b: bool, x: bool, y: bool) -> str | None:
        pressed = frozenset(
            name for name, down in (("a", a), ("b", b), ("x", x), ("y", y)) if down
        )
        if not pressed:
            confirmed = self._candidate if self._active and not self._cancelled else None
            self.reset()
            return confirmed

        if not self._active:
            self._active = True

        if len(pressed) > 2:
            self._cancelled = True
            self._candidate = None
        elif len(pressed) == 2 and not self._cancelled:
            chord = self.CHORDS.get(pressed)
            if self._candidate is None:
                self._candidate = chord
            elif chord != self._candidate:
                self._cancelled = True
                self._candidate = None
        return None


class HandIntentStream:
    """Publish independent, freshness-qualified close intent for both hands."""

    def __init__(self) -> None:
        # A controller can outlive and reconnect to this publisher. Seed from
        # the host monotonic clock so a restarted streamer never replays a
        # lower sequence that the safety controller must reject as stale.
        self.sequence = time.monotonic_ns()
        self.last_source_timestamp_ns: int | None = None
        self.hysteresis = TriggerHysteresis()

    def publish(self, socket, reader, *, hold: bool = False) -> None:
        _, left_trigger, right_trigger, left_grip, right_grip = get_controller_inputs(reader)
        try:
            source_timestamp_ns = int(reader.get_timestamp_ns())
        except Exception:
            source_timestamp_ns = 0
        valid = source_timestamp_ns > 0 and source_timestamp_ns != self.last_source_timestamp_ns
        if valid:
            self.last_source_timestamp_ns = source_timestamp_ns
        # Preserve the familiar teleop fist control on the side grip while
        # also accepting the index trigger. The external hand protocol is a
        # binary open/close contract, so the stronger input is the close
        # demand passed through hysteresis.
        left_close = float(np.clip(max(left_trigger, left_grip), 0.0, 1.0))
        right_close = float(np.clip(max(right_trigger, right_grip), 0.0, 1.0))
        left_closed = self.hysteresis.update("left", left_close, valid)
        right_closed = self.hysteresis.update("right", right_close, valid)
        self.sequence += 1
        socket.send(
            encode_hand_message(
                HAND_INTENT_TOPIC,
                {
                    "schema": HAND_INTENT_SCHEMA,
                    "sequence": self.sequence,
                    "monotonic_ns": time.monotonic_ns(),
                    "source": "pico",
                    "hold": hold,
                    "left": {
                        "valid": valid,
                        "closed": left_closed,
                        "trigger": left_close,
                    },
                    "right": {
                        "valid": valid,
                        "closed": right_closed,
                        "trigger": right_close,
                    },
                },
            )
        )


### Parse 3 point pose from SMPL
#
# OFFSETS: Rotation corrections applied to each keypoint to align SMPL joint frames
# with the desired robot/visualization coordinate convention.
#
# Index mapping (based on [0, 22, 23, 12].index(joint_id)):
#   - OFFSETS[0]: Root/Pelvis (joint 0)
#   - OFFSETS[1]: Left Wrist (joint 22)
#   - OFFSETS[2]: Right Wrist (joint 23)
#   - OFFSETS[3]: Neck (joint 12) - more stable than Head (joint 15) for body tracking
#
# Scipy euler rotation convention:
#   - Lowercase "xyz" = EXTRINSIC rotations (about the FIXED/ORIGINAL frame's axes)
#   - Uppercase "XYZ" = INTRINSIC rotations (about the ROTATING body's axes)
#
# For EXTRINSIC "xyz" with angles [a, b, c]:
#   All rotations are about the ORIGINAL frame's axes (before any rotation):
#     R_total = R_z(c) @ R_y(b) @ R_x(a)   (matrix multiplication order)
#   Applied as: first rotate 'a' about original X, then 'b' about original Y, then 'c' about original Z
#
# For INTRINSIC "XYZ" with angles [a, b, c]:
#   Each rotation is about the CURRENT (rotated) frame's axis:
#     R_total = R_x(a) @ R_y(b) @ R_z(c)   (matrix multiplication order)
#   Applied as: first rotate 'a' about X, then 'b' about NEW Y, then 'c' about NEW Z
#
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root: yaw -90° about fixed Z
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist: roll +90° about fixed X
    sRot.from_euler(
        "xyz", [-90, 0, 180], degrees=True
    ),  # R-Wrist: roll -90° about fixed X, then yaw 180° about fixed Z
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck: yaw -90° about fixed Z
]


def _compute_rel_transform(pose, world_frame, scalar_first=True):
    """
    Transform a pose from Unity coordinate frame to robot coordinate frame.

    Args:
        pose: np.ndarray shape (7,) - [x, y, z, qx, qy, qz, qw] in Unity frame
        world_frame: np.ndarray shape (7,) - reference frame to compute relative transform
        scalar_first: bool - if True, quaternion is [qw, qx, qy, qz]; if False, [qx, qy, qz, qw]

    Returns:
        rel_pos: np.ndarray (3,) - position in robot frame
        rel_rot: np.ndarray (4,) - quaternion [qw, qx, qy, qz] in robot frame

    Coordinate transform matrix Q converts Unity (Y-up, left-handed) to Robot (Z-up, right-handed):
        Unity:  X-right, Y-up, Z-forward
        Robot:  X-forward, Y-left, Z-up
    """
    world_frame = world_frame.copy()

    # Q transforms Unity coordinates to Robot coordinates
    # Unity [x, y, z] -> Robot [-x, z, y]
    Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
    pose[:3] = Q @ pose[:3]
    world_frame[:3] = Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


def _process_3pt_pose(smpl_pose_np):
    """
    Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full SMPL body joint poses.

    NOTE: We use Neck (joint 12) instead of Head (joint 15) because:
      - Neck is more rigidly coupled to the torso
      - Head has high DoF (looking around) which doesn't reflect body pose
      - Neck provides more stable tracking for upper body orientation

    Args:
        smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints, each [x, y, z, qx, qy, qz, qw]
                      in Unity frame (scalar-last quaternion format)

    Returns:
        vr_3pt_pose: np.ndarray shape (3, 7) - 3 keypoints in robot frame
                     Each row is [x, y, z, qw, qx, qy, qz] (scalar-FIRST quaternion format)
                     Row 0: Left Wrist (SMPL joint 22)
                     Row 1: Right Wrist (SMPL joint 23)
                     Row 2: Neck (SMPL joint 12)

                     IMPORTANT: Positions and orientations are RELATIVE TO ROOT (pelvis).

    Processing Steps:
        1. Transform all 24 joints from Unity frame to robot frame
        2. Extract 4 keypoints: Root(0), L-Wrist(22), R-Wrist(23), Neck(12)
        3. Apply per-joint rotation OFFSETS to align joint frames
        4. Make L-Wrist, R-Wrist, Neck relative to Root (both position and orientation)
        5. Return only the 3 non-root keypoints

    Note: Position calibration (wrist offsets, neck kinematic chain) is done in
          ThreePointPose.apply_calibration() to ensure consistency with calibrated
          orientations.
    """

    # Defensive copy: _compute_rel_transform modifies pose[:3] in-place, which would
    # corrupt the caller's array (e.g. PicoReader._latest) and cause wrong results
    # if the same sample is processed more than once.
    smpl_pose_np = smpl_pose_np.copy()

    # =========================================================================
    # STEP 1: Transform all joints from Unity frame to robot frame
    # =========================================================================
    # Input: smpl_pose_np[i] = [x, y, z, qx, qy, qz, qw] in Unity frame (scalar-last)
    # Output: body_poses[i] = [x, y, z, qw, qx, qy, qz] in robot frame (scalar-first)
    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos  # Position in robot frame
        body_poses[i, 3:] = orn  # Quaternion [qw, qx, qy, qz] in robot frame

    # =========================================================================
    # STEP 2 & 3: Extract 4 keypoints and apply rotation OFFSETS
    # =========================================================================
    # We only care about these SMPL joint indices:
    #   - Joint 0:  Root/Pelvis (reference frame)
    #   - Joint 22: Left Wrist
    #   - Joint 23: Right Wrist
    #   - Joint 12: Neck (more stable than Head joint 15)
    #
    # kp_poses maps these to indices 0, 1, 2, 3 respectively
    positions = np.array([[p[0], p[1], p[2]] for p in body_poses])
    kp_poses = np.zeros((4, 7), dtype=np.float32)

    for i, pose in enumerate(body_poses):
        if i not in [0, 22, 23, 12]:
            continue  # Skip joints we don't care about

        pos = positions[i]

        # Map SMPL joint index to our keypoint index (0-3)
        # rel_i: 0=Root, 1=L-Wrist, 2=R-Wrist, 3=Neck
        rel_i = [0, 22, 23, 12].index(i)

        # Extract quaternion and apply rotation offset
        # pose[3:7] is [qw, qx, qy, qz] (scalar-first from _compute_rel_transform)
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])

        # Apply offset: new_rotation = original_rotation * OFFSET
        # This post-multiplies the offset (intrinsic rotation)
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )

        kp_poses[rel_i, 3:] = rot_quat  # Store as scalar-last temporarily for scipy compatibility
        kp_poses[rel_i, :3] = pos

    # =========================================================================
    # STEP 4: Make positions and orientations RELATIVE TO ROOT
    # =========================================================================
    # This transforms everything into the root's local coordinate frame.
    # After this step:
    #   - Root's position would be (0,0,0) and orientation identity (but we don't return root)
    #   - Other keypoints are expressed relative to root
    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()  # Still scalar-last for scipy

    for i in range(1, 4):
        # Position: subtract root position, then rotate by inverse of root orientation
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)

        # Orientation: compute relative rotation (root_inv * keypoint_rot)
        # Result stored as scalar-FIRST [qw, qx, qy, qz]
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    # =========================================================================
    # STEP 5: Return only L-Wrist, R-Wrist, Neck (skip Root)
    # =========================================================================
    # NOTE: Position and orientation calibration (including neck position via kinematic
    #       chain) is done in ThreePointPose.apply_calibration() to ensure consistency
    #       between calibrated orientation and computed neck position.
    # kp_poses[1:] = indices 1, 2, 3 = L-Wrist, R-Wrist, Neck
    # Each row: [x, y, z, qw, qx, qy, qz] relative to root, scalar-first quaternion
    return kp_poses[1:]


# =============================================================================
# VR 3-Point Pose Visualization Functions
# =============================================================================


def run_vr3pt_visualizer_test():
    """
    Standalone test for VR 3-point pose visualizer using PyVista.
    Run this to verify the reference frames are displayed correctly.
    """
    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Visualizer Test (PyVista)")
    print("=" * 60)
    print("\nExpected reference frames (all with RGB axes for XYZ):")
    print("  1. WHITE ball at origin (0, 0, 0) - World frame")
    print("  2. CYAN ball at (0, 0, 0.35) - Looking forward (identity)")
    print("  3. MAGENTA ball at (0, 0.4, 0.25) - Looking left (yaw +90°)")
    print("  4. YELLOW ball at (0.4, 0, 0.15) - Looking down (pitch +90°)")
    print("\nClose the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_static()


def run_vr3pt_live_visualizer():
    """
    Live visualizer for real VR 3-point pose data from Pico.
    Captures one frame from Pico and displays it alongside reference frames.
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use live visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Live Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Capturing VR 3-point pose...")

    # Capture body poses and compute vr_3pt_pose
    body_poses = xrt.get_body_joints_pose()
    body_poses_np = np.array(body_poses)

    # Process to get 3-point pose (L-Wrist, R-Wrist, Neck)
    vr_3pt_pose = _process_3pt_pose(body_poses_np)

    print(f"\nCaptured vr_3pt_pose shape: {vr_3pt_pose.shape}")
    print(f"  L-Wrist: pos={vr_3pt_pose[0, :3]}, quat_wxyz={vr_3pt_pose[0, 3:]}")
    print(f"  R-Wrist: pos={vr_3pt_pose[1, :3]}, quat_wxyz={vr_3pt_pose[1, 3:]}")
    print(f"  Neck:    pos={vr_3pt_pose[2, :3]}, quat_wxyz={vr_3pt_pose[2, 3:]}")

    print("\nDisplaying visualization...")
    print("Close the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_with_vr_pose(vr_3pt_pose)


def run_vr3pt_realtime_visualizer(update_hz: int = 10):
    """
    Real-time visualizer for VR 3-point pose data from Pico.
    Continuously updates the visualization with live data.

    Args:
        update_hz: Update rate in Hz (default 10)
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use realtime visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Real-time Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Starting real-time visualization...")
    print(f"Update rate: {update_hz} Hz")
    print("Close the window or press 'q' to exit.")
    print("=" * 60)

    # Use the VR3PtPoseVisualizer for real-time visualization with G1 robot
    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.create_realtime_plotter(interactive=True)

    try:
        while visualizer.is_open:
            # Get new data from Pico
            body_poses = xrt.get_body_joints_pose()
            body_poses_np = np.array(body_poses)
            vr_3pt_pose = _process_3pt_pose(body_poses_np)

            # Update visualization
            visualizer.update_vr_poses(vr_3pt_pose)
            visualizer.render()

            time.sleep(1.0 / update_hz)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()


def process_smpl_joints(body_pose, global_orient, transl):
    """Process SMPL parameters to compute local joints.

    Args:
        body_pose: Body pose tensor, shape (T, 69)
        global_orient: Global orientation tensor, shape (T, 3)
        transl: Translation tensor, shape (T, 3)

    Returns:
        Dictionary with processed joints and parameters
    """
    # Convert global_orient to quaternion and apply transformations (robust if utils missing)
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    if smpl_root_ytoz_up is not None:
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    # Compute joints and vertices using SMPL model (single forward pass)
    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
    )  # (*, 24, 3)

    # Apply base rotation removal and compute local joints
    if remove_smpl_base_rot is not None:
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)
    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)
    global_orient_6d = global_orient_mat[..., :2].reshape(1, 6)

    return {
        "smpl_pose": body_pose,
        "joints": joints,
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
        "global_orient_6d": global_orient_6d,
        "adjusted_transl": transl,
    }


def generate_finger_data(hand: str, trigger: float, grip: float) -> np.ndarray:
    """
    Generate finger position data from Pico controller button states.

    Args:
        hand: "left" or "right"
        trigger: Trigger button value (0-1)
        grip: Grip button value (0-1)

    Returns:
        Array of shape [25, 4, 4] representing fingertip positions
    """
    fingertips = np.zeros([25, 4, 4])

    thumb = 0
    middle = 10
    # Control thumb based on shoulder button state (index 4 is thumb tip)
    fingertips[4 + thumb, 0, 3] = 1.0  # open thumb
    if trigger > 0.5:
        fingertips[4 + middle, 0, 3] = 1.0  # close middle

    return fingertips


# Joystick deadzone threshold
JOYSTICK_DEADZONE = 0.15


class YawAccumulator:
    """Accumulates yaw heading angle based on joystick input."""

    def __init__(self, yaw_gain: float = 1.5, deadzone: float = JOYSTICK_DEADZONE):
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()

    def reset(self):
        """Reset facing direction to default (1,0,0)."""
        self.heading = [1.0, 0.0, 0.0]
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0
        print("YawAccumulator: reset yaw angle to 0.0")

    def yaw_angle(self) -> float:
        """Get current yaw angle in radians."""
        return self.yaw_angle_rad

    def yaw_angle_change(self) -> float:
        """Get current yaw angle change in radians."""
        return self.dyaw

    def update(self, rx: float, dt: float) -> list[float]:
        """
        Update facing direction based on right stick x-axis input.

        Args:
            rx: Right stick x-axis value (-1 to 1)
            dt: Time delta in seconds

        Returns:
            Facing direction as [x, y, 0.0]
        """
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = [np.cos(self.yaw_angle_rad), np.sin(self.yaw_angle_rad), 0.0]
        return self.heading


def compute_from_body_poses(parent_indices: list, device, body_poses_np: np.ndarray):
    """
    Compute local joints and body orientation from provided body_poses_np.
    """
    positions = body_poses_np[:, :3]
    global_quats = body_poses_np[:, [6, 3, 4, 5]]

    # Convert to local rotations
    global_rots = sRot.from_quat(global_quats, scalar_first=True)
    global_rots = global_rots * sRot.from_euler("y", 180, degrees=True)

    local_rots = []
    for i in range(24):
        if parent_indices[i] == -1:
            local_rots.append(global_rots[i])
        else:
            local_rot = global_rots[parent_indices[i]].inv() * global_rots[i]
            local_rots.append(local_rot)

    pose_aa = np.array([rot.as_rotvec() for rot in local_rots])

    body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(device).unsqueeze(0)
    global_orient = torch.from_numpy(pose_aa[0]).float().to(device).unsqueeze(0)
    transl = torch.from_numpy(positions[0]).float().to(device).unsqueeze(0)

    return process_smpl_joints(body_pose, global_orient, transl)


# def compute_latest_frame(parent_indices: list, device) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Pull body data from XRoboToolkit, compute local SMPL joints and body orientation.
#     Returns (smpl_joints_local_np [24,3], global_orient_quat_np [4,])
#     """
#     body_poses = xrt.get_body_joints_pose()
#     body_poses_np = np.array(body_poses)
#     return compute_from_body_poses(parent_indices, device, body_poses_np)


def init_hand_ik_solvers():
    """Initialize hand IK solvers if available."""
    if G1GripperInverseKinematicsSolver is not None:
        left_solver = G1GripperInverseKinematicsSolver(side="left")
        right_solver = G1GripperInverseKinematicsSolver(side="right")
        print("Hand IK solvers initialized")
        return left_solver, right_solver
    print("Warning: Hand IK solvers not available")
    return None, None


# Readers that expose `get_controller_data()` returning the IsaacTeleop
# controller_data dict schema (left/right trigger/squeeze, thumbstick, clicks).
# Tuple form keeps the dispatch sites uniform if/when a second reader speaks
# the same schema.
_ISAAC_TELEOP_READERS = (input_readers.IsaacTeleopReader,)


def get_controller_inputs(reader=None):
    """Fetch controller button/trigger states from XRoboToolkit or IsaacTeleop."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return False, 0.0, 0.0, 0.0, 0.0
        return (
            False,
            float(ctrl.get("left_trigger_value", 0.0)),
            float(ctrl.get("right_trigger_value", 0.0)),
            float(ctrl.get("left_squeeze_value", 0.0)),
            float(ctrl.get("right_squeeze_value", 0.0)),
        )
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    left_menu_button = xrt.get_left_menu_button()
    return left_menu_button, left_trigger, right_trigger, left_grip, right_grip


def get_controller_axes(reader=None):
    """Fetch joystick axes (lx, ly, rx, ry). Falls back to zeros if not available."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return 0.0, 0.0, 0.0, 0.0
        left_thumbstick = ctrl.get("left_thumbstick", [0.0, 0.0])
        right_thumbstick = ctrl.get("right_thumbstick", [0.0, 0.0])
        return (
            float(left_thumbstick[0]),
            float(left_thumbstick[1]),
            float(right_thumbstick[0]),
            float(right_thumbstick[1]),
        )
    if xrt is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        left_axis = xrt.get_left_axis()  # expected [x, y]
        right_axis = xrt.get_right_axis()  # expected [x, y]
        lx = float(left_axis[0]) if len(left_axis) >= 1 else 0.0
        ly = float(left_axis[1]) if len(left_axis) >= 2 else 0.0
        rx = float(right_axis[0]) if len(right_axis) >= 1 else 0.0
        ry = float(right_axis[1]) if len(right_axis) >= 2 else 0.0
        return lx, ly, rx, ry
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def get_menu_buttons(reader=None):
    """Fetch both menu buttons (left, right). Falls back to False if not available."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        return False, False
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_menu_button")
    right = _safe_btn("get_right_menu_button")
    return left, right


def get_axis_clicks(reader=None):
    """Fetch both axis click buttons (left, right). Falls back to False if not available."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return False, False
        return (
            float(ctrl.get("left_thumbstick_click", 0.0)) > 0.5,
            float(ctrl.get("right_thumbstick_click", 0.0)) > 0.5,
        )
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_axis_click")
    right = _safe_btn("get_right_axis_click")
    return left, right


def get_face_buttons(reader=None):
    """Fetch primary face buttons A and X. Returns (a_pressed, x_pressed)."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return False, False
        return (
            float(ctrl.get("right_primary_click", 0.0)) > 0.5,
            float(ctrl.get("left_primary_click", 0.0)) > 0.5,
        )
    if xrt is None:
        return False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        x_pressed = bool(xrt.get_X_button())
        return a_pressed, x_pressed
    except Exception:
        return False, False


def get_abxy_buttons(reader=None):
    """Fetch A,B,X,Y face buttons as booleans (a,b,x,y)."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return False, False, False, False
        return (
            float(ctrl.get("right_primary_click", 0.0)) > 0.5,
            float(ctrl.get("right_secondary_click", 0.0)) > 0.5,
            float(ctrl.get("left_primary_click", 0.0)) > 0.5,
            float(ctrl.get("left_secondary_click", 0.0)) > 0.5,
        )
    if xrt is None:
        return False, False, False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        b_pressed = bool(xrt.get_B_button())
        x_pressed = bool(xrt.get_X_button())
        y_pressed = bool(xrt.get_Y_button())
        return a_pressed, b_pressed, x_pressed, y_pressed
    except Exception:
        return False, False, False, False


def compute_hand_joints_from_inputs(
    left_solver, right_solver, left_trigger, left_grip, right_trigger, right_grip
) -> tuple[np.ndarray, np.ndarray]:
    """Compute left/right hand joints using IK solvers, or zeros if unavailable."""
    if left_solver is not None and right_solver is not None:
        left_finger_data = generate_finger_data("left", left_trigger, left_grip)
        right_finger_data = generate_finger_data("right", right_trigger, right_grip)
        left_hand_joints = left_solver({"position": left_finger_data})
        right_hand_joints = right_solver({"position": right_finger_data})
    else:
        left_hand_joints = np.zeros((1, 7), dtype=np.float32)
        right_hand_joints = np.zeros((1, 7), dtype=np.float32)
    return left_hand_joints, right_hand_joints


def _quat_lerp_normalized(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linear interpolate two quaternions and renormalize. Input shape (4,), xyzw order.
    Ensures shortest path by flipping sign if dot < 0.
    """
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q


def _interp_pose_axis_angle(
    prev_pose: np.ndarray, curr_pose: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Interpolate axis-angle joint poses by converting to quats, lerp-normalize, then back.
    prev_pose, curr_pose: (21,3) axis-angle (rotvec)
    Returns (21,3) axis-angle.
    """
    prev_quats = sRot.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()  # (N,4) xyzw
    curr_quats = sRot.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for i in range(prev_quats.shape[0]):
        out_quats[i] = _quat_lerp_normalized(prev_quats[i], curr_quats[i], alpha)
    out_pose = sRot.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape)
    return out_pose


class PicoReader:
    """
    Background reader that pulls Pico/XRT data as fast as possible and computes dt/FPS.
    """

    STALE_TIMEOUT = 2.0

    def __init__(self, max_queue_size: int = 15):
        del max_queue_size
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()
        self._last_new_data_time = time.monotonic()
        self._disconnected = threading.Event()

    def start(self):
        if self._thread.is_alive():
            return
        self._stop.clear()
        self._last_new_data_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    @property
    def disconnected(self) -> bool:
        return self._disconnected.is_set()

    def clear_disconnect(self):
        self._disconnected.clear()
        self._last_new_data_time = time.monotonic()
        self._last_stamp_ns = None
        self._fps_ema = 0.0

    def reconnect(self) -> None:
        """Reconnect the native XRT client without replacing this reader object."""
        self.stop()
        with self._lock:
            self._latest = None
        _close_xrt()
        _connect_xrt_body_stream()
        self.clear_disconnect()
        self.start()

    def get_timestamp_ns(self) -> int:
        if xrt is None:
            return 0
        return int(xrt.get_time_stamp_ns())

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            try:
                body_available = xrt.is_body_data_available()
            except Exception as exc:
                print(f"[PicoReader] availability error: {exc}")
                body_available = False

            if not body_available:
                self._flag_disconnect_if_stale("No body data")
                time.sleep(0.001)
                continue
            try:
                stamp_ns = xrt.get_time_stamp_ns()
            except Exception as exc:
                print(f"[PicoReader] timestamp error: {exc}")
                self._flag_disconnect_if_stale("Timestamp read failed")
                time.sleep(0.01)
                continue
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                self._flag_disconnect_if_stale("Body timestamps stopped")
                time.sleep(0.000001)
                continue
            # Compute device-based dt/fps using timestamp deltas (ns -> s)
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns
            t_realtime = time.time()
            t_monotonic = time.monotonic()
            try:
                body_poses = xrt.get_body_joints_pose()

                sample = {
                    "body_poses_np": np.array(body_poses),
                    "timestamp_realtime": t_realtime,
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample
                self._last_new_data_time = time.monotonic()
                if self._disconnected.is_set():
                    print("[PicoReader] Fresh body data received; connection restored")
                    self._disconnected.clear()
                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] dt_ts: {device_dt*1000.0:.2f} ms, fps: {self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as e:
                print(f"[PicoReader] read error: {e}")

    def _flag_disconnect_if_stale(self, reason: str) -> None:
        if (
            time.monotonic() - self._last_new_data_time > self.STALE_TIMEOUT
            and not self._disconnected.is_set()
        ):
            print(f"[PicoReader] {reason} for {self.STALE_TIMEOUT:.1f}s; reconnecting")
            self._disconnected.set()


def _pose_stream_common(
    socket,
    hand_socket,
    buffer_size: int,
    num_frames_to_send: int,
    target_fps: int,
    use_cuda: bool,
    record_dir: str,
    record_format: str,
    stop_event: threading.Event | None = None,
    log_prefix: str = "PoseLoop",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    reader=None,
):
    """Shared pose streaming loop used by run_pico."""
    if reader is None:
        if xrt is None:
            raise ImportError(
                "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run pose streaming."
            )

        # Create reader and start it
        reader = PicoReader(max_queue_size=buffer_size)
        reader.start()

    # Create 3-point pose processor with visualization settings
    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix=log_prefix,
    )

    streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix=log_prefix,
    )

    if stop_event is None:
        stop_event = threading.Event()
    hand_intent = HandIntentStream()

    try:
        while not stop_event.is_set():
            streamer.run_once()
            hand_intent.publish(hand_socket, reader)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup resources
        reader.stop()
        three_point.close()


class ThreePointPose:
    """
    Encapsulates everything around calculating 3-point pose from SMPL input.

    This includes:
    - Processing SMPL poses to extract 3-point VR pose (L-Wrist, R-Wrist, Neck)
    - Calibration logic to align VR poses with G1 robot
    - Optional visualization of 3-point poses

    Calibration is done in two steps:
    1. Neck orientation: Captures initial neck orientation to align subsequent poses as upright
    2. Wrist positions: Aligns wrist positions to match G1 robot key frame positions
    """

    # Kinematic chain constants for neck position (matches VR3PtPoseVisualizer)
    TORSO_LINK_OFFSET_Z = 0.05  # meters from root to torso_link
    NECK_LINK_LENGTH = 0.35  # meters from torso_link to neck along neck's local Z

    def __init__(
        self,
        enable_vis_vr3pt: bool = False,
        with_g1_robot: bool = True,
        enable_waist_tracking: bool = False,
        enable_smpl_vis: bool = False,
        log_prefix: str = "ThreePointPose",
        robot_model=None,
    ):
        """
        Initialize 3-point pose processor.

        Args:
            enable_vis_vr3pt: Whether to enable VR 3pt pose visualization (requires display)
            with_g1_robot: Whether to include G1 robot in visualization
            enable_waist_tracking: Whether to enable waist tracking in visualization
            enable_smpl_vis: Whether to render SMPL body joints in the VR3pt visualizer
            log_prefix: Prefix for log messages
            robot_model: Optional pre-instantiated RobotModel. If None, will create one.
                        Used for FK-based calibration (no display required).
        """
        self.log_prefix = log_prefix
        self.with_g1_robot = with_g1_robot
        self.enable_waist_tracking = enable_waist_tracking
        self.enable_smpl_vis = enable_smpl_vis

        # Robot model for FK-based calibration (headless, no display required)
        self._robot_model = robot_model
        if self._robot_model is None:
            from gear_sonic.data.robot_model.instantiation.g1 import (
                instantiate_g1_robot_model,
            )

            self._robot_model = instantiate_g1_robot_model()
            print(f"[{log_prefix}] Robot model loaded for FK calibration")

        # Optional visualization (requires display + PyVista)
        self.vr3pt_visualizer = None
        if enable_vis_vr3pt:
            if VR3PtPoseVisualizer is None:
                raise ImportError(
                    "VR3PtPoseVisualizer could not be imported but --vis_vr3pt was requested. "
                    "Ensure pyvista is installed: pip install pyvista"
                )
            self.vr3pt_visualizer = VR3PtPoseVisualizer(
                axis_length=0.08,
                ball_radius=0.015,
                with_g1_robot=with_g1_robot,
                robot_model=self._robot_model,
                enable_waist_tracking=enable_waist_tracking,
                enable_smpl_vis=enable_smpl_vis,
            )
            self.vr3pt_visualizer.create_realtime_plotter(interactive=True)
            g1_str = " with G1 robot" if with_g1_robot else ""
            waist_str = " + waist tracking" if enable_waist_tracking else ""
            smpl_str = " + SMPL body" if enable_smpl_vis else ""
            print(f"[{log_prefix}] VR 3pt pose visualization enabled{g1_str}{waist_str}{smpl_str}")

        # Calibration state — triggered explicitly by calibrate_now() or reset_with_measured_q()
        self._calibration_pending = False
        self._calibration_neck_quat_inv: np.ndarray | None = None  # inv(initial neck quat)
        self._calibration_lwrist_offset: np.ndarray | None = None  # position offset
        self._calibration_rwrist_offset: np.ndarray | None = None
        self._calibration_lwrist_rot_offset: sRot | None = None  # orientation offset
        self._calibration_rwrist_rot_offset: sRot | None = None
        # Override robot q for FK during recalibration (e.g. measured joints for VR 3PT)
        self._override_robot_q: np.ndarray | None = None

    @property
    def is_pending(self) -> bool:
        """Check if calibration is pending."""
        return self._calibration_pending

    @property
    def robot_model(self):
        """Shared G1 model used by calibration and arm IK."""
        return self._robot_model

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration has been captured."""
        return self._calibration_neck_quat_inv is not None

    def process_smpl_pose(
        self,
        smpl_pose_np: np.ndarray,
        smpl_joints_local: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Process SMPL pose to extract and calibrate 3-point VR pose.

        Args:
            smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints
            smpl_joints_local: Optional np.ndarray shape (24, 3) - SMPL local joint
                               positions for body visualization. If provided and SMPL
                               visualization is enabled, the joint spheres are updated.

        Returns:
            vr_3pt_pose: np.ndarray shape (3, 7) - Calibrated 3-point pose
                         [L-Wrist, R-Wrist, Neck], each row [x, y, z, qw, qx, qy, qz]
        """
        # Extract raw 3-point pose from SMPL
        vr_3pt_pose_raw = _process_3pt_pose(smpl_pose_np)

        # Capture calibration on first valid frame (or after reset)
        if self._calibration_pending:
            self._capture_calibration(vr_3pt_pose_raw)

        # Apply calibration to get the final pose
        vr_3pt_pose = self._apply_calibration(vr_3pt_pose_raw)

        if self.vr3pt_visualizer is not None:
            self.vr3pt_visualizer.update_from_vr_pose(vr_3pt_pose, waist_scale=1.0)
            if smpl_joints_local is not None:
                self.vr3pt_visualizer.update_smpl_joints(smpl_joints_local)
            self.vr3pt_visualizer.render()

        return vr_3pt_pose

    def close(self) -> None:
        """Close and cleanup visualizer resources."""
        if self.vr3pt_visualizer is not None:
            try:
                self.vr3pt_visualizer.close()
            except Exception as e:
                print(f"[{self.log_prefix}] Warning: Error closing VR3pt visualizer: {e}")

    def calibrate_now(self, body_poses_np: np.ndarray) -> bool:
        """Calibrate using current SMPL frame against FK of all-zero body joints.
        Operator should be in zero-reference pose when calling this."""
        try:
            vr_3pt_pose_raw = _process_3pt_pose(body_poses_np)
            self._override_robot_q = np.zeros(29, dtype=np.float64)
            self._capture_calibration(vr_3pt_pose_raw)
            print(f"[{self.log_prefix}] Calibration completed (zero-pose reference)")
            return True
        except Exception as e:
            print(f"[{self.log_prefix}] Calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _capture_calibration(self, vr_3pt_pose: np.ndarray) -> None:
        """Capture calibration offsets from vr_3pt_pose against G1 FK reference.
        If neck calibration already exists (e.g. from calibrate_now), it is preserved
        to avoid jumps from SMPL noise during recalibration."""

        # Step 1: Neck orientation — only capture if not already set
        if self._calibration_neck_quat_inv is None:
            neck_quat_wxyz = vr_3pt_pose[2, 3:].copy()
            neck_rot = sRot.from_quat(neck_quat_wxyz, scalar_first=True)
            self._calibration_neck_quat_inv = neck_rot.inv().as_quat(scalar_first=True)
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Step 2: Rotate VR wrist positions/orientations by neck inverse
        lwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[0, :3].copy())
        rwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[1, :3].copy())
        lwrist_rot_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
        rwrist_rot_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)

        # Step 3: Get G1 FK reference poses
        if self._robot_model is None:
            raise RuntimeError(
                "Robot model is required for calibration but was not loaded. "
                "Ensure the G1 robot model and URDF are available."
            )
        if get_g1_key_frame_poses is None:
            raise RuntimeError(
                "get_g1_key_frame_poses could not be imported. "
                "Ensure gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer is available."
            )

        # Convert 29-DOF override to full model config if needed
        if self._override_robot_q is not None:
            robot_q = self._robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=self._override_robot_q[:29]
            )
        else:
            robot_q = None
        g1_poses = get_g1_key_frame_poses(self._robot_model, q=robot_q)

        g1_lwrist_pos = g1_poses["left_wrist"]["position"]
        g1_rwrist_pos = g1_poses["right_wrist"]["position"]
        g1_lwrist_rot = sRot.from_quat(
            g1_poses["left_wrist"]["orientation_wxyz"], scalar_first=True
        )
        g1_rwrist_rot = sRot.from_quat(
            g1_poses["right_wrist"]["orientation_wxyz"], scalar_first=True
        )

        # Compute position offsets: calibrated = neck_corrected - offset
        self._calibration_lwrist_offset = lwrist_pos_corrected - g1_lwrist_pos
        self._calibration_rwrist_offset = rwrist_pos_corrected - g1_rwrist_pos

        # Compute orientation offsets: calibrated = rot_offset * neck_corrected
        self._calibration_lwrist_rot_offset = g1_lwrist_rot * lwrist_rot_corrected.inv()
        self._calibration_rwrist_rot_offset = g1_rwrist_rot * rwrist_rot_corrected.inv()

        self._calibration_pending = False
        self._override_robot_q = None

        # Log summary
        source = "override q" if g1_lwrist_pos.any() else "default/zero"
        print(
            f"[{self.log_prefix}] Calibration captured (FK ref: {source}):\n"
            f"  L-Wrist pos offset: [{self._calibration_lwrist_offset[0]:.4f}, "
            f"{self._calibration_lwrist_offset[1]:.4f}, {self._calibration_lwrist_offset[2]:.4f}]\n"
            f"  R-Wrist pos offset: [{self._calibration_rwrist_offset[0]:.4f}, "
            f"{self._calibration_rwrist_offset[1]:.4f}, {self._calibration_rwrist_offset[2]:.4f}]"
        )

    def _apply_calibration(self, vr_3pt_pose: np.ndarray) -> np.ndarray:
        """Apply stored calibration offsets to raw VR 3-point pose."""
        if self._calibration_neck_quat_inv is None:
            return vr_3pt_pose

        calibrated = vr_3pt_pose.copy()
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Neck orientation: calibrated = inv(initial) * current
        neck_rot = sRot.from_quat(vr_3pt_pose[2, 3:], scalar_first=True)
        calibrated[2, 3:] = (calib_inv_rot * neck_rot).as_quat(scalar_first=True)

        # Wrist positions: rotate by neck inverse, then subtract offset
        if self._calibration_lwrist_offset is not None:
            calibrated[0, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[0, :3]) - self._calibration_lwrist_offset
            )
        if self._calibration_rwrist_offset is not None:
            calibrated[1, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[1, :3]) - self._calibration_rwrist_offset
            )

        # Wrist orientations: rot_offset * (neck_inv * current)
        if self._calibration_lwrist_rot_offset is not None:
            lw_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
            calibrated[0, 3:] = (self._calibration_lwrist_rot_offset * lw_corrected).as_quat(
                scalar_first=True
            )
        if self._calibration_rwrist_rot_offset is not None:
            rw_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)
            calibrated[1, 3:] = (self._calibration_rwrist_rot_offset * rw_corrected).as_quat(
                scalar_first=True
            )

        # Neck position via kinematic chain: root → torso_link (+Z) → neck (along calibrated Z)
        neck_z = sRot.from_quat(calibrated[2, 3:], scalar_first=True).apply([0, 0, 1])
        calibrated[2, :3] = (
            np.array([0, 0, self.TORSO_LINK_OFFSET_Z]) + self.NECK_LINK_LENGTH * neck_z
        ).astype(np.float32)

        return calibrated

    def _clear_calibration(self):
        """Clear all calibration state."""
        self._calibration_neck_quat_inv = None
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._override_robot_q = None

    def reset(self) -> None:
        """Reset calibration. Next process_smpl_pose() call will recalibrate."""
        self._clear_calibration()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Calibration reset, will re-calibrate on next frame")

    def reset_with_measured_q(self, body_q_measured: np.ndarray) -> None:
        """Recalibrate wrist offsets using measured robot joints (29 DOFs).
        Preserves neck calibration to avoid jumps from SMPL noise.
        Next process_smpl_pose() will recompute wrist offsets against FK of these joints."""
        # Preserve neck calibration — only clear wrist offsets
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._override_robot_q = body_q_measured.copy()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Wrist recalibration pending (neck preserved, measured q)")


class PoseStreamer:
    """Encapsulates the pose streaming loop state and logic."""

    def __init__(
        self,
        socket,
        reader: "PicoReader | input_readers.IsaacTeleopReader",
        three_point: ThreePointPose,
        num_frames_to_send: int,
        target_fps: int,
        use_cuda: bool,
        record_dir: str,
        record_format: str,
        log_prefix: str = "PoseLoop",
    ):
        self.socket = socket
        self.reader = reader
        self.num_frames_to_send = num_frames_to_send
        self.target_fps = target_fps
        self.record_dir = record_dir
        self.log_prefix = log_prefix

        # Injected dependencies
        self.reader = reader
        self.three_point = three_point

        self.device = (
            torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
        )

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.record_idx = 0

        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()
        self.parent_indices = [
            -1,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            9,
            9,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
        ][:24]

        self.step = 0
        self.last_fps_report = time.time()
        self.fps_counter = 0
        # NOTE: Sleep budget set to 95% of the ideal frame period so that the actual
        # FPS lands closer to target_fps despite per-frame processing overhead.
        self.frame_time = 0.95 / max(1, target_fps)
        self.frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))

        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.next_target_ns = None
        self.frame_start = time.time()

        # Data collection button state tracking (edge-triggered)
        self.toggle_data_collection_last = False
        self.toggle_data_abort_last = False

        self.buffer_cleared = (
            True  # Start with buffer cleared - wait for full buffer before first send
        )
        self.yaw_accumulator = YawAccumulator()

    def reset_yaw(self):
        """Called when entering pose mode. Resets yaw only.
        Calibration is triggered separately by the operator (A+B+X+Y → calibrate_now)."""
        self.yaw_accumulator.reset()

    def on_mode_exit(self):
        self.frame_buffer.clear()
        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.next_target_ns = None
        self.buffer_cleared = True
        self.step = 0

    def run_once(self):
        """Execute one iteration of the pose streaming loop."""
        sample = self.reader.get_latest()

        if sample is None:
            time.sleep(0.005)
            return

        latest_data = compute_from_body_poses(
            self.parent_indices, self.device, sample["body_poses_np"]
        )
        left_menu_button, left_trigger, right_trigger, left_grip, right_grip = get_controller_inputs(
            self.reader
        )
        # Get A and B button states for data collection control
        a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons(self.reader)

        # Data collection controls avoid grip/trigger inputs so starting a
        # recording cannot alter the demonstrated hand pose.
        # X+B = start/stop-success, Y+A = discard the active take.
        toggle_data_collection_tmp = x_pressed and b_pressed
        toggle_data_abort_tmp = y_pressed and a_pressed

        # Detect rising edge
        toggle_data_collection = toggle_data_collection_tmp and not self.toggle_data_collection_last
        toggle_data_abort = toggle_data_abort_tmp and not self.toggle_data_abort_last
        self.toggle_data_collection_last = toggle_data_collection_tmp
        self.toggle_data_abort_last = toggle_data_abort_tmp

        left_hand_joints, right_hand_joints = compute_hand_joints_from_inputs(
            self.left_hand_ik_solver,
            self.right_hand_ik_solver,
            left_trigger,
            left_grip,
            right_trigger,
            right_grip,
        )
        smpl_pose_np = (
            latest_data["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)[0]
        ).astype(np.float32)
        smpl_joints_np = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0].astype(np.float32)
        )
        body_quat_np = (
            latest_data["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)
        )
        curr_stamp_ns = int(sample.get("timestamp_ns", 0))
        step_ns = int(1e9 / max(1, self.target_fps))
        if self.prev_stamp_ns is None:
            self.prev_stamp_ns = curr_stamp_ns
            self.prev_smpl_pose_np = smpl_pose_np
            self.prev_smpl_joints_np = smpl_joints_np
            self.prev_body_quat_np = body_quat_np
            self.next_target_ns = curr_stamp_ns
            return
        if curr_stamp_ns <= self.prev_stamp_ns:
            return
        if self.next_target_ns is None:
            self.next_target_ns = self.prev_stamp_ns + step_ns
        if self.next_target_ns < self.prev_stamp_ns:
            self.next_target_ns = self.prev_stamp_ns
        if self.next_target_ns > curr_stamp_ns:
            return
        denom = float(curr_stamp_ns - self.prev_stamp_ns)
        alpha = float(self.next_target_ns - self.prev_stamp_ns) / denom if denom > 0.0 else 1.0
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        use_joints = (1.0 - alpha) * self.prev_smpl_joints_np + alpha * smpl_joints_np
        use_pose = _interp_pose_axis_angle(self.prev_smpl_pose_np, smpl_pose_np, alpha).astype(
            np.float32
        )
        use_body_quat = _quat_lerp_normalized(self.prev_body_quat_np, body_quat_np, alpha).astype(
            np.float32
        )
        N = len(self.frame_buffer["frame_index"])

        ##### From @Jiefeng for directly setting the joint position ######
        joint_pos = np.zeros(29)
        body_pose = use_pose.reshape(-1, 21, 3)

        SMPL_L_ELBOW_IDX = 17
        SMPL_L_WRIST_IDX = 19
        SMPL_R_ELBOW_IDX = 18
        SMPL_R_WRIST_IDX = 20

        # G1_L_ELBOW_IDX = 0
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27

        # G1_R_ELBOW_IDX = 0
        G1_R_WRIST_ROLL_IDX = 24  # Done
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28
        smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
        smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
        smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
        smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

        g1_l_elbow_axis = np.array([0, 1, 0])
        g1_l_elbow_q_twist, g1_l_elbow_q_swing = decompose_rotation_aa(
            smpl_l_elbow_aa, g1_l_elbow_axis
        )

        g1_r_elbow_axis = np.array([0, 1, 0])
        g1_r_elbow_q_twist, g1_r_elbow_q_swing = decompose_rotation_aa(
            smpl_r_elbow_aa, g1_r_elbow_axis
        )

        # Move elbow roll/yaw into wrist while preserving wrist pitch from SMPL
        l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )
        r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )

        l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
        r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

        g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
        g1_l_wrist_pitch = -l_wrist_euler[:, 1]
        g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

        g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
        g1_r_wrist_pitch = -r_wrist_euler[:, 1]
        g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

        joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0]
        joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0]
        joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0]

        joint_pos[G1_R_WRIST_ROLL_IDX] = g1_r_wrist_roll[0]
        joint_pos[G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch[0]
        joint_pos[G1_R_WRIST_YAW_IDX] = g1_r_wrist_yaw[0]

        # Process SMPL pose to get calibrated 3-point VR pose and update visualization
        # Pass SMPL local joints for optional body visualization in the VR3Pt viewer
        smpl_joints_for_vis = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0]
            if self.three_point.enable_smpl_vis
            else None
        )
        vr_3pt_pose = self.three_point.process_smpl_pose(
            sample["body_poses_np"], smpl_joints_local=smpl_joints_for_vis
        )
        ##### From @Jiefeng for directly setting the joint position ######

        self.frame_buffer["smpl_pose"].append(use_pose)
        self.frame_buffer["smpl_joints"].append(use_joints)
        self.frame_buffer["body_quat_w"].append(use_body_quat)
        self.frame_buffer["frame_index"].append(int(self.step))
        self.frame_buffer["joint_pos"].append(joint_pos)
        pico_dt = float(sample.get("dt", 0.0))
        pico_fps = float(sample.get("fps", 0.0))
        N = len(self.frame_buffer["frame_index"])

        # Wait for buffer to be completely filled before sending first message after clearing
        buffer_is_full = len(self.frame_buffer["frame_index"]) >= self.num_frames_to_send
        if buffer_is_full and self.buffer_cleared:
            # Buffer is now full with fresh data, can start sending
            self.buffer_cleared = False

        # Get joystick axes for yaw accumulation
        _, _, rx, _ = get_controller_axes(self.reader)
        self.yaw_accumulator.update(rx, self.frame_time)

        # Only send if buffer is full and we're not waiting for fresh data
        if buffer_is_full and not self.buffer_cleared:
            numpy_data = {
                "smpl_pose": np.stack((self.frame_buffer["smpl_pose"]), axis=0),
                "smpl_joints": np.stack((self.frame_buffer["smpl_joints"]), axis=0),
                "body_quat_w": np.stack((self.frame_buffer["body_quat_w"]), axis=0),
                "joint_pos": np.stack((self.frame_buffer["joint_pos"]), axis=0),
                "joint_vel": np.zeros((N, 29)),
                "vr_position": vr_3pt_pose[:, :3].flatten(),
                "vr_orientation": vr_3pt_pose[:, 3:].flatten(),
                "frame_index": np.array((self.frame_buffer["frame_index"]), dtype=np.int64),
                "left_trigger": np.array([left_trigger], dtype=np.float32),
                "right_trigger": np.array([right_trigger], dtype=np.float32),
                "left_grip": np.array([left_grip], dtype=np.float32),
                "right_grip": np.array([right_grip], dtype=np.float32),
                "pico_dt": np.array([pico_dt], dtype=np.float32),
                "pico_fps": np.array([pico_fps], dtype=np.float32),
                "timestamp_realtime": np.array(
                    [sample.get("timestamp_realtime", 0.0)], dtype=np.float64
                ),
                "timestamp_monotonic": np.array(
                    [sample.get("timestamp_monotonic", 0.0)], dtype=np.float64
                ),
                "publisher_monotonic_ns": np.array(
                    [time.monotonic_ns()], dtype=np.int64
                ),
                "left_hand_joints": left_hand_joints.reshape(-1).astype(np.float32),
                "right_hand_joints": right_hand_joints.reshape(-1).astype(np.float32),
                "toggle_data_collection": np.array([toggle_data_collection], dtype=bool),
                "toggle_data_abort": np.array([toggle_data_abort], dtype=bool),
                "heading_increment": np.array(
                    [self.yaw_accumulator.yaw_angle_change()], dtype=np.float32
                ),
            }

            packed_message = pack_pose_message(numpy_data, topic="pose")
            self.socket.send(packed_message)

            if self.record_dir:
                out_path = os.path.join(self.record_dir, f"pose_{self.record_idx:06d}.npz")
                np.savez_compressed(out_path, **numpy_data)
                self.record_idx += 1

        self.step += 1
        self.next_target_ns += step_ns
        self.prev_stamp_ns = curr_stamp_ns
        self.prev_smpl_pose_np = smpl_pose_np
        self.prev_smpl_joints_np = smpl_joints_np
        self.prev_body_quat_np = body_quat_np
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_report >= 5.0:
            fps = self.fps_counter / (current_time - self.last_fps_report)
            print(f"[{self.log_prefix}] FPS: {fps:.2f}, Step: {self.step}")
            self.fps_counter = 0
            self.last_fps_report = current_time
        elapsed = time.time() - self.frame_start
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self.frame_start = time.time()


XRT_SERVICE_HOST = "127.0.0.1"
XRT_SERVICE_PORT = 60061
XRT_CONNECT_ATTEMPT_SECONDS = 8.0


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex((host, port)) == 0


def _ensure_robotics_service() -> None:
    """Start the local XRT bridge only when there is not already one listening."""
    if _port_is_open(XRT_SERVICE_HOST, XRT_SERVICE_PORT):
        return

    print("[XRT] Starting RoboticsService bridge...")
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if _port_is_open(XRT_SERVICE_HOST, XRT_SERVICE_PORT):
            print("[XRT] RoboticsService bridge is ready")
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"RoboticsService did not open {XRT_SERVICE_HOST}:{XRT_SERVICE_PORT}"
    )


def _close_xrt() -> None:
    if xrt is None:
        return
    try:
        xrt.close()
    except Exception as exc:
        print(f"[XRT] SDK close warning: {exc}")


def _connect_xrt_body_stream() -> None:
    """Keep resubscribing until live, advancing full-body timestamps arrive."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run Pico streaming."
        )

    attempt = 0
    while True:
        attempt += 1
        try:
            _ensure_robotics_service()
            print(f"[XRT] SDK subscription attempt {attempt}")
            xrt.init()
            deadline = time.monotonic() + XRT_CONNECT_ATTEMPT_SECONDS
            first_stamp_ns = None
            while time.monotonic() < deadline:
                if xrt.is_body_data_available():
                    stamp_ns = int(xrt.get_time_stamp_ns())
                    if first_stamp_ns is None:
                        first_stamp_ns = stamp_ns
                    elif stamp_ns != first_stamp_ns:
                        print("[XRT] Live full-body stream connected")
                        return
                time.sleep(0.1)
        except KeyboardInterrupt:
            _close_xrt()
            raise
        except Exception as exc:
            print(f"[XRT] Connection attempt failed: {exc}")

        _close_xrt()
        print(
            "[XRT] No live body frames. Pane 1 will keep retrying; "
            "start/restart streaming on the PICO to this PC."
        )
        time.sleep(1.0)


def _init_input_source(
    input_source: str,
    buffer_size: int,
) -> "PicoReader | input_readers.IsaacTeleopReader":
    """Create, start, and wait for readiness of the requested teleop input source."""
    if input_source == "isaac-teleop":
        reader = input_readers.IsaacTeleopReader(max_queue_size=buffer_size)
        reader.start()
        print("Using Isaac Teleop (in-process CloudXR / DeviceIO), waiting for data...")
        while reader.get_latest() is None:
            print("waiting for Isaac Teleop body data (connect the headset to CloudXR)...")
            time.sleep(1)
        return reader

    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run Pico streaming."
        )

    print("Waiting for live body tracking data...")
    _connect_xrt_body_stream()

    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()
    return reader


def run_pico(
    buffer_size: int = 15,
    port: int = 5556,
    hand_intent_port: int = DEFAULT_HAND_INTENT_PORT,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    input_source: str = "xrt",
):
    """Run body tracking with real-time visualization and ZMQ streaming."""
    reader = _init_input_source(input_source, buffer_size)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    hand_socket = context.socket(zmq.PUB)
    hand_socket.setsockopt(zmq.SNDHWM, 1)
    hand_socket.bind(f"tcp://*:{hand_intent_port}")
    time.sleep(0.1)
    print(f"ZMQ socket bound to port {port}")
    if build_command_message is not None and build_planner_message is not None:
        try:
            socket.send(build_command_message(start=False, stop=False, planner=False))
            socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0))
        except Exception as e:
            print(f"Warning: failed to send initial command/planner messages: {e}")
    try:
        _pose_stream_common(
            socket=socket,
            hand_socket=hand_socket,
            buffer_size=buffer_size,
            num_frames_to_send=num_frames_to_send,
            target_fps=target_fps,
            use_cuda=use_cuda,
            record_dir=record_dir,
            record_format=record_format,
            stop_event=None,
            log_prefix="Main",
            enable_vis_vr3pt=enable_vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=enable_waist_tracking,
            enable_smpl_vis=enable_smpl_vis,
            reader=reader,
        )
    finally:
        reader.stop()
        if input_source == "xrt":
            _close_xrt()
        socket.close()
        hand_socket.close()
        context.term()
        print("Threads stopped, ZMQ socket closed")


class FeedbackReader:
    """Reads feedback from robot via ZMQ and processes measured upper body position to use as frozen targets."""

    def __init__(self, zmq_feedback_host: str = "localhost", zmq_feedback_port: int = 5557):
        self.poller = ZMQPoller(host=zmq_feedback_host, port=zmq_feedback_port, topic="g1_debug")

        self.upper_body_joint_indices = self._get_upper_body_joint_indices()

        self.upper_body_position_target = None
        self.upper_body_planner_target = None
        self.left_hand_position_target = None
        self.right_hand_position_target = None
        # Full body joint configuration (29 DOFs) as measured from robot,
        # used for FK when recalibrating VR 3PT tracking against actual robot pose
        self.full_body_q_measured: np.ndarray | None = None
        self.last_body_feedback_monotonic: float | None = None

    def _get_upper_body_joint_indices(self) -> list[int]:
        # TODO: get from robot model, not hardcoded
        # robot_model = instantiate_g1_robot_model()
        # return robot_model.get_joint_group_indices("upper_body")
        return [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

    def poll_feedback(self, *, retain_last: bool = False) -> bool:
        """Poll once and report whether a complete body sample was received."""
        upper_body, planner_target, left_hand, right_hand, full_body = (
            self._process_upper_body_position_targets()
        )
        if full_body is None and retain_last:
            return False
        self.upper_body_position_target = upper_body
        self.upper_body_planner_target = planner_target
        self.left_hand_position_target = left_hand
        self.right_hand_position_target = right_hand
        self.full_body_q_measured = full_body
        if full_body is None:
            self.last_body_feedback_monotonic = None
            return False
        self.last_body_feedback_monotonic = time.monotonic()
        print("[PlannerLoop] Saved upper body position target:", upper_body)
        return True

    def _process_upper_body_position_targets(
        self,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        data = self.poller.get_data()

        if data is None:
            print("[PlannerLoop] No feedback data received")
            return None, None, None, None, None

        unpacked = msgpack.unpackb(data, raw=False)
        full_body_q = None
        if "body_q_measured" in unpacked:
            candidate = np.asarray(unpacked["body_q_measured"], dtype=np.float64)
            if candidate.shape == (29,) and np.all(np.isfinite(candidate)):
                full_body_q = candidate
                body_q = candidate[self.upper_body_joint_indices]
            else:
                print("[PlannerLoop] body_q_measured feedback is invalid")
                body_q = None
        else:
            print("[PlannerLoop] body_q_measured not in feedback data")
            body_q = None

        if "body_q_target" in unpacked:
            body_q_target = np.asarray(unpacked["body_q_target"], dtype=np.float64)
            if body_q_target.shape == (29,) and np.all(np.isfinite(body_q_target)):
                planner_target = body_q_target[self.upper_body_joint_indices]
            else:
                print("[PlannerLoop] body_q_target feedback is invalid")
                planner_target = None
        else:
            print("[PlannerLoop] body_q_target not in feedback data")
            planner_target = None

        if "left_hand_q_measured" in unpacked:
            left_hand_q = unpacked["left_hand_q_measured"]
        else:
            print("[PlannerLoop] left_hand_q_measured not in feedback data")
            left_hand_q = None

        if "right_hand_q_measured" in unpacked:
            right_hand_q = unpacked["right_hand_q_measured"]
        else:
            print("[PlannerLoop] right_hand_q_measured not in feedback data")
            right_hand_q = None

        return body_q, planner_target, left_hand_q, right_hand_q, full_body_q


class PlannerStreamer:
    """Encapsulates the planner control loop state and logic."""

    def __init__(
        self,
        socket,
        reader: "PicoReader | input_readers.IsaacTeleopReader",
        three_point: ThreePointPose,
        poll_hz: int = 50,
        zmq_feedback_host: str = "localhost",
        zmq_feedback_port: int = 5557,
        initial_mode: LocomotionMode = LocomotionMode.IDLE,
        ik_upper_body: bool = False,
    ):
        self.socket = socket
        self.reader = reader
        self.three_point = three_point
        self.feedback_reader = FeedbackReader(
            zmq_feedback_host=zmq_feedback_host, zmq_feedback_port=zmq_feedback_port
        )

        self.dt = 1.0 / max(1, poll_hz)
        # Persistent gait selection. Zero stick input still sends IDLE, but
        # movement begins in this gait as soon as the joystick leaves deadzone.
        self.mode = LocomotionMode(initial_mode)
        # Persistent facing buffer (unit vector on XY plane)
        self.yaw_accumulator = YawAccumulator()
        self.last_xrt_timestamp = None

        # Hand IK solvers for trigger-controlled hand open/close in VR 3PT mode
        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()
        self.ik_upper_body = None
        if ik_upper_body:
            from gear_sonic.utils.teleop.ik_upper_body import IkUpperBodyRetargeter

            self.ik_upper_body = IkUpperBodyRetargeter(three_point.robot_model)

    def reset_yaw(self):
        """Called when entering planner mode. Resets state for fresh start."""
        self.yaw_accumulator.reset()

    def save_upper_body_position_target(self):
        """Poll feedback and save upper body position target."""
        self.feedback_reader.poll_feedback()

    def poll_fresh_feedback(self, *, require_planner_target: bool = False) -> bool:
        """Refresh robot feedback used to seed a safe upper-body transition."""
        deadline = time.monotonic() + 0.1
        while (
            not self.feedback_reader.poll_feedback(retain_last=True)
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        measured = self.feedback_reader.upper_body_position_target
        received_at = self.feedback_reader.last_body_feedback_monotonic
        if (
            measured is None
            or np.asarray(measured).shape != (UPPER_BODY_WIDTH,)
            or received_at is None
            or time.monotonic() - received_at > 0.25
        ):
            print("[PlannerLoop] Cannot transition without fresh upper-body feedback")
            return False
        if require_planner_target:
            target = self.feedback_reader.upper_body_planner_target
            if target is None or np.asarray(target).shape != (UPPER_BODY_WIDTH,):
                print("[PlannerLoop] Cannot return to idle without a planner joint target")
                return False
        return True

    def recalibrate_for_vr3pt(self) -> bool:
        """
        Recalibrate VR 3-point pose tracking using the robot's current measured joints.

        Polls the g1_debug feedback to get the robot's actual joint state, then
        schedules recalibration so VR tracking aligns with the robot's current pose.
        This prevents sudden jumps when entering VR 3PT mode from PLANNER mode.
        """
        if not self.poll_fresh_feedback():
            print("[PlannerLoop] Cannot enter VR 3PT without fresh robot feedback")
            return False
        measured_q = self.feedback_reader.full_body_q_measured
        if measured_q is None or np.asarray(measured_q).shape != (29,):
            print("[PlannerLoop] Cannot enter VR 3PT without complete 29-DOF feedback")
            return False
        self.three_point.reset_with_measured_q(measured_q)
        print("[PlannerLoop] VR 3PT recalibration scheduled with measured robot pose")
        return True

    def prepare_ik_upper_body(self) -> bool:
        """Seed calibrated arm IK from fresh measured robot feedback."""
        if self.ik_upper_body is None:
            raise RuntimeError("upper-body IK was not initialized")
        if not self.poll_fresh_feedback():
            return False
        measured_q = self.feedback_reader.full_body_q_measured
        received_at = self.feedback_reader.last_body_feedback_monotonic
        if (
            measured_q is None
            or received_at is None
            or time.monotonic() - received_at > 0.25
        ):
            print(
                "[PlannerLoop] Cannot enter upper-body IK mode without fresh "
                "29-DOF robot feedback"
            )
            return False
        self.three_point.reset_with_measured_q(measured_q)
        self.ik_upper_body.reset(measured_q)
        print("[PlannerLoop] Upper-body IK seeded from measured robot pose")
        return True

    def run_once(
        self,
        stream_mode: StreamMode,
        *,
        face_command: str | None = None,
        upper_body_override: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        force_locomotion_idle: bool = False,
    ) -> bool:
        """Execute one iteration of the planner control loop."""
        try:
            # Avoid sending old commands if XRT timestamp hasn't advanced, in case of headset disconnect
            xrt_timestamp = self.reader.get_timestamp_ns()
            transition_or_base_pose = (
                upper_body_override is not None
                or stream_mode == StreamMode.PLANNER_IDLE_BASE_POSE
            )
            if xrt_timestamp == self.last_xrt_timestamp and not transition_or_base_pose:
                return False
            self.last_xrt_timestamp = xrt_timestamp

            # Face chords are disambiguated by the manager and confirmed on
            # release. A+B selects the next mode; X+Y selects the previous.
            if face_command == "ab":
                self.mode = LocomotionMode(min(LocomotionMode.INJURED_WALK, self.mode + 1))
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")
            if face_command == "xy":
                self.mode = LocomotionMode(max(LocomotionMode.IDLE, self.mode - 1))
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")

            # Read axes/joysticks to control movement, facing, speed and mode
            lx, ly, rx, ry = get_controller_axes(self.reader)

            # Facing from RIGHT stick: continuous yaw based on rx (right = turn right, left = turn left)
            facing = self.yaw_accumulator.update(rx, self.dt)

            raw_mag = np.hypot(lx, ly)
            raw_mag = np.clip(raw_mag, 0.0, 1.0)
            if np.abs(raw_mag) < JOYSTICK_DEADZONE:
                mag = 0.0
                speed = -1.0
                mode_to_send = LocomotionMode.IDLE
            else:
                mag = (raw_mag - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
                if mag > 1.0:
                    mag = 1.0
                mode_to_send = self.mode

                if self.mode == LocomotionMode.SLOW_WALK:
                    speed = 0.1 + 0.5 * mag  # 0.1 .. 0.6
                elif self.mode == LocomotionMode.WALK:
                    speed = -1.0
                elif self.mode == LocomotionMode.RUN:
                    speed = 1.5 + 3 * mag  # 1.5 .. 4.5
                else:
                    speed = mag  # default 0 .. 1.0

            denom = raw_mag if raw_mag > 0.0 else 1.0
            scale = mag / denom
            movement_local = np.array([-lx, ly]) * scale
            perp_x, perp_y = -facing[1], facing[0]
            rotation_facing = np.array([[perp_x, perp_y], [facing[0], facing[1]]])
            movement_global = rotation_facing @ movement_local

            movement = [movement_global[0], movement_global[1], 0.0]

            if force_locomotion_idle or stream_mode == StreamMode.PLANNER_IDLE_BASE_POSE:
                mode_to_send = LocomotionMode.IDLE
                movement = [0.0, 0.0, 0.0]
                speed = -1.0

            upper_body_position = None
            upper_body_velocity = None
            upper_body_mask = None
            left_hand_position = None
            right_hand_position = None
            if stream_mode == StreamMode.PLANNER_IDLE_BASE_POSE:
                upper_body_position = IDLE_BASE_UPPER_BODY_RAD.copy()
                upper_body_velocity = np.zeros(UPPER_BODY_WIDTH, dtype=np.float64)
                upper_body_mask = IDLE_BASE_UPPER_BODY_MASK.copy()
                left_hand_position = self.feedback_reader.left_hand_position_target
                right_hand_position = self.feedback_reader.right_hand_position_target

            if upper_body_override is not None:
                upper_body_position, upper_body_velocity, upper_body_mask = upper_body_override
                upper_body_position = np.asarray(upper_body_position, dtype=np.float64)
                upper_body_velocity = np.asarray(upper_body_velocity, dtype=np.float64)
                upper_body_mask = np.asarray(upper_body_mask, dtype=bool)
                if (
                    upper_body_position.shape != (UPPER_BODY_WIDTH,)
                    or upper_body_velocity.shape != (UPPER_BODY_WIDTH,)
                    or upper_body_mask.shape != (UPPER_BODY_WIDTH,)
                ):
                    raise ValueError("upper-body override must contain three 17-element arrays")

            vr_3pt_position = None
            vr_3pt_orientation = None
            vr_3pt_compliance = None
            if stream_mode == StreamMode.PLANNER_VR_3PT:
                sample = self.reader.get_latest()
                if sample is not None:
                    vr_3pt_pose = self.three_point.process_smpl_pose(sample["body_poses_np"])
                    vr_3pt_position = (vr_3pt_pose[:, :3].flatten()).tolist()
                    vr_3pt_orientation = vr_3pt_pose[:, 3:].flatten().tolist()

                # Compute hand joints from trigger/grip inputs so operator can
                # control hand open/close while in VR 3PT mode
                (
                    left_menu_button,
                    left_trigger,
                    right_trigger,
                    left_grip,
                    right_grip,
                ) = get_controller_inputs(self.reader)
                lh_joints, rh_joints = compute_hand_joints_from_inputs(
                    self.left_hand_ik_solver,
                    self.right_hand_ik_solver,
                    left_trigger,
                    left_grip,
                    right_trigger,
                    right_grip,
                )
                left_hand_position = lh_joints.reshape(-1).astype(np.float32).tolist()
                right_hand_position = rh_joints.reshape(-1).astype(np.float32).tolist()

            if stream_mode == StreamMode.PLANNER_IK_UPPER:
                if self.ik_upper_body is None:
                    raise RuntimeError("upper-body IK was not initialized")
                sample = self.reader.get_latest()
                if sample is None:
                    return False
                vr_3pt_pose = self.three_point.process_smpl_pose(sample["body_poses_np"])
                upper_body_position, upper_body_velocity, upper_body_mask = (
                    self.ik_upper_body.update(vr_3pt_pose)
                )

            msg = build_planner_message(
                mode_to_send.value,
                movement,
                facing,
                speed=speed,
                height=-1.0,
                upper_body_position=upper_body_position,
                upper_body_velocity=upper_body_velocity,
                upper_body_mask=upper_body_mask,
                left_hand_position=left_hand_position,
                right_hand_position=right_hand_position,
                vr_3pt_position=vr_3pt_position,
                vr_3pt_orientation=vr_3pt_orientation,
                vr_3pt_compliance=vr_3pt_compliance,
                publisher_monotonic_ns=time.monotonic_ns(),
            )
            self.socket.send(msg)
            return True
        except Exception as e:
            import traceback

            print(f"[PlannerLoop] error: {e}")
            traceback.print_exc()
            raise


def run_pico_manager(
    port: int = 5556,
    hand_intent_port: int = DEFAULT_HAND_INTENT_PORT,
    buffer_size: int = 15,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    zmq_feedback_host: str = "localhost",
    zmq_feedback_port: int = 5557,
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    input_source: str = "xrt",
    teleop_mode: str = "pose",
    initial_locomotion_mode: str = "idle",
    recording_status_host: str = "localhost",
    recording_status_port: int = 5581,
    teleop_control_host: str = "localhost",
    teleop_control_port: int = DEFAULT_TELEOP_CONTROL_PORT,
    idle_base_transition_duration: float = 2.0,
):
    """
    Manager: publishes body and latest-only hand intent from one fixed-rate loop.
    Controller input:
      A+X: Save while recording; otherwise twice within 2s advances toward teleop
      A+B+X+Y: Start the policy from OFF
      B+Y: Step back from teleop to base pose, then from base pose to idle
      X+B: Start/stop-success recording
      Y+A: The only explicit discard gesture
    """
    reader = _init_input_source(input_source, buffer_size)
    if teleop_mode not in {"pose", "vr3pt", "ik-upper"}:
        raise ValueError("teleop_mode must be 'pose', 'vr3pt', or 'ik-upper'")
    if not np.isfinite(idle_base_transition_duration) or idle_base_transition_duration <= 0:
        raise ValueError("idle_base_transition_duration must be positive and finite")
    try:
        initial_mode = LocomotionMode[initial_locomotion_mode.upper()]
    except KeyError as exc:
        raise ValueError(
            "initial_locomotion_mode must be idle, slow_walk, walk, or run"
        ) from exc
    if initial_mode not in {
        LocomotionMode.IDLE,
        LocomotionMode.SLOW_WALK,
        LocomotionMode.WALK,
        LocomotionMode.RUN,
    }:
        raise ValueError("initial_locomotion_mode must be idle, slow_walk, walk, or run")
    teleop_stream_mode = {
        "pose": StreamMode.POSE,
        "vr3pt": StreamMode.PLANNER_VR_3PT,
        "ik-upper": StreamMode.PLANNER_IK_UPPER,
    }[teleop_mode]

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    hand_socket = context.socket(zmq.PUB)
    hand_socket.setsockopt(zmq.SNDHWM, 1)
    hand_socket.bind(f"tcp://*:{hand_intent_port}")
    recording_status_socket = context.socket(zmq.SUB)
    recording_status_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    recording_status_socket.setsockopt(zmq.CONFLATE, 1)
    recording_status_socket.connect(
        f"tcp://{recording_status_host}:{recording_status_port}"
    )
    teleop_control_socket = context.socket(zmq.SUB)
    teleop_control_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    teleop_control_socket.setsockopt(zmq.CONFLATE, 1)
    teleop_control_socket.connect(
        f"tcp://{teleop_control_host}:{teleop_control_port}"
    )
    time.sleep(0.1)
    print(
        f"[Manager] Body ZMQ socket bound to port {port}; "
        f"latest-only hand intent bound to port {hand_intent_port}; "
        f"target rate {target_fps} Hz"
    )

    # Print available locomotion modes
    try:
        print("[Manager] Available modes:")
        for mode in LocomotionMode:
            print(f"  {mode.value}: {mode.name}")
    except Exception:
        pass

    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix="PoseLoop",
    )

    pose_streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix="PoseLoop",
    )
    planner_streamer = PlannerStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        poll_hz=target_fps,
        zmq_feedback_host=zmq_feedback_host,
        zmq_feedback_port=zmq_feedback_port,
        initial_mode=initial_mode,
        ik_upper_body=teleop_mode == "ik-upper",
    )
    hand_intent = HandIntentStream()

    # Stable manager states for planner-based teleoperation:
    #
    #              A+X                         A+X + calibration
    #     PLANNER ------> IDLE_BASE_POSE --------------------------> TELEOP
    #             <------                 <--------------------------
    #               B+Y                              B+Y
    #
    # The arrows touching IDLE_BASE_POSE use a private two-second joint
    # interpolation. They are not additional public FSM states. During both
    # directions locomotion is idle and hand intent is held.
    #
    # Full-body POSE retains its legacy switching behavior; the safe base-pose
    # path applies to the planner-owned VR_3PT and IK upper-body modes.
    #
    #   A+B+X+Y is start-only. Once running, stop through the UI or terminate
    #   the process; the headset gesture cannot transition back to OFF.
    #   POSE_PAUSE: left_menu_button held --> POSE_PAUSE, released --> POSE
    #
    print(
        f"Manager controls: A+X=save while recording; otherwise twice within 2s="
        f"advance toward {teleop_mode.upper()} teleop, B+Y=step back toward idle, "
        "X+B=record/save, Y+A=only discard, "
        f"A+B+X+Y=start policy (start-only); initial gait={initial_mode.name}"
    )
    current_mode = StreamMode.OFF
    vr3pt_parent_mode = StreamMode.PLANNER
    upper_body_transition: JointPoseTransition | None = None
    transition_destination: StreamMode | None = None
    recorder_is_recording = False
    recorder_command_timestamp = 0.0
    safe_idle_requested = False
    last_teleop_control_sequence = -1
    policy_started = False
    face_chords = FaceChordTracker()
    ax_double_press = DoublePressTracker(window_seconds=2.0)
    manager_period = 1.0 / max(target_fps, 1)
    manager_deadline = time.monotonic()
    rate_report_started = manager_deadline
    manager_report_count = 0
    planner_report_count = 0
    manager_deadline_misses = 0
    try:
        prev_ax_pressed = False
        prev_by_pressed = False
        prev_start_combo = False
        prev_left_axis_click = False
        while True:
            if reader.disconnected:
                print(
                    "[Manager] Teleop frames lost; suspending teleop input. "
                    "SONIC remains running."
                )
                if current_mode == StreamMode.POSE:
                    pose_streamer.on_mode_exit()
                current_mode = StreamMode.OFF
                upper_body_transition = None
                transition_destination = None

                # OFF is local manager state only. Never send stop=True from the
                # headset process: SONIC lifetime is owned by the operator UI or
                # direct process termination (for example Ctrl-C).
                for _ in range(3):
                    socket.send(
                        pack_pose_message(
                            {
                                "stream_mode": np.array(
                                    [StreamMode.OFF.value], dtype=np.int32
                                ),
                                "toggle_data_collection": np.array([False], dtype=bool),
                                "toggle_data_abort": np.array([False], dtype=bool),
                                "publisher_monotonic_ns": np.array(
                                    [time.monotonic_ns()], dtype=np.int64
                                ),
                            },
                            topic="manager_state",
                        )
                    )
                    time.sleep(0.05)

                if isinstance(reader, PicoReader):
                    reader.reconnect()
                else:
                    print("[Manager] Waiting for teleop input to reconnect...")
                    while reader.disconnected:
                        time.sleep(0.5)

                hand_intent = HandIntentStream()
                prev_ax_pressed = False
                prev_by_pressed = False
                prev_start_combo = False
                prev_left_axis_click = False
                face_chords.reset()
                ax_double_press.reset()
                print(
                    "[Manager] Teleop reconnected; SONIC is still running and "
                    "teleop remains OFF. "
                    "Use A+B+X+Y to recalibrate/start when ready."
                )
                manager_deadline = time.monotonic()
                continue

            # Poll Pico controller for buttons/axes
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons(reader)

            # Synchronize with the authoritative exporter state. Ignore status
            # packets older than a locally emitted command to avoid briefly
            # reverting the optimistic state during PUB/SUB propagation.
            while recording_status_socket.poll(0):
                recorder_status = recording_status_socket.recv_json()
                if float(recorder_status.get("timestamp", 0.0)) >= recorder_command_timestamp:
                    recorder_is_recording = bool(recorder_status.get("recording", False))

            safe_idle_abort_requested = False
            while teleop_control_socket.poll(0):
                command = teleop_control_socket.recv_json()
                if not isinstance(command, dict):
                    print("[Manager] Ignoring malformed teleop-control command")
                    continue
                try:
                    sequence = int(command.get("sequence", -1))
                except (TypeError, ValueError):
                    print("[Manager] Ignoring teleop-control command without a sequence")
                    continue
                if (
                    command.get("action") == "safe_idle"
                    and sequence > last_teleop_control_sequence
                ):
                    last_teleop_control_sequence = sequence
                    safe_idle_requested = True
                    if recorder_is_recording:
                        recorder_is_recording = False
                        recorder_command_timestamp = time.time()
                        safe_idle_abort_requested = True
                        print("[Manager] Safe-idle request aborting active recording")
                    print(
                        "[Manager] UI requested smooth return through "
                        "IDLE_BASE_POSE to IDLE"
                    )

            left_menu_button, _, _, _, _ = get_controller_inputs(reader)
            left_axis_click, _ = get_axis_clicks(reader)

            # Confirm a two-button chord only after all face buttons are
            # released. A third button cancels it, so the four-button policy
            # gesture cannot leak a subset while being pressed or released.
            face_command = face_chords.update(
                bool(a_pressed), bool(b_pressed), bool(x_pressed), bool(y_pressed)
            )
            start_combo = bool(a_pressed) and bool(b_pressed) and bool(x_pressed) and bool(y_pressed)

            # During recording, a completed A+X is reserved for save. Otherwise,
            # two completed A+X gestures advance one step toward teleoperation.
            ax_pressed = False
            if face_command == "ax" and not recorder_is_recording:
                ax_pressed = ax_double_press.register()
                if not ax_pressed:
                    print(
                        "[Manager] A+X registered; press A+X again within 2s "
                        "to advance toward teleop"
                    )

            # Mode changes are disabled while recording. This makes Y+A the
            # only gesture that can discard a take.
            by_pressed = face_command == "by" and not recorder_is_recording

            new_mode = current_mode
            requested_transition: StreamMode | None = None
            planner_based_teleop = teleop_stream_mode in {
                StreamMode.PLANNER_VR_3PT,
                StreamMode.PLANNER_IK_UPPER,
            }

            # An interpolation owns the upper body until its final sample has
            # been sent. Do not accept another mode gesture halfway through it.
            if upper_body_transition is not None:
                pass
            elif safe_idle_requested:
                if current_mode in {
                    StreamMode.PLANNER_VR_3PT,
                    StreamMode.PLANNER_IK_UPPER,
                }:
                    requested_transition = StreamMode.PLANNER_IDLE_BASE_POSE
                elif current_mode == StreamMode.OFF and policy_started:
                    # A headset reconnect resets only the local FSM to OFF;
                    # SONIC and the C++ timeout arm hold remain active. The UI
                    # may therefore recover that hold through the normal base
                    # pose path without restarting the policy.
                    requested_transition = StreamMode.PLANNER_IDLE_BASE_POSE
                elif current_mode == StreamMode.PLANNER_IDLE_BASE_POSE:
                    requested_transition = StreamMode.PLANNER
                elif current_mode == StreamMode.PLANNER:
                    safe_idle_requested = False
                    print("[Manager] Safe-idle return complete")
                else:
                    safe_idle_requested = False
                    print(
                        f"[Manager] Safe-idle request unavailable from {current_mode.name}; "
                        "planner-based teleop must be running"
                    )
            elif current_mode == StreamMode.OFF:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.PLANNER
                    policy_started = True
                    if planner_based_teleop:
                        print(
                            "[Manager] Planner started in IDLE; operator calibration "
                            "is deferred until IDLE_BASE_POSE -> TELEOP"
                        )
                    else:
                        # Legacy full-body POSE uses its zero-reference calibration.
                        sample = reader.get_latest()
                        if sample is not None:
                            three_point.calibrate_now(sample["body_poses_np"])
                        else:
                            print("[Manager] WARNING: No SMPL data available for calibration")

            elif current_mode == StreamMode.PLANNER:
                if planner_based_teleop and ax_pressed and not prev_ax_pressed:
                    requested_transition = StreamMode.PLANNER_IDLE_BASE_POSE
                elif not planner_based_teleop and ax_pressed and not prev_ax_pressed:
                    new_mode = teleop_stream_mode
                elif (
                    not planner_based_teleop
                    and left_axis_click
                    and not prev_left_axis_click
                ):
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.POSE:
                if ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.PLANNER
                elif left_menu_button:
                    new_mode = StreamMode.POSE_PAUSE

            elif current_mode == StreamMode.PLANNER_IDLE_BASE_POSE:
                if planner_based_teleop and ax_pressed and not prev_ax_pressed:
                    new_mode = teleop_stream_mode
                elif planner_based_teleop and by_pressed and not prev_by_pressed:
                    requested_transition = StreamMode.PLANNER

            elif current_mode == StreamMode.POSE_PAUSE:
                if not left_menu_button:
                    new_mode = StreamMode.POSE

            elif current_mode == StreamMode.PLANNER_VR_3PT:
                if planner_based_teleop and by_pressed and not prev_by_pressed:
                    requested_transition = StreamMode.PLANNER_IDLE_BASE_POSE
                elif (
                    not planner_based_teleop
                    and left_axis_click
                    and not prev_left_axis_click
                ):
                    new_mode = vr3pt_parent_mode
                elif not planner_based_teleop and ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.POSE

            elif current_mode == StreamMode.PLANNER_IK_UPPER:
                if by_pressed and not prev_by_pressed:
                    requested_transition = StreamMode.PLANNER_IDLE_BASE_POSE

            if recorder_is_recording and (
                new_mode != current_mode or requested_transition is not None
            ):
                attempted_mode = requested_transition or new_mode
                new_mode = current_mode
                requested_transition = None
                print(
                    "[Manager] Mode change ignored while recording: "
                    f"{current_mode.name} -> {attempted_mode.name}; "
                    "save with A+X or X+B, or discard with Y+A first"
                )

            if requested_transition is not None:
                needs_planner_target = requested_transition == StreamMode.PLANNER
                if planner_streamer.poll_fresh_feedback(
                    require_planner_target=needs_planner_target
                ):
                    transition_start = np.asarray(
                        planner_streamer.feedback_reader.upper_body_position_target,
                        dtype=np.float64,
                    )
                    transition_goal = (
                        np.asarray(
                            planner_streamer.feedback_reader.upper_body_planner_target,
                            dtype=np.float64,
                        )
                        if needs_planner_target
                        else IDLE_BASE_UPPER_BODY_RAD
                    )
                    upper_body_transition = JointPoseTransition(
                        transition_start,
                        transition_goal,
                        started_at=time.monotonic(),
                        duration_s=idle_base_transition_duration,
                    )
                    transition_destination = requested_transition
                    ax_double_press.reset()
                    print(
                        "[Manager] Smooth upper-body transition: "
                        f"{current_mode.name} -> {requested_transition.name} "
                        f"({idle_base_transition_duration:.2f}s)"
                    )

            # Handle instantaneous mode transitions before running the loop.
            if upper_body_transition is None and new_mode != current_mode:
                # A mode change invalidates any unconfirmed gesture from the
                # previous interaction context. Confirmed pairs are already reset.
                ax_double_press.reset()
                if current_mode == StreamMode.POSE:
                    pose_streamer.on_mode_exit()

                if new_mode == StreamMode.PLANNER_VR_3PT:
                    vr3pt_parent_mode = current_mode

                if new_mode == StreamMode.POSE:
                    pose_streamer.reset_yaw()
                elif new_mode == StreamMode.PLANNER and current_mode not in {
                    StreamMode.PLANNER_VR_3PT,
                    StreamMode.PLANNER_IK_UPPER,
                }:
                    # Only reset yaw when freshly entering PLANNER from POSE,
                    # not when returning from VR_3PT sub-mode
                    planner_streamer.reset_yaw()
                elif new_mode == StreamMode.PLANNER_VR_3PT:
                    # Recalibrate VR tracking against the robot's actual current pose
                    # (read via g1_debug feedback + FK) to prevent sudden jumps
                    if not planner_streamer.recalibrate_for_vr3pt():
                        new_mode = current_mode
                elif new_mode == StreamMode.PLANNER_IK_UPPER:
                    if not planner_streamer.prepare_ik_upper_body():
                        new_mode = current_mode

            # Run one iteration. Smooth transitions are emitted as planner
            # overrides while the stable public source state remains unchanged.
            transition_completed = False
            if upper_body_transition is not None:
                transition_sample = upper_body_transition.sample(time.monotonic())
                planner_sent = planner_streamer.run_once(
                    StreamMode.PLANNER,
                    upper_body_override=(
                        transition_sample.position,
                        transition_sample.velocity,
                        IDLE_BASE_UPPER_BODY_MASK,
                    ),
                    force_locomotion_idle=True,
                )
                planner_report_count += int(planner_sent)
                transition_completed = transition_sample.complete
            elif new_mode == StreamMode.POSE:
                pose_streamer.run_once()
            elif new_mode in PLANNER_STREAM_MODES:
                planner_sent = planner_streamer.run_once(
                    new_mode,
                    face_command=face_command,
                )
                planner_report_count += int(planner_sent)

            # Make sure to send command messages after loop iteration to ensure data arrives before mode switch
            if upper_body_transition is None and new_mode != current_mode:
                if new_mode in PLANNER_STREAM_MODES:
                    socket.send(build_command_message(start=True, stop=False, planner=True))
                elif new_mode == StreamMode.POSE:
                    socket.send(build_command_message(start=True, stop=False, planner=False))

                print(f"[Manager] StreamMode switch: {current_mode.name} -> {new_mode.name}")
                current_mode = new_mode

            if transition_completed:
                completed_destination = transition_destination
                if completed_destination is None:
                    raise RuntimeError("upper-body transition has no destination")
                print(
                    f"[Manager] StreamMode switch: {current_mode.name} -> "
                    f"{completed_destination.name}"
                )
                current_mode = completed_destination
                upper_body_transition = None
                transition_destination = None

            # Mode-independent: send manager_state for data exporter
            recording_action = recording_face_action(
                face_command,
                recorder_is_recording=recorder_is_recording,
                recording_mode_ready=(
                    upper_body_transition is None
                    and current_mode == teleop_stream_mode
                ),
            )
            toggle_dc_requested = face_command == "xb"
            toggle_dc = recording_action in {"start", "save"}
            toggle_da = recording_action == "discard" or safe_idle_abort_requested
            if toggle_dc_requested and not toggle_dc:
                print(
                    "[Manager] Recorder start rejected: use A+X twice within 2s to enter "
                    f"{teleop_stream_mode.name} first"
                )
            if toggle_dc:
                was_recording = recorder_is_recording
                recorder_is_recording = recording_action == "start"
                recorder_command_timestamp = time.time()
                gesture = "A+X" if face_command == "ax" else "X+B"
                print(
                    f"[Manager] Recorder {gesture} -> "
                    f"{'stop/save' if was_recording else 'start'}"
                )
            elif toggle_da:
                recorder_is_recording = False
                recorder_command_timestamp = time.time()
                if safe_idle_abort_requested:
                    print("[Manager] Recorder UI safe-idle -> discard")
                else:
                    print("[Manager] Recorder Y+A -> discard")
            socket.send(
                pack_pose_message(
                    {
                        "stream_mode": np.array(
                            [
                                (
                                    transition_destination
                                    if transition_destination is not None
                                    else current_mode
                                ).value
                            ],
                            dtype=np.int32,
                        ),
                        "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                        "toggle_data_abort": np.array([toggle_da], dtype=bool),
                        "publisher_monotonic_ns": np.array(
                            [time.monotonic_ns()], dtype=np.int64
                        ),
                    },
                    topic="manager_state",
                )
            )
            manager_report_count += 1
            hand_intent.publish(
                hand_socket,
                reader,
                hold=(
                    upper_body_transition is not None
                    or current_mode == StreamMode.PLANNER_IDLE_BASE_POSE
                ),
            )

            prev_ax_pressed = ax_pressed
            prev_by_pressed = by_pressed
            prev_start_combo = start_combo
            prev_left_axis_click = left_axis_click

            # Pace every manager mode from one monotonic scheduler. Planner no
            # longer owns a separate 20 Hz sleep, so planner and manager state
            # share the requested (normally 50 Hz) cadence without drift.
            manager_deadline += manager_period
            now = time.monotonic()
            remaining = manager_deadline - now
            if remaining > 0:
                time.sleep(remaining)
            elif -remaining >= manager_period:
                skipped = int((-remaining) / manager_period) + 1
                manager_deadline_misses += skipped
                manager_deadline += skipped * manager_period
                time.sleep(max(0.0, manager_deadline - time.monotonic()))

            report_now = time.monotonic()
            report_elapsed = report_now - rate_report_started
            if report_elapsed >= 5.0:
                print(
                    f"[Manager] manager_state/hand_intent: "
                    f"{manager_report_count / report_elapsed:.2f} Hz; "
                    f"planner: {planner_report_count / report_elapsed:.2f} Hz; "
                    f"deadline misses: {manager_deadline_misses}"
                )
                rate_report_started = report_now
                manager_report_count = 0
                planner_report_count = 0

    except KeyboardInterrupt:
        print("\nStopping manager...")
    finally:
        # Cleanup resources
        reader.stop()
        if input_source == "xrt":
            _close_xrt()
        three_point.close()
        recording_status_socket.close()
        teleop_control_socket.close()
        socket.close()
        hand_socket.close()
        context.term()
        print("[Manager] Shutdown complete")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=15, help="Sliding window buffer size")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port (default: 5556)")
    parser.add_argument(
        "--hand-intent-port",
        type=int,
        default=DEFAULT_HAND_INTENT_PORT,
        help=f"Latest-only hand-intent port (default: {DEFAULT_HAND_INTENT_PORT})",
    )
    parser.add_argument(
        "--num_frames_to_send", type=int, default=5, help="Number of frames to send (default: 200)"
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Target loop FPS (default: 50)")
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA for tensors and model (default: CPU)"
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="Directory to save sent batches (default: disabled)",
    )
    parser.add_argument(
        "--record_format",
        type=str,
        default="npz",
        help="Recording format: 'npz' or 'bin' (default: npz)",
    )
    parser.add_argument(
        "--manager",
        action="store_true",
        help="Run manager with planner and pose threads (interactive)",
    )
    parser.add_argument(
        "--zmq_feedback_host",
        type=str,
        default="localhost",
        help="ZMQ feedback host (default: localhost)",
    )
    parser.add_argument(
        "--zmq_feedback_port",
        type=int,
        default=5557,
        help="ZMQ feedback port (default: 5557)",
    )
    parser.add_argument(
        "--recording-status-host",
        type=str,
        default="localhost",
        help="Recorder status publisher host (default: localhost)",
    )
    parser.add_argument(
        "--recording-status-port",
        type=int,
        default=5581,
        help="Recorder status publisher port (default: 5581)",
    )
    parser.add_argument(
        "--teleop-control-host",
        type=str,
        default="localhost",
        help="Browser teleop-control publisher host (default: localhost)",
    )
    parser.add_argument(
        "--teleop-control-port",
        type=int,
        default=DEFAULT_TELEOP_CONTROL_PORT,
        help=f"Browser teleop-control publisher port (default: {DEFAULT_TELEOP_CONTROL_PORT})",
    )
    parser.add_argument(
        "--vr3pt_test",
        action="store_true",
        help="Run VR 3-point pose visualizer test (reference frames only)",
    )
    parser.add_argument(
        "--vr3pt_live",
        action="store_true",
        help="Capture one frame of VR 3-point pose and visualize with reference frames",
    )
    parser.add_argument(
        "--vr3pt_realtime",
        action="store_true",
        help="Run standalone real-time VR 3-point pose visualizer",
    )
    parser.add_argument(
        "--vis_vr3pt",
        action="store_true",
        help="Enable inline VR 3-point pose visualization in pose streaming mode",
    )
    parser.add_argument(
        "--vr3pt_hz",
        type=int,
        default=10,
        help="Update rate for real-time VR visualization in Hz (default: 10)",
    )
    parser.add_argument(
        "--no_g1",
        action="store_true",
        help="Disable G1 robot visualization in VR 3pt pose view (G1 is shown by default)",
    )
    parser.add_argument(
        "--waist_tracking",
        action="store_true",
        help="Enable G1 robot waist to follow VR head orientation (disabled by default for performance)",
    )
    parser.add_argument(
        "--vis_smpl",
        action="store_true",
        help="Enable SMPL body joint visualization (24 joint spheres) in the VR3pt viewer",
    )
    parser.add_argument(
        "--input-source",
        type=str,
        default="xrt",
        choices=["xrt", "isaac-teleop"],
        help=(
            "Input source: 'xrt' for XRoboToolkit SDK (default), "
            "'isaac-teleop' for in-process IsaacTeleop / CloudXR DeviceIO"
        ),
    )
    parser.add_argument(
        "--teleop-mode",
        choices=["pose", "vr3pt", "ik-upper"],
        default="pose",
        help=(
            "Teleop mode selected by double A+X: full-body pose, learned VR 3-point planner, "
            "or deterministic arm IK with planner-owned legs/waist"
        ),
    )
    parser.add_argument(
        "--initial-locomotion-mode",
        choices=["idle", "slow_walk", "walk", "run"],
        default="idle",
        help="Initial joystick gait in planner and VR 3-point modes",
    )
    parser.add_argument(
        "--idle-base-transition-duration",
        type=float,
        default=2.0,
        help="Seconds for each arm transition into or out of idle base pose (default: 2.0)",
    )
    args = parser.parse_args()

    # Standalone VR3Pt test modes (exit after finishing)
    if args.vr3pt_test:
        print("Running VR 3-point pose visualizer test...")
        run_vr3pt_visualizer_test()
        print("VR 3-point pose visualizer test completed")
        exit(0)

    if args.vr3pt_live:
        print("Running VR 3-point pose live capture...")
        run_vr3pt_live_visualizer()
        print("VR 3-point pose live visualizer completed")
        exit(0)

    if args.vr3pt_realtime:
        print("Running VR 3-point pose real-time visualizer...")
        run_vr3pt_realtime_visualizer(update_hz=args.vr3pt_hz)
        print("VR 3-point pose real-time visualizer completed")
        exit(0)

    # Main execution modes
    # G1 robot visualization is enabled by default when vis_vr3pt is used
    with_g1_robot = not args.no_g1

    if args.manager:
        run_pico_manager(
            port=args.port,
            hand_intent_port=args.hand_intent_port,
            buffer_size=args.buffer_size,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            zmq_feedback_host=args.zmq_feedback_host,
            zmq_feedback_port=args.zmq_feedback_port,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
            input_source=args.input_source,
            teleop_mode=args.teleop_mode,
            initial_locomotion_mode=args.initial_locomotion_mode,
            recording_status_host=args.recording_status_host,
            recording_status_port=args.recording_status_port,
            teleop_control_host=args.teleop_control_host,
            teleop_control_port=args.teleop_control_port,
            idle_base_transition_duration=args.idle_base_transition_duration,
        )
    else:
        # Run legacy single-thread pose streaming
        run_pico(
            buffer_size=args.buffer_size,
            port=args.port,
            hand_intent_port=args.hand_intent_port,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
            input_source=args.input_source,
        )
