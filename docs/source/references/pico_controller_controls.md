# Pico controller controls for G1

These are the default controls for `--teleop-mode vr3pt`, including the
data-collection launcher's VR 3-point body modes. They use directly tracked
controllers and headset heading, without the SMPL body pose or A+X calibration.
`--legacy-vr-controls` selects the older SMPL/A+X flow in the Pico manager.
Full-body POSE and upper-body IK keep their existing controls.

- **A+B+X+Y:** start SONIC and enter VR control with both arms and hands held.
- **Middle-finger side button, left or right:** release, then hold to calibrate
  and enable that arm. Each engagement captures a fresh headset heading and
  controller reference against the held robot target. The other arm retains
  its own reference. Calibration is local and does not need a network round trip.
- **Release that side button:** hold that arm's last commanded target and hold
  that hand at its latest measured position.
- **Index trigger, with that arm enabled:** fully close on press, fully open
  on release, using the existing 0.60/0.40 hysteresis. There is no proportional
  aperture control. After each grip engagement, release then press the trigger
  to resume hand input; until then its held target is preserved.
- **A:** open both hands and smoothly return the arms and waist to the existing
  ready/base pose with neutral wrists (0° wrist roll, pitch and yaw).
  Locomotion remains available at the selected speed, and an
  active recording continues. Release and repress the side buttons after the
  return to resume arm control.
- **B:** smoothly recall the measured arm pose captured on 2026-09-11
  (sample 53586), using VR_3PT wrist position/orientation targets. Hold the grippers
  and preserve the waist target. Locomotion and an active
  recording remain available. Release and repress the side buttons afterward.
- **Left stick click:** toggle locomotion. Center both sticks afterward to arm it.
- **Left stick up/down:** forward/backward while locomotion is enabled.
- **Right stick left/right:** turn while locomotion is enabled.
- **Simultaneous forward/backward and turn input:** stop until only one action
  is requested. Left stick horizontal and right stick vertical are unused.
- **X:** smoothly recall the arm pose captured from measured robot feedback on
  2026-09-11. Uses VR_3PT wrist position/orientation targets, just like A's home
  return. Preserve the current waist target and hold both grippers. Locomotion
  and recording remain available; release/repress the side buttons afterward.
  X replaces the speed toggle; walking speed follows the configured initial gait
  (the data-collection launcher starts in slow mode).
- **Y:** smoothly recall the measured arm pose captured on 2026-09-11 at 16:49:57
  Europe/Berlin. Uses the same VR_3PT return as X, preserving the current waist
  target and holding both grippers. Locomotion and recording remain available;
  release/repress the side buttons afterward.
- **X+B:** start recording, or save the active recording.
- **Y+B:** stop and save the active recording as a failed episode.
- **Y+A:** discard the active recording and delete its temporary files.
- **UI safe-idle:** stop locomotion, hold hands, and return to idle. AXBY is
  required to re-enter teleop afterward; locomotion starts disabled.
- **UI stop / keyboard O:** retain the existing SONIC stop controls.

Solo face buttons and recording chords execute only when all face buttons are
released. AXBY cannot also invoke A, B, X, Y, or a recording chord; XB cannot invoke
B's resting return or X's saved pose. YA cannot invoke A's home or Y's saved pose,
and YB cannot invoke either saved pose. Any third face button cancels the
recording/pose action, including staggered releases. A+X has no action in this profile.

The headset defines the forward direction captured on each arm engagement;
turning it while holding a grip does not change that arm's reference. Head
motion itself does not drive the waist in this profile. Keep the headset facing
the direction you intend to use as forward when engaging, including when it is
worn around the neck. Reposition the controller while released, then engage
again to acquire a new reference without a target jump.

Calibrated controller targets pass through directly. Releasing a grip holds
that arm's last emitted target immediately. A fresh press captures a new
reference without moving the held target.

An independent watchdog latches a hold when the latest valid Pico sample is
**200 ms old**, checked on each manager tick (normally 50 Hz). It holds the last
commanded arm targets, stops locomotion, gates hand input, and cancels any A/B/X/Y
return. Fresh data alone cannot resume motion: release **both** side buttons and center both
sticks, then hold a side button to recalibrate and enable that arm. A+X is not
needed. A prolonged disconnect keeps the exact last emitted target across SDK
reconnection, without replacing it with robot visualization feedback. Pose presets
require an explicit B/X/Y command. The 200 ms check requires the manager loop to be running; publisher
failure still uses SONIC's existing receiver timeout. Holding Cartesian targets
does not mechanically lock the joints; SONIC continues balancing the robot.

The hand-intent protocol is now `sonic.hand_intent.v3` to carry independent
left/right hold commands. Update and restart **both the Pico manager and the
hand controller/server**, including the remote hand server on Orin. Updated
hand controllers also accept v2 publishers; older controllers reject v3.

The controls are covered by offline pose, manager, and simulated-hand tests.
Physical headset/robot behavior still needs validation on the robot.
