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
  ready/base pose. Locomotion remains available at the selected speed, and an
  active recording continues. Release and repress the side buttons after the
  return to resume arm control.
- **B:** smoothly return the arms to the saved arms-on-legs planner resting
  pose. Hold the grippers and preserve the waist target. Locomotion and an active
  recording remain available. Release and repress the side buttons afterward.
- **Left stick click:** toggle locomotion. Center both sticks afterward to arm it.
- **Left stick up/down:** forward/backward while locomotion is enabled.
- **Right stick left/right:** turn while locomotion is enabled.
- **Simultaneous forward/backward and turn input:** stop until only one action
  is requested. Left stick horizontal and right stick vertical are unused.
- **X:** toggle normal/slow speed while locomotion is enabled. Center the sticks
  after switching. Slow mode uses a 0.1–0.6 m/s walking command and
  halves the turn rate; normal mode uses SONIC's normal walking speed. The
  existing data-collection launcher initially selects slow speed.
- **X+B:** start recording, or save the active recording.
- **Y+A:** discard the active recording using the existing exporter semantics.
- **UI safe-idle:** stop locomotion, hold hands, and return to idle. AXBY is
  required to re-enter teleop afterward; locomotion starts disabled.
- **UI stop / keyboard O:** retain the existing SONIC stop controls.

Solo face buttons and recording chords execute only when all face buttons are
released. AXBY cannot also invoke A, B, X, or a recording chord; XB cannot invoke
B's resting return. A+X and B+Y have
no action in this profile.

The headset defines the forward direction captured on each arm engagement;
turning it while holding a grip does not change that arm's reference. Head
motion itself does not drive the waist in this profile. Keep the headset facing
the direction you intend to use as forward when engaging, including when it is
worn around the neck. Reposition the controller while released, then engage
again to acquire a new reference without a target jump.

VR motion filtering is **off by default**: calibrated targets pass through
without speed, acceleration, jerk, or tracking-jump filtering. To enable it,
add `--no-disable-vr-motion-limiter` to `launch_data_collection.py`, or
`--enable-vr-motion-limiter` to the Pico manager. When enabled, releasing a grip
brakes that arm smoothly; a press during braking waits for the arm to stop.

An independent watchdog latches a hold when the latest valid Pico sample is
**100 ms old**, checked on each manager tick (normally 50 Hz). It holds the last
commanded arm targets, stops locomotion, gates hand input, and cancels any A/B
return. With the optional filter enabled, arms brake into the hold. Fresh data
alone cannot resume motion: release **both** side buttons and center both
sticks, then hold a side button to recalibrate and enable that arm. A+X is not
needed. A prolonged disconnect keeps the held target; returning to the legs
requires B. The 100 ms check requires the manager loop to be running; publisher
failure still uses SONIC's existing receiver timeout. Holding Cartesian targets
does not mechanically lock the joints; SONIC continues balancing the robot.

The hand-intent protocol is now `sonic.hand_intent.v3` to carry independent
left/right hold commands. Update and restart **both the Pico manager and the
hand controller/server**, including the remote hand server on Orin. Updated
hand controllers also accept v2 publishers; older controllers reject v3.

The controls are covered by offline pose, manager, and simulated-hand tests.
Physical headset/robot behavior still needs validation on the robot.
