# Atlas G1-D controls compared with this G1 stack

Reviewed 2026-09-10 against `MicroAGI-Labs/research-atlas`, branch
`g1d-teleop`, commit `dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9`.
The remote exposed `g1d-teleop`; the ElevenLabs guide in that branch still
names `g1d-teleop-11labs`. This comparison uses the available branch's code
and `deploy/g1d/elevenlabs.env`, including its enabled Pico mobile controls.

This is the source review and original implementation proposal. The selected
G1 adaptation is now implemented; see [current Pico controls](pico_controller_controls.md).
It uses binary DEX 1 control, keeps locomotion available during A's home return,
and omits Atlas's lift, pinch, presets, and audio controls. The comparison below
records the stack before this adaptation. No robot or hardware tests were run.

## Controller commands

Face-button actions in the recording-enabled path resolve after the complete
gesture is released. For example, X+B must not also execute X's speed toggle
or B's grasp toggle. Grip and trigger inputs remain sustained analog controls.
The default mobile mode is `arm-hand`, with all mobile stick outputs zero.
Arms/hands can remain engaged independently of the selected mobile mode.

| Input | Atlas behavior | Adaptation to this G1 + SONIC + DEX 1 stack |
|---|---|---|
| Left middle-finger grip | Calibrate and enable left arm while held; also enable left hand input | Arm clutch exists, but Atlas uses a different pose source and per-press alignment. Hand gating would be new. |
| Right middle-finger grip | Same for right arm/hand | Same changes, independently. |
| Release a grip | Hold that arm and that hand; release/repress required after startup or input loss | Our arm target brakes to a hold. DEX 1 currently continues following its trigger independently; exact Atlas hand behavior needs per-side hold and trigger re-arming. |
| Left/right index trigger | Analog finger closure while the corresponding grip is engaged | Our DEX 1 path provides binary open/close through hysteresis. Adopt grip gating if matching Atlas; proportional aperture is separate backend work. |
| A alone | Open both hands and move arms/waist to home; invalidate engagement | Adapt the existing smooth base/idle return, add an explicit open request, and require release/repress. Use a G1-specific home pose. |
| Y alone | Open hands and move to wider palms-up preset; current preset includes 26.5-degree waist pitch | Requires a G1-specific pose and validated transition. Do not copy G1-D/Revo2 joint angles or waist pitch directly. |
| B alone | Toggle both BrainCo hands between full-hand and pinch, only when both triggers are released | DEX 1 has one actuated gripper DOF per hand; these two finger configurations have no direct equivalent. Leave unused initially. |
| Left stick click | Select locomotion mode; click again to return to arm-hand; switches directly from height mode | Feasible new input-mode gate around the existing SONIC locomotion commands. |
| Right stick click | Select height/waist mode; click again to return to arm-hand; switches directly from locomotion | Feasible UI state, but its body outputs need adaptation to biped control. |
| Left stick up/down in locomotion | Base forward/backward velocity | Route through SONIC walking. Keep G1-specific gait and speed limits. |
| Right stick left/right in locomotion | Base yaw rate | Route through existing SONIC facing control. |
| Right stick up/down in locomotion | Raise/lower the G1-D lift | No physical lift on this G1. A planner body-height command is a possible substitute, subject to policy support and simulation verification. |
| X alone in locomotion | Toggle normal/slow speed | Feasible replacement for broad gait cycling. Active ElevenLabs profile uses 0.40/0.15 m/s base caps; those are G1-D values, not proposed G1 limits. |
| Left stick up/down in height mode | Raise/lower lift | Same biped-height adaptation requirement. |
| Left stick left/right in height mode | Integrate waist-yaw target | Must coordinate with SONIC's torso/head target; not an independent direct motor command. |
| Right stick up/down in height mode | Integrate waist-pitch target | Same coordination requirement, with G1-specific limits. |
| Right stick left/right in height mode | No action | Can match. |
| Left stick left/right in locomotion | No action | Our current mapping uses this for strafing. Exact Atlas parity would remove that control. |
| X+B | Start recording or stop/save the active episode | Already present. Preserve recorder acknowledgement and freshness rules. |
| Y+A | Discard the active recording | Already present as a discard flag; Atlas removes the unfinished recording. Destructive cleanup semantics differ. |
| X+Y | Toggle AVP upper-body control in optional hybrid mode | Not used by the Pico-only ElevenLabs profile. Our current X+Y selects the previous gait; reserve it only if hybrid control is added. |
| A+X, B+Y, A+B, A+B+X+Y | No corresponding actions in the reviewed Pico episode/button dispatcher | These already have mode/start/gait roles here. Replacing them requires changes to the startup/mode state machine, not just extra bindings. Preserve an explicit SONIC start action. |
| Menu/system buttons | No application action in the reviewed Pico control structure | Do not repurpose headset system controls implicitly. Our left-menu pause applies to legacy full-body POSE mode. |

Sources: [mobile input state machine](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/atlas-core/src/atlas/teleop/pico_mobile_base.py),
[grip gate](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/atlas-core/src/atlas/teleop/pico_upper_body.py),
[face gesture dispatcher](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/atlas-core/src/atlas/teleop/pico_episode.py),
[hardware runtime and hand mapping](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/atlas-core/scripts/robot/g1d_teleop_hardware.py),
[active profile](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/deploy/g1d/elevenlabs.env),
[current Y pose](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/atlas-core/configs/controllers/g1d_palms_up.toml).

## Calibration: the important architectural difference

Atlas reads the headset plus left/right controller poses directly from
XRoboToolkit. This stack currently derives its VR targets from the estimated
SMPL wrists and neck relative to the live pelvis.

For every newly engaged Atlas arm, the runtime creates a separate retargeter.
It captures the current headset position and yaw, controller pose, and robot
wrist pose for that engagement. The headset-derived axes stay fixed for that
arm until release. The other arm retains its own calibration. Thus, capturing
fresh headset alignment per press does not require replacing one shared
calibration for both arms.

The Pico configuration removes only headset yaw from the translation mapping;
it does not use the complete neck-orientation correction in our current
`ThreePointPose`. Wrist rotation follows a local rotation delta. In standalone
Pico mode Atlas supplies a neutral head target to IK and controls waist through
sticks, rather than forwarding headset motion to the torso.

This corrects the earlier discussion: a fixed shared neck correction is a
property of our current implementation, not a requirement for per-arm clutches.
To adopt Atlas's behavior, add direct controller/headset samples to the reader
and store a separate heading/pose anchor per arm. Merely remapping buttons
would not reproduce its tracking behavior. Headset yaw must still be sensible
in the operator's actual wearing position; source inspection does not establish
that wearing the headset around the neck yields reliable tracking.

Atlas also latches measured arm joint positions on release and resets its IK
filter on engagement. Our SONIC path holds conditioned Cartesian targets.
Exact measured-joint freezing is not equivalent on a balancing biped; retain
the current continuous commanded-target handoff unless a separate controller
change is deliberately validated.

Sources: [direct Pico sampling](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/atlas-core/src/atlas/teleop/pico_source.py#L170),
[reference math](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/atlas-core/src/atlas/teleop/g1d_retargeting.py#L150),
[per-side engagement](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/atlas-core/scripts/robot/g1d_teleop_hardware.py#L5080).

## Transition and recording details worth adopting

- A mode or speed change requires neutral sticks before movement resumes.
- Startup, presets, and source loss require released grips before fresh engagement.
- Grip release freezes a hand's target. After re-engaging, a fresh trigger
  release/press is needed before absolute trigger control resumes.
- A/Y presets open the hands. They must be treated as explicit object-release
  actions, not ordinary arm clutch release.
- Atlas leaves base/lift live during A/Y return but suspends arm, hand, and waist
  input. Our current return forces locomotion idle; adopting Atlas's live-base
  behavior requires separate biped validation.
- Atlas uses a 200 ms stale-input hold. Our existing publisher braking and
  receiver watchdog have different timing and recovery behavior; these should
  remain a coordinated end-to-end policy rather than copied constants.
- Recording commands run independently of the controller. Confirmed outcomes
  trigger cached spoken clips. This is not a live ElevenLabs voice-command UI.
- The reviewed branch discards an unfinished episode at 240 seconds without
  stopping teleop. This is an optional recorder feature, not a controller button.

## Terminal and launcher commands

These are inventories of the Atlas entrypoints, not commands to run against
the current G1 stack. `elevenlabs.sh` loads the Pico demo profile and delegates
to `thor.sh` except for `config`.

| Atlas command | Purpose / matching capability here |
|---|---|
| `deploy/g1d/elevenlabs.sh config` | Print active profile; can provide similar consolidated launcher output. |
| `... build` | Build Atlas image; our stack has different environments and deployment binaries. |
| `... static-check` | Static image checks; adapt to this stack's configuration checks. |
| `... offline-test` | Offline image tests; retain our own teleop regression suite. |
| `... image-status` | Inspect image revision/digests; specific to Atlas packaging. |
| `... preflight` | Read-only DDS discovery and deployment checks. |
| `... run` | Start robot controller after Atlas's explicit hardware admission; not a replacement for SONIC startup. |
| `... status` | Show tmux/container status. |
| `... attach` | Attach to the control tmux session. |
| `... logs` | Follow controller container logs. |
| `... stop` | Refuses implicit shutdown and prints the required explicit stop command. |
| `G1D_CONFIRM_STOP=CONTROLS_RELEASED deploy/g1d/elevenlabs.sh stop-confirmed` | Actual confirmed controller shutdown. This differs from the older guide's short `stop` example. |
| `deploy/g1d/thor.sh pedal-probe [device]` | Inspect optional AVP pedal input; not needed for Pico-only controls. |
| `deploy/g1d/thor.sh arm-sdk-probe` | Hardware compatibility probe publishing zero blend authority; unrelated to mapping buttons. |
| `deploy/g1d/data_collection.sh build` | Build separate recording image. |
| `... offline-test` | Exercise recording/export without hardware. |
| `... start "task description"` | Start camera/recorder/viewer session. |
| `... status`, `... attach`, `... logs` | Inspect or attach to the recording session. |
| `... stop` | Stop the camera/recorder session; does not stop a separate controller. |
| `... dataset-tool audit\|preview\|export ...` | Offline dataset inspection/export. |
| Recording terminal `c` | Start/save; this stack already has a ZMQ keyboard equivalent. |
| Recording terminal `x` | Discard; this stack already has a ZMQ keyboard equivalent. |
| Recording terminal `q` | Exit camera/recorder process; not a robot emergency stop. |
| Simulator terminal `1`–`7` | Select baseline, collision, robust, combined, unitree, unitree-collision, unitree-pink-barrier controller variants; simulator-only, not Pico hardware buttons. |

Optional standalone AVP uses the middle pedal to engage and the right pedal
for home return. Hybrid mode uses X+Y to enable/disable AVP upper-body tracking.
Neither applies to the Pico-only demo profile.

Sources: [Thor command dispatcher](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/deploy/g1d/thor.sh),
[ElevenLabs wrapper](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/deploy/g1d/elevenlabs.sh),
[recording commands](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/deploy/g1d/data_collection.sh),
[recording behavior](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/deploy/g1d/DATA_COLLECTION.md),
[simulator keys](https://github.com/MicroAGI-Labs/research-atlas/blob/dbc02c4eede71aec6b9ebbeea75ff1f9e4a4fba9/atlas-core/src/atlas/teleop/terminal_keys.py).

## Proposed implementation order

1. Add direct headset/controller sampling and per-arm headset-yaw calibration
   on grip engagement. Preserve current SONIC target continuity, motion limits,
   and explicit policy startup. Remove the A+X calibration prerequisite from
   the new input profile after initial target preparation.
2. Introduce a single gesture dispatcher for A/Y presets, X speed selection,
   and existing X+B / Y+A recording. Add stick-mode selection and neutral gates;
   account explicitly for the current strafe mapping and legacy gait chords.
3. Add per-hand hold and fresh-trigger re-arming to the external hand protocol
   and controller. Existing `hold` is global, so independent hand clutches need
   a compatible protocol/controller extension, not only a publisher edit.
4. Implement a G1 home preset first. Add a separate G1 palms-up preset only
   after checking reachability, gripper mounting, and the transition in simulation.
5. Evaluate height/waist mode through SONIC separately. Keep lift and BrainCo
   pinch commands unavailable until a meaningful G1/DEX 1 behavior is defined.

The main code boundaries are `gear_sonic/scripts/pico_manager_thread_server.py`,
the input readers and `vr_arm_clutch.py`, gesture dispatch, and the external
hand protocol/controller. This is a broader change than the initial arm-only
clutch; button parity, pose-source parity, and robot-controller parity are
different pieces of work.
