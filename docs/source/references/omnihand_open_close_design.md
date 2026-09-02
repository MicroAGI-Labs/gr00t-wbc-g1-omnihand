# Dex3/OmniHand controller design

Status: implemented baseline, reviewed against Atlas commit
[`14b1406`](https://github.com/MicroAGI-Labs/research-atlas-v0/tree/14b1406c8d3069870e298341be0cd204cee5515d).
The implementation does not move hardware unless the physical controller is
started with `--enable-command`; the launcher also requires an explicit
`OMNIHAND` acknowledgement.

## Review findings and clarified scope

The original description was unusually thorough, but these acceptance details
were missing or ambiguous:

- **Simulation meant two different things.** This change now provides both the
  deterministic controller/data path and Atlas's combined `scene_49dof.xml`:
  29 body actuators, 20 O10 active actuators, collision meshes, and twelve
  nonlinear equalities for the passive joints. Atlas's provenance still marks
  physical mass, collision geometry, and mount transforms as unqualified, so
  this is functional MuJoCo support rather than a high-fidelity sim-to-real claim.
- **The future policy producer has no wire endpoint or arbitration rule.** Only
  `pico_open_close` is admitted now. Before enabling `source=policy`, specify its
  endpoint, lease/priority relative to PICO, transition behavior, rate, and stale
  policy. A `source` string alone is not authority arbitration.
- **Clock domains need an explicit rule.** Remote `monotonic_ns` values are
  provenance only; watchdog age is measured from local receive time because
  monotonic clocks are not comparable across hosts.
- **Reconnect session semantics were unclear.** Transport reconnects retain the
  controller process session ID, discard the old target, seed a new measured
  hold, and require an intent received after reconnect. Process restart creates
  a new session ID and the collector must relock before another episode.
- **“Exact SDK names” assumes the Python binding exposes names.** Product and
  active-DOF identity are mandatory. Exact name/order comparison is also made
  when `get_joint_names()` exists; otherwise the pinned SDK/API order is the
  admitted contract and commissioning must verify channel/sign motion.
- **No thermal/current trip values are approved.** Non-zero vendor error masks
  stop tracking. Temperature/current are recorded but cannot be used as invented
  safety limits; a physical power cutoff and vendor-approved thresholds remain
  commissioning requirements.
- **Schema variability is per exporter process, not per episode.** The collector
  locks one profile/session before creating the LeRobot schema. Switching between
  Dex3 and O10 requires restarting the exporter and creating a new dataset.
- **Partial-hand operation and collection differ.** The controller supports a
  selected side for diagnostics, but integrated 43/49-channel collection is
  bilateral and rejects partial config/state.
- **External Dex3 is not part of this OmniHand milestone.** The existing C++
  Dex3 owner remains the rollback path. The shared Dex3 profile and discussion
  below reserve a compatible future boundary; no second Dex3 publisher was added.

## Decisions

- SONIC controls the 29-DOF body. Its legacy C++ path can retain Dex3 ownership,
  or a sibling end-effector controller can own both OmniHands.
- PICO publishes independent left/right click intent. A pose mapper converts it
  to complete 7- or 10-DOF joint targets before the controller.
- GR00T can later send predicted joint angles through the same full-joint target
  interface, bypassing the PICO open/close mapper.
- The controller is the hand authority and feeds the collector: intent,
  requested joint targets, applied joint targets, measured joints, and health.
- The click is recorded in addition to full joint data; it never replaces the
  GR00T hand action.
- Keep the current C++ Dex3 path as explicit rollback. Never run it together
  with the external Dex3 controller.
- No SONIC ONNX/TRT, body observation, or ZED changes. The hand geometry is the
  pinned Atlas/vendor asset bundle, not a locally reconstructed model.

## Architecture

```text
PICO body pose --------------------> SONIC body controller ------------+
                                                                        |
PICO click --> open/close mapper --> full-joint hand target             v
                                          |                       collector
GR00T joint output (future) ---------------+                             ^
                                          v                             |
                              end-effector controller ------------------+
                                   | requested/applied/measured/config
                                   +--> OmniHand O10 SocketCAN
```

“Sibling controller” means the same control-plane level as SONIC, not the same
executable. Process isolation keeps the Python-only OmniHand SDK and its
reconnect lifecycle out of the body controller. Dex3 remains in the legacy C++
owner until a separately reviewed external backend exists.

### Ownership modes in C++

Add `--hand-control legacy-dex3|external|none`:

- `legacy-dex3`: current C++ implementation, unchanged rollback path.
- `external`: C++ performs no hand init/read/write; external controller owns
  either backend.
- `none`: no hands.

Guard all current Dex3 initialization, lifecycle open/close, state reads,
`SetMaxCloseRatio`, `setAllJointsCommand`, `writeOnce`, logging, and command
copies. Publish `hand_control` in `robot_config`. The launcher rejects dual Dex3
ownership.

## Target and recording contract

The controller accepts a model-qualified target:

```text
source: pico_open_close | policy
profile: dex3.v1 | omnihand_o10.v1
sequence + monotonic timestamp
joint_names[left/right]
position_rad[left/right]
valid[left/right]
```

For the initial PICO behavior, close at trigger `>=0.60`, open at `<=0.40`, and
retain the prior state inside the deadband. Grip is ignored. Publish continuously
with a monotonic sequence; unavailable controller input is invalid, not “open.”

Record all of the following:

| Meaning | Dataset field | OmniHand width | Dex3 width |
|---|---|---:|---:|
| PICO source intent | `teleop.hand_closed` | 2 | 2 |
| Full desired joint pose; primary GR00T label | hand part of `action.wbc` | 20 | 14 |
| Target actually sent after safety limiting | `control.hand_applied_position` | 20 | 14 |
| Measured active joints | hand part of `observation.state` | 20 | 14 |

Canonical combined order, preserving the existing Dex3 dataset layout:

```text
left_leg[6], right_leg[6], waist[3], left_arm[7], left_hand[D],
right_arm[7], right_hand[D]
```

Thus `action.wbc` and `observation.state` are 43 values for Dex3 and 49 for
O10. Assemble by declared joint name, never by vector width alone.

The click-only dataset contains two commanded poses per side expressed in full
joint space. It can train open/close but not continuous dexterity. Any later
continuous joint-angle source can use the same controller/collector contract;
designing one is outside this scope.

The collector has `--hand-profile auto`, waits for controller config/state, and
locks the profile and session ID before creating an episode schema. It never
imports a hardware SDK. Missing/stale/wrong-width state blocks recording or
invalidates the frame; do not substitute zeros.

For collector wrist FK, keep the existing 43-DOF G1+Dex3 model internally with
neutral finger placeholders: the wrist frames are upstream of finger joints.
Assemble the real 43/49-value dataset arrays separately. The simulator itself
loads the pinned O10 MJCF/meshes; the collector does not need to import them.

## ZMQ protocol

Use topic-prefixed MessagePack with literal schema IDs and reject unknown major
versions.

### PICO to controller

Reuse the PICO PUB socket on port `5556`, topic `hand_intent`. The controller
SUB uses `RCVHWM=1` and `CONFLATE=1`.

`sonic.hand_intent.v1` fields:

```text
schema, sequence, monotonic_ns, source="pico"
left/right: {valid, closed, trigger}
```

There are deliberately no hardware names or joint vectors here.

### Controller to collector

Controller PUB port `5570`:

- `hand_config`, republished every two seconds: schema, session ID, backend,
  profile, selected sides, units, ordered joint names, position/velocity limits,
  open/closed poses, close scale, SDK version/commit.
- `hand_state`: schema, session/sequence/timestamp, backend/profile/mode,
  target source, intent sequence and booleans, requested/applied/measured
  positions, validity/connectivity, input/feedback ages, errors, temperature,
  and current.

Unselected sides are absent/null, never fake feedback.

## Controller safety contract

Provide two distinct commands, not a teleoperation “dry-run backend”:

- `probe`: structurally read-only interface/device/identity/feedback diagnostics.
- `run`: real controller, entered through the launcher's real-robot confirmation.

Mock devices are internal to tests/simulation only.

Startup and operation:

1. Validate backend/profile, sides, transport, product, DOF, names, shapes, and
   finite feedback.
2. Bound startup feedback. Clamp only deviations `<=0.01 rad`; abort above it.
3. Seed the first applied target from measured feedback and issue only that hold.
4. Require a fresh, increasing target before tracking.
5. Map click to the full pose, apply close scale, clip position, then slew from
   the previous applied target using `velocity_limit * min(dt, 0.1)`.
6. Publish requested, applied, and measured vectors separately.

Watchdog/recovery:

- Default target timeout: `0.5 s`, recorded and configurable.
- Stale input holds the last bounded target; it never opens automatically.
- SDK read/write loss releases all handles and reconnects on a bounded cadence.
- Reconnect reads and holds the new measured pose, discards old targets, and
  requires a post-reconnect target before tracking.
- Shutdown stops writes and retains the last bounded target; do not auto-open.
- Poll device health at about 5 Hz; latch non-zero error masks and leave tracking.
- Publish temperature/current, but do not invent trip thresholds: vendor-approved
  limits remain a commissioning gate.

## Hand profiles

### OmniHand O10 (`omnihand_o10.v1`)

Ten active joints per side; passive/coupled joints are not command or dataset
channels. Exact SDK order and admitted limits are from Atlas
[`model.py`](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/atlas/hands/omnihand/model.py):

| i | Right joint | min | max | Left joint | min | max | initial slew rad/s |
|---:|---|---:|---:|---|---:|---:|---:|
| 0 | `R_thumb_roll_joint` | -.0296 | 1.1213 | `L_thumb_roll_joint` | -1.1213 | .0296 | .164 |
| 1 | `R_thumb_abad_joint` | -1.6423 | .0453 | `L_thumb_abad_joint` | -.0453 | 1.6423 | .164 |
| 2 | `R_thumb_mcp_joint` | 0 | .8412 | `L_thumb_mcp_joint` | -.8412 | 0 | .308 |
| 3 | `R_index_abad_joint` | -.1640 | 0 | `L_index_abad_joint` | 0 | .1640 | .164 |
| 4 | `R_index_pip_joint` | 0 | 1.4835 | `L_index_pip_joint` | 0 | 1.4835 | .308 |
| 5 | `R_middle_pip_joint` | 0 | 1.4835 | `L_middle_pip_joint` | 0 | 1.4835 | .308 |
| 6 | `R_ring_abad_joint` | 0 | .1692 | `L_ring_abad_joint` | -.1692 | 0 | .164 |
| 7 | `R_ring_pip_joint` | 0 | 1.4835 | `L_ring_pip_joint` | 0 | 1.4835 | .308 |
| 8 | `R_pinky_abad_joint` | 0 | .1850 | `L_pinky_abad_joint` | -.1850 | 0 | .164 |
| 9 | `R_pinky_pip_joint` | 0 | 1.4835 | `L_pinky_pip_joint` | 0 | 1.4835 | .308 |

Open is ten zeros. Initial closed fixture, derived from Atlas
[`poses.py`](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/atlas/hands/omnihand/poses.py):

```text
right [ .728845, -.903265,  .757080, 0, 1.335150, 1.335150, 0, 1.335150, 0, 1.335150]
left  [-.728845,  .903265, -.757080, 0, 1.335150, 1.335150, 0, 1.335150, 0, 1.335150]
```

This is a kinematic fixture, not a hardware-qualified production grasp. Apply a
profile `close_scale` and commission at reduced travel. Use explicit
`set_all_active_joint_angles(list[10])`, not a vendor gesture, so targets remain
limit-able and recordable.

The initial slew values above come from the official SDK O10 API table. Atlas's
URDF-derived velocity values are much higher; do not adopt them without hardware
characterization. O10 position/velocity/current mixed mode is unavailable, and
the “torque” field is current in mA, not N-m. Use position commands only.

### Dex3 (`dex3.v1`)

Seven joints per side in explicit driver order:

```text
thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
```

Use the existing model-qualified names. Existing C++ limits:

```text
left  min [-1.05, -.724, 0, -1.57, -1.75, -1.57, -1.75]
      max [ 1.05, 1.05, 1.75, 0, 0, 0, 0]
right min [-1.05, -1.05, -1.75, 0, 0, 0, 0]
      max [ 1.05,  .742, 0, 1.57, 1.75, 1.57, 1.75]
```

Open is zero. Preserve midpoint close targets:

```text
left  [0, .163, .875, -.785, -.875, -.785, -.875]
right [0, -.154, -.875, .785, .875, .785, .875]
```

Match current gains (`kp=1.5`, `kd=0.1`) and measured-relative maximum write
delta (`0.25 rad`) initially. Current C++ normal commands encode `status=1,
timeout=0`; the bundled Python example uses `timeout=1`. Confirm the mode bit by
readback and restrained motion before promoting external Dex3.

## OmniHand installation and Thor host contract

Pin the official
[`agillink_omnihand_sdk`](https://github.com/AgibotTech/agillink_omnihand_sdk/tree/026740d9fdd8ba32b0605fa702a992b322076f1b)
as `external_dependencies/agillink_omnihand_sdk` at commit
`026740d9fdd8ba32b0605fa702a992b322076f1b` (v1.1.8, Mulan PSL v2).

Physical runtime:

- Dedicated uv-managed Python 3.12 environment: `.venv_omnihand`.
- Wheel: `linux/aarch64/python/omnihand-1.1.8-cp312-cp312-linux_aarch64.whl`.
- SHA-256: `3d089768492729d793c5e4b29ef23620a39fd26af27924a2f3ff78ca8f6d93ae`.
- Install the wheel explicitly after verifying the hash; do not declare an
  optional-submodule path in `pyproject.toml`.
- Keep the vendor SDK outside `.venv_teleop`; verify import and `probe` without
  moving hardware.

O10 needs separate 24 V power through CAN-FD; USB is data only. Never hot-plug a
powered hand.

| Side | gs_usb serial | Interface | Device ID |
|---|---|---|---:|
| Right | `2082395E534B50052` | `can10` | 1 |
| Left | `205B3973534B50042` | `can11` | 1 |

Use serial-bound `systemd-networkd`; do not use Thor's built-in `mttcan` `can4`
or `can5`. Required settings: 1 Mbps arbitration/80% sample point, 5 Mbps data/
75% sample point, FD on, tx queue 1000. Application startup verifies interface,
`gs_usb`, link parameters, product `OMNIHAND_2025`, DOF=10, exact names, and
finite feedback; it never silently reconfigures the host. `can10` observation is
not proof of bilateral setup—validate `can11` independently.

SDK construction uses
`OmniHand2025.create_hand_socketcan(HandType.LEFT/RIGHT, 1, interface)`, then
`init()`. The Python binding has no public close; reconnect drops the final
object reference. Native startup output writes fd 1, so JSON diagnostics must
redirect fd 1 to stderr around SDK construction.

## Implementation surface

Expected production/setup surface: 13 runtime/integration files, 7 host/vendor
paths, plus focused tests. Tests may be consolidated without collapsing runtime
boundaries.

New controller package:

1. `gear_sonic/end_effectors/{__init__,profiles,protocol,controller}.py`
2. `gear_sonic/end_effectors/backends/{__init__,base,sim,mujoco,omnihand}.py`
3. `gear_sonic/end_effectors/mujoco_driver.py`

Modify integration:

4. `gear_sonic/scripts/pico_manager_thread_server.py`
5. `gear_sonic/scripts/run_data_exporter.py`
6. `gear_sonic/data/features_sonic_vla.py`
7. `gear_sonic/scripts/launch_data_collection.py`
8. `gear_sonic/utils/mujoco_sim/{base_sim,configs,unitree_sdk2py_bridge}.py`
9. `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`

Add reproducible setup:

10. Atlas `g1_omnihand` bundle with vendor checksums/provenance
11. Installer-managed SDK checkout pinned to the reviewed commit (no gitlink)
12. `install_scripts/install_omnihand.sh`
13. four Thor `.link/.network` files
14. guarded CAN network install and bring-up scripts

Focused tests:

- profile/name/order/limit/pose invariants;
- protocol and PICO hysteresis/sequence rejection;
- first-hold, slew, watchdog, reconnect, partial-connect, and mock no-I/O;
- collector 43/49 layouts, requested/applied/measured distinction, stale/profile
  rejection, and unchanged wrist FK;
- C++ ownership modes and Thor network configuration.

## Delivery gates

1. **Offline:** mock controller, independent clicks, full-joint targets, safety
   state machine, 43/49 dataset assembly, vendor checksum verification, Atlas
   49-actuator scene load, active/passive dynamics, render, and SONIC smoke step.
2. **O10 read-only:** pinned install, both stable interfaces, identity/order,
   bilateral finite feedback/health, reconnect; no write path through `probe`.
3. **O10 restrained:** one side, low close scale, channel/sign verification,
   open/close, watchdog/reconnect hold, then bilateral. Do not test point/pinch;
   Atlas records a right middle-PIP pinch-to-point jam.
4. **Dex3 parity:** DDS readback, mode-bit decision, restrained comparison with
   `legacy-dex3`, and proof of single publisher ownership.
5. **Integrated collection:** exact metadata and full intent/requested/applied/
   measured signals for `dex3`, `omnihand`, and `none`; ONNX/TRT and ZED remain
   byte-for-byte untouched.

## Unresolved commissioning gates

- Use official O10 API slew values until the API-vs-URDF discrepancy is resolved.
- Qualify the proposed O10 fist with reduced `close_scale` before full travel.
- Confirm Dex3 timeout bit (`0` deployed C++ vs `1` Python example).
- Obtain vendor-approved temperature/current shutdown thresholds.
- Verify left `can11`; only right `can10` was previously observed active.
- Measure the physical hand mass, collision geometry, and wrist-adapter transform
  before treating the imported model as a high-fidelity contact/dynamics twin.

## Authoritative references

Atlas implementation at the audited commit:

- O10 [model/order/limits](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/atlas/hands/omnihand/model.py),
  [poses](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/atlas/hands/omnihand/poses.py),
  [hardware adapter](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/atlas/hands/omnihand/hardware.py),
  [safety/recovery](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/atlas/hands/omnihand/live_sink.py), and
  [health model](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/atlas/hands/omnihand/health.py).
- Hand [profiles](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/atlas/hands/profiles.py),
  [recording adapter](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/atlas/hands/recording_adapter.py), and
  [wire contract](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/atlas/hands/teleop/wire.py).
- Combined G1+O10 [scene](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/assets/robots/g1_omnihand/scene_49dof.xml),
  [generated model](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/assets/robots/g1_omnihand/g1_29dof_with_omnihand.xml),
  and [asset provenance](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/assets/robots/g1_omnihand/provenance.json).
- Thor [host config](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/configs/hosts/thor.yaml),
  [CAN installer](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/tools/install_omnihand_can_network.sh),
  [CAN bring-up](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/tools/bringup_omnihand_canfd.sh), and
  [OmniHand runbook](https://github.com/MicroAGI-Labs/research-atlas-v0/blob/14b1406c8d3069870e298341be0cd204cee5515d/docs/projects/omnihand-2025/README.md).

Pinned vendor reference:

- Official O10 [Python API](https://github.com/AgibotTech/agillink_omnihand_sdk/blob/026740d9fdd8ba32b0605fa702a992b322076f1b/doc/en/API_PYTHON_O10.md).

Current target integration points:

- [PICO publisher](../../../gear_sonic/scripts/pico_manager_thread_server.py)
- [collector](../../../gear_sonic/scripts/run_data_exporter.py)
- [dataset schema](../../../gear_sonic/data/features_sonic_vla.py)
- [launcher](../../../gear_sonic/scripts/launch_data_collection.py)
- [MuJoCo O10 driver](../../../gear_sonic/end_effectors/mujoco_driver.py)
- [C++ SONIC controller](../../../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp)
- [current Dex3 driver](../../../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/dex3_hands.hpp)
