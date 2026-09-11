# Data collection: PR handbook

This is the operating and design guide for the `data-collection` branch and
[PR #30](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/pull/30).
It covers the G1 robot, PICO controller teleoperation, DEX 1 grippers on Orin,
cameras and recording on Thor, and the browser dashboard. It also identifies
the supported local-hand, OmniHand, simulation, and body-control alternatives.

The daily configuration is **direct controller tracking, independently enabled
arms, slow walking up to 0.6 m/s, DEX 1 grippers on Orin, ego and wrist videos,
and a 50 Hz recording target**. Those choices come from the wrapper below;
some defaults of the generic launcher are different.

This guide describes the implementation in this branch. A configured rate is
not evidence of sustained hardware throughput, and the offline tests do not
establish physical tracking accuracy or the quality of a recorded demonstration.

## Contents

1. [Daily launch](#daily-launch)
2. [Machine and process ownership](#machine-and-process-ownership)
3. [Understanding the terminal dashboard](#understanding-the-terminal-dashboard)
4. [Operating a recording session](#operating-a-recording-session)
5. [Controller controls and reference frames](#controller-controls-and-reference-frames)
6. [Hand connection and fault recovery](#hand-connection-and-fault-recovery)
7. [Cameras and timestamps](#cameras-and-timestamps)
8. [How the collector constructs a row](#how-the-collector-constructs-a-row)
9. [Optional sender-time recording](#optional-sender-time-recording)
10. [Dataset schema and action meaning](#dataset-schema-and-action-meaning)
11. [Saving outcomes and uploading](#saving-outcomes-and-uploading)
12. [Installation and alternative launches](#installation-and-alternative-launches)
13. [Troubleshooting by symptom](#troubleshooting-by-symptom)
14. [Design decisions and tradeoffs](#design-decisions-and-tradeoffs)
15. [Code tour and cleanup boundaries](#code-tour-and-cleanup-boundaries)
16. [Validation and remaining work](#validation-and-remaining-work)

## Daily launch

On the prepared Thor checkout:

```bash
cd /home/unitree/worktrees/gr00t-wbc-g1-pr3-wrist-cameras
bash tools/collect_dex1_orin.sh --check-only
```

`--check-only` validates local prerequisites and prints the remote hand command.
It does not start services, open the grippers, or check remote SSH authentication.
A successful check therefore does not establish that Orin, the cameras, or the
robot are ready.

To start the dashboard:

```bash
bash tools/collect_dex1_orin.sh
```

The wrapper selects these launcher options:

| Setting | Daily wrapper selection |
|---|---|
| Hands | `--hand-backend dex1` |
| Hand host | `--hand-server-host 192.168.123.164` |
| User interface | `--remote-ui` |
| Cameras to record | Ego view plus `--record-wrist-cameras` |
| Collection frequency | `--data-exporter-frequency 50` |
| Body mode, inherited from launcher | `vr3pt-slow-planner` |
| PICO input, inherited from launcher | `xrt` |
| Initial walking speed, passed by launcher | `slow_walk` |
| Time alignment, inherited from launcher | Latest available samples |

The wrapper does **not** select ZED stereo/depth recording or sender-time
alignment. Append their flags when deliberately collecting those schemas:

```bash
bash tools/collect_dex1_orin.sh --record-zed-stereo
```

Launching replaces the existing `sonic_data_collection` tmux session. Finish
and flush an active recording before relaunching. The launcher checks SSH
access before replacing the dashboard, then manages the camera service and
starts the requested processes. Installing or editing code does not update
Python processes that are already running.

The worktree directory's historical name does not determine the active branch.
Use these read-only checks when unsure which checkout is being used:

```bash
git branch --show-current
git log -1 --oneline
git status --short --branch
```

The intended branch here is `data-collection`. PR #30 replaces PR #3; the
older PR remains review history, not another deployment configuration.

### Open the browser

On Thor, open `http://127.0.0.1:8080`. From a laptop, run this in a separate
terminal, replacing `THOR_HOST` with the reachable Thor hostname or address:

```bash
ssh -N -L 8080:127.0.0.1:8080 unitree@THOR_HOST
```

Then open the same loopback URL on the laptop. Keep the SSH tunnel running.
The server binds to loopback by default; `--remote-ui` means a browser served
through the tunnel, not a public HTTP listener.

The daily wrapper enables the browser dataset workflow. Recording saves locally
without a Hugging Face connection. Set the task prompt before recording and
select an upload destination when ready to upload saved episodes.
The default organization is `MicroAGI-Labs`, with private datasets selected by
default. The Thor account must already have Hub credentials with the required
access. Credentials are used on Thor, not entered into this README or sent as
part of the teleop command stream.

## Machine and process ownership

```text
PICO headset/controllers
          |
          v
Thor: PICO manager -- body/planner targets --> Thor: C++ SONIC --> G1 body
          |                                         |
          | hand intent                             | robot state/config
          v                                         v
Orin: hand supervisor/controller --------------> Thor: collector
          |                        hand state       ^       |
          v                                         |       v
     two native DEX 1 workers                 camera server  local dataset
          |                                         ^       |
          v                                         |       v
     two USB grippers                      ZED + USB wrists  Hub uploader

Thor: browser viewer -- recording commands --> collector
                     -- reconnect request --> hand supervisor
                     -- idle request -------> PICO manager
                     -- disconnect ---------> SONIC tmux pane
```

Thor owns body inference, headset input, cameras attached to Thor, the collector,
the browser, video encoders, and upload work. Orin owns only the remote DEX 1
stack and its USB adapters. The laptop browser displays and controls the Thor
service; it is not the recorder. A separately configured camera host is possible
in latest-sample mode, but the daily configuration uses local cameras.

There is one owner of physical hand commands. With DEX 1 or OmniHand selected,
the launcher passes `--hand-control external` to SONIC. The external hand
controller owns those devices, while C++ owns the body. This prevents both
paths from driving the same end effectors.

### Default endpoints

Ports below are TCP/ZMQ unless identified as HTTP. A publisher generally binds
on the named host and consumers connect to it. These are defaults, not a port
scan of the current machine.

| Port | Owner in the daily setup | Consumers and purpose |
|---|---|---|
| 5555 | Thor camera server | Collector and viewer; image/depth packets |
| 5556 | Thor PICO manager | SONIC and collector; `manager_state`, `planner`, and mode-dependent `pose` |
| 5557 | Thor C++ deployment | Collector; `g1_debug` and periodically repeated `robot_config` |
| 5569 | Thor PICO manager | Orin hands; dedicated latest-only `hand_intent` |
| 5570 | Orin hand controller | Thor collector and UI; `hand_config` and `hand_state` |
| 5572 | Thor browser control hub | Orin supervisor; `hand_control` reconnect requests |
| 5573 | Thor browser teleop controls | PICO manager; return-to-idle requests |
| 5574 | Orin clock server, opt-in | Thor recorder; read-only clock exchange for sender-time recording |
| 5574 | Thor PICO body diagnostics | Browser; a separate socket on a different host from the Orin clock server |
| 5580 | Thor browser recording controls | Collector; recording and dataset-configuration commands |
| 5581 | Thor collector | Browser; recorder, quality, finalizer, upload, and rate status |
| 8080 | Thor HTTP viewer, loopback | Local browser or SSH-forwarded laptop browser |

The simulator may also use port 5571 for external-hand feedback. Robot DDS
transport, vendor headset transport, SSH, and optional visualization endpoints
are separate from this table. Changing an endpoint requires updating both
ends; the launcher forwards its supported port flags to the processes it owns.

## Understanding the terminal dashboard

The launcher creates one tiled tmux window named `data_collection` inside the
`sonic_data_collection` session. Pane numbers identify roles; their positions
on screen change with terminal size and tmux layout.

| Pane | Daily hardware setup | What the output means |
|---|---|---|
| 0 | C++ SONIC deployment | Model loading, robot communications, inference and body-control diagnostics. Starts through `gear_sonic_deploy/deploy.sh`. |
| 1 | PICO manager | Headset connection, controller state, mode transitions, tracking and planner commands. A shell supervisor restarts unexpected worker exits. |
| 2 | Data exporter | Waits for config, admits or blocks frames, reports recording events, local finalization, and upload progress/errors. |
| 3 | Browser camera viewer | HTTP server, camera subscription, recorder commands/status, and hand controls. With another launch profile this can be the native viewer or an unused pane. |
| 4 | Remote hands over SSH | Orin supervisor, Python hand controller, and native gripper logs. This is not a second body controller. |
| 5 | Camera service journal | `journalctl --follow` for `composed_camera_server.service`. This pane observes an independently owned system service. |

Simulation inserts the simulator at pane 4. External hands, if enabled, then
occupy pane 5. Hardware camera-log panes appear after any hand pane; they are
omitted in simulation. Reusing an existing remote hand service with
`--no-start-hand-server` omits the launcher-owned hand pane.

### Why multiple processes and threads appear

The launcher starts components in separate environments because their
requirements differ: headset vendor bindings in `.venv_teleop`, collection and
browser dependencies in `.venv_data_collection`, camera SDKs in `.venv_camera`,
and the prepared remote hand runtime in `.venv_hands` on Orin.

The hand server is a supervisor, which starts a Python controller, which starts
one native worker per gripper. Native DEX 1 workers run motor loops at 200 Hz;
the Python hand loop and published state target 50 Hz. A separately paced
publisher can repeat a held state, so 50 status messages are not necessarily
50 new device measurements.

Within the exporter, video encoding uses one worker thread per video, local
finalization uses a worker thread, and Hub upload uses a worker thread plus an
isolated upload subprocess. These are background tasks owned by the collector,
not additional collectors. The camera server also has independent capture
workers so one camera does not own the capture loop of the other cameras.

PICO's native XRT extension can abort its worker. The shell in pane 1 restarts
an unexpected exit after two seconds. A clean exit or Ctrl-C stops that loop.
The hand supervisor has its own recovery policy described below; it is not an
unconditional restart loop for every kind of failure.

### Useful tmux commands

```bash
# Inspect the actual roles and current commands without changing the session.
tmux list-panes -t sonic_data_collection:data_collection \
  -F '#{pane_index} #{pane_current_command} #{pane_current_path}'

# Reattach to a running dashboard.
tmux attach -t sonic_data_collection

# Read the last 100 lines from the recorder pane.
tmux capture-pane -p -t sonic_data_collection:data_collection.2 -S -100
```

Click a pane to focus it. `Ctrl+b`, then an arrow changes focus. `Ctrl+b`, then
`z` zooms or unzooms a pane; `Ctrl+b`, then `d` detaches while everything keeps
running. Ctrl-C affects the focused process. `Ctrl+\` kills the entire session;
it is not a graceful recording-save command.

Closing the camera-log pane stops the log follower, not the camera service.
Closing a launcher-owned remote hand pane ends its SSH session and stops its
remote worker. Closing the laptop browser only closes the client. Detaching
tmux leaves the complete dashboard running.

## Operating a recording session

1. Launch the intended profile and open the browser. Confirm that the required
   cameras are visible and each hand has fresh feedback. Read the recorder
   status as well as the camera picture: a live preview alone does not establish
   that robot configuration, teleop mode, or dataset setup is ready.
2. Set the Hub repository and task prompt in the browser. Wait for the recorder
   to acknowledge configuration. The default prompt describes a tote/conveyor
   task; replace it when collecting a different task.
3. Press and release **A+B+X+Y** to start SONIC and enter direct VR control.
   Both arms and hands initially remain held. Locomotion starts disabled.
4. Release the side grips, face the intended forward direction with the headset,
   then hold the left or right middle-finger side button to enable that arm.
   Each arm calibrates independently against its held target.
5. Release and then press that hand's index trigger to arm open/close input.
   Check the hand status and response before collecting a take.
6. Click the left stick to enable walking, then center both sticks to arm it.
   The daily launch starts in slow mode. Use forward/back or turn separately.
7. Press and release **X+B**, or use the browser recording control, to begin the
   episode. Perform the demonstration. Recording state and teleop state are
   independent: pressing a record button does not start SONIC or engage grips.
8. Press and release **X+B** again before the four-minute limit to accept the
   take, **Y+B** to save it as failed, or **Y+A** to discard it.
   The default `--max-episode-duration-s 240` automatically discards a take
   that reaches four minutes and removes its temporary files.
   Wait for local finalization to complete. Inspect the validation result and
   upload status; an accepted take can still fail a later disk operation.
9. Begin another take when the recorder is ready. Background uploads can
   continue while recording; a backed-up local finalizer can temporarily block
   the next start.

A and B arm-return commands can be used during a recording. They do not end
that episode. A grasp-free episode may fail the default hand-activity check;
for a task intentionally requiring no hand action, configure the exporter with
`--no-require-hand-activity` rather than assuming all saved takes pass.

### Ending the session

Save the active take as successful or failed and wait for the local finalizer. Stop the
exporter with Ctrl-C in pane 2 and let its cleanup run. It drains local saves
and waits up to the configured upload timeout, 300 seconds by default, for Hub
work. A forced process kill can interrupt this cleanup.

Use the browser's idle/stop controls according to the intended robot state.
**Safe idle** requests an arm return and disables locomotion; **Disconnect
SONIC** is a separate, confirmed action that sends Ctrl-C to the deployment
pane. Once the recorder has exited and the robot has been handled, the remaining
dashboard can be closed:

```bash
tmux kill-session -t sonic_data_collection
```

The independently managed physical camera service can remain running afterward.

## Controller controls and reference frames

This table is for the default direct-controller `vr3pt` profile. The older
SMPL-based and upper-body IK profiles have their own entry/calibration sequence;
do not mix their A+X instructions into the daily direct-controller workflow.

| Input | Effect |
|---|---|
| A+B+X+Y | Start SONIC and enter VR mode with both arms/hands held. |
| Left/right middle-finger side grip | After release, hold to capture that arm's reference and enable it. |
| Release side grip | Hold that arm's last emitted target and that hand's latest measured position. |
| Index trigger with its arm enabled | Binary full close on press, full open on release; 0.60/0.40 hysteresis. Release then press after grip engagement. |
| A | Open both hands; smoothly return arms and waist to the ready/base pose. Walking and recording remain available. |
| B | Recall the measured arm pose captured on 2026-09-11 (sample 53586), hold grippers, and preserve waist target. Walking and recording remain available. |
| Y | Recall the measured arm pose captured on 2026-09-11 at 16:49:57 Europe/Berlin, hold grippers, and preserve waist target. Walking and recording remain available. |
| Left stick click | Toggle locomotion; center both sticks before movement can resume. |
| Left stick up/down | Forward/back while locomotion is enabled. |
| Right stick left/right | Turn while locomotion is enabled. |
| Forward/back and turn together | Command no movement until only one action is requested. |
| X | Recall the measured arm pose captured on 2026-09-11 (sample 55826), hold grippers, and preserve waist target. Walking speed follows the configured initial gait. |
| X+B | Start recording or accept the active episode. |
| Y+B | Stop and save the active episode as failed. |
| Y+A | Discard the active episode and delete its temporary recording files. |
| UI safe idle | Stop locomotion, hold hands, return to idle; AXBY is needed to re-enter. |
| UI Disconnect SONIC / existing keyboard O stop | Stop controls, separate from the face-button start/mode controls. |

Solo face buttons and recording chords execute when **all face buttons have
been released**. AXBY does not also fire A, B, X, or a recording chord. XB does
not also invoke B's arm return. A+X and B+Y have no action in this default profile.

Slow walking maps the active stick range to a **0.1–0.6 m/s command**, with half
the normal turn rate. It does not force a constant 0.6 m/s velocity. Actual robot
speed depends on the controller and physical conditions. Left-stick horizontal
and right-stick vertical axes are unused here.

### What calibration actually captures

Each grip engagement captures the current controller pose and headset heading
against that arm's already held robot target. Subsequent controller displacement
is interpreted relative to that reference. This permits releasing a grip,
repositioning a controller, and engaging again without jumping the target.
The other arm retains its independent reference.

Headset heading defines forward at engagement. Turning the headset while a grip
is held does not rotate that arm's reference. Head motion does not continuously
drive the waist in this profile. If the headset is worn around the neck, its
orientation still matters when engaging a grip.

Controller targets pass through without the removed optional motion conditioner,
velocity/acceleration/jerk filters, or trace/tuning features. The explicitly
requested A/B/X/Y returns still interpolate over two seconds by default, controlled
by `--idle-base-transition-duration`. A generated return trajectory and filtering
every live controller sample are different operations.

### Tracking loss and re-entry

A manager watchdog latches a hold when the latest valid PICO sample reaches
200 ms old, checked on each manager tick, normally at 50 Hz. It holds the last
commanded arm targets, stops locomotion, gates hand input, and cancels an active
A/B/X/Y return. Fresh packets alone do not resume motion.

Release **both** side grips and center both sticks, then hold the desired grip
to capture a new reference. A+X is not required. Direct-controller disconnects
retain the held target; B/X/Y explicitly recall their saved arm poses. Cartesian
hold is still a target tracked by SONIC while it balances, not a mechanical
joint lock. If the manager itself stops executing, the receiver's disconnect
handling is the relevant fallback; the Python watchdog cannot run in a dead
process.

The hand intent schema is `sonic.hand_intent.v3`, which carries independent
left/right holds. Update and restart the PICO manager **and** the hand controller,
including Orin's checkout, when moving from the earlier protocol. Current hand
controllers also accept v2; old controllers do not understand v3.

## Hand connection and fault recovery

The stack already retries recoverable connection failures. Routine USB transport
loss does not require repeated clicks on **Reconnect Hands**. A persistent
motor fault follows a different path and will not be cleared by repeated
transport retries.

### Connection sequence

On startup or recoverable reconnect, the hand controller opens both configured
devices, validates feedback, seeds targets from measured positions, and sends a
zero-delta hold. It discards intent queued during the outage before accepting
fresh input. It does not replay old close commands against a newly connected
motor.

DEX 1 uses serial identities instead of whichever `ttyUSB` number Linux happens
to assign. The configured pair is:

| Side | Serial adapter identity |
|---|---|
| Left | `FTBQ776H`, `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBQ776H-if00-port0` |
| Right | `FTBWJBC1`, `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBWJBC1-if00-port0` |

These map the physical pair used for this setup. Moving adapters between hosts
does not change their identities; substituting hardware requires checking the
mapping. A visible USB adapter establishes neither motor power nor successful
adapter-to-motor communication.

| Condition | Automatic behavior | What the operator should inspect |
|---|---|---|
| Missing adapter or recoverable USB reply failure | Retry at the configured interval, normally one second. | Power, USB connection, serial identity, adapter-to-motor cable, and Hands pane error. |
| Recoverable external SDK/controller failure | Supervisor replaces the worker when it exits with the recoverable code, 75. | Repeated exception text and whether feedback becomes fresh. |
| DEX 1 health/control fault | Fault is latched; automatic motion reconnection stops. | Reported motor mask and health values; address the cause, then request reconnect. |
| Native worker wedged while supervisor remains alive | Browser reconnect can terminate and replace that worker. | Hands pane and whether a new PID starts after the request. |
| Remote SSH session exits | The launcher-owned hand pane ends; it is not an SSH reconnect daemon. | Robot network and SSH connection; restore the remote service/session. |
| Old hand-intent protocol | Decoder rejects incompatible messages. | Matching branch/protocol on Thor and Orin, followed by process restart. |

The parent supervisor owns the reconnect command socket outside the vendor
worker. The UI can therefore request a clean restart even when a native SDK
call is stuck. It does not repair power, cables, motor temperature, voltage,
or an unresolved hardware error. An unexpected supervisor exit also cannot
be repaired by publishing requests to that absent supervisor.

DEX 1 native workers distinguish transport telemetry from motor error bits;
`controller.py` treats the low four error bits as motor faults. Read the actual
error and device values before assigning a cause to a numeric mask. Do not
infer a specific physical failure solely from “disconnected” in the browser.

The default DEX 1 stroke time is 1.35 seconds, with an allowed range of 1.35–30
seconds. The launcher forwards `--dex1-transition-duration` to its local or
SSH-launched worker. For a separately managed hand service, configure the
service itself. The hand target watchdog is separate from PICO's 200 ms tracking
watchdog; its default target timeout is 0.5 seconds.

Physical motor health checks, hardware limits, and stop behavior remain part
of the device/controller path. The cleanup removed optional VR pose filtering;
it did not remove the native checks used to admit physical hand commands.

## Cameras and timestamps

### Capture, publication, reception, and recording are different rates

Each camera has a capture worker. The composed server publishes newly available
camera data, keeping original per-camera capture timestamps. The receiver merges
streams independently so a new ego image does not erase a wrist update simply
because those cameras completed at different times.

The `thor-jr` wrist profile selects the physical JR0001/JR0002 pair by stable
`/dev/v4l/by-id` paths. It requests 1280×720 MJPEG capture at 60 FPS, two V4L
buffers, continuous draining, and 640×480 RGB output. The profile also selects
60 Hz server publication. There is no fallback to an arbitrary video device
when a configured side is missing or conflicting.

ZED supports HD720 capture at 60 FPS in this setup. Generic camera-server
publication defaults to 30 Hz unless overridden or the wrist profile sets it.
The recorder's 50 Hz setting does not change camera hardware FPS or service
configuration. A 30 FPS camera can appear in a 50-row/s dataset through reuse
of its latest image.

Read-only checks for the actual service configuration and device identities:

```bash
systemctl cat composed_camera_server.service
systemctl status composed_camera_server.service --no-pager
journalctl -u composed_camera_server.service -n 100 --no-pager
ls -l /dev/v4l/by-id/
```

Configure the camera server on the host that physically owns the cameras.
`--record-wrist-cameras` adds required videos to the recording schema; it does
not attach USB devices, change the installed service, or select the JR profile.
See the [camera setup tutorial](docs/source/tutorials/data_collection.md) for
service installation and the fixed wrist profile.

### What each timestamp means

| Timestamp family | Meaning and limitation |
|---|---|
| Per-image wall timestamp | Host capture boundary, stored as `capture.<camera>_source_timestamp_ns` where defined. Useful for source identity; not a calibrated hardware exposure timestamp. |
| Per-image `capture_monotonic_ns` | Host monotonic capture time, independent for each camera. Used by sender-time selection. |
| Camera publisher timestamp/sequence | Packet publication identity; one packet can include channels captured at different times. |
| Receiver monotonic timestamp | Arrival at the recorder. Independent of the publisher's clock on another host. |
| LeRobot `timestamp` | Logical row time, `frame_index / fps`. Does not prove physical simultaneity or continuous acquisition. |

A republished cached image keeps its capture identity. It must not count as a
new exposure or advance that camera's sender-time history. Stereo/depth channels
from one ZED grab retain the same source timestamp. Separate USB cameras are
not hardware-triggered together by this implementation.

The UI's publisher rate estimates source cadence; receiver rate counts distinct
captures consumed by the collector. A stalled wrist can drop to zero while the
ego camera keeps running. The below-45-Hz highlighting is diagnostic, not an
episode pass/fail threshold. Inspect the required cameras individually.

### ZED depth: transport versus dataset

The camera transport can carry rectified RGB eyes and a lossless float32 metric
depth array. Native orientation is retained; no default 180-degree rotation is
applied. The right-eye RGB is the existing `ego_view`.

`--record-zed-stereo` adds `ego_view_left` and `ego_view_depth` **video features**.
The recorded depth feature is a 640×480, three-channel uint8 visualization:
the SDK's depth view when supplied, otherwise a conversion of metric depth
using the exporter's 0–10 m display range. It passes through the video encoder.
It is **not a lossless metric-depth dataset** and should not be used as one.
Metric-depth archival would require a separate storage feature and validation.

## How the collector constructs a row

The collector is ROS-free. It subscribes to C++ robot state/configuration, the
PICO manager's body-control stream, external hand state when configured, and
the camera server. SONIC inference and physical hand control continue in their
own processes; recording is a consumer of their outputs.

### Startup fixes the schema

The exporter first waits for `robot_config` from C++. The default timeout is
zero, meaning wait indefinitely. When body configuration says hands are external,
it also waits for `hand_config`; that default is also indefinite. A terminal
apparently waiting at this stage may be missing an upstream publisher rather
than performing disk work.

It resolves the hand profile, derives joint names and widths, adds the selected
camera features, and creates or resumes the local dataset. The launcher and
standalone exporter read `~/.config/sonic/recording.json` for `root_output_dir`
and `dataset_name`; explicit CLI options override those values. Without a saved
configuration, an omitted name creates a timestamped directory under `outputs/`.
The explicit name selects
a local directory; it is not automatically the Hub repository name.

The schema includes the configured FPS, hand profile, and selected video fields.
Sender-time and latest-sample datasets cannot be silently mixed: synchronization
feature mismatches are rejected before video writers open. Treat changes in
hands, cameras, or timing configuration as a new dataset unless compatibility
has been checked.

### Default technique: latest available samples

At each loop tick, the collector:

1. Polls robot state, pose/planner/manager messages, external hands, and cameras.
2. Updates its latest values and receipt/source diagnostics.
3. If recording, checks the selected mode, freshness, shape, finite required
   numeric values, nonzero motion token, required cameras, and external hand
   validity/feedback.
4. Builds one row from the available snapshot, including measured observations,
   requested actions, applied hand commands, teleop targets, and provenance.
5. Queues its video frames and appends the numerical fields to the episode
   buffer, then checks recording commands and publishes status.

The loop follows an absolute monotonic deadline. At 50 Hz, deadlines are 20 ms
apart. This avoids accumulating a small sleep error every iteration. If blocking
work causes a large overrun, it advances the deadline rather than replaying an
arbitrary backlog of ticks.

This is an **asynchronous latest-sample collector**. It does not wait for every
source to have the same timestamp, interpolate measurements, or guarantee that
an image and a motor observation describe the exact same instant. Samples can
be reused while they remain valid; capture fields preserve evidence of reuse.
A future training pipeline must decide which recorded action/observation pairing
and horizon to use. The exporter does not shift action labels into the future.

### Freshness flags and admission

| Required source | Default warning threshold |
|---|---|
| Robot state | 100 ms |
| Required camera frames | 100 ms |
| External hand state | 200 ms, with additional validity/device-age checks |
| Active planner or SMPL stream | 200 ms |

Latest mode uses receiver-local monotonic freshness and available per-source
age/identity checks. It does not subtract a raw Orin monotonic timestamp from a
Thor monotonic timestamp. Receipt freshness alone cannot remove network delay;
source and receive fields are kept for diagnosis. Sender-time mode below uses
a common mapped acquisition timeline instead.

In VR3PT/IK planner modes the row requires planner VR positions and orientations.
POSE mode requires the SMPL pose stream. Recording must start in the selected
teleop mode. Base-pose mode 3 is admitted as an intentional pause within an
existing episode, but a successful episode must also contain the selected
teleop mode.

In latest-sample mode, a stale camera, robot, or teleop sample remains in the
recording with its original source timestamps. It does not fail the episode.
The quality report in `meta/episode_quality.jsonl` stores
`validation.frame_diagnostics`: each reason has a count, maximum age, and
zero-based inclusive `frame_ranges` identifying saved table rows and video
frames to review or crop. Hand warnings remain in `validation.hand_diagnostics`.
`flagged_frame_count` counts affected rows once across all warnings.

Missing or malformed required measurements cannot form a row. The recorder
reports these omissions in `validation.input_gaps`, with elapsed recording
times and the next saved frame index (which can equal episode length at the
end). Such gaps remain validation errors after the first saved row; the take
is still preserved on Stop & Save. Optional sender-time recording retains its
strict synchronization admission checks described below.

### Why 50 Hz does not mean 50 unique images

Suppose a camera exposes at 30 FPS and the recorder emits rows at 50 Hz. Some
neighboring rows must share an image. Even a camera configured for 60 FPS may
stall or deliver late, so a 50 Hz loop does not prove a unique exposure per row.
Repeated packet publication is also different from a new device measurement.

The row counter, publisher frequency, distinct receiver frequency, source
sequence, and source timestamp answer different questions. Keep all of them
when assessing a run. The previous `minimum_recording_rate_hz` argument remains
for CLI compatibility; rates are diagnostic and do not determine acceptance.

## Optional sender-time recording

Enable this explicitly with a fresh evaluation dataset:

```bash
bash tools/collect_dex1_orin.sh \
  --sender-time-recording \
  --dataset-name sender-time-evaluation
```

This adds a delay to **recording selection**, not to PICO commands or SONIC
control. The default path remains latest-sample recording. The detailed
[sender-time reference](docs/source/references/sender_time_recording.md) is
also useful when bringing up the remote clock endpoint.

### Selection algorithm

At recording start, the collector creates a target grid in Thor's monotonic
clock. At 50 Hz, target times are 20 ms apart. It keeps a bounded history of
32 samples per required stream, including each camera independently.

For a target `T`, the recorder waits until `T + 100 ms` by default. It then
requires every relevant producer to have advanced beyond T, proving that the
history has progressed past the requested instant. It selects the latest
sample at or before T that also satisfies the source age limit. Once selected,
all row-building helpers use that selection rather than later mutable snapshots.

For a mapped sample time `t` with estimated uncertainty `u`, selection requires
`t + u <= T`. The newest advancing sample must satisfy `t - u > T`. A sample
that might be from the future is not used just because its central estimate is
slightly before T. No image or joint interpolation is performed.

If sources are unavailable, the recorder waits up to another 250 ms by default.
It then records a skipped target/gap diagnostic and advances the target grid.
Stop drains targets through the stop-request time before finalizing. A gap
makes the take unsuccessful under the current quality policy; it is retained.

### Mapping Orin's clock

Camera, C++ state, and teleop publishers must be local to the recorder in this
mode, addressed as loopback. Remote hand state is supported through a dedicated
read-only clock service on Orin, normally port 5574.

For one exchange, Thor sends at `t1`, Orin receives at `t2` and replies at `t3`,
and Thor receives at `t4`. The estimated remote-minus-local offset is:

```text
offset = ((t2 - t1) + (t3 - t4)) / 2
network_round_trip = (t4 - t1) - (t3 - t2)
mapped_time = remote_source_time - offset
```

The implementation uses integer nanoseconds. It selects a low-round-trip
estimate from recent exchanges, requires at least three exchanges within two
seconds, and rejects an estimate above 5 ms uncertainty. It budgets half the
network round trip plus assumed 100 ppm relative clock drift; selection adds
age-dependent uncertainty as well. This is a measurement and mapping scheme,
not PTP, hardware exposure synchronization, or a change to either system clock.

The hand message's Linux boot identity must match the clock service. Reboots,
stale exchanges, missing producer timestamps, and backward mappings block
admission or mark episode errors. Missing producer timestamps are never silently
replaced with receipt timestamps.

The launcher enables the clock service on its remote hand server when requested.
With `--no-start-hand-server`, configure the existing server with the matching
clock port or start `gear_sonic.data.clock_sync` on that hand host. Run one clock
service per port, not both mechanisms on the same endpoint.

### Limits and recorded diagnostics

`capture.sync_target_monotonic_ns` records T. Per-stream `capture.sync.*` columns
retain the original source time, mapped time, receipt time, offset, and
uncertainty. Unused stream fields are `-1`. Status includes history depth,
overflow, late samples, skipped targets, and input errors.

Histories and transport queues are bounded; the mode is not a lossless archive
of every source packet. A history that is too short for a larger configured
delay can lose necessary samples. Reusing a previous camera exposure remains
possible even when all selected samples are causally before T.

LeRobot/video time still uses `frame_index / fps`. If target times jump over a
recording gap, the emitted video timeline remains contiguous. Use the recorded
target times to recover real elapsed acquisition timing. Validate this mode
under the intended camera and upload load before relying on its timing quality.

## Dataset schema and action meaning

The recorder writes LeRobot v2.1-style episode Parquet files, video files, and
metadata. Exact paths come from the dataset metadata. The usual layout is:

```text
outputs/<dataset-name>/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.ego_view/episode_000000.mp4
  videos/chunk-000/observation.images.left_wrist/episode_000000.mp4
  videos/chunk-000/observation.images.right_wrist/episode_000000.mp4
  meta/info.json
  meta/modality.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/episode_quality.jsonl
  recovery/                       # present after a recoverable finalizer failure
  .upload_snapshots/               # temporary immutable upload staging
```

Additional ZED videos exist only when requested. Empty or partially initialized
datasets may not yet have episode/video files. Episode tables and videos are
not interchangeable with a raw sensor packet log.

### Observations, intent, and applied commands

| Feature | Meaning |
|---|---|
| `observation.state` | Measured body and hand joint positions, assembled using the selected profile and recorded joint names. |
| `observation.eef_state` | Two wrist poses from robot-model forward kinematics: position plus wxyz quaternion for each side, 14 values. |
| `observation.body_joint_velocity` | 29 measured body joint velocities. |
| `observation.root_orientation` | Base IMU quaternion, wxyz. |
| `observation.base_angular_velocity` / `observation.projected_gravity` | Angular velocity and derived body-frame gravity. |
| `action.wbc` | Body's last policy position targets, scaled and offset by C++, combined with requested hand targets. |
| `action.motion_token` | 64-value SONIC motion token. All-zero tokens are rejected during admission. |
| `action.dex1_{left,right}_raw` or `action.omnihand_{left,right}_raw` | Requested native hand targets. |
| Corresponding `observation.*_raw` | Measured native hand positions. |
| `control.hand_applied_position` | Applied hand position command, distinct from both requested intent and measured motion. |
| `teleop.*` | Mode-dependent targets, hand intent, planner fields, VR 3-point fields, and SMPL compatibility fields. |
| `observation.images.*` | Selected RGB or depth-visualization video features. |
| `episode.success` | Per-row outcome flag written at finalization: 1 for accepted-valid, 0 for failed. |
| `capture.*` | Source identities, sequences, timestamps, and optional synchronization metadata. |

DEX 1 has one active motor per hand: its assembled state has 31 entries
(29 body plus two hand values). OmniHand O10 has ten per hand: 49 entries.
Use the stored feature names and `meta/modality.json` slices; do not assume
that the complete state is simply a 29-element body prefix followed by both
hands. The assembly uses robot joint groups, with hands adjacent to their arms.

Position units are meters for Cartesian features and radians for joint features;
angular velocities are radians per second. Dataset quaternion fields use wxyz.
The native transport and third-party libraries can have other conventions,
which the adapters convert at their boundaries.

Requested close/open intent, the intermediate applied command, and actual
measured hand motion can differ because strokes take time and motor feedback
is asynchronous. Training on requested actions and auditing actual execution
are different uses of these columns. `action.wbc` is not measured robot state
and does not imply an independently measured end-to-end actuator delay.

In planner VR3PT/IK modes, unused SMPL compatibility fields can be zeros. That
is not itself evidence that the demonstration is empty. The modality file
decides which stored fields are exposed to GR00T; keeping provenance columns
in Parquet does not automatically add them as model inputs.

### Reproducibility metadata

`meta/info.json` includes `script_config`, camera/hand selections, the recording
synchronization configuration, and capture metadata. The recorder stores the
Git commit and dirty state, configured artifact paths, and SHA-256 hashes of
resolved model/encoder/planner/observation files and policy parameters where
available. Missing files are identified instead of receiving invented hashes.

These fields help identify the runtime configuration. They are not a substitute
for tracking physical mounting, camera calibration, headset setup, firmware,
or the actual binaries and processes used during a run. In particular, an old
process can keep running after the worktree has moved to a new commit.

## Saving outcomes and uploading

### Persistent local folder and manual upload

This Thor is configured to append to `/home/unitree/recordings/g1_teleop` on
subsequent launches. The machine-local configuration is
`/home/unitree/.config/sonic/recording.json`. Changing cameras, hand profile, or
FPS requires a compatible schema or a different dataset folder.

Upload the saved episodes manually with:

```bash
/home/unitree/recordings/upload_to_hf.sh MicroAGI-Labs/g1-teleop --dry-run
/home/unitree/recordings/upload_to_hf.sh MicroAGI-Labs/g1-teleop
```

The repository name is an example; choose the desired organization/name.
The script defaults to a private dataset. Dry-run checks local files without
contacting Hugging Face. It snapshots committed episodes, including accepted
takes with failed validation, and excludes active recordings and recovery
files. Recording can continue during this standalone upload. A save changing
metadata during snapshot creation causes a retryable error. Rerun the command
to include episodes saved after the snapshot. Existing remote episodes must
match local data before the script will append more.

For another checkout, invoke
`.venv_data_collection/bin/python gear_sonic/scripts/upload_local_recordings.py ORG/NAME`
and optionally pass `--dataset-dir /path/to/dataset`.

### Three separate milestones

1. **Accepted:** the collector evaluates the take and transfers ownership of
   its episode buffer and video writers to the finalizer. UI/audio acknowledgment
   can happen here.
2. **Finalized locally:** the finalizer has written episode data, flushed videos,
   and committed dataset metadata. `last_finalized_episode` advances here.
3. **Uploaded:** the background uploader has successfully sent a snapshot.
   `last_uploaded_episode` advances here.

“Recording saved” is an early acknowledgment, not proof that all three
milestones have completed. Read the finalizer and Hub status before assuming
that a take is durable locally or available remotely.

### Video and local finalization

Video writers open lazily on an episode's first frame. Each uses a bounded
50-frame queue and an encoding thread. Queue capacity is reduced if needed to
fit a 256 MiB aggregate raw-image queue budget per take (about 220 MiB for five
640×480 RGB cameras). Encoder working memory and measurement buffers are extra.
Images are compressed continuously as H.264 (veryfast, CRF 23, zerolatency).
Every camera queue is checked before admitting a synchronized row; a full queue
stops the take as unsuccessful rather than silently dropping a camera frame.

Each take owns a unique `.recording/<id>/` directory on the dataset disk.
On acceptance, video containers are closed and moved to their final paths,
then the table and metadata are committed. No full-episode raw image buffer
is retained.

At stop, `detach_episode()` hands the completed buffer and existing writers to
the finalizer and creates an empty next buffer. It does not eagerly open the
next episode's video files. Thus a later failure to create a new video cannot
steal ownership of the completed take.

The finalizer serializes local saves. Its default admission threshold allows
two pending jobs; a new recording is blocked when it is at capacity or has a
latched failure. It still accepts ownership of a take already detached by the
collector, so an earlier failure cannot abandon that later buffer.

The exporter checks episode timestamps before file work, preserves the source
buffer, and restores in-memory metadata state when finalization raises. On
failure, the finalizer attempts to stop its writers and persists the owned
measurements, video references, and validation result to
`recovery/episode_<index>.pkl` using a temporary
file, flush/fsync, and replacement. The error remains visible and blocks new
recordings. Recovery is for deliberate inspection/repair; it is not an automatic
replay service or a guarantee against power loss during any multi-file write.

### Success and failure both save

**Y+B**, the `f` command, and the browser's **Stop & Save Failure** button all
stop the take and save its measurements and videos as a failure. Successful and
failed saves advance the episode index. **Y+A**, `x`, the browser's **Discard**
button, and the four-minute limit delete the active take's temporary files and
do not add an episode. The next take can reuse that unsaved episode number.

Recording chords trigger once after all face buttons are released. Their full
button union is tracked so a third button cancels the recording/pose action,
even with staggered releases. All four buttons remain SONIC-only. Y+B cannot
also invoke B/Y's saved pose or the legacy return-to-base action during recording.

Failed takes have `episode.success = 0`, an entry in `failed_episode_indices`,
and a quality report in `meta/episode_quality.jsonl`. Failure reports preserve
frame flags and gaps for cropping, with a `failure_reason` such as
`operator_marked_failure`. Failed quality checks,
technical stops, and shutdown also preserve captured data as unsuccessful.
All saved outcomes are included in a later manual upload. An empty take has no
data to save and is skipped.

New failures are not added to the legacy `discarded_episode_indices` removal
list. Older datasets may retain those historical markers; the processor's
`--remove-discarded` option only acts on that legacy list.

Validation includes required-input errors, allowed teleop modes, sender-time
gaps when enabled, and optional external-hand activity. With
`--require-hand-activity`, at least one requested hand joint must span 0.02 rad.
This tests requested command activity, not proof
that an object was grasped or that the intended task succeeded. Stream rates
remain diagnostic.

### Upload technique and failure behavior

After local finalization, the uploader copies metadata into an immutable
snapshot and hard-links finalized episode files through that episode index,
falling back to copies if hard-linking is unavailable. Future active video
files are excluded. Uploading the live dataset directory would race with
recording and metadata changes; the snapshot fixes that boundary.

Snapshots are cumulative. When upload falls behind, queued snapshots are
coalesced into the newest complete one. Failed uploads retry with exponential
backoff, starting at one second and capped at 30 seconds. Network backlog does
not block local recording once a Hub dataset is configured. A failure to stage
an upload is reported separately and does not turn a committed local episode
into a failed disk save; a later cumulative snapshot can include it.

The upload helper runs in a separate process with lowered scheduling priority
and at most two available CPUs selected for its affinity. This reduces CPU
competition; it does not reserve hardware resources or guarantee real-time
recording. Video encoding, disk bandwidth, thermal limits, and camera traffic
can still affect the collector.

The active uploader does not restore its in-memory configuration/queue on a
fresh process start. Preserve local data and inspect upload state after an
interrupted session. Do not assume that resuming a local dataset also resumes
its previous Hub configuration. The browser requires an empty remote dataset
for initial selection and locks configuration once local episodes exist; upload
recovery of an existing collection is a separate operation.

### Preparing training data

For default VR3PT or upper-body IK recordings, disable the legacy SMPL cleaning
rule because those modes intentionally contain zero SMPL compatibility fields:

```bash
.venv_data_collection/bin/python gear_sonic/scripts/process_dataset.py \
  --dataset-path outputs/my_dataset \
  --output-path outputs/my_dataset_cleaned \
  --no-remove-stale-smpl
```

This writes a separate processed dataset. The processor removes discarded
records by default; use `--no-remove-discarded` when retaining them for analysis.
For full-SMPL recordings, stale-SMPL cleaning can be appropriate. Check the
stored mode and quality metadata rather than applying one cleaning recipe to
all profiles. Merging also validates configuration compatibility; compatible
joint widths alone are not sufficient.

## Installation and alternative launches

The daily commands assume a prepared checkout, downloaded controller artifacts,
camera SDK installation, device permissions, and a reachable robot network.
They are not a from-scratch operating-system installer. Use the
[deployment installation guide](docs/source/getting_started/installation_deploy.md)
and [model download guide](docs/source/getting_started/download_models.md) for
C++/model setup, then install the components appropriate to each host.

| Host/component | Installer or reference | Result |
|---|---|---|
| Thor collection/browser | `bash install_scripts/install_data_collection.sh` | `.venv_data_collection` with LeRobot, video, and UI dependencies. |
| Thor PICO | `bash install_scripts/install_pico.sh` | `.venv_teleop` and supported headset dependencies; see the PICO setup guide for the headset application. |
| Thor physical cameras | `bash install_scripts/install_camera_server.sh` | Camera environment/service setup; ZED SDK and Python bindings require their documented installation. |
| Orin DEX 1 server | `bash install_scripts/install_hand_server.sh` | `.venv_hands` and pinned native worker build, without the full inference/collection stack. |
| Thor local DEX 1 | `bash install_scripts/install_dex1.sh` | Native worker in `build/dex1`; Python uses `.venv_data_collection`. |
| Thor physical OmniHand | `bash install_scripts/install_omnihand.sh` | OmniHand environment and SDK setup. |
| Simulation | `bash install_scripts/install_mujoco_sim.sh` | Simulation environment; model/scene assets must also be present. |

Use [DEX 1 setup](docs/source/references/dex1_teleop.md) for pinned SDK details,
offline worker self-tests, permissions, and remote preparation. The remote
checkout defaults to `/home/unitree/gr00t-wbc-g1-omnihand`; it is a path on Orin,
not the historical Thor worktree path. Override it with `--hand-server-repo`
when the prepared remote checkout lives elsewhere.

Pair SSH from Thor once:

```bash
bash tools/setup_orin_ssh.sh 192.168.123.164
```

This setup step asks for the remote account password and installs a dedicated
SSH key. Subsequent launches use noninteractive authentication. Normal launch
probes authentication before replacing the tmux session. It also configures
SSH keepalives; that detects a lost connection but does not automatically launch
a new SSH session after the old one has ended.

### Local DEX 1 hands

With the grippers connected to Thor and its native worker prepared:

```bash
bash tools/teleop_dex1.sh --record-wrist-cameras
```

This still enables the browser, but omits the remote hand host. The generic
Python launcher defaults to DEX 3, so do not omit `--hand-backend dex1` when
invoking it directly for this pair.

### Existing remote hand service

Start the prepared service on Orin, substituting Thor's wired address:

```bash
.venv_hands/bin/python -m gear_sonic.end_effectors.server \
  --teleop-host THOR_IP --enable-command
```

Then reuse it from Thor:

```bash
bash tools/collect_dex1_orin.sh --no-start-hand-server
```

The launcher still connects the recorder and UI to Orin but no longer owns the
remote server's lifecycle. For sender-time recording, add the clock service
configuration on Orin too. This mode requires an actual existing service;
it does not create one implicitly.

### Physical OmniHand or simulation

```bash
# Physical O10 hands, with their prepared SDK environment and CAN interfaces.
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py \
  --hand-backend omnihand --remote-ui

# Prepared MuJoCo setup using the generic launcher's default DEX 3 hand path.
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py \
  --sim --remote-ui
```

DEX 1 currently requires physical USB grippers; `--sim --hand-backend dex1` is
rejected. OmniHand simulation uses its external-hand controller and requires
the corresponding scene assets, including downloaded LFS meshes.

With `--remote-ui`, simulation is headless with EGL offscreen rendering.
The launcher stops the physical camera system service for simulation and starts
it for hardware, keeping the normal camera endpoint stable. It needs the
configured service and noninteractive permission for that switch. For a camera
service managed separately, use `--no-manage-camera-service` and ensure the
selected endpoint is ready yourself. This option changes service ownership,
not the recording schema.

### Other body modes and headset sources

| Launcher selection | PICO mode / recording mode | Intended use |
|---|---|---|
| `--body-control-mode vr3pt-slow-planner` | `vr3pt` / 5 | Default learned VR 3-point upper-body tracking with planner locomotion. |
| `--body-control-mode ik-upper-slow-planner` | `ik-upper` / 6 | Upper-body IK plus planner; requires the IK dependencies. |
| `--body-control-mode full-smpl` | `pose` / 1 | Existing learned full-body SMPL tracking. |

Use the [body-mode controls](docs/source/tutorials/data_collection.md) when
selecting the older flows. For a prepared CloudXR/Isaac Teleop setup, append
`--pico-input-source isaac-teleop`; XRT is the daily default. The
[Isaac Teleop setup guide](docs/source/tutorials/isaac_teleop_publisher_setup.md)
contains its separate prerequisites.

### Models and a local-only collector

The launcher's empty model override flags delegate to `deploy.sh`. Current
shell defaults are checkpoint prefix `policy/release/model`, observation config
`policy/release/observation_config.yaml`, planner
`planner/target_vel/V2/planner_sonic.onnx`, and motion data `reference/example/`,
resolved from the deployment directory. `--deploy-checkpoint`,
`--deploy-obs-config`, `--deploy-planner`, and `--deploy-motion-data` select
alternatives. Choose matching models/configuration rather than inferring a
model revision from the branch name. Reference lookahead is only one component
of latency, not an end-to-end measurement.

For collection without the browser's mandatory Hub setup, use a launch without
`--remote-ui`, or run the exporter separately against an already running stack:

```bash
.venv_data_collection/bin/python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt 'pick up the cup' \
  --hand-state-host 192.168.123.164 \
  --record-wrist-cameras \
  --required-stream-mode 5
```

Run one collector per intended session; this is an alternative to the launched
pane 2, not a command to start beside it. The standalone exporter does not start
SONIC, PICO, cameras, or hands. Its frequency flag is
`--data-collection-frequency`; the launcher's forwarding flag is
`--data-exporter-frequency`. Do not assume every exporter option is exposed by
the launcher. The standalone process also needs a recording-command source;
see the [manual collection tutorial](docs/source/tutorials/data_collection.md).

## Troubleshooting by symptom

| Symptom | Likely boundary to inspect | Next check |
|---|---|---|
| Browser does not open | HTTP process or laptop tunnel | Pane 3, then `curl http://127.0.0.1:8080/healthz` on Thor; confirm the laptop tunnel uses the same HTTP port. |
| Camera pictures work, recorder is waiting | Upstream config or dataset setup | Pane 2 for `robot_config` / `hand_config` wait; browser dataset acknowledgment. |
| Record button does nothing useful | Wrong teleop mode, Hub unset, or finalizer blocked | `/recording/status` and pane 2; enter the selected teleop mode and read the specific reason. |
| Hands continually reconnect | Transport/device issue | Hands pane on the device-owning host; inspect stable serial paths, power, and exact exception. |
| Hands say fault and do not retry | Latched motor/health/control fault | Resolve the reported condition, then request a clean reconnect. Repeated clicks cannot fix a persistent physical fault. |
| Reconnect button has no effect | Supervisor absent, old protocol, or broken network | Pane 4 / remote service status and protocol versions; a button cannot restart an absent remote SSH session. |
| Arm does not resume after tracking loss | Rearm sequence incomplete | Release both grips, center both sticks, then re-engage the intended grip. |
| Hand trigger appears ignored | Per-arm or trigger gate held | Engage that arm, release trigger, then press it; inspect hand validity. |
| Walking remains stopped | Locomotion disabled, sticks not neutral, or conflicting axes | Toggle with left-stick click, center both sticks, then request only forward/back or turn. |
| Wrist rate is zero but ego is live | Individual wrist capture or mapping | JR side/device identity and camera journal; required missing wrists block frame admission. |
| Configured 50 Hz but lower receiver rate | Source cadence, duplicates, or collection load | Compare source and distinct receiver rates, timestamps, loop-overrun logs, encoding and disk activity. |
| Valid-looking take is unsuccessful | Quality policy | `meta/episode_quality.jsonl`, hand command range, input gaps, and allowed modes. |
| Save acknowledged but no committed episode | Background finalization | Finalizer status, disk error, recovery directory, and video writer errors. |
| Upload retrying while recording works | Background network/Hub work | Hub error and pending status; preserve locally committed files. |
| Sender-time recording waits indefinitely | Missing producer timestamps or clock mapping | Per-stream input errors, camera capture timestamps, Orin clock identity, and fresh clock exchanges. |
| Cleaning removes an entire VR3PT take | Legacy stale-SMPL removal enabled | Reprocess the original into a new directory with `--no-remove-stale-smpl`. |
| New code seems absent | Long-running process or wrong worktree | Compare Git HEAD and pane paths; restart the affected components at a session boundary. |

Useful read-only status endpoints on Thor:

```bash
curl -s http://127.0.0.1:8080/recording/status
curl -s http://127.0.0.1:8080/hands/status
journalctl -u composed_camera_server.service -n 100 --no-pager
ip route get 192.168.123.164
```

A healthy HTTP endpoint proves only that the viewer process is responding.
Read the nested recorder/hand freshness and error values before diagnosing the
rest of the stack. Camera service logs remain available even if the dashboard
was closed because that service has independent ownership.

## Design decisions and tradeoffs

| Decision | Why it exists | Consequence |
|---|---|---|
| Direct controller references with per-arm grips | Permit immediate local calibration and independent repositioning. | The operator must understand grip/rearm state and headset heading at engagement. |
| Planner locomotion separate from arm tracking | Keep walking intent explicit while manipulating. | Turning and forward/back are mutually exclusive in the daily controls. |
| Slow mode capped at 0.6 m/s command | Provide a useful collection walking range. | Actual speed is not guaranteed by the command value. |
| Remove optional VR motion filtering | Keep live target mapping direct and eliminate tuning/trace code no longer used. | Tracking quality is exposed directly; explicit returns and loss handling remain. |
| External hand ownership | Support DEX 1/O10 without coupling their SDKs into body inference. | Hand status and body state are separate streams that must be checked/aligned. |
| Native motor workers on the USB host | Keep high-rate motor I/O local and isolate native failures. | More processes, explicit lifecycle ownership, and a remote boundary for hand timestamps. |
| Latest-only hand intent | Avoid replaying stale commands after slow SDK work or reconnect. | Intermediate input events can be dropped; the protocol represents current intent. |
| Supervisor outside the vendor worker | Make a clean restart possible when SDK teardown is unreliable or blocked. | Transport recovery and latched hardware faults need different handling. |
| Independent camera workers and source identities | Preserve asynchronous wrist/ego updates and distinguish cached frames. | No claim of simultaneous exposures or one new image for every dataset row. |
| Latest-sample recording by default | Keep the common capture loop simple with bounded work and no synchronization wait. | Cross-stream acquisition times can differ. |
| Sender-time mode opt-in | Permit explicit causal selection and remote clock diagnostics without changing control latency. | Requires new timing metadata, bounded histories, extra recording delay, and a validated clock estimate. |
| Separate observations, requested actions, and applied hand commands | Preserve both training intent and execution evidence. | Consumers must choose the correct columns rather than treating every action as measured motion. |
| Background video/finalization/upload | Keep expensive work away from ordinary control/status handling. | There are several completion milestones and explicit queue/error states. |
| Immutable cumulative upload snapshots | Prevent uploads from observing partially changing dataset metadata/video files. | Snapshot staging costs disk work; interrupted jobs require explicit recovery awareness. |
| Retain unsuccessful takes | Preserve failure evidence and permit later inspection/filtering. | Training preparation must respect quality flags. |
| Keep supported alternate modes and backends | They remain callable configurations with their own dependencies and tests. | “Unused in today's DEX 1 session” is not equivalent to unreachable code. |

## Code tour and cleanup boundaries

These are the main paths to review when changing behavior. Links resolve to the
same checkout on GitHub; the code is the authority when a configuration changes.

| Concern | Entry points |
|---|---|
| Daily wrappers | [collect_dex1_orin.sh](tools/collect_dex1_orin.sh), [teleop_dex1.sh](tools/teleop_dex1.sh) |
| Launch defaults, SSH, panes, service switch | [launch_data_collection.py](gear_sonic/scripts/launch_data_collection.py) |
| PICO modes, supervision inputs, watchdog and hand intent | [pico_manager_thread_server.py](gear_sonic/scripts/pico_manager_thread_server.py) |
| Controller button semantics | [pico_controls.py](gear_sonic/utils/teleop/pico_controls.py) |
| Arm engagement/held target and generated returns | [vr_arm_clutch.py](gear_sonic/utils/teleop/vr_arm_clutch.py), [pose_transition.py](gear_sonic/utils/teleop/pose_transition.py) |
| C++ planner/pose receive and disconnect behavior | [zmq_endpoint_interface.hpp](gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp) |
| C++ state/config publication | [zmq_output_handler.hpp](gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/zmq_output_handler.hpp) |
| Hand process ownership and protocol | [server.py](gear_sonic/end_effectors/server.py), [supervisor.py](gear_sonic/end_effectors/supervisor.py), [protocol.py](gear_sonic/end_effectors/protocol.py) |
| Hand control/feedback and profiles | [controller.py](gear_sonic/end_effectors/controller.py), [profiles.py](gear_sonic/end_effectors/profiles.py), [DEX 1 adapter](gear_sonic/end_effectors/backends/dex1.py) |
| Camera capture and independent reception | [composed_camera.py](gear_sonic/camera/composed_camera.py), [sensor_server.py](gear_sonic/camera/sensor_server.py), [drivers](gear_sonic/camera/drivers/) |
| Collector orchestration, row admission and assembly | [run_data_exporter.py](gear_sonic/scripts/run_data_exporter.py) |
| Dataset schema and modality mapping | [features_sonic_vla.py](gear_sonic/data/features_sonic_vla.py) |
| Producer-time selection and remote mapping | [sender_sync.py](gear_sonic/data/sender_sync.py), [clock_sync.py](gear_sonic/data/clock_sync.py) |
| Video and local dataset persistence | [video_writer.py](gear_sonic/data/video_writer.py), [exporter.py](gear_sonic/data/exporter.py) |
| Finalization and upload ownership | [episode_finalizer.py](gear_sonic/data/episode_finalizer.py), [hub_uploader.py](gear_sonic/data/hub_uploader.py), [upload_dataset_snapshot.py](gear_sonic/scripts/upload_dataset_snapshot.py) |
| Browser/status/control routes | [run_camera_web_viewer.py](gear_sonic/scripts/run_camera_web_viewer.py) |
| Post-processing | [process_dataset.py](gear_sonic/scripts/process_dataset.py) |

The follow-up cleanup makes the finalizer and uploader library modules the
single implementations imported by the collector. Earlier copies in those
modules were only exercised by their own tests while the live collector used
different classes embedded in its script. Keeping both obscured which behavior
was deployed. Relevant tests now target the implementation the collector uses;
tests specific to the retired alternate implementation were removed. The unused
`causal_sync.py` helper was also removed; active producer-time selection lives
in `sender_sync.py`.

The prior cleanup removed the optional VR motion conditioner, its trace/analyzer
and tuning flags, associated protocol fields, and unreachable helpers. Native
motor checks and tracking-loss handling remain. SMPL, IK, Isaac Teleop,
simulation, camera alternatives, training, and other maintained project paths
remain because they have supported uses outside the daily configuration.

The PR does not require resurrecting retired experimental branches or their
worktrees. Branch names and local SDK/build artifacts are operational state,
not runtime feature switches. The upstream project README remains available
for the broader controller, training, and model documentation.

## Validation and remaining work

The automated suite covers controller/chord semantics, tracking re-entry,
hand protocols and recovery, independent camera reception, sender-time
selection/clock mapping, dataset admission, video/Parquet finalization,
upload snapshots, and browser controls. Relevant tests are under
[gear_sonic/tests](gear_sonic/tests/) and
[end-effectors tests](gear_sonic/end_effectors/tests/).

From a prepared checkout, run the Python suite without the optional MuJoCo
integration test:

```bash
PYTHONPATH="$PWD" .venv_data_collection/bin/python -m pytest -q \
  gear_sonic/tests gear_sonic/end_effectors/tests \
  --ignore=gear_sonic/end_effectors/tests/test_omnihand_mujoco.py
```

Use a prepared environment containing MuJoCo for that separate integration
suite. Native disconnect regression checks require the deployment build
prerequisites. Some unrelated native FK tests require reference fixtures that
are not supplied by this checkout. The PR description records the checks run
for the reviewed commit; do not treat historical test counts as measurements
of the currently running robot.

The remaining hardware work is concrete:

- Complete a take under the intended ZED/wrist/hand configuration and inspect
  unique camera captures, actual row cadence, timestamps, and quality metadata.
- Validate direct-controller re-engagement, heading references, and requested
  arm returns on the actual headset/robot setup.
- Exercise transport recovery and confirm that actual motor health faults are
  reported distinctly from USB/network loss.
- Evaluate sender-time offset uncertainty and skipped targets under load before
  choosing it for production collection.
- Confirm that video encoding, local finalization, and sustained uploading keep
  up for the intended episode length and camera count.

Known implementation limits include visualization-only recorded depth, bounded
source histories and encoder queues, no automatic restoration of the active
uploader's queue/configuration after process restart, and no guarantee that a
multi-file episode commit survives arbitrary power loss. These are specific
engineering boundaries, not reasons to mislabel the configured collection rate
or successful offline tests as end-to-end hardware validation.
