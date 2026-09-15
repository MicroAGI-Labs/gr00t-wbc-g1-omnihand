# Teleop recovery

Use the main checkout on Thor, `/home/unitree/repos/gr00t-wbc-g1-omnihand`,
on `data-collection`. The daily launcher is `bash tools/collect_dex1_orin.sh`.
It starts the body, PICO, recorder, viewer and remote hands together; use it
at a session boundary, with the robot ready for normal startup.

## Capture the fault before restarting

```bash
cd /home/unitree/repos/gr00t-wbc-g1-omnihand
python gear_sonic/scripts/check_teleop.py --hand-server-host 192.168.123.164
```

This takes a five-second status sample and saves recovery advice, pane logs,
checkout revision, disk space, camera-service logs, and Orin USB topology and
kernel events under `outputs/teleop-diagnostics/`. It uses HTTP GET, SSH and
filesystem reads; it never sends robot commands, opens motor serial ports,
restarts services, or changes USB settings. Failed checks are retained too.
It works with the system Python, even if the teleop environment is broken.

Use `--duration 30` for intermittent status loss, or `--output-dir PATH` for a
new destination. Existing capture directories are never overwritten. Omit
`--hand-server-host` if Orin SSH is unavailable. Local and remote UTC times and
the Orin boot ID are recorded: verify clock alignment before correlating logs.
Motor and input watchdogs use local monotonic clocks.

## Recover the affected component

| Symptom | Recovery |
|---|---|
| PICO pane repeats `VR target has a zero quaternion`; four buttons do nothing | Restore streaming to Thor with the headset and both controllers awake and tracking. Wait for `Live controller stream connected`. Release all face buttons, then press/release A+B+X+Y. |
| Brief tracking loss during VR3PT | Let the 200 ms watchdog hold the targets. Restore tracking, release both grips and center both sticks, then engage the desired grip again. Re-arm its trigger with release then press. |
| PICO reconnects after a prolonged loss | The last commanded arm targets remain held. Resume through the same grip/stick sequence. A/B/X/Y are explicit pose commands, not reconnect commands. |
| An arm moves in the opposite direction | Release its grip. Confirm the main checkout includes the heading correction and the manager was started afterward; face the intended forward direction and re-engage. If the problem is joystick turning, record that separately—the joystick uses a different path. |
| Hands say RECONNECTING | Transport retries are automatic. Check USB/kernel evidence and motor power/cabling if replies do not return. Repeated clicks do not repair a disappearing adapter. |
| Hands say FAULT | Inspect the error first. Health violations, invalid position and persistent failure to acknowledge disable latch. Resolve the reported cause, then use **Reconnect hands**. |
| Startup reports `communication timeout not cleared` | Reconnect already sends the documented timeout-clear value while disabled. A flag that remains set for one second faults; inspect motor communication before retrying. General motor error codes are not erased. |
| Hands are connected but do not follow triggers | Check PICO input and grip/trigger re-engagement. A connected hand service can legitimately be holding without fresh PICO intent. |
| Hand reports a shifted encoder position after power loss | Startup admits the measured profile's unique adjacent-turn representation. If it still faults, inspect the reading and physical position; do not widen the profile or change calibration blindly. |
| Recorder is OFFLINE | Inspect its pane/exit error. Preserve the local dataset and any `recovery/` files before restarting the recorder with the existing dataset path. The viewer can be online while the recorder is down. |
| Brief stale camera/hand frames | Save the wanted take. Inspect `meta/episode_quality.jsonl` and frame ranges for cropping. Retained hand positions keep their original timestamps and do not represent new measurements during loss. |
| Camera server rate looks normal but a view freezes | Inspect the individual head and wrist camera rows. These count distinct frames using each camera's capture timestamp; a live combined publisher does not prove that all views are fresh. |
| Capture stops because a video queue fills | Keep the saved failed take and inspect loop timing, CPU/disk load and camera cadence. A full queue is a throughput failure; increasing the memory cap alone does not repair it. |
| Previous save is pending | The next take can record while one previous finalizer job remains pending. At capacity, wait for a save to finish. Investigate a reported finalizer error rather than discarding saved data. |
| Upload stalls | Continue local collection. Run the separate manual uploader against `/home/unitree/recordings/g1_teleop`; verify committed remote episodes before assuming upload completion. |

Default recording controls: **X+B** starts/saves success, **Y+B** saves failure,
**Y+A** deletes a discard. A take reaching **240 seconds is discarded**.
Saved failures retain videos and frame-quality annotations. Local recording
does not require selecting a Hugging Face dataset.

## Diagnose head-camera freezes

First compare the individual head-camera rates with the aggregate publisher and
wrist rates. If head timestamps stop while the others advance, the delay is
upstream of recording. Preserve the service logs before restarting:

```bash
journalctl -u composed_camera_server.service --since '-10min' --no-pager
```

For the recorder's loopback camera connection, frame age includes capture, publication,
transport queuing and decoding. A newly received frame can still be old. The
background receiver drains encoded backlog before expensive image decoding,
keeps the newest frame plus its normal reserve, and decodes repeated camera
captures only once. `camera_buffer.wire_backlog_dropped` in recorder status records
discarded queued publications; it is separate from a camera capture failure.
Backlog recovery can skip captures, so inspect frame-quality annotations too.

The ZED driver emits `[ZED capture delay]` when a successful read or the interval
between captures reaches 100 ms. The event includes monotonic capture time,
total read duration, and time spent in `grab`, each RGB retrieval/copy, depth-view
retrieval/copy, and metric-depth retrieval/copy. A long `grab` identifies an SDK
capture/depth-computation delay; a long retrieval identifies that stage. A long
capture interval with a short current read needs correlation with the previous
read and scheduling. The optional SDK dropped-frame count is cumulative and
also includes superseded frames; it does not establish a USB fault.

Correlate camera-service restarts and body/planner model startup with these
events. Exclude intentional service downtime from steady-state rate checks.
Saving every video frame does not prove that every saved image was fresh.

## Isolate recurring USB loss

The September 11 captures show FTDI USB protocol errors followed by adapter
removal; earlier captures also show removal of the shared hub. The later
topology places both adapters below a GenesysLogic hub and a Realtek hub:

- Robot-left: `FTBQ776H`, interface 00, motor 0.
- Robot-right: `FTBWJBC1`, interface 00, motor 1.

USB device presence and successful motor communication are separate checks.
The adapters had runtime power control `on`; autosuspend was not established
as the cause. Linux documents `EPROTO` as an ambiguous transfer failure, often
associated with a device, cable or firmware problem; it does not identify the
failed component. [Kernel USB error documentation](https://docs.kernel.org/driver-api/usb/error-codes.html)

With the robot supported, grippers unloaded and the hand service stopped:

1. Capture a baseline, then change one connection at a time.
2. Bypass the external hub where physically possible, using a direct Orin port
   and a known-good USB cable. Keep the same adapter/motor pair and serial IDs.
3. Repeat the normal hand startup and a controlled open/close/hold trial; capture
   diagnostics afterward. Look for new USB errors and reconnects, not only a
   green status at the end.
4. If the fault follows a cable or port, replace that component. If it follows
   the adapter across known-good connections, investigate the adapter and its
   supply. If both adapters disappear together, inspect their shared upstream
   connection and hub power first.

Software recovery can recover a usable connection; it cannot establish that an
intermittent cable/hub fault is repaired. The startup worker retries only
disabled commands for up to one second, then enables a measured hold after a
healthy disabled acknowledgement. A persistent refusal to disable still faults.
The same handshake waits for the communication-timeout flag to clear before
enabling; it does not clear motor health errors or change calibration.

## Validate reconnect motion before resuming task collection

With the robot supported and the grippers empty, verify both arms hold after
grip release, then interrupt and restore only PICO streaming. Confirm that
reconnection alone does not change the commanded targets. Release both grips
and center the sticks; the next engagement should calibrate without a target
jump. Test left/right and forward/back controller displacements separately,
including downward headset pitch. Test joystick turning separately too.
Offline regressions cover these command paths; this procedure checks the actual
headset, tracking origin, controller stream and robot together.

## Temperature warnings

The configured 110 C body warning is a software alert, not a verified firmware
shutdown threshold. Unitree's SDK provides separate casing and winding checks
(85 C and 120 C defaults) and recommends passive mode after a check trips.
It does not document the actual derating or torque-cutoff thresholds for this
robot's firmware. Those remain a question for Unitree, rather than a threshold
to infer by heating the robot. [Unitree SDK checks](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/g1/common/terminations.hpp)
