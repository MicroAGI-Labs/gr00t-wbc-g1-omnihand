# Data Collection for VLA

Record teleop demonstrations as [LeRobot](https://github.com/huggingface/lerobot) datasets for post-training with [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T). The data exporter runs alongside the SONIC deployment and VR teleop stack, capturing robot state, mode-dependent planner/VR or SMPL targets, hand state/actions, and camera images at a configurable frequency.

For the complete Thor/Orin operating procedure and design explanation, start
with the [Data-collection PR handbook](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/data-collection/README_DATA_COLLECTION.md).

```{admonition} Deployment model
:class: important
Run the camera server on the computer physically connected to the cameras. In the Thor setup, the C++ deployment, PICO server, camera server, data exporter, and viewer all run from the same repo clone on Thor; use `localhost` for their ZMQ connections. No camera process is needed on the G1 Orin.
```

```{admonition} Supported cameras
:class: note
The composed camera server supports **ZED**, **Luxonis OAK**, RealSense, and generic USB cameras. A ZED publishes both rectified RGB eyes plus a lossless float32 depth map. ZED frames stay in native camera orientation; no 180-degree image rotation is applied by default.

A 3D-printable mount for the head/ego-view **OAK-D W** camera is available under [`hardware/camera_mount/`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/hardware/camera_mount/README.md) — see its README for print settings, the bill of materials, and how it mounts on the G1.
```

```{admonition} Prerequisites
:class: note
1. **Completed the [Quick Start](../getting_started/quickstart.md)** — you can run the sim2sim loop (includes [installing the deployment](../getting_started/installation_deploy.md) and [downloading model checkpoints](../getting_started/download_models.md)).
2. **Completed the [VR Teleop Setup](../getting_started/vr_teleop_setup.md)** — PICO hardware is calibrated and `.venv_teleop` is ready.
3. **Camera server running on the camera host** — see [Camera Server Setup](#camera-server-setup-on-the-camera-host) below. For simulation, the MuJoCo sim loop publishes camera images automatically — no camera server needed.
```

---

## One-Time Setup (Thor or Workstation)

On the computer running deployment and collection (Thor in the onboard setup), run the install script from the repo root to create a dedicated virtual environment with all data collection dependencies (LeRobot, PyAV, OpenCV, etc.):

```sh
bash install_scripts/install_data_collection.sh
```

This creates `.venv_data_collection` using Python 3.10 via `uv`. It installs `gear_sonic[data_collection]` which includes `lerobot`, `av`, `opencv-python`, and other required packages. It also installs `espeak` (system package) for voice feedback during recording.

```{tip}
This environment is separate from `.venv_teleop` and `.venv_sim` — the data exporter has heavier ML dependencies that are not needed for teleop or simulation.
```

---

(camera-server-setup-on-the-camera-host)=
## Camera Server Setup (On the Camera Host)

The camera server must run where the camera USB cable is connected. For the onboard ZED setup, that is Thor. It can publish locally to the exporter over `localhost` while using the same ZMQ interface as a remote setup.

### Step 1: Clone the repo on the camera host

One clone is sufficient when every process runs on Thor:

```sh
git clone https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand.git
cd gr00t-wbc-g1-omnihand
```

### Step 2: Run the install script

The install script creates the virtual environment, installs the common camera
dependencies, detects the selected camera type, and can install a systemd service.

```sh
bash install_scripts/install_camera_server.sh
```

The script will:

1. Create `.venv_camera` with `gear_sonic[camera]` (DepthAI, ZMQ, msgpack, OpenCV, tyro).
2. Detect the selected OAK or ZED camera and list its device ID.
3. Prompt you for each camera position (ego view, and optionally left/right wrist) and its device ID.
4. Ask whether to install the camera server as a **systemd service** (recommended). If you answer **y**, it generates the unit file, installs, enables, and starts the service automatically.

After the script finishes, verify the service is running:

```sh
sudo systemctl status composed_camera_server.service
journalctl -u composed_camera_server.service -f
```

```{note}
The Stereolabs ZED SDK is a system dependency and is not installed by pip. After installing it on Thor, add its Python API to the camera environment with `.venv_camera/bin/python /usr/local/zed/get_python_api.py`.
```

### Manual setup (alternative)

If you prefer not to use the install script, or need to reconfigure:

**Finding camera device IDs:**

List connected ZED cameras and their serial numbers:

```sh
source .venv_camera/bin/activate
python -c "import pyzed.sl as sl; print(sl.Camera.get_device_list())"
```

For OAK cameras, list MxIDs with:

```sh
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

**Starting the camera server manually:**

```sh
source .venv_camera/bin/activate

# ZED: capture and publish HD720 at 60 FPS; the exporter samples at 50 Hz
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera zed \
    --ego-view-device-id <YOUR_ZED_SERIAL> \
    --zed-camera-resolution HD720 \
    --zed-camera-fps 60 \
    --fps 60 \
    --port 5555

# Multiple cameras (ego view + wrist cameras)
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera oak --ego-view-device-id <EGO_MXID> \
    --left-wrist-camera oak --left-wrist-device-id <LEFT_WRIST_MXID> \
    --right-wrist-camera oak --right-wrist-device-id <RIGHT_WRIST_MXID> \
    --port 5555
```

**ZED plus JR USB wrist cameras:**

For the fixed pair mounted on this Thor, use the saved profile:

```sh
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera zed --wrist-camera-profile thor-jr --port 5555
```

The mapping uses the **robot's** left and right, based on the current mounted
camera views:

| Dataset stream | Camera serial | Persistent device path |
| --- | --- | --- |
| `left_wrist` | JR0001 | `/dev/v4l/by-id/usb-JR0001_JR0001_JR0001-video-index0` |
| `right_wrist` | JR0002 | `/dev/v4l/by-id/usb-JR0002_JR0002_JR0002-video-index0` |

These identities survive reboot, unplug/replug, and changes to `/dev/videoN`
numbering or hub ports. The profile never falls back to another camera if one
is missing, and rejects conflicting side assignments. Keep the cameras in
these mounting positions; swapping the physical cameras requires updating the
profile. No additional udev rule is needed for the existing JR devices.

`thor-jr` fixes wrist capture to 1280×720 MJPEG at 60 FPS and the shared publisher
to 60 FPS. For other hardware, leave `--wrist-camera-profile custom` (the default)
and use the explicit `--left-wrist-camera`, `--left-wrist-device-id`,
`--right-wrist-camera`, `--right-wrist-device-id`, and `--usb-camera-*` options.

The JR cameras capture MJPEG at 1280×720/60 FPS with two V4L2 buffers. The
USB driver continuously drains capture and outputs RGB at 640×480, matching the
existing wrist dataset schema and ZED output. The camera server publishes at
60 FPS. Independently arriving images retain their original timestamps; a
camera waiting for its next capture does not discard another camera's frame.
The recorder's receiver buffers each stream separately and selects frames at
50 Hz. There is no 50 Hz throttle in the camera driver.

Configure the camera service's `ExecStart` with these arguments for persistent
use. For a manually running server, pass `--no-manage-camera-service` to the
launcher. Only one process should open each physical camera; close standalone
USB viewers before starting the shared server. The teleop dashboard consumes
that shared stream and automatically displays the wrist views.

The dashboard's stream-rate table includes separate left and right wrist rows.
Publisher Hz estimates each camera's capture cadence from its source timestamps;
receiver Hz counts distinct frames sampled by the collector. Cached frames do
not count as new arrivals, and a stalled wrist drops to 0 Hz independently of
the other cameras. Active receiver rates below 45 Hz are highlighted.

Enable optional collection with:

```sh
python gear_sonic/scripts/launch_data_collection.py \
    --hand-backend dex1 --remote-ui --record-wrist-cameras \
    --data-exporter-frequency 50
```

This is the single launch command once the environments, deploy binary, DEX 1
worker, and camera service are installed in the selected checkout. The camera
service must use that checkout and `--ego-view-camera zed --wrist-camera-profile
thor-jr` to publish all three streams. The launcher starts the service if needed;
`--record-wrist-cameras` enables recording but does not configure the service.
Use `--check-only` to check launch prerequisites without starting the stack.
The browser UI is at `http://127.0.0.1:8080` on Thor (or through SSH forwarding).

When the DEX 1 USB adapters are on Orin, add `--hand-server-host 192.168.123.164`
to that launch command. This starts the prepared Orin hand server over SSH and
routes hand status back to the collector and UI; Thor no longer needs the
gripper USB devices. See [remote DEX 1 setup](../references/dex1_teleop.md) for
the Orin installation and the option to reuse a separately managed hand service.

Without `--record-wrist-cameras`, datasets contain only the existing ego view.
With it, both wrist images must be present, correctly sized, and fresh within
`--camera-max-age` (0.1 seconds in the exporter). A stalled required camera
blocks recording even if another camera keeps publishing. Wrist datasets also
store `capture.left_wrist_source_timestamp_ns` and
`capture.right_wrist_source_timestamp_ns`: host wall-clock timestamps taken
immediately after capture returns, before resize/conversion, in nanoseconds.
These identify reused images; they are not hardware exposure timestamps.

Run `python -m gear_sonic.camera.composed_camera --help` for all options including `--fps`, `--use-mjpeg`, and `--mjpeg-quality`.

**Manual systemd setup:**

```sh
# 1. Edit the service file to match your camera setup
nano systemd/composed_camera_server.service

# 2. Copy to systemd, enable, and start
sudo cp systemd/composed_camera_server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable composed_camera_server.service
sudo systemctl start composed_camera_server.service
```

Once the systemd service is running, the camera server starts automatically whenever the camera host boots.

### Connecting to the Camera Server

When the exporter and camera server both run on Thor, keep the default camera host, `localhost`:

```sh
# Data exporter
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the cup"

# Camera viewer (to verify the feed)
python gear_sonic/scripts/run_camera_viewer.py
```

Pass `--camera-host <THOR_IP>` only when a client runs on another computer.

### Frame Rates

- Existing 30 FPS cameras publish only new frames. The 50 Hz exporter reuses its cached latest image when no new image arrived.
- ZED and configured JR USB wrists capture and publish at 60 FPS. The exporter receiver selects each stream independently at 50 Hz, discarding excess queued captures without interpolating images.
- The dataset timeline is controlled by `--data-collection-frequency` (50 Hz by default).

The tmux launcher accepts the same host setting:

```sh
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host <THOR_IP> \
    --task-prompt "pick up the cup"
```

### ZMQ message format

The camera server publishes a single msgpack-encoded payload per frame cycle containing all camera images:

```python
{
    "timestamps": {"ego_view": 1712345678.123, "left_wrist": 1712345678.125},
    "images": {"ego_view": "<base64-jpeg>", "left_wrist": "<base64-jpeg>"}
}
```

Images are JPEG-compressed (quality 80) and either base64-encoded strings or raw JPEG bytes (when MJPEG on-device encoding is enabled). The data exporter's `ComposedCameraClientSensor` handles both formats automatically.

---

## Architecture

In the daily setup, body control, PICO, cameras, and recording run on Thor.
External DEX 1 hands can run on Orin.

```text
PICO manager ------> SONIC deploy ------> G1 body
    |                     |
    |                     +-----------------> Data exporter
    +---------------------------------------> Data exporter
    +----> External hand controller --------> Data exporter
                    |                         ^
                    v                         |
                Grippers          ZED / wrists -> Camera server
```

| Source | Runs on | ZMQ Topic | Default Port | Provides |
|---|---|---|---|---|
| C++ deployment | Thor | `g1_debug` | 5557 | Joint positions, velocities, IMU quaternion |
| C++ deployment | Thor | `robot_config` | 5557 | Robot configuration at startup |
| PICO teleop streamer | Thor | `manager_state`, `planner`, `pose` | 5556 | Mode and active planner/VR or SMPL targets |
| PICO teleop streamer | Thor | `hand_intent` | 5569 | Latest left/right open-close intent |
| External hand controller | Thor, or Orin for remote DEX 1 | `hand_config`, `hand_state` | 5570 | Hand profile, connection, feedback, and health state |
| Browser UI | Thor | `hand_control` | 5572 | Manual clean-reconnect request |
| Camera server | Thor | ZMQ camera packets | 5555 | Images, depth payloads, and capture timestamps |

---

## Running Data Collection

There are two ways to run the data collection stack: an **all-in-one tmux launcher** (recommended) or **manual multi-terminal setup**.

### Option A: All-in-One Tmux Launch (Recommended)

The launcher starts one tiled tmux window with four core panes and optional
simulator, hand, and camera-log panes. Screen positions depend on terminal size.

| Pane | Role |
|---|---|
| 0 | C++ deployment |
| 1 | PICO manager and its worker restart loop |
| 2 | Data exporter, including background finalization/upload |
| 3 | Browser or native camera viewer |
| 4 on daily hardware setup | Local hand supervisor or SSH to the Orin hand server |
| 5 on daily hardware setup | Read-only camera service journal |

Simulation inserts its process at pane 4 and moves any hand pane after it.
Reusing an existing remote hand service omits the launcher-owned hand pane.

```{note}
Requires `tmux` to be installed (`sudo apt install tmux`).
```

**For simulation** (the launcher starts `run_sim_loop.py` in an extra pane in the same window):

```bash
python gear_sonic/scripts/launch_data_collection.py --sim
```

**For a real robot with the camera server on the same Thor:**

```bash
python gear_sonic/scripts/launch_data_collection.py \
    --task-prompt "pick up the cup"
```

**With wrist cameras** (records ego view + left/right wrist camera streams):

```bash
python gear_sonic/scripts/launch_data_collection.py \
    --task-prompt "pick up the cup" \
    --record-wrist-cameras
```

Add `--record-zed-stereo` when the ego camera is a ZED to record its left eye
and depth visualization alongside the existing right-eye ego video. Although
the camera transport supports float32 metric depth, the recorder stores
`observation.images.ego_view_depth` as a three-channel uint8 video, not lossless
metric depth.

**With physical OmniHands and the browser controls:**

```bash
python gear_sonic/scripts/launch_data_collection.py \
    --hand-backend omnihand \
    --remote-ui
```

**With the measured physical DEX 1 pair and browser controls:**

```bash
bash tools/teleop_dex1.sh
```

Build its worker once with `bash install_scripts/install_dex1.sh` if this
checkout is not prepared. See [DEX 1 teleoperation](../references/dex1_teleop.md)
for device profiles, setup, fault behavior, and the first hardware check.

```{tip}
No need to activate a virtual environment first — the launcher automatically detects and uses `.venv_data_collection` if the required dependencies are not in the current Python.
```

The launcher auto-attaches to the tmux session.  Use `Ctrl+b` then arrow keys to switch between panes.

Common options:

| Flag | Default | Description |
|---|---|---|
| `--task-prompt` | Tote/conveyor task from `hub_config.py` | Language task description (e.g., `"pick up the cup"`) |
| `--dataset-name` | *(auto: timestamp)* | Dataset name; omit to auto-generate |
| `--sim / --no-sim` | `False` | Run deploy.sh in sim mode (also starts the sim loop) |
| `--camera-host` | `localhost` | Camera server host; set the Thor IP only for a remote client |
| `--camera-port` | `5555` | Camera server port |
| `--remote-ui` | `False` | Serve camera, recording, and hand recovery controls on loopback |
| `--no-camera-viewer` | *(viewer on)* | Disable the camera viewer pane |
| `--body-control-mode` | `vr3pt-slow-planner` | `vr3pt-slow-planner`, `ik-upper-slow-planner`, or `full-smpl` |
| `--omnihand-close-scale` | `1.0` | Use the full calibrated OmniHand closing range |
| `--omnihand-transition-duration` | `0.2` | Open/close transition duration in seconds |
| `--idle-base-transition-duration` | `2.0` | Arm interpolation time into and out of the teleop alignment pose |
| `--hand-control-port` | `5572` | Manual hand reconnect command port |
| `--data-exporter-frequency` | `50` | Recording frequency (Hz) |
| `--deploy-checkpoint` | *(default)* | Custom checkpoint path for deploy.sh |
| `--deploy-obs-config` | *(default)* | Custom observation config for deploy.sh |
| `--deploy-planner` | *(default)* | Custom planner model path for deploy.sh |
| `--deploy-motion-data` | *(default)* | Custom motion data path for deploy.sh |
| `--record-wrist-cameras` | `False` | Record left/right wrist camera streams in the dataset |
| `--record-zed-stereo` | `False` | Record ZED left-eye RGB and depth visualization videos |
| `--no-text-to-speech` | *(on)* | Disable voice feedback via espeak |

Run `python gear_sonic/scripts/launch_data_collection.py --help` for all options.

The three supported body-control configurations are selected at launch:

```bash
# Learned VR 3-point upper body + slow-walk planner (default)
python gear_sonic/scripts/launch_data_collection.py \
    --body-control-mode vr3pt-slow-planner

# Deterministic PICO wrist-to-arm IK + slow-walk planner
python gear_sonic/scripts/launch_data_collection.py \
    --body-control-mode ik-upper-slow-planner

# Existing learned full-body SMPL tracking; planner gait is used only in PLANNER mode
python gear_sonic/scripts/launch_data_collection.py --body-control-mode full-smpl
```

In the default `vr3pt` configuration, **A+B+X+Y** starts SONIC and enters
VR control with both arms and hands held. There is no A+X calibration step.
Release then hold each **middle-finger side button** to calibrate and enable
that arm and hand independently. Each engagement captures fresh controller
poses and headset heading against the held robot target. Releasing brings
that arm to a hold and holds that hand. After engagement, release then press
the **index trigger** to resume binary hand control: pressed closes, released
opens.

**A** opens both hands and smoothly returns arms and waist to the existing
ready/base pose. It keeps locomotion available and continues an active
recording. The return follows the configured smooth
transition duration; release/repress the side buttons afterward.

Calibrated VR targets pass through directly. Pico data stale for 100 ms latches
the last arm targets, stops locomotion, and cancels any arm return. After fresh
data returns, release both side buttons and center both sticks, then press a
side button to recalibrate that arm. Long disconnects keep holding.

**B** smoothly returns the arms to the saved arms-on-legs planner resting pose
while holding the grippers and preserving the waist target. Locomotion and
recording remain available. Release/repress the side buttons to resume tracking.

**Left stick click** toggles locomotion. **Left stick up/down** commands
forward/backward, and **right stick left/right** commands turning. Only one
of these four actions is accepted at a time; simultaneous walking and turning
inputs stop motion until unambiguous. **X** toggles normal/slow speed. Center
the sticks after a mode/speed change. **X+B** records/saves; **Y+A** discards.
See [all Pico controller controls](../references/pico_controller_controls.md)
for input-loss recovery, trigger re-arming, and hand-server compatibility.

Fresh robot feedback seeds initial VR targets; a grip press anchors against the
last emitted target, preserving continuity despite measured tracking error.
The manager sends Cartesian VR targets throughout home return, and SONIC keeps
its whole-body balancing control. The headset provides alignment only and does
not continuously drive the waist in this profile.

The `ik-upper` configuration and Pico manager's `--legacy-vr-controls` option
retain the older flow: **AXBY** starts in planner idle; double **A+X** returns to
base; another double **A+X** calibrates and enters teleop. **B+Y** steps back
through base to idle, with locomotion stopped during transitions. Full-body
`full-smpl` retains its POSE/PLANNER switching behavior.

After updating an existing checkout, rebuild the deployment and refresh the
teleop environment once so the masked arm command and IK dependencies match:

```bash
bash install_scripts/install_pico.sh
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
```

```{tip}
The launcher automatically enables **mouse support** in the tmux session — click to select panes, scroll with the mouse wheel, and drag to resize pane borders.
```

#### OmniHand disconnect recovery

The launcher automatically performs a clean hand-worker restart if either
OmniHand loses feedback or its CAN transport fails. A process restart is used
because the vendor SDK has no transport teardown API; this releases all native
SDK state before both devices are admitted again. Reconnection begins from the
measured hand positions, sends a zero-delta hold, discards intent queued during
the outage, and waits for a fresh PICO hand command.

When `--remote-ui` is enabled, the browser displays the current per-hand status
and a **Reconnect Hands** button. Use the button if a hand is physically back
online but automatic recovery has not completed. The button is handled by the
parent supervisor, so it remains usable even when the native SDK worker is
wedged. It requests the same clean worker restart and does not stop SONIC or
the rest of the teleoperation stack.

The browser also provides a red **Disconnect SONIC** button. It requires an
explicit confirmation and sends Ctrl-C directly to the SONIC deployment pane,
so it remains available during a PICO/XRT outage. The PICO face buttons remain
start/mode controls and cannot stop a running deployment.

**Session management:**

| Action | Command |
|---|---|
| Switch panes | `Ctrl+b`, then arrow keys |
| Detach (keep running) | `Ctrl+b`, then `d` |
| Reattach | `tmux attach -t sonic_data_collection` |
| Kill session | `Ctrl+\` in any pane, or `tmux kill-session -t sonic_data_collection` |

### Option B: Manual Multi-Terminal Setup

If you prefer individual control over each process, run them in separate terminals:

**Terminal 1 — MuJoCo Simulator** *(skip for real robot)*:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
    --enable-image-publish --enable-offscreen --camera-port 5555
```

The `--enable-image-publish` and `--enable-offscreen` flags are required so the
sim renders camera images and streams them over ZMQ on the specified port.
The data exporter subscribes to this port the same way it subscribes to a
physical camera server.

For real robot deployment, skip this terminal and see [VR Whole-Body Teleop](vr_wholebody_teleop.md) instead.

**Terminal 2 — C++ Deployment** (from `gear_sonic_deploy/`):

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim
# Wait until you see "Init done"
```

**Terminal 3 — PICO Teleop Streamer:**

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

**Terminal 4 — Data Exporter:**

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cup"
```

**Terminal 5 (optional) — Camera Viewer:**

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py
```

All options are provided via CLI flags — no interactive prompts.  Key flags:

| Flag | Default | Description |
|---|---|---|
| `--task-prompt` | Tote/conveyor task from `hub_config.py` | Language task description for this session |
| `--dataset-name` | *(auto: timestamp)* | Dataset name.  Omit to create a new one, or pass an existing name to append episodes |
| `--data-collection-frequency` | `50` | Recording frequency (Hz) |
| `--root-output-dir` | `outputs` | Parent directory for saved datasets |

```{tip}
Datasets are saved under `<root-output-dir>/<dataset-name>/`.  If `--dataset-name`
is not specified, a new name is generated from the local date and time
(`YYYY-MM-DD-HH-MM-SS-microseconds`). Each launch creates a separate dataset;
pass an explicit name only when you want to append to an existing dataset.
```

### Recording Controls

There are two ways to control recording: **PICO VR controllers** (recommended during teleop) or **keyboard over ZMQ**.

**PICO VR Controllers (via `manager_state` topic):**

| Input | Action |
|---|---|
| **X + B** | **Toggle on release** — starts a new episode only while the selected teleop mode is active, or stops and saves the current one |
| **Y + A** | **Discard on release** — saves the active episode flagged for removal during post-processing |

Recording is locked to the launch-selected teleop mode (POSE, VR3PT, or IK
upper). In the default VR controller profile, **A** returns home within VR3PT
and the episode continues, including the return. **Y+A** never triggers A's
home action, and **X+B** never triggers X's speed change. UI safe-idle aborts
an active take. Legacy VR base navigation can keep a take open; other legacy
mode changes require saving or discarding first.

**Keyboard over ZMQ:**

| Key | Action |
|---|---|
| `c` | **Toggle** recording (same as X + B; start requires the selected teleop mode) |
| `x` | **Discard** episode (same as Y + A — flagged for removal) |

```{note}
Keyboard commands are sent via a separate ZMQ publisher (default port `5580`). The data exporter subscribes to this channel automatically. You can send keys from any ZMQ publisher on that port, or integrate with the C++ deployment's keyboard handler.
```

---

## Camera Viewer

A standalone camera viewer is available for monitoring camera feeds and recording raw video independently of the data exporter.

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555
```

The viewer connects to the same ZMQ camera server used by the data exporter and displays all detected camera streams in a tiled OpenCV window.

**Controls** (OpenCV window must be focused):

| Key | Action |
|---|---|
| `R` | Start/stop video recording |
| `Q` | Quit |

Recordings are saved to `camera_recordings/rec_<timestamp>/` with one MP4 per camera stream. This is useful for:
- Verifying camera placement and image quality before starting data collection
- Recording reference videos alongside the LeRobot dataset
- Debugging camera server connectivity

Run `python gear_sonic/scripts/run_camera_viewer.py --help` for all options.

---

## CLI Options

All options can be viewed with `--help`:

```bash
python gear_sonic/scripts/run_data_exporter.py --help
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--task-prompt` | Tote/conveyor task from `hub_config.py` | Language task description for annotation |
| `--dataset-name` | *(auto: timestamp)* | Dataset name; omit to auto-generate, or reuse an existing name to append |
| `--data-collection-frequency` | `50` | Recording frequency in Hz |
| `--camera-host` | `localhost` | Camera server hostname |
| `--camera-port` | `5555` | Camera server port |
| `--sonic-zmq-host` | `localhost` | SMPL pose publisher host |
| `--sonic-zmq-port` | `5556` | SMPL pose publisher port |
| `--state-zmq-host` | `localhost` | Robot state publisher host |
| `--state-zmq-port` | `5557` | Robot state publisher port |
| `--root-output-dir` | `outputs` | Root directory for saved datasets |
| `--text-to-speech / --no-text-to-speech` | `True` | Voice feedback via espeak |

---

## Output Format

Datasets are saved in the [LeRobot v2.1](https://github.com/huggingface/lerobot) format under `<root-output-dir>/<dataset-name>/`:

```text
outputs/2026-04-03-14-30-00-G1-robot01/
├── data/
│   ├── chunk-000/episode_000000.parquet  # Per-episode tabular data
│   └── ...
├── videos/
│   ├── chunk-000/observation.images.ego_view/
│   │   ├── episode_000000.mp4   # H264-encoded ego camera video
│   │   └── ...
│   ├── chunk-000/observation.images.left_wrist/   # (only with --record-wrist-cameras)
│   └── chunk-000/observation.images.right_wrist/  # (only with --record-wrist-cameras)
└── meta/
    ├── info.json                # Dataset metadata (fps, features, sizes)
    ├── modality.json            # GR00T modality configuration
    ├── episodes.jsonl           # Per-episode metadata
    ├── episode_quality.jsonl    # Save-time validation and success status
    └── tasks.jsonl              # Task prompt definitions
```

### Recorded Data Channels

Each frame contains:

| Feature | Shape | Description |
|---|---|---|
| `observation.state` | `(49,)` with OmniHand | Measured body and hand joint positions (rad) |
| `observation.body_joint_velocity` | `(29,)` | Measured body joint velocities (rad/s) |
| `observation.root_orientation` | `(4,)` | Base IMU quaternion (wxyz) |
| `observation.base_angular_velocity` | `(3,)` | Base IMU angular velocity (rad/s) |
| `observation.projected_gravity` | `(3,)` | Gravity vector in body frame |
| `observation.omnihand_{left,right}_raw` | `(10,)` | Native measured OmniHand positions (rad) |
| `observation.images.ego_view` | `(480, 640, 3)` | Ego camera image (saved as MP4 video) |
| `observation.images.ego_view_left` | `(480, 640, 3)` | ZED rectified left-eye image (with `--record-zed-stereo`) |
| `observation.images.ego_view_depth` | `(480, 640, 3)` | ZED depth visualization video, not metric-depth storage (with `--record-zed-stereo`) |
| `observation.images.left_wrist` | `(480, 640, 3)` | Left wrist camera (only with `--record-wrist-cameras`) |
| `observation.images.right_wrist` | `(480, 640, 3)` | Right wrist camera (only with `--record-wrist-cameras`) |
| `action.motion_token` | `(64,)` | SONIC universal motion token |
| `teleop.{left,right}_hand_joints` | `(10,)` | Requested native OmniHand actions (rad) |
| `action.omnihand_{left,right}_raw` | `(10,)` | Named copies of the requested native actions |
| `control.hand_applied_position` | `(20,)` with OmniHand | Applied hand position command, distinct from requested and measured positions |
| `episode.success` | `(1,)` | Successful save (`1`) or discarded/invalid (`0`) |
| `capture.*` | `(1,)` | Source sequences and source/receive timestamps |
| `task_index` | `(1,)` | Index of the language task in `tasks.jsonl` |

The auxiliary velocity, raw-action, timing, and validity columns are retained
for provenance but are not added to the default `UNITREE_G1_SONIC` modality.
GR00T continues to train on the compact registered state/action contract.

### Automatic episode quality gates

While recording, stale camera, robot, active teleop, hand, malformed token,
NaN, or wrong-shaped required samples are not admitted. A dropout after an
episode has begun marks that episode invalid. On save, external-hand collection verifies that at
least one requested hand joint changed by `--minimum-hand-motion-rad` (default
`0.02`). Invalid episodes are preserved with `episode.success = 0`, listed in
`discarded_episode_indices`, and explained in `meta/episode_quality.jsonl`.
The voice announces "Recording failed validation" and the UI shows the reason.
These episodes are still saved and uploaded to the configured Hugging Face
dataset as unsuccessful. An operator discard still announces "Recording discarded".

Use `--no-require-hand-activity` only for a task that genuinely contains no
hand motion. Stream rates are diagnostic and do not determine episode acceptance.

---

## Post-Processing Datasets

After recording, you can clean and merge datasets using the processing script.
All commands below run in the **data collection virtual environment**:

```bash
source .venv_data_collection/bin/activate
```

### Remove Discarded Episodes

Episodes discarded during collection (`x` key or Y + A) are saved to disk
but flagged in `meta/info.json`. By default, the processing script removes these
flagged episodes so they are excluded from fine-tuning:

```bash
# Clean a default VR3PT dataset; keep its intentional zero SMPL fields.
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/my_dataset \
    --output-path outputs/my_dataset_cleaned \
    --no-remove-stale-smpl
```

To keep discarded episodes (e.g., for inspection), pass `--no-remove-discarded`.

### Remove Stale SMPL Frames

Teleop pauses or ZMQ frame drops create frames where `teleop.smpl_pose` is all
zeros.  The processing script detects these and also removes consecutive
frozen (identical) lead-in frames that precede them:

```bash
# Clean a single dataset in-place
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/my_dataset

# Clean and write to a new directory (non-destructive)
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/my_dataset \
    --output-path outputs/my_dataset_cleaned
```

```{warning}
If you collected data using **VR 3-point tracking** or **upper-body IK**,
the `teleop.smpl_pose` column will be all zeros because both planner modes use
retargeted upper-body targets instead of sending a full-body SMPL motion to
SONIC. In this case, you **must** disable SMPL cleaning to avoid dropping all
frames:

    python gear_sonic/scripts/process_dataset.py \
        --dataset-path outputs/my_dataset \
        --output-path outputs/my_dataset_cleaned \
        --no-remove-stale-smpl
```

### Merge Multiple Datasets

Combine several recording sessions into a single dataset.  The script
validates that all sessions share the same `script_config` (robot
configuration) before merging:

```bash
# Merge by listing datasets on the command line
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/session1 outputs/session2 outputs/session3 \
    --output-path outputs/merged_dataset

# Or use a text file (one dataset path per line, # for comments)
python gear_sonic/scripts/process_dataset.py \
    --dataset-list datasets.txt \
    --output-path outputs/merged_dataset
```

SMPL cleaning is applied by default during merging.  When enabled, the script
removes entire frames where the SMPL teleop pose is stuck at zeros — this
happens during operator pauses or ZMQ packet-drop periods where the SMPL
stream stops updating.  Consecutive frozen (identical) frames that lead into
a zero block are also removed, since they represent stale data right before
the dropout.  To skip this cleaning and merge only, add `--no-remove-stale-smpl`.

---

## Next Steps: Fine-tune and Deploy

The output dataset is directly compatible with the [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) post-training pipeline. To fine-tune a VLA model on your collected data and deploy it for autonomous inference, see the [VLA Workflow tutorial](vla_workflow.md).
