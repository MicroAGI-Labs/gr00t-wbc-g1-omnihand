# DEX 1 teleoperation

The physical DEX 1 pair uses the same headset input, SONIC body controller,
camera/recording UI, and hand-reconnect button as the working OmniHand stack.
Each gripper has one active motor. Press either the index trigger or side grip
to close the corresponding gripper; release both to open it. Input uses the
existing 0.60/0.40 hysteresis and pause/idle behavior.

## Start

On the prepared Thor checkout, run:

```bash
cd /home/unitree/repos/gr00t-wbc-g1-omnihand
python gear_sonic/scripts/launch_data_collection.py --hand-backend dex1 --remote-ui
```

It starts the existing tmux dashboard with SONIC in `external` hand mode,
the headset streamer, DEX 1 controller, recorder, browser UI, and camera logs.
Open `http://127.0.0.1:8080` on Thor. For a browser on another computer, forward
the UI port with `ssh -N -L 8080:127.0.0.1:8080 unitree@<thor-host>` and open
the same address there. Body arming and headset mode selection follow the
existing teleop workflow; launching the UI does not bypass them.

The wrapper uses the launcher's existing XRT input default. If the working
headset setup uses CloudXR / Isaac Teleop, append
`--pico-input-source isaac-teleop`. Other launcher flags can also be appended.
`--check-only` checks prerequisites without opening serial devices or starting
any processes or services.

For another checkout with the standard teleop/data-collection environments
already installed, build the DEX 1 worker once:

```bash
bash install_scripts/install_dex1.sh
bash tools/teleop_dex1.sh
```

The installer pins Unitree's `dex1_1_service` SDK at
`2986d26eefa4136d4777493e6fd0b8bac7a4c6ae`, builds under `build/dex1`, and runs
an offline packet/control self-test. ARM64 uses the SDK's bundled libserialport
packages extracted locally; it does not install a system service. On x86-64,
`libserialport-dev` must already be installed. The DEX 1 Python adapter uses
`.venv_data_collection`; no additional Python environment is needed.

## Optional Orin hand server

The default stays **all on Thor**: `bash tools/teleop_dex1.sh` starts the local
hand supervisor and uses the grippers connected to Thor. The same checkout can
instead use a standalone hand service on Orin. Body control, headset input,
cameras, dashboard, and recording remain on Thor.

| Connection | Publisher / listener | Subscriber / client |
|---|---|---|
| TCP 5569, hand intent | Thor headset streamer | Orin hand controller |
| TCP 5570, hand config/state | Orin hand controller | Thor recorder and dashboard |
| TCP 5572, reconnect | Thor dashboard | Orin hand supervisor |

Use the wired robot network. The Orin is `192.168.123.164`; determine Thor's
address on that network with `ip route get 192.168.123.164` (the `src` address).
The hand server runs 200 Hz motor loops locally and publishes state at 50 Hz.
Only intent, status, and reconnect messages cross the network.

On Orin, with this branch checked out:

```bash
# Installs a separate .venv_hands and builds the pinned native worker.
# No CUDA, Torch, ROS, system Python changes, or motor I/O are involved.
bash install_scripts/install_hand_server.sh

# Offline compatibility check; does not open the grippers.
.venv_hands/bin/python -m gear_sonic.end_effectors.server \
    --teleop-host THOR_IP --check-only
```

The installer accepts `HAND_SERVER_PYTHON=/path/to/python3.10` and otherwise
finds Python 3.10 on PATH or in the existing uv-managed ARM64 installation.
For an Orin without working package-index access, transfer wheels from Thor
and set `HAND_SERVER_WHEELHOUSE=/path/to/wheels`. The pinned dependencies are
NumPy 1.26.4, pyzmq 27.2.0, and msgpack 1.2.2. `DEX1_SDK_SOURCE` can point to a
local git repository or bundle containing the pinned SDK commit for an offline
native build.

After moving the two gripper USB adapters to Orin, start the hand service there:

```bash
.venv_hands/bin/python -m gear_sonic.end_effectors.server \
    --teleop-host THOR_IP --enable-command
```

The existing `/dev/serial/by-id` assignments below preserve the physical sides.
The Orin user needs read/write access to those serial devices. Start only one
hand controller for the pair. The service owns its native workers and retains
the measured startup holds, watchdogs, torque caps, and explicit fault recovery.
Use `--dex1-transition-duration` on this server to configure its stroke timing;
the launcher's duration option controls only locally launched hands.

Then on Thor:

```bash
bash tools/teleop_dex1.sh --hand-server-host 192.168.123.164
# Optional wrist recording remains independent:
bash tools/teleop_dex1.sh --hand-server-host 192.168.123.164 --record-wrist-cameras
```

`--hand-server-host` connects to an **existing** service; it does not SSH, deploy,
or start a second hand controller. Local hand USB/worker checks and the local
hands pane are omitted. The recorder and browser use the specified hand-state
host, and the existing Reconnect hands button reaches the Orin supervisor.
`--check-only` on the Thor launcher validates Thor's launch prerequisites; the
Orin command above checks the hand installation itself.

For startup at boot, edit the checkout paths and `THOR_IP` in
`systemd/dex1_hand_server.service`, then install it on Orin:

```bash
sudo install -m 644 systemd/dex1_hand_server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dex1_hand_server.service
journalctl -u dex1_hand_server.service -f
```

Stop a manually launched server before enabling the service. The installer does
not enable it automatically. To return to all-on-Thor operation, stop the Orin
service, reconnect the USB adapters to Thor, and omit `--hand-server-host`.
A standalone server can also run on Thor with `--teleop-host 127.0.0.1`; pass
`--hand-server-host 127.0.0.1` to reuse it rather than starting another worker.

Input and feedback watchdogs use local monotonic receipt times and the hand
server's locally computed state age. They do not subtract Orin timestamps from
Thor timestamps or depend on synchronized wall clocks. Recording preserves
original timestamps and identifies their host domains in the capture metadata;
remote source times must not be treated as directly comparable to recorder
monotonic times. Timestamp alignment for downstream cross-host analysis is a
separate operation. If Thor reboots while Orin stays running, use Reconnect
hands before resuming; the restarted publisher's sequence must not be confused
with the previous boot. After restarting the entire Orin service, restart the
recorder so it locks the new hand session/configuration.

The installer and native offline self-test passed on the Orin NX with Ubuntu
20.04 and Python 3.10. A no-I/O fake-hand run across the actual Thor/Orin network
verified bilateral intent/state, hold after input silence, supervisor reconnect,
and resumed input with the same recording session.

The remote deployment still needs a physical side/direction and network-loss
check with the actual grippers on Orin. Offline builds, fake-motor tests, and
network checks do not replace that first hardware validation.

## Device profile

The initial `dex1.v1` profile is for this measured pair, in **output-shaft
radians**, including the existing encoder offsets:

| Hand | USB adapter, interface | Motor ID | Closed | Open | Position guard |
|---|---|---:|---:|---:|---|
| Left | FTBQ776H, 00 | 0 | 0.12 | 5.30 | -0.10 to 5.75 |
| Right | FTBWJBC1, 00 | 1 | -2.36 | 2.84 | -2.60 to 3.10 |

Left and right are from the robot's perspective.
Serial paths use `/dev/serial/by-id`, so USB enumeration cannot swap the hands.
Changing grippers or adapters requires measuring and updating the profile;
these values are not universal factory calibration. No calibration is written.
Do not run the stock DDS gripper service or the standalone movement tools at
the same time. Each native worker claims its port exclusively against new opens.

The default stroke duration is 1.5 seconds. `--dex1-transition-duration` accepts
1.35–30 seconds. Targets replace the current trajectory immediately and start a
new smooth ramp from measured position. The tested 1.0 N·m output torque cap is
retained. Closing against an object can hold at that cap; an obstructed opening
still trips the following-error guard. This is torque-limited grasping, not a
calibrated jaw-force controller.

## Process and fault behavior

The Python controller receives headset intent and publishes hand config/state
at 50 Hz. A C++ worker per gripper runs serial feedback and torque control at
200 Hz. Its loop continues when a position target is unchanged. The worker
initializes SDK command fields explicitly, including zero firmware gains, so
it does not depend on local changes to the SDK constructors.

- Startup reads both grippers before requesting measured-position holds.
- Stale headset input (0.5 seconds), pause, and safe idle cancel motion and
  hold measured position. The paused controller continues its heartbeat.
- A missing Python heartbeat (0.25 seconds), invalid/replayed command,
  feedback/control delay (60 ms), motor error, position/speed/torque violation,
  or voltage/temperature limit stops the worker. The other hand is stopped by
  the Python controller as well.
- Worker failures latch in the UI. Check the Hands pane and device, then use
  **Reconnect hands**. Reconnect starts fresh workers, seeds measured holds,
  and discards queued input before accepting fresh headset intent.
- Normal shutdown, EOF, and handled SIGINT/SIGTERM/SIGHUP send three stop
  packets per motor. SIGKILL, power loss, and an unresponsive transport cannot
  guarantee delivery of a software stop packet.

The initial motor checks retain the utility's 24–64 V supply range, housing
temperature below 55 °C, drive temperature below 80 °C, output speed below
9 rad/s, and measured output torque below 1.30 N·m. These are the local test
profile's limits, not newly established manufacturer operating ratings.

## Recording and validation

The exporter locks the profile declared by the hand controller. DEX 1 records
31 channels for `observation.state` and `action.wbc`: 29 body joints and one
motor per hand. Requested, interpolated/applied, and measured positions remain
distinct. Native aliases are `observation.dex1_{left,right}_raw` and
`action.dex1_{left,right}_raw`; metadata identifies `unitree_g1_dex1_sonic`.
Existing OmniHand and Dex3 dataset names/layouts remain compatible. This does
not make existing OmniHand-trained policies compatible with DEX 1.

Offline validation includes the real SDK packet/control self-test, worker
tests compiled against a stationary fake motor on pseudo-terminals, controller
and recorder regression tests, launcher command checks, and an HTTP/ZMQ UI
status/reconnect test. The fake motor exercises contact, release, watchdogs,
invalid commands, pause, port ownership, and shutdown without motor I/O.

The operator's initial integrated teleop run identified reversed left/right
mapping; the profile above includes that correction. After relaunching, confirm
each side and direction with clear jaws, then verify pause/reconnect and a
low-force grasp. DEX 1 MuJoCo integration is not included;
the launcher rejects `--sim --hand-backend dex1`.
