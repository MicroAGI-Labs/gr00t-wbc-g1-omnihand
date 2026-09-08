# DEX 1 teleoperation

The physical DEX 1 pair uses the same headset input, SONIC body controller,
camera/recording UI, and hand-reconnect button as the working OmniHand stack.
Each gripper has one active motor. Press either the index trigger or side grip
to close the corresponding gripper; release both to open it. Input uses the
existing 0.60/0.40 hysteresis and pause/idle behavior.

## Start

On the prepared Thor worktree, run this single command:

```bash
bash /home/unitree/worktrees/gr00t-wbc-g1-dex1-teleop/tools/teleop_dex1.sh
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

## Device profile

The initial `dex1.v1` profile is for this measured pair, in **output-shaft
radians**, including the existing encoder offsets:

| Hand | USB adapter, interface | Motor ID | Closed | Open | Position guard |
|---|---|---:|---:|---:|---|
| Right | FTBQ776H, 00 | 0 | 0.12 | 5.30 | -0.10 to 5.75 |
| Left | FTBWJBC1, 00 | 1 | -2.36 | 2.84 | -2.60 to 3.10 |

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

The prior standalone tool demonstrated bilateral open/close motion on this
pair. **The integrated teleop path still needs its first supervised hardware
run.** Start with clear jaws, confirm each side and direction, then verify
pause/reconnect and a low-force grasp. DEX 1 MuJoCo integration is not included;
the launcher rejects `--sim --hand-backend dex1`.
