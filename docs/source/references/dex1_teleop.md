# DEX1 teleoperation

The merged DEX1 stack uses the existing headset, SONIC body controller, camera
server, recorder, browser UI, and hand reconnect control. Each gripper has one
active motor; the hand controller receives the existing open/close intent and
publishes state at 50 Hz while native workers run the motor loop at 200 Hz.

## Local Thor setup

Build and validate the worker, then launch the dashboard:

```bash
bash install_scripts/install_dex1.sh
python gear_sonic/scripts/launch_data_collection.py \
    --hand-backend dex1 --record-wrist-cameras
```

Use `--check-only` to validate configuration without opening serial devices or
starting tmux. The DEX1 worker requires explicit command enablement through the
launcher and retains the measured holds, watchdogs, torque limits, and manual
Reconnect hands recovery path.

## Orin hand server

Move both USB adapters to the Orin, install the standalone CPU service there,
and pair SSH once from Thor:

```bash
bash install_scripts/install_hand_server.sh
bash tools/setup_orin_ssh.sh 192.168.123.164
```

Then launch the daily configuration from Thor:

```bash
bash tools/collect_dex1_orin.sh
```

Only hand intent, state, and reconnect messages cross the wired network. The
Orin owns the motor workers; Thor continues to own body control, cameras,
recording, and the browser UI. Use `--no-start-hand-server` when the Orin
service is managed separately with systemd. Check the service with:

```bash
.venv_hands/bin/python -m gear_sonic.end_effectors.server \
    --teleop-host THOR_IP --check-only
```

The physical profile is device-specific: left is JR motor 0 with the measured
`-0.10..5.75` guard, and right is motor 1 with the `-2.60..3.10` guard. Do not
run another gripper service on those serial ports. Confirm physical left/right
motion during the first hardware run; USB enumeration alone does not establish
the motor connection.
