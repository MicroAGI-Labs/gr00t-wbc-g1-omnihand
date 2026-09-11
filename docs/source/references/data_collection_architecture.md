# Data collection code tour

For the full operator and design guide, read the
[Data-collection PR handbook](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/data-collection/README_DATA_COLLECTION.md).
It covers daily launch, every dashboard process, controller state transitions,
hand recovery, both collector techniques, timestamps, dataset fields, background
saving/uploading, and the reasons behind these choices. This page is a short
code index.

The `data-collection` branch connects PICO teleoperation, SONIC whole-body
control, external hands, cameras, and dataset recording. Start with
[the collection guide](../tutorials/data_collection.md) for launch commands and
[the controller reference](pico_controller_controls.md) for operator controls.

## Runtime path

| Component | Entry point or implementation | Responsibility |
| --- | --- | --- |
| Dashboard launcher | `gear_sonic/scripts/launch_data_collection.py` | Starts the configured teleop, deploy, exporter, viewer, and hand processes in tmux; selects hardware or simulation. |
| PICO publisher | `gear_sonic/scripts/pico_manager_thread_server.py` | Reads controller samples, calibrates each arm on grip engagement, sends direct wrist targets and locomotion commands, and publishes hand intent. |
| Controller semantics | `gear_sonic/utils/teleop/pico_controls.py`, `vr_arm_clutch.py`, `pose_transition.py` | Owns button gestures, neutral-stick rearming, arm reference frames, and explicit home/rest transitions. Slow walking reaches 0.6 m/s at full stick. |
| SONIC receiver | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp` | Accepts planner and VR commands, feeds whole-body control, and handles publisher loss. |
| External hands | `gear_sonic/end_effectors/server.py`, `supervisor.py`, `controller.py`, `backends/` | Runs local or SSH-hosted hand workers, applies measured holds, reports feedback, and recovers transport failures. |
| Cameras | `gear_sonic/camera/sensor_server.py`, `composed_camera.py`, `drivers/` | Publishes selected wrist, ZED RGB, and depth streams with source timestamps and device identities. |
| Recorder | `gear_sonic/scripts/run_data_exporter.py`, `gear_sonic/data/features_sonic_vla.py`, `exporter.py` | Selects inputs, builds observations/actions with timing provenance, and manages episode data. |
| Finalization and upload | `gear_sonic/data/episode_finalizer.py`, `hub_uploader.py`, `gear_sonic/scripts/upload_dataset_snapshot.py` | Owns detached buffers, commits local episodes, and uploads immutable snapshots in the background. |
| Sender-time synchronization | `gear_sonic/data/clock_sync.py`, `sender_sync.py` | Provides optional clock exchange and selection by producer timestamps. |
| Operator UI | `gear_sonic/scripts/run_camera_web_viewer.py`, `gear_sonic/utils/data_collection/` | Displays camera/recorder/hand state and sends recording, dataset, and recovery commands. |

## Direct control and fault handling

Live calibrated VR poses are sent directly. There is no optional Cartesian
smoothing, speed/acceleration/jerk limiter, tracking-jump filter, or filter-trial
logger. Grip release holds the last emitted arm target; a fresh press captures
a new reference. Explicit home/rest commands still interpolate over their
configured duration.

The direct-controller watchdog remains independent of pose filtering: 200 ms
without a valid controller sample latches arm holds, stops locomotion, gates
hand intent, and cancels an active return. Fresh data requires release and
recalibration before tracking resumes. SONIC also handles loss of the publisher.
Motor health checks, measured startup holds, hardware position/torque limits,
and manual stop controls remain part of the hardware control path.

The launcher also supports full-body SMPL, upper-body IK, Isaac Teleop input,
simulation, and multiple hand backends. These have explicit entry points and
tests. Selecting the DEX1 hardware profile does not make those paths dead code.

## Validation

The `Recording pipeline tests` workflow runs `gear_sonic/tests` and
`gear_sonic/end_effectors/tests`, excluding the MuJoCo test module. Coverage
includes direct pose publication, independent arm engagement, input loss,
recording gestures, hand transport recovery, camera provenance, synchronization,
episode finalization, and uploads. Native planner disconnect tests live in
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/`.

Offline tests do not establish physical tracking quality or recording cadence.
Those require an explicit hardware collection session.
