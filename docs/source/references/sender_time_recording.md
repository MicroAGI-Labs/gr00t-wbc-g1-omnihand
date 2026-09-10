# Sender-time recording

This opt-in mode aligns recorded rows using producer timestamps. The default
recording mode is unchanged. It does not delay teleop commands.

At 50 Hz, the recorder keeps a history for each required camera, robot state,
manager state, hands, and active pose/planner stream. It records each target
100 ms later, after all required streams advance beyond it, using their latest
samples at or before the target. It waits up to another 250 ms for unavailable
inputs, then reports skipped targets. Stop drains targets through the stop time.

Robot state, camera frames, pose, and hand snapshots use their host measurement
timestamps. Planner and manager use publication timestamps. Camera timestamps
are taken at the host capture boundary; they are not hardware exposure clocks.
Missing producer timestamps block recording; receipt time is never substituted.

## Enable

Update the camera publisher and the remote hand checkout with this implementation
before enabling the new mode. The camera publisher must send a separate
`capture_monotonic_ns` for each image; the hand publisher must send `clock_id`.
Running processes need to load the updated code during the next planned restart.

Add these arguments to the existing data-collection launch command:

```bash
--sender-time-recording --dataset-name sender-time-evaluation
```

The launcher forwards `--clock-port 5574` when starting a remote hand server.
When using `--no-start-hand-server`, enable that flag on the separately managed
updated hand server, or run the read-only clock service on the hand host:

```bash
.venv_hands/bin/python -m gear_sonic.data.clock_sync --port 5574
```

Only one clock service should bind the chosen port. `--hand-clock-port` changes
the port used by the launcher/recorder. No additional dependencies are required.

Use a new dataset name for initial evaluation. Resuming with a mismatched
synchronization schema is rejected before video writers open. The existing
stereo, depth, and wrist recording flags retain their meanings.

## Clock mapping

The initial implementation requires camera, robot, and teleop publishers on the
recorder host, addressed via `localhost`, `127.0.0.1`, or `::1`. Remote hands are
supported through a four-timestamp clock exchange on their host. Other remote
producer arrangements require additional mapping support.

The client collects fresh exchanges, chooses the lowest-RTT estimate, and maps
the remote monotonic clock into the recorder's monotonic clock. It requires
three exchanges within two seconds and at most 5 ms estimated uncertainty.
The uncertainty includes half the network round-trip time and an assumed
100 ppm drift budget. These settings need validation on the actual hosts; they
are not a measured hardware synchronization guarantee. Boot identity mismatches
and expired/uncertain mappings block hand admission.

Selection accounts for uncertainty: a sample whose time interval could extend
past the target is excluded. The advancing sample must be entirely beyond the
target. Receiver timestamps remain available for diagnosing delivery delay.

## Recorded diagnostics and limits

New `capture.sync.*` fields preserve each selected source timestamp, mapped time,
receipt time, applied clock offset, and uncertainty as integer nanoseconds.
`capture.sync_target_monotonic_ns` identifies the common target. Unused streams
are marked `-1`. Status and episode validation include input errors, skipped
targets, history overflow, and late samples. A synchronization gap makes a take
unsuccessful under this branch's existing quality policy; recordings are retained.

Camera captures are deduplicated independently, so republishing a cached wrist
image cannot advance its history. Stereo/depth channels from one grab retain the
same source timestamp. Repeated camera exposures across rows remain possible.

Histories are bounded to 32 samples per stream. Camera/transport/encoder queues
are also bounded; the system does not promise lossless capture of every source
packet. Larger delays or higher source rates may require larger histories.
The existing encoder and finalizer implementation is retained in this change.

LeRobot/video time stays `frame_index / fps`. Use the common target field to
detect real acquisition-time gaps; continuous video time does not represent a
gap's elapsed duration. First validate an evaluation dataset under the intended
camera and upload load before using this mode for production collection.
