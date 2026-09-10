# Recording pipeline comparison and migration recommendation

Reviewed 2026-09-10.

**Recommendation:** adopt main's delayed, causal frame assembly on this branch, while preserving this branch's independent camera ingestion, stereo/depth support, recording controls, and dataset features. Extend synchronization to each required camera. Copying main's recording files wholesale would remove useful behavior from this branch.

This is a code comparison and offline verification, not an implementation or a hardware recording benchmark. No running service was restarted or reconfigured.

## 1. Versions compared

| Version | Exact baseline |
| --- | --- |
| Wrist-camera branch | `/home/unitree/worktrees/gr00t-wbc-g1-pr3-wrist-cameras`, branch `feat/pr3-wrist-cameras`, HEAD `9ec0ded`, including the working-tree changes present during review |
| Current main | Freshly fetched `origin/main`, `e29fe46890170f1afc09a28eed40564bc1e2b3f3` |
| Local branch named `main` | `a4273da`, outdated; **not** the baseline for this recommendation |

The working tree contains existing edits to the exporter, launcher, UI, and teleop code. In particular, the uncommitted exporter change restores missing metadata indexes only for a provably empty dataset. Preserve this work during integration. The branch references below describe the inspected working tree; main references are pinned to the fetched commit.

## 2. What “waiting for every stream” means on main

The description is substantially correct, with two qualifications: the delay applies to **every recording tick**, not only startup; and “every stream” means the streams required by the selected teleop mode and dataset.

At 50 Hz, main creates target times spaced 20 ms apart, beginning at the recording start command. For each target `T`:

1. Wait until the recorder clock reaches at least `T + 100 ms`.
2. Require every relevant stream's newest received timestamp to be **strictly greater than `T`**. This newest timestamp is called its watermark.
3. Select the newest sample with timestamp **less than or equal to `T`** from each stream's history.
4. Check selected samples for age, missing values, and hardware faults, then assemble the dataset row.
5. If a required stream has not advanced, allow an additional 250 ms before recording a gap. The normal waiting deadline is therefore `T + 350 ms`; this is an admission deadline, not a hard bound on scheduling or disk I/O.

The later sample proves that the stream has advanced; its values do **not** enter the earlier row. Missing or stale past samples and explicit faults can reject a target without consuming the full wait timeout.

Always required: robot state, a camera message, and manager state. External hands are required when configured. Manager state **selected at `T`** determines whether the row also needs a pose or planner message. Main does not wait for an inactive pose stream during a planner-only mode. The camera message is one synchronization stream containing multiple images; main does not independently watermark each physical camera. [M1, M2]

Example, using recorder receipt times in milliseconds:

| Stream | Latest sample at/before target 1000 | Later sample proving advancement | Value used in row |
| --- | --- | --- | --- |
| Camera | 980 | 1020 | Camera at 980 |
| Robot | 990 | 1010 | Robot at 990 |
| Manager | 995 | 1015 | Manager at 995 |
| Planner | 992 | 1012 | Planner at 992 |
| Hand | 985 | 1005 | Hand at 985 |

If the hand's later sample has not arrived, the row waits even though a hand sample already exists. Once it arrives and the delay has elapsed, the row uses the hand sample at 985, not 1005.

## 3. Actual differences

| Area | Wrist-camera branch | Current main | Consequence |
| --- | --- | --- | --- |
| Frame assembly | Polls sources and validates the current `latest_*` messages before adding a row. | Selects all row inputs against a historical target time. | Branch freshness checks do not establish a common time boundary. |
| Histories | Camera histories exist; robot, hand, pose, planner, and manager are held as latest values in the collector. | `CausalSynchronizer` keeps up to 32 admitted samples per stream. | Main can choose past inputs after newer ones arrive. |
| Intentional delay | No explicit all-stream synchronization delay. Camera reserve introduces variable buffering. | 100 ms per target, configurable. | Main deliberately adds recording latency without changing teleop control latency. |
| Camera receiver | Separate deques of distinct captures per camera, capacity 5; trims to depth 2 before consuming one and retains previous selections. | Bounded FIFO of whole camera messages, capacity 5; recording drains all pending messages into causal history. | Branch handles cameras independently; main retains candidate bundles for time-based lookup. |
| Camera publisher | Caches each camera's latest frame and publishes on any new arrival, retaining original timestamps. | Requires fresh frames from all configured cameras in the same `read()` call; otherwise returns no message after draining their queues. | Replacing the branch publisher can lose useful arrivals from cameras running at different phases. |
| Transport | Robot and hand SUB sockets conflate to latest; mixed teleop socket is non-conflated with receive HWM 20; background camera socket is non-conflated with HWM 5. | Same basic transport policy, followed by collector histories. | Main is not a lossless capture queue for every producer packet. |
| Manager state | Uses current mode; no explicit manager freshness check in `_validate_recording_inputs()`. | Requires a fresh causal manager sample and selects additional streams using that sample's mode. | Main avoids combining an old target with a newer mode transition. |
| Freshness defaults | Robot 100 ms; cameras 100 ms; teleop 200 ms; hands 200 ms, primarily measured against current collector time. | Robot 100 ms; cameras 250 ms; teleop 200 ms; hands 200 ms, measured against the target, with per-camera age estimation. | Migrating changes the meaning of “age”; thresholds need an explicit choice. |
| Camera rate policy | Rolling rates are diagnostic; legacy minimum recording rate does not fail episodes. | Enforces a 25 Hz minimum camera publish/receive rate as well as camera availability and age. | This is a dataset acceptance policy change, not just synchronization. |
| Missing input | Blocks rows until inputs recover; errors after the first row accumulate in episode validation. | Waits within a bound, skips target ticks, and records gap counts/reasons; distinguishes teleop interruptions from hardware failures. | Main exposes timeline gaps and permits some teleop interruptions as warnings. |
| Stop recording | Hands buffered episode to the finalizer immediately on stop. | Records a stop target and drains pending target ticks through that boundary before finalizing. | A delayed recorder needs this drain to preserve its final portion. |
| Finalization | Background episode saving and uploading already exist; local finalizer/uploader classes live in the runner. | Separate finalizer/uploader modules, finalizer completion results, and recovery if a handoff is rejected. | Queuing completed episodes is separate from synchronizing input samples. |
| Video pressure | Bounded queue, but `add_frame()` and shutdown waits have no timeout. | 250 ms enqueue timeout by default, timed shutdown, worker-owned flushing/closing. | Main bounds several encoder waits; neither version guarantees successful writing under overload. |
| Dataset timing | Extensive source/receipt timing and sequence fields, including per-camera source timestamps; no explicit common sync target. | Adds common target, selected receipt times, selected ages, and synchronization/drop diagnostics. | Preserve the branch's richer source fields and add main's alignment fields. |
| Camera payload/features | Independent wrists plus stereo ZED and optional depth payload/recording support. | Selected ZED RGB view and wrist support; different camera timing schema. | A wholesale replacement would regress branch functionality and schemas. |

Sources: branch [B1–B5]; main [M1–M6].

There are several different queues here: transport queues, camera receiver queues, per-stream synchronization histories, video encoder queues, and completed-episode finalizer/upload queues. Having the latter two does not make the inputs in a row synchronized. Both collectors also declare `obs_act_buffer`, but that declaration is not the all-stream synchronization mechanism.

## 4. What main guarantees—and what remains unresolved

**Main gives a consistent boundary in the recorder's receipt-time clock.** Selected inputs precede the same target and pass explicit age checks. This is stronger than assembling whatever latest values happen to be available during sequential polling. It does not mean the camera exposure, hand feedback, and robot measurement happened simultaneously.

Robot, hand, and teleop receipt times are assigned when the collector polls/decodes their messages. Camera receipt time is assigned in the background receiver after decoding. Scheduling and transport delay affect those timestamps, and conflation can discard intermediate robot/hand messages before they reach the history. A watermark does not prove that every upstream sample was received. [M1, M2, B4]

**Remote clocks require care.** The hand server runs on the Orin while the collector runs on Thor. Their monotonic clocks have different origins. Use collector receipt time for the initial integration, and keep source timestamps as provenance. Main's camera-age estimate combines collector time since receipt with capture-to-publish duration measured entirely on the camera host. It avoids subtracting monotonic clocks from different hosts, but it does not measure network transit or exact exposure alignment. [M3]

**A new bundle is not a new capture from every camera.** This is especially relevant when retaining the branch's cached asynchronous publisher. A moving right wrist could advance a bundle's timestamp while the left wrist remains frozen. The branch already avoids resetting a cached image's original receipt time; preserve that property and extend it into synchronization. [B2, B5]

**Neither pipeline manufactures a new exposure for every 50 Hz row.** Reusing the same past image can be valid when a camera is slower than the recorder. Record duplicate identity and age; never describe 50 dataset rows per second as 50 unique camera captures without measuring it.

**Dataset time and acquisition time differ after gaps.** Both exporters default the LeRobot `timestamp` to `frame_index / fps`, and the reviewed frame builders do not override it. Main preserves skipped target intervals in `capture.sync_target_monotonic_ns` and validation metadata, but ordinary dataset/video time remains a contiguous fixed-rate timeline. Training code must use the acquisition metadata or split at gaps if it needs true elapsed time. [M1, M6, B3]

## 5. Recommended implementation

### A. Port delayed selection, preserving this branch's ingestion

Bring over `CausalSampleBuffer`, `CausalSynchronizer`, target scheduling, bounded waiting, gap accounting, stop/drain handling, and their regression tests. Change the frame builder to receive an explicit selected-samples object. Every recorded observation, action, mode, and timing field must derive from that selection rather than reading live `latest_*` state partway through assembly.

Keep the branch's asynchronous camera publisher. Replace the camera receiver's trim-and-pop path **for recording** with a drain of distinct per-camera captures into synchronization histories. Keep the live preview's existing low-latency sampling behavior.

For the current stereo/wrist setup, synchronize robot, manager, hand, active planner/pose, and each required camera channel. Give ZED left/right/depth from one grab a shared capture identity so related outputs stay paired. Deduplicate cached images by original capture identity and never advance a camera watermark solely because another camera caused a bundle to be republished.

Start with main's 100 ms delay and 250 ms additional wait as configurable values. Retain the branch's 100 ms camera freshness budget initially **relative to the target**, while measuring actual per-camera ages; do not silently relax it to main's 250 ms. Keep robot/teleop/hand limits explicit. These are initial evaluation settings, not validated hardware optima.

Size histories from stream rate, delay, wait budget, and scheduler margin. Main's 32 samples cover approximately 533 ms at 60 Hz or 640 ms at 50 Hz; at a higher merged-message rate the same capacity covers much less time. Per-camera queues make that budget easier to reason about. Add capacity/drop diagnostics and avoid retaining duplicate decoded image arrays.

### B. Reconcile metadata and recording policy explicitly

Preserve stereo/depth/wrist features, robot and hand source sequences/times, dataset modality mappings, chosen teleop mode, intentional base-mode pauses, hand-activity validation, and existing browser controls. These are separate from frame synchronization.

Use per-camera `capture_monotonic_ns` metadata as on main while retaining compatibility with the branch's scalar `sample_monotonic_ns`, `timestamps`, and `depths`. Add target time, selected receipt time/age, capture identity, reuse count, and gap reason. Unknown source timing must remain marked unknown.

Create a dataset with the new schema for the first migrated recording. Do not append new feature keys to an existing immutable dataset schema. Define supported legacy behavior explicitly rather than silently losing synchronization metadata.

Use separate outcomes for normal teleop pauses, missing tracking, camera/robot/hand failures, operator discard, and encoder errors. Keep recordings and diagnostics available, but distinguish a completed local save from a take that passed quality checks. Adopt main's stop/drain and finalizer completion reporting; preserve this branch's operator workflow and its current empty-dataset metadata repair.

### C. Adopt bounded writer/finalizer handling as a separate change

Bring over the timed writer enqueue/shutdown and failed-handoff ownership restoration with their tests. Reconcile exporter transactions and upload behavior separately from synchronization. This keeps a synchronization regression distinguishable from an encoding or save failure.

## 6. Verification and acceptance criteria

Completed during this review:

- Ran main's three `test_causal_sync.py` tests from an isolated temporary copy: **3 passed**. They cover waiting for all streams, stale past samples, history bounds, out-of-order rejection, and trimming.
- Ran an additional offline five-stream scenario matching the example above: the target waited for hand advancement, then selected only past samples.
- Ran this branch's three focused wrist-camera tests for independent publisher arrivals, asynchronous 60 Hz → 50 Hz receiver sampling, and detection of cached/stalled images: **3 passed**, 20 unrelated tests deselected.
- Inspected main's collector tests for future-watermark exclusion, bounded gaps, 50 Hz camera selection, teleop dropout handling, hand reconnect invalidation, and rejected finalizer handoff. These integration tests were inspected, not run.

Before enabling the migrated recorder on hardware, require:

| Scenario | Acceptance criterion |
| --- | --- |
| Deterministic 60 Hz asynchronous cameras → 50 Hz recorder | Every row uses the newest admitted capture at/before its target; bounded memory; no avoidable loss caused by sampling before selection. |
| One camera freezes while others continue | Its own identity and watermark stop; another camera cannot make it appear fresh; gaps and affected camera are reported. |
| Hand stream delayed or disconnected | Delay waits within budget; disconnect does not produce a valid row from stale or invalid hand feedback. |
| Mode change, headset interruption, base pause | Historical manager state controls row requirements; intended pause policy is preserved; hardware failures remain errors. |
| Recording start/stop | First target is the start boundary; pending targets through stop are resolved; no post-stop targets enter the episode. |
| Encoder stalls, disk/save errors | Bounded failure reporting; completed buffers retain an owner; UI does not claim local commit before completion. |
| Schema/restart | Stereo, depth, wrists, and timing round-trip; new-schema recording and supported dataset reopening work; existing nonempty datasets remain intact. |
| Hardware load test | Measure per-camera unique capture rate, row rate, selected ages/skew, target delay, gaps, queue drops, encoder backlog, and memory under the actual enabled streams and uploads. |

Run the new assembly path first in an offline replay or a shadow collector that emits diagnostics without motor commands or production dataset writes. Compare selected source identities for the same input trace. Then validate real recordings and loaded dataset/video frame counts. No hardware synchronization performance claim is made by this review.

## 7. Decision

**Yes, implementing main's recording model here is feasible and worthwhile.** The useful combination is this branch's independent, timestamp-preserving camera ingestion plus main's explicit delayed selection and bounded recovery. The migration spans receiver APIs, frame assembly, timestamp schema, and stop/save semantics; it is more than adding a startup sleep or increasing a queue size.

Use separate reviewable changes for (1) causal synchronization with per-camera histories, (2) schema/UI/policy integration, and (3) writer/finalizer reliability. Keep the live teleop control path outside the recorder's intentional delay.

## Source references

- **B1:** [Branch collector: configuration, validation, frame building, and run loop](../../../gear_sonic/scripts/run_data_exporter.py), especially `_validate_recording_inputs`, `_add_data_frame`, `_add_capture_features`, `_finish_recording`, and `run`.
- **B2:** [Branch camera publisher and receiver](../../../gear_sonic/camera/composed_camera.py), especially `ComposedCameraSensor.read`, `_buffer_camera_frames`, and `_sample_camera_frames`.
- **B3:** [Branch exporter](../../../gear_sonic/data/exporter.py), [video writer](../../../gear_sonic/data/video_writer.py), [features](../../../gear_sonic/data/features_sonic_vla.py), and [camera wire schema](../../../gear_sonic/camera/sensor_server.py).
- **B4:** [Robot state subscriber](../../../gear_sonic/utils/data_collection/zmq_state_subscriber.py).
- **B5:** [Branch wrist-camera regression tests](../../../gear_sonic/tests/test_wrist_cameras.py) and [recording quality tests](../../../gear_sonic/end_effectors/tests/test_recording_quality.py).
- **M1:** [Main collector at e29fe46](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/e29fe46890170f1afc09a28eed40564bc1e2b3f3/gear_sonic/scripts/run_data_exporter.py), especially lines 613–656, 763–834, 1067–1225, and 1347–1391.
- **M2:** [Main causal synchronizer](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/e29fe46890170f1afc09a28eed40564bc1e2b3f3/gear_sonic/data/causal_sync.py).
- **M3:** [Main camera publisher, FIFO, and age estimation](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/e29fe46890170f1afc09a28eed40564bc1e2b3f3/gear_sonic/camera/composed_camera.py).
- **M4:** [Main recording integration tests](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/e29fe46890170f1afc09a28eed40564bc1e2b3f3/gear_sonic/tests/test_recording_pipeline.py) and [synchronizer unit tests](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/e29fe46890170f1afc09a28eed40564bc1e2b3f3/gear_sonic/tests/test_causal_sync.py).
- **M5:** [Main video writer](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/e29fe46890170f1afc09a28eed40564bc1e2b3f3/gear_sonic/data/video_writer.py) and [episode finalizer](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/e29fe46890170f1afc09a28eed40564bc1e2b3f3/gear_sonic/data/episode_finalizer.py).
- **M6:** [Main dataset exporter](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/e29fe46890170f1afc09a28eed40564bc1e2b3f3/gear_sonic/data/exporter.py) and [feature definitions](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/e29fe46890170f1afc09a28eed40564bc1e2b3f3/gear_sonic/data/features_sonic_vla.py).
