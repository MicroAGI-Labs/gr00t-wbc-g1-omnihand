# Focused G1 teleop repository

## Outcome and open requirements

A new operator can prepare the supported robot, check readiness, teleoperate and
collect a correctly described dataset through one documented workflow. Retain
NVIDIA GEAR-SONIC body control and its attribution.

This plan is not a claim that deployment is complete. Confirm the new camera
model, required hand types, host layout and task-specific collection rules.
The current Thor/Orin configuration is the baseline.

## Findings and work needed

| Area | Existing code | Next change |
|---|---|---|
| Launch | DEX 1 wrapper and Python launcher | Put host/device choices in a documented robot profile. |
| Setup | Separate collection, PICO, camera, hand and native installers | Provide repeatable setup by host and pin tested SDK/model versions. |
| Readiness | Local prerequisite checks | Add remote/device readiness, stream freshness, model and disk checks. |
| Hands | Shared external-hand protocol, DEX 1 and OmniHand adapters | Define tested combinations, calibration, joint names, units and ownership. Remote launch currently requires physical DEX 1. |
| Cameras | Composed service and separate drivers | Specify the new camera, device identities, calibration, streams and recording representation. |
| Collection | Admission checks, quality metadata, finalizer and uploader | Define profile-specific rules and separate task success from automatic checks. |

The intended operator flow is setup, readiness check, launch, then the browser.
These should share a versioned robot profile. Reuse existing runtime components;
keep credentials outside tracked profiles. Proposed interfaces are not implemented
commands yet.

## Cleanup boundaries

Keep active body inference/planning, robot assets, headset input, hand adapters,
cameras, collector, browser, deployment tools, dataset processing and their tests.
Preserve licenses, third-party notices and citation.

MotionBricks and controller-training entry points are removal candidates after
checking imports, configured assets, packaging, CI and documentation. Their
absence from the daily launch alone does not prove they are unused.

Do not delete `decoupled_wbc` wholesale:
`gear_sonic/utils/teleop/ik_upper_body.py` imports its robot model and IK solver.
If that optional mode is retained, preserve or extract its dependency closure.
Do not delete `gear_sonic/trl` or model directories based on their names; trace
runtime imports and asset paths first. Keep simulation fixtures needed to validate
retained functionality. Avoid runtime package renames during the first pruning pass.

## Collection contract to settle

- Required streams, allowed control modes, freshness/gap limits and measured
  acquisition-rate expectations.
- Task prompt, episode boundaries, operator success labels and unsuccessful-take
  handling. Current discard preserves nonempty takes as unsuccessful.
- Hand-specific activity criteria. Requested hand motion does not prove a grasp
  succeeded. DEX 1 has 31 assembled joints; OmniHand O10 has 49. Preserve joint names
  and modality slices and reject incompatible dataset combinations.
- Alignment mode, calibration identity, units, model hashes and software provenance.
- Local finalization, upload completion, recovery and training quality filters.
  The current uploader does not restore its queue automatically after restart.
- Whether metric depth is required: recorded ZED depth currently uses visualization
  video, not lossless metric depth.

Multiple hand backends means selecting the appropriate backend per robot profile;
simultaneous mixed hand types are not assumed supported.

## Implementation and merge sequence

1. Confirm hardware and rules; define supported profiles and deployment artifacts.
2. Implement profile-based setup/readiness/launch around existing components.
3. Remove unrelated code in reviewable groups, updating imports, packaging, CI
   and docs alongside each removal.
4. Run retained Python tests and relevant native/simulation checks; validate fresh
   installation on each supported host platform.
5. Validate engagement, rearming, stopping and hand recovery on the robot. Record
   successful and unsuccessful takes; inspect video, Parquet, timestamps and quality
   metadata; verify local finalization and upload under the intended camera load.
6. Merge the PR after validation. Recheck ancestry and the resulting file tree
   immediately before merging. Removing inherited files does not need a force-push.
