# G1 teleoperation and data collection

Teleoperate the G1 and collect demonstrations using NVIDIA GEAR-SONIC whole-body
control, PICO controllers, interchangeable hand backends, and ego/wrist cameras.
This fork is being focused on deploying and operating our robot.

**Start with the [operator handbook](README_DATA_COLLECTION.md).** It covers
installation by host, controls, recording rules, camera timing, hand recovery,
and troubleshooting. The [deployment and cleanup plan](docs/teleop_repo_plan.md)
identifies work remaining before a fresh checkout is easy to deploy.

## Current collection setup

| Component | Configuration |
|---|---|
| Body | NVIDIA GEAR-SONIC C++ deployment on Thor |
| Input | PICO, direct controller tracking, independent arm engagement |
| Hands | DEX 1 on Orin; local DEX 1 and OmniHand O10 alternatives |
| Cameras | ZED ego camera and USB wrist cameras on Thor |
| Data | LeRobot episodes, 50 Hz target, timestamps and quality metadata |
| Interface | Browser dashboard, local finalization and background Hub uploads |

The daily wrapper selects slow planner walking with a 0.6 m/s command cap.
The recording target is not a measured throughput guarantee. See the handbook
for configuration differences and remaining hardware validation.

## Start on a prepared robot

From this checkout on Thor:

```bash
bash tools/collect_dex1_orin.sh --check-only
bash tools/collect_dex1_orin.sh
```

The first command checks local prerequisites, not remote hand/camera readiness.
The second replaces an existing collection tmux session: finish recording and
local finalization before relaunching. Open `http://127.0.0.1:8080` on Thor,
or use the handbook's SSH tunnel from a laptop.

For a new machine, follow [installation by host](README_DATA_COLLECTION.md#installation-and-alternative-launches).
The wrapper assumes installed environments, controller artifacts, camera SDKs,
device permissions and networking. It is not yet a complete installer.

## Collect a demonstration

1. Check body, hands, cameras and recorder status in the dashboard.
2. Select the dataset and task prompt before recording.
3. Follow the handbook's engagement, walking and recording controls.
4. Accept or discard the take, then wait for local finalization. Check upload
   completion separately before relying on the remote copy.

Discarded nonempty takes are retained and marked unsuccessful. Prepare training
data using the documented quality filters. Hand activity checks do not establish
that the task succeeded. See the handbook for required inputs and dataset meanings.

## Development and NVIDIA foundation

- [Runtime code map](README_DATA_COLLECTION.md#code-tour-and-cleanup-boundaries)
- [Validation and hardware work](README_DATA_COLLECTION.md#validation-and-remaining-work)
- [NVIDIA foundation and model reference](README_UPSTREAM.md)
- [Citation](CITATION.cff), [license](LICENSE), and [third-party notices](legal/)

MotionBricks has been removed. SONIC training and older controller code remain
pending dependency review; their presence does not make them part of the operator
workflow.
