# Data collection PR cleanup

## Scope

Preserve the basic NVIDIA stack, including MotionBricks, SONIC training,
Decoupled WBC, models, assets, documentation and existing tooling. Cleanup is
limited to code introduced or changed in the data collection PR (#30).
Changes to a shared file must target the PR's additions, not unrelated upstream code.

The baseline for the audit is `origin/main...data-collection`; cleanup work lives
on `cleanup/teleop-deployment`. Keep teleop, cameras, multiple hand backends and
recording behavior intact unless a behavior change is explicitly agreed.

## Restart and first findings

The previous broad cleanup and README rewrite have been reverted. Before applying
this scoped cleanup, the complete tracked tree was verified identical to the PR.

- Removed the PR-added unused `Optional` import from the exporter.
- Removed the PR-added `ClockEstimate.to_local` helper, whose only caller was a
  test. Production already maps timestamps in `SenderSynchronizer.observe`.
  The clock-exchange test now checks that real production mapping instead.
- Left existing upstream unused imports in the Sphinx configuration alone.

Continue reviewing the PR's launcher, browser, hand/camera adapters and recorder
for duplicated or unreachable code. A low reference count is only a review hint;
it is not sufficient evidence to remove a supported mode or backend.

## Validation

Run the existing teleop, recording and hand tests explicitly; keep the upstream
root pytest configuration intact. Native control changes also require their
relevant regressions. Offline tests do not replace hardware validation.

First scoped pass: 382 tests passed under Python 3.10, excluding the optional
MuJoCo integration file. Unused-import checks for the edited runtime modules
and `git diff --check` passed. NVIDIA trees and original root configuration
were compared against `data-collection` and match exactly. Hardware was not run.
