# OmniHand O10 assets

This directory pins the official OmniHand O10 description package for offline
development. It includes visual and collision URDFs, Xacro sources, STL meshes,
and the vendor's native MuJoCo models for both hands.

Use `omnihand_description/assets/MJCF/scene.xml` for bilateral simulation and
`omnihand_description/assets/MJCF/omnihand_left.xml` for a standalone left
hand. Use `mjcf/omnihand_right.xml` for a standalone right hand; this
Atlas-owned wrapper supplies the shared asset definitions omitted by the
vendor right entry point without modifying vendor kinematics. Prefer these
MJCF files over a URDF-to-MJCF conversion: they contain the
official nonlinear equality constraints for the six passive joints in each
hand.

The source tree under `omnihand_description/` is preserved without fixes.
`provenance.json` records its source, archive hash, license evidence, validation
state, and the known upstream standalone-right loading defect. File integrity
can be checked from this directory with:

```bash
sha256sum -c omnihand_description.sha256
```

Physical mass, collision, and mount measurements are not yet qualified. They
are not required for kinematic retargeting work, but they remain gates for
high-fidelity contact and whole-body dynamics claims.
