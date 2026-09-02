# OmniHand simulation asset provenance

Imported from `MicroAGI-Labs/research-atlas-v0` at commit
`14b1406c8d3069870e298341be0cd204cee5515d`.

The `omnihand_description/` vendor tree is unchanged. Verify it from this
directory with:

```bash
sha256sum -c omnihand_description.sha256
```

The Atlas-generated `g1_29dof_with_omnihand.xml` has one repository-layout
adjustment: G1 body mesh paths use the byte-identical existing `../g1/meshes/`
tree instead of Atlas's `../g1_inspire/meshes/g1/` location. OmniHand geometry,
inertials, actuators, contacts, joint limits, and nonlinear equalities are
unchanged.

The runtime `scene_49dof.xml` omits Atlas's world-fixed red `com_marker`
debug site. It has no dynamics, contact, sensor, or actuator role; removing it
keeps the third-person operator view free of a misleading gantry-like marker.

See `provenance.json` for the original vendor archive, hash, license evidence,
and fidelity limitations. In particular, mass properties, collision geometry,
and the physical wrist-adapter mount have not been validated against hardware.
