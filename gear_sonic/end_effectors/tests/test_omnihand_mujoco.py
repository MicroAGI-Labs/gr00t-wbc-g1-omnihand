from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import zmq

from gear_sonic.end_effectors.mujoco_driver import OmniHandMuJoCoDriver
from gear_sonic.end_effectors.profiles import OMNIHAND_O10
from gear_sonic.end_effectors.protocol import HAND_STATE_SCHEMA, HAND_STATE_TOPIC, encode

REPO = Path(__file__).resolve().parents[3]
ASSET_ROOT = REPO / "gear_sonic/data/robot_model/model_data/g1_omnihand"
SCENE = ASSET_ROOT / "scene_49dof.xml"


def _controller_state(left: list[float], right: list[float]) -> bytes:
    return encode(
        HAND_STATE_TOPIC,
        {
            "schema": HAND_STATE_SCHEMA,
            "session_id": "mujoco-test",
            "sequence": 1,
            "monotonic_ns": 1,
            "backend": "sim",
            "profile": OMNIHAND_O10.name,
            "mode": "tracking",
            "sides": {
                side: {
                    "valid": True,
                    "connected": True,
                    "applied_position_rad": values,
                }
                for side, values in (("left", left), ("right", right))
            },
        },
    )


def test_atlas_combined_asset_has_body_active_passive_and_contact_contract():
    provenance = json.loads((ASSET_ROOT / "provenance.json").read_text())
    assert provenance["asset"] == "omnihand_o10_official_description"
    assert provenance["model_properties"]["actuators_per_hand"] == 10
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    assert (model.nq, model.nv, model.nu, model.neq) == (68, 67, 49, 12)
    for side in ("left", "right"):
        profile = OMNIHAND_O10.side(side)
        for name in profile.joint_names:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        prefix = "L" if side == "left" else "R"
        assert model.joint(f"{prefix}_index_dip_joint").id >= 0
    hand_body_ids = {model.body(name).id for name in ("L_palm", "R_palm", "L_index_pip_link", "R_index_pip_link")}
    assert any(
        int(model.geom_bodyid[geom]) in hand_body_ids and int(model.geom_contype[geom]) != 0
        for geom in range(model.ngeom)
    )


def test_atlas_scene_exposes_head_floor_contact_for_fall_reset():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    floor_geom = model.geom("floor").id
    head_mesh = model.mesh("head_link").id
    head_collision_geoms = {
        geom
        for geom in range(model.ngeom)
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        and model.geom_dataid[geom] == head_mesh
        and model.geom_contype[geom] != 0
        and model.geom_conaffinity[geom] != 0
    }
    assert len(head_collision_geoms) == 1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "com_marker") == -1

    head_touched_floor = False
    for _ in range(2_000):
        mujoco.mj_step(model, data)
        head_touched_floor = any(
            {int(contact.geom1), int(contact.geom2)} == {floor_geom, next(iter(head_collision_geoms))}
            for contact in data.contact
        )
        if head_touched_floor:
            break

    assert head_touched_floor
    # Demonstrates why pelvis-height-only fall detection misses this contact.
    assert data.qpos[2] > 0.2


def test_atlas_o10_actuators_move_active_and_nonlinearly_coupled_joints():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.gravity[:] = 0.0
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    context = zmq.Context()
    driver = OmniHandMuJoCoDriver(
        model,
        data,
        state_endpoint="inproc://omnihand-state-test",
        feedback_endpoint="inproc://omnihand-feedback-test",
        context=context,
    )
    try:
        left_target = OMNIHAND_O10.left.target(True, close_scale=0.35)
        right_target = OMNIHAND_O10.right.target(False)
        driver._accept_state(_controller_state(left_target.tolist(), right_target.tolist()))
        for _ in range(800):
            torque = driver.compute_torques()
            data.ctrl[:] = 0.0
            for offset, side in ((0, "left"), (10, "right")):
                actuators = driver.hand_actuators[side]
                data.ctrl[actuators] = np.clip(
                    torque[offset : offset + 10],
                    model.actuator_ctrlrange[actuators, 0],
                    model.actuator_ctrlrange[actuators, 1],
                )
            mujoco.mj_step(model, data)
        left = data.qpos[driver.hand_qpos["left"]]
        right = data.qpos[driver.hand_qpos["right"]]
        active_index = float(data.qpos[model.joint("L_index_pip_joint").qposadr[0]])
        passive_index = float(data.qpos[model.joint("L_index_dip_joint").qposadr[0]])
        assert np.max(np.abs(left)) > 0.3
        np.testing.assert_allclose(right, right_target, atol=0.02)
        assert active_index > 0.3
        assert passive_index > active_index
    finally:
        driver.close()
        context.term()
