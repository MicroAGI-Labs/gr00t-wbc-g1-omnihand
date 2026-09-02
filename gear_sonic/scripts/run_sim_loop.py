"""Entry point for running a MuJoCo simulation loop with the G1 robot model.

Parses a YAML-based WBC config via tyro CLI, instantiates the G1 robot model,
and launches the simulator (optionally with offscreen image publishing).
"""

from typing import Dict

import tyro

from gear_sonic.data.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel

ArgsConfig = SimLoopConfig


class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model
        self.config = config

        init_channel(config=self.config)

        # Create simulator using factory
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name

    if config.enable_image_publish:
        assert (
            config.enable_offscreen
        ), "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()

    camera_configs = {}
    if config.enable_offscreen and config.enable_image_publish:
        if config.stream_camera == "third_person":
            camera_configs = {
                "third_person": {
                    "height": 480,
                    "width": 640,
                    "tracking_body": "pelvis",
                },
                # Data collection always stores the canonical head-mounted
                # camera as observation.images.ego_view.  Keep publishing it
                # when the remote UI also requests the third-person view.
                "ego_view": {
                    "height": 480,
                    "width": 640,
                    "mjcf_name": "head_camera",
                },
            }
        else:
            camera_configs = {
                "ego_view": {
                    "height": 480,
                    "width": 640,
                    "mjcf_name": "head_camera",
                }
            }

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
        camera_configs=camera_configs,
    )
    if config.smoke_once:
        try:
            sim_wrapper.sim.sim_env.sim_step()
            env = sim_wrapper.sim.sim_env
            print(
                "MuJoCo smoke step OK: "
                f"nu={env.mj_model.nu}, body={len(env.obs['body_q'])}, "
                f"left_hand={len(env.obs.get('left_hand_q', ()))}, "
                f"right_hand={len(env.obs.get('right_hand_q', ()))}"
            )
        finally:
            sim_wrapper.sim.close()
        return
    # Start simulator as independent process
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
