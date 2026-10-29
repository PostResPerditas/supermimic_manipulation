"""Independent single-world ARX5 + OSC Door contact scene."""

import math

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from clab.robots.arx5.binding import Arx5Binding
from clab.scenes import workcell
from clab.scenes.robosuite_table import ground_cfg

from .props import door_cfg


def door_observation(env, target_angle: float):
    door = env.scene["door"]
    return torch.cat(
        (
            door.data.joint_pos,
            door.data.joint_vel,
            torch.full_like(door.data.joint_pos, target_angle),
        ),
        dim=-1,
    )


def make_env_cfg(
    *,
    target_angle: float = 0.30,
    damping: float = 0.1,
    frictionloss: float = 0.0,
) -> ManagerBasedRlEnvCfg:
    if not math.isfinite(target_angle) or not 0 < target_angle <= 0.4:
        raise ValueError("target_angle must be in (0, 0.4] radians")
    if any(not math.isfinite(v) or v < 0 for v in (damping, frictionloss)):
        raise ValueError("Door damping and frictionloss must be finite and nonnegative")
    layout = workcell.load_layout()
    robot, action = Arx5Binding.robot_and_action_cfg(layout)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1,
            entities={
                "robot": robot,
                "table": workcell.table_cfg(layout),
                "ground": ground_cfg(-layout["table"]["height"]),
                "room": workcell.room_cfg(layout),
                "door": door_cfg(damping=damping, frictionloss=frictionloss),
            },
            sensors=(
                ContactSensorCfg(
                    name="robot_door_contact",
                    primary=ContactMatch(mode="subtree", pattern="arx5", entity="robot"),
                    secondary=ContactMatch(mode="subtree", pattern="door", entity="door"),
                    fields=("found", "force", "dist"),
                    reduce="netforce",
                    global_frame=True,
                    history_length=Arx5Binding.decimation,
                ),
            ),
        ),
        actions={"arx5_osc": action},
        observations={
            "actor": ObservationGroupCfg(
                terms={
                    "door": ObservationTermCfg(
                        func=door_observation,
                        params={"target_angle": target_angle},
                    )
                },
                enable_corruption=False,
            )
        },
        # This is an asset experiment: default resets only, no bank, no training.
        auto_reset=False,
        decimation=Arx5Binding.decimation,
        episode_length_s=10.0,
        sim=SimulationCfg(
            nconmax=256,
            njmax=1024,
            mujoco=MujocoCfg(
                timestep=Arx5Binding.physics_dt,
                integrator="implicitfast",
                cone="elliptic",
                impratio=5.0,
                iterations=16,
                ls_iterations=20,
            ),
        ),
        viewer=ViewerConfig(
            lookat=(0.35, 0.0, 0.3),
            distance=1.15,
            azimuth=-70.0,
            elevation=-20.0,
            origin_type=ViewerConfig.OriginType.WORLD,
        ),
    )
    Arx5Binding.configure_gripper_friction(cfg.scene, 2.0)
    return cfg
