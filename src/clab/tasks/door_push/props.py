"""Load the unmodified robosuite unlocked Door through MjSpec."""

from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

DOOR_XML = Path(__file__).resolve().parents[4] / "third_party/robosuite/assets/objects/door.xml"


def door_spec(*, damping: float = 0.1, frictionloss: float = 0.0) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(DOOR_XML))
    # robosuite's enclosing model uses radians; standalone MJCF defaults to degrees.
    spec.compiler.degree = False
    spec.modelname = "door_push"
    spec.bodies[1].name = "root"
    hinge = spec.joint("hinge")
    # Door._load_model -> DoorObject(friction=0.0, damping=0.1, lock=False).
    # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/door.py
    hinge.damping[0] = damping
    hinge.frictionloss = frictionloss
    # Keep the source 0..0.4 rad range, explicit body inertials, panel, cylinders
    # and handle. All source geoms already collide; no duplicate collision shell.
    # VFS bytes survive Entity's mocap wrapper and the subsequent Scene attachment.
    for texture in spec.textures:
        path = (DOOR_XML.parent / texture.file).resolve()
        texture.file = path.name
        spec.assets[path.name] = path.read_bytes()
    return spec


def door_cfg(*, damping: float = 0.1, frictionloss: float = 0.0) -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: door_spec(damping=damping, frictionloss=frictionloss),
        init_state=EntityCfg.InitialStateCfg(
            # The original frame has an additional -pi/2 yaw and 0.22 m offset.
            # Here the closed panel is at x=0.42, centre y~0, hinge y~0.255.
            # Its bottom clears the tabletop by 0.02 m; +X pushes open (+hinge).
            pos=(0.42, -0.08, 0.31),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={"hinge": 0.0},
        ),
    )
