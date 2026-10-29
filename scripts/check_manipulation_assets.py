#!/usr/bin/env python3
"""CPU / mjlab-Warp asset contact experiment and interactive trajectory viewer.

Examples (server: prefix with bash server_uv.sh uv run --python 3.12 --no-sync):
  python scripts/check_manipulation_assets.py --backend cpu --output reports/door_cpu
  python scripts/check_manipulation_assets.py --backend both --output reports/door
  python scripts/check_manipulation_assets.py --backend cpu --view viser
  python scripts/check_manipulation_assets.py --task box_in_bin --view viser

Only the robot is initialized by IK. During dynamics, actions go through OSC;
the Door has no actuator and its joint state is never written after reset.
The browser shows recorded physics with a time slider, playback and overlays.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

if "--image" in sys.argv or "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mjlab.scene import Scene  # noqa: E402

from clab.controllers.osc import operational_space_torque  # noqa: E402
from clab.controllers.pose import pose_target_action  # noqa: E402
from clab.robots.arx5.description import ARM_JOINTS, HOME, JOINTS  # noqa: E402
from clab.robots.kinematics import SiteIK  # noqa: E402


def make_config(args):
    if args.task == "door_push":
        from clab.tasks.door_push.config import make_env_cfg

        cfg = make_env_cfg(
            target_angle=args.target_angle, damping=args.damping, frictionloss=args.frictionloss
        )
        # Solve a reproducible pre-contact pose, not a trained policy/reset bank.
        scene = Scene(cfg.scene, device="cpu")
        model = scene.compile()
        ik = SiteIK(model, tuple(f"robot/{j}" for j in ARM_JOINTS), "robot/grasp_point")
        arm = ik.solve(
            np.array((0.34, -0.10, 0.32)),
            np.array((1.0, 0.0, 0.0, 0.0)),
            np.array(HOME[:6]),
            iterations=300,
        )
        if arm is None:
            raise RuntimeError("Cannot reach Door pre-contact pose")
        cfg.scene.entities["robot"].init_state.joint_pos = dict(
            zip(JOINTS, (*arm, 0.002, 0.002), strict=True)
        )
        if args.no_robot_contact:
            previous = cfg.scene.spec_fn

            def exclude_contact(spec):
                if previous is not None:
                    previous(spec)
                for body in ("arx5", *(f"link{i}" for i in range(1, 9))):
                    for door_body in ("frame", "door", "latch"):
                        spec.add_exclude(bodyname1=f"robot/{body}", bodyname2=f"door/{door_body}")

            cfg.scene.spec_fn = exclude_contact
    else:
        from clab.tasks.box_in_bin.config import make_env_cfg

        cfg = make_env_cfg(play=True)
        # Use the exact existing scene/OSC, with deterministic default poses and
        # no reset bank or domain randomization for the asset-only comparison.
        cfg.events = {"reset_default": cfg.events["reset_default"]}
        cfg.terminations = {}
        cfg.auto_reset = False
    cfg.scene.num_envs = 1
    cfg.seed = 1
    # View from the open side of the workcell, inside the visual side wall.
    cfg.viewer.azimuth = -70.0
    cfg.viewer.distance = 1.15
    return cfg


def compile_cpu(cfg):
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    return model, data


class CpuOsc:
    """CPU reference for the existing OSC: same processing, gains and torque code."""

    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.ee = model.body("robot/link6").id
        self.dofs = np.array([model.joint(f"robot/{j}").dofadr[0] for j in ARM_JOINTS])
        self.motors = [model.actuator(f"robot/{j}_torque").id for j in ARM_JOINTS]
        self.fingers = [model.actuator(f"robot/joint{i}_position").id for i in (7, 8)]
        self.jp, self.jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        self.mass = np.zeros((model.nv, model.nv))
        self.kp = np.array(cfg.motion_stiffness)
        self.kd = 2 * np.sqrt(self.kp) * np.array(cfg.motion_damping_ratio)

    def process(self, data, action, dt):
        from mjlab.utils.lab_api.math import apply_delta_pose

        from clab.controllers.osc import limit_vector_norm

        a = torch.as_tensor(action, dtype=torch.float32).reshape(1, 7)
        delta = a[:, :6].clone().clamp(-self.cfg.input_limit, self.cfg.input_limit)
        delta[:, :3] *= self.cfg.position_scale
        delta[:, 3:] *= self.cfg.orientation_scale
        delta[:, :3] = limit_vector_norm(delta[:, :3], self.cfg.linear_velocity_limit * dt)
        delta[:, 3:] = limit_vector_norm(delta[:, 3:], self.cfg.angular_velocity_limit * dt)
        pos, quat = apply_delta_pose(
            torch.tensor(data.xpos[self.ee], dtype=torch.float32)[None],
            torch.tensor(data.xquat[self.ee], dtype=torch.float32)[None],
            delta,
        )
        self.pos, self.quat = pos, quat
        self.grip = self.cfg.closed_position if action[6] < 0 else self.cfg.open_position

    def apply(self, data):
        from mjlab.utils.lab_api.math import compute_pose_error

        pos_error, rot_error = compute_pose_error(
            torch.tensor(data.xpos[self.ee], dtype=torch.float32)[None],
            torch.tensor(data.xquat[self.ee], dtype=torch.float32)[None],
            self.pos,
            self.quat,
        )
        mujoco.mj_jac(self.model, data, self.jp, self.jr, data.xipos[self.ee], self.ee)
        jac = np.vstack((self.jp, self.jr))[:, self.dofs]
        mujoco.mj_fullM(self.model, data, self.mass)
        acceleration = self.kp * torch.cat((pos_error, rot_error), -1).numpy()[0]
        acceleration -= self.kd * (jac @ data.qvel[self.dofs])
        torque = operational_space_torque(
            torch.tensor(jac, dtype=torch.float32)[None],
            torch.tensor(self.mass[np.ix_(self.dofs, self.dofs)], dtype=torch.float32)[None],
            torch.tensor(acceleration, dtype=torch.float32)[None],
            solver=self.cfg.inertia_solver,
            rcond=self.cfg.inertia_rcond,
            uncouple_pos_ori=self.cfg.uncouple_pos_ori,
        ).numpy()[0]
        data.ctrl[self.motors] = np.clip(torque, -self.cfg.effort_limit, self.cfg.effort_limit)
        data.ctrl[self.fingers] = self.grip


def scripted_action(data, model, action_cfg, initial_pos, initial_quat, t, args):
    target = initial_pos.copy()
    if args.task == "door_push" and args.motion == "push":
        # 1 s settle, 4 s straight push, then hold. No door-state feedback.
        target[0] += 0.23 * np.clip((t - 1.0) / 4.0, 0.0, 1.0)
    ee = model.body("robot/link6").id
    action = pose_target_action(
        torch.tensor(data.xpos[ee], dtype=torch.float32)[None],
        torch.tensor(data.xquat[ee], dtype=torch.float32)[None],
        torch.tensor(target, dtype=torch.float32)[None],
        torch.tensor(initial_quat, dtype=torch.float32)[None],
        action_cfg=action_cfg,
        gripper_command=-1.0 if args.task == "door_push" else 1.0,
    )
    return action.numpy()[0]


def sample(model, data, t, args, frame_start, overflow=0):
    contacts = []
    for c in data.contact:
        names = [model.geom(int(g)).name or "unnamed" for g in c.geom]
        robot_door = any(n.startswith("robot/") for n in names) and any(
            n.startswith("door/") for n in names
        )
        if robot_door:
            contacts.append(" | ".join(names))
    ee = data.body("robot/link6").xpos
    row = dict(
        time_s=float(t),
        ee_x=float(ee[0]),
        ee_y=float(ee[1]),
        ee_z=float(ee[2]),
        robot_door_contact=bool(contacts),
        contact_pairs="; ".join(sorted(set(contacts))),
        ncon=int(data.ncon),
        min_contact_distance_m=min((float(c.dist) for c in data.contact), default=0.0),
        overflow=int(overflow),
    )
    if args.task == "door_push":
        j = model.joint("door/hinge")
        row.update(
            angle_rad=float(data.qpos[j.qposadr[0]]),
            angular_velocity_rad_s=float(data.qvel[j.dofadr[0]]),
            target_reached=bool(data.qpos[j.qposadr[0]] >= args.target_angle),
            frame_displacement_m=float(np.linalg.norm(data.body("door/frame").xpos - frame_start)),
        )
    row["finite"] = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    row["max_abs_joint_speed"] = float(np.max(np.abs(data.qvel)))
    violation = 0.0
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            q = data.qpos[model.jnt_qposadr[j]]
            violation = max(violation, model.jnt_range[j, 0] - q, q - model.jnt_range[j, 1])
    row["max_joint_limit_violation"] = float(violation)
    row["warnings"] = "; ".join(
        f"{mujoco.mjtWarning(i).name}:{int(w.number)}"
        for i, w in enumerate(data.warning)
        if w.number
    )
    return row


def snapshot(data):
    return (
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
        data.mocap_pos.copy(),
        data.mocap_quat.copy(),
    )


def restore(model, data, state):
    for name, values in zip(
        ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"), state, strict=True
    ):
        getattr(data, name)[:] = values
    mujoco.mj_forward(model, data)


def run(cfg, backend, args):
    env = None
    try:
        if backend == "cpu":
            model, data = compile_cpu(cfg)
            osc = CpuOsc(model, cfg.actions["arx5_osc"])
        else:
            from mjlab.envs import ManagerBasedRlEnv

            from clab.simulation import configure_cuda_graph

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Warp validation needs the server CUDA GPU; use --backend cpu locally"
                )
            env = ManagerBasedRlEnv(cfg, device=args.device)
            configure_cuda_graph(env, "plain")
            env.reset(seed=1)
            model, data = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
            import mujoco_warp as mjwarp

            mjwarp.get_data_into(data, model, env.sim.wp_data)
        if args.task == "door_push":
            assert model.nv == 9 and model.nu == 8, "Door must have one passive hinge"
            hinge = model.joint("door/hinge")
            np.testing.assert_allclose(hinge.range, (0, 0.4))
            np.testing.assert_allclose(model.jnt_axis[hinge.id], (0, 0, 1))
            assert not np.any(model.actuator_trnid[:, 0] == hinge.id), "Door must be passive"
            for name in ("frame", "door", "latch"):
                inertia = model.body_inertia[model.body(f"door/{name}").id]
                assert np.all(inertia > 0) and 2 * inertia.max() <= inertia.sum()
        ee = model.body("robot/link6").id
        initial_pos, initial_quat = data.xpos[ee].copy(), data.xquat[ee].copy()
        frame_start = data.body("door/frame").xpos.copy() if args.task == "door_push" else None
        dt = cfg.decimation * cfg.sim.mujoco.timestep
        rows = [sample(model, data, 0, args, frame_start)]
        states = [snapshot(data)]
        for i in range(round(args.seconds / dt)):
            action = scripted_action(
                data, model, cfg.actions["arx5_osc"], initial_pos, initial_quat, i * dt, args
            )
            subrows = []
            if backend == "cpu":
                osc.process(data, action, dt)
                for sub in range(cfg.decimation):
                    osc.apply(data)
                    mujoco.mj_step(model, data)
                    # Match mjlab's substep timing: derived arrays remain one
                    # physics substep old until the final control-step forward.
                    subrows.append(
                        sample(
                            model, data, (i + (sub + 1) / cfg.decimation) * dt, args, frame_start
                        )
                    )
                mujoco.mj_forward(model, data)
                row = sample(model, data, (i + 1) * dt, args, frame_start)
                row["robot_door_contact"] = any(r["robot_door_contact"] for r in subrows)
                row["contact_pairs"] = "; ".join(
                    sorted({r["contact_pairs"] for r in subrows if r["contact_pairs"]})
                )
                row["min_contact_distance_m"] = min(r["min_contact_distance_m"] for r in subrows)
            else:
                env.step(torch.tensor(action, device=args.device)[None])
                mjwarp.get_data_into(data, model, env.sim.wp_data)
                row = sample(
                    model,
                    data,
                    (i + 1) * dt,
                    args,
                    frame_start,
                    int(env.sim.data.overflow[0].item()),
                )
                if args.task == "door_push":
                    # Force/distance history covers all 24 physics substeps;
                    # named pairs above describe final-substep Warp contacts.
                    sensor = env.scene["robot_door_contact"].data
                    row["robot_door_contact"] |= bool(sensor.found.any().item()) or bool(
                        (sensor.force_history.abs() > 1e-5).any().item()
                    )
                    row["min_contact_distance_m"] = min(
                        row["min_contact_distance_m"], float(sensor.dist_history.min().item())
                    )
            rows.append(row)
            states.append(snapshot(data))
            if not row["finite"] or row["warnings"] or row["overflow"]:
                break
        result = summarize(model, rows, backend, args)
        print(json.dumps(result, indent=2), flush=True)
        if args.output:
            prefix = Path(f"{args.output}_{backend}")
            prefix.parent.mkdir(parents=True, exist_ok=True)
            prefix.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
            with prefix.with_suffix(".csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if args.image or args.video:
            render(model, data, states, cfg, args, backend)
        return result, model, states, rows
    finally:
        if env is not None:
            env.close()


def summarize(model, rows, backend, args):
    result = dict(
        task=args.task,
        backend=backend,
        mujoco_version=mujoco.__version__,
        motion=args.motion,
        robot_contact_enabled=not args.no_robot_contact,
        seconds_completed=rows[-1]["time_s"],
        initial_contacts=rows[0]["ncon"],
        all_finite=all(r["finite"] for r in rows),
        warnings=sorted({r["warnings"] for r in rows if r["warnings"]}),
        overflow=max(r["overflow"] for r in rows),
        min_contact_distance_m=min(r["min_contact_distance_m"] for r in rows),
        max_joint_limit_violation=max(r["max_joint_limit_violation"] for r in rows),
        max_abs_joint_speed=max(r["max_abs_joint_speed"] for r in rows),
    )
    checks = dict(
        finite=result["all_finite"],
        no_warnings=not result["warnings"],
        no_overflow=result["overflow"] == 0,
        no_deep_penetration=result["min_contact_distance_m"] > -0.005,
        joint_limits=result["max_joint_limit_violation"] < 0.02,
        completed=abs(result["seconds_completed"] - args.seconds) < 0.051,
    )
    if args.task == "door_push":
        hinge = model.joint("door/hinge")
        dof = hinge.dofadr[0]
        result.update(
            target_angle_rad=args.target_angle,
            angle_initial_rad=rows[0]["angle_rad"],
            angle_final_rad=rows[-1]["angle_rad"],
            angle_peak_rad=max(r["angle_rad"] for r in rows),
            angular_speed_peak_rad_s=max(abs(r["angular_velocity_rad_s"]) for r in rows),
            contact_steps=sum(r["robot_door_contact"] for r in rows),
            frame_displacement_m=max(r["frame_displacement_m"] for r in rows),
            hinge_damping=float(model.dof_damping[dof]),
            hinge_frictionloss=float(model.dof_frictionloss[dof]),
            hinge_range_rad=hinge.range.tolist(),
            door_body_mass_kg=float(model.body_mass[model.body("door/door").id]),
            authored_inertials={
                name: {
                    "mass_kg": float(model.body_mass[model.body(f"door/{name}").id]),
                    "principal_inertia_kg_m2": model.body_inertia[
                        model.body(f"door/{name}").id
                    ].tolist(),
                }
                for name in ("frame", "door", "latch")
            },
            collision_geoms={
                name: {
                    "type": mujoco.mjtGeom(
                        int(model.geom_type[model.geom(f"door/{name}").id])
                    ).name,
                    "size_m": model.geom_size[model.geom(f"door/{name}").id].tolist(),
                    "friction": model.geom_friction[model.geom(f"door/{name}").id].tolist(),
                    "condim": int(model.geom_condim[model.geom(f"door/{name}").id]),
                }
                for name in ("panel", "r_frame", "l_frame", "handle_base", "handle")
            },
        )
        checks["frame_fixed"] = result["frame_displacement_m"] < 1e-7
        if args.motion == "push" and not args.no_robot_contact:
            checks["physical_push"] = (
                result["contact_steps"] > 0 and result["angle_peak_rad"] >= args.target_angle
            )
        else:
            checks["passive_door_stays_closed"] = max(abs(r["angle_rad"]) for r in rows) < 1e-4
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result


def render(model, data, states, cfg, args, backend):
    import imageio.v2 as imageio

    camera = mujoco.MjvCamera()
    camera.lookat[:] = cfg.viewer.lookat
    camera.distance = cfg.viewer.distance
    camera.azimuth = cfg.viewer.azimuth
    camera.elevation = cfg.viewer.elevation
    option = mujoco.MjvOption()
    option.geomgroup[3] = int(args.collisions)
    # Keep software rendering reasonably small as well as GPU rendering.
    model.vis.quality.shadowsize = 1024
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    with mujoco.Renderer(model, height=640, width=960) as renderer:

        def frame(state):
            restore(model, data, state)
            renderer.update_scene(data, camera=camera, scene_option=option)
            return renderer.render().copy()

        if args.image:
            path = Path(f"{args.output}_{backend}.png")
            imageio.imwrite(path, np.concatenate((frame(states[0]), frame(states[-1])), axis=1))
            print(f"Initial / final image: {path}", flush=True)
        if args.video:
            path = Path(f"{args.output}_{backend}.mp4")
            with imageio.get_writer(path, fps=20) as writer:
                for state in states:
                    writer.append_data(frame(state))
            print(f"Physics video: {path}", flush=True)


def view(model, states, rows, cfg, args):
    data = mujoco.MjData(model)
    restore(model, data, states[0])
    if args.view == "native":
        from mujoco.viewer import launch_passive

        with launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = cfg.viewer.lookat
            viewer.cam.distance = cfg.viewer.distance
            viewer.cam.azimuth = cfg.viewer.azimuth
            viewer.cam.elevation = cfg.viewer.elevation
            i = 0
            while viewer.is_running():
                restore(model, data, states[i])
                viewer.sync()
                time.sleep(0.05)
                i = (i + 1) % len(states)
    else:
        import viser
        from mjviser import ViserMujocoScene

        server = viser.ViserServer(host="127.0.0.1", port=args.port)
        scene = ViserMujocoScene(server, model, num_envs=1)
        scene.camera_tracking_enabled = False
        scene.create_visualization_gui(
            camera_distance=cfg.viewer.distance,
            camera_azimuth=cfg.viewer.azimuth,
            camera_elevation=abs(cfg.viewer.elevation),
        )

        @server.on_client_connect
        def set_camera(client):
            az = np.deg2rad(cfg.viewer.azimuth)
            el = np.deg2rad(abs(cfg.viewer.elevation))
            centre = np.asarray(cfg.viewer.lookat)
            client.camera.look_at = centre
            client.camera.position = centre + cfg.viewer.distance * np.array(
                (-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el))
            )

        server.gui.add_markdown(
            "Recorded physics: orbit the view, scrub time, inspect contacts / collision hulls."
        )
        play = server.gui.add_checkbox("Play", initial_value=False)
        index = server.gui.add_slider(
            "Frame (20 Hz)", min=0, max=len(states) - 1, step=1, initial_value=0
        )
        status = server.gui.add_markdown("")
        print(
            f"Replay: http://localhost:{args.port} (SSH: ssh -L {args.port}:127.0.0.1:{args.port} server)",
            flush=True,
        )
        try:
            while True:
                i = int(index.value)
                restore(model, data, states[i])
                scene.update_from_mjdata(data)
                r = rows[i]
                status.content = f"t={r['time_s']:.2f}s; door={r.get('angle_rad', 0):.4f} rad; contact={r['robot_door_contact']}"
                time.sleep(0.05)
                if play.value:
                    index.value = (i + 1) % len(states)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--task", choices=("door_push", "box_in_bin"), default="door_push")
    p.add_argument("--backend", choices=("cpu", "warp", "both"), default="cpu")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target-angle", type=float, default=0.30)
    p.add_argument("--damping", type=float, default=0.1)
    p.add_argument("--frictionloss", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=7.0)
    p.add_argument("--motion", choices=("push", "hold"), default="push")
    p.add_argument("--no-robot-contact", action="store_true", help="Door causal contact control")
    p.add_argument("--output", type=Path, help="Prefix for CSV/JSON and requested images/video")
    p.add_argument("--view", choices=("none", "native", "viser"), default="none")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--image", action="store_true", help="Side-by-side initial/final PNG")
    p.add_argument("--video", action="store_true")
    p.add_argument("--collisions", action="store_true", help="Render ARX5 collision geoms too")
    args = p.parse_args()
    if args.task == "box_in_bin":
        args.motion = "hold"
    if args.seconds < 0.05 or not np.isfinite(args.seconds):
        p.error("--seconds must be finite and >= 0.05")
    if (args.image or args.video) and args.output is None:
        p.error("--image / --video require --output")
    if args.task != "door_push" and args.no_robot_contact:
        p.error("--no-robot-contact applies only to Door")
    if args.image or args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    cfg = make_config(args)
    backends = ("cpu", "warp") if args.backend == "both" else (args.backend,)
    results = [run(cfg, b, args) for b in backends]
    passed = all(r[0]["passed"] for r in results)
    if len(results) == 2 and args.task == "door_push":
        a, b = results
        n = min(len(a[3]), len(b[3]))
        error = np.array([r["angle_rad"] for r in a[3][:n]]) - np.array(
            [r["angle_rad"] for r in b[3][:n]]
        )
        comparison = dict(
            cpu_warp_angle_rmse_rad=float(np.sqrt(np.mean(error**2))),
            final_angle_difference_rad=abs(a[0]["angle_final_rad"] - b[0]["angle_final_rad"]),
            peak_angle_difference_rad=abs(a[0]["angle_peak_rad"] - b[0]["angle_peak_rad"]),
        )
        comparison["trend_consistent"] = (
            passed
            and comparison["cpu_warp_angle_rmse_rad"] < 0.08
            and comparison["final_angle_difference_rad"] < 0.08
        )
        passed &= comparison["trend_consistent"]
        print(json.dumps(comparison, indent=2), flush=True)
        if args.output:
            Path(f"{args.output}_comparison.json").write_text(
                json.dumps(comparison, indent=2) + "\n"
            )
    if args.view != "none":
        _, model, states, rows = results[-1]
        view(model, states, rows, cfg, args)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
