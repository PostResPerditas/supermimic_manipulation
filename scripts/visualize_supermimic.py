#!/usr/bin/env python3
"""Visualize an emoPointer/supermimic_manipulation checkpoint.

Uses the project's own evaluation environment, reset bank and inference loader.
Designed for mjlab 1.5.3 as pinned by the repository (not upstream `play`).
"""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="box_in_bin", choices=("box_in_bin", "lemon", "mug_hang"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--reset-bank", type=Path, default=None)
    parser.add_argument("--mode", choices=("viser", "native", "video"), default="viser")
    parser.add_argument("--reset-type", type=int, choices=range(4), default=0)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-graph-mode", choices=("auto", "plain"), default="plain")
    parser.add_argument("--video-dir", type=Path, default=Path("reports/visualizations"))
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "video":
        # Must be set before importing MuJoCo/mjlab.
        os.environ.setdefault("MUJOCO_GL", "egl")

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    from clab.algorithms.inference import load_policy
    from clab.contracts import checkpoint_environment
    from clab.tasks.registry import load_task

    torch.set_grad_enabled(False)
    task = load_task(args.task)
    checkpoint = (args.checkpoint or Path(f"checkpoints/{args.task}/model_004000.pt")).resolve(strict=True)
    bank = (args.reset_bank or Path(f"banks/{args.task}_test.pt")).resolve(strict=True)

    # Match all environment settings against the checkpoint, like `clab eval policy`.
    saved, training_env_config = checkpoint_environment(checkpoint, None)
    cfg = task.make_eval_cfg(saved, bank)
    cfg.observations["actor"].enable_corruption = False
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.width = args.video_width
    cfg.viewer.height = args.video_height

    video_path = args.video_dir / f"{args.task}_type{args.reset_type}_seed{args.seed}-step-0.mp4"
    if args.mode == "video" and video_path.exists():
        raise FileExistsError(f"Refusing to replace video: {video_path}")

    env = ManagerBasedRlEnv(
        cfg=cfg,
        device=args.device,
        render_mode="rgb_array" if args.mode == "video" else None,
    )
    try:
        task.robot.validate_env(env)
        reset_term = task.reset_term(env)
        reset_term.type_override = args.reset_type
        policy = load_policy(
            env, checkpoint, algorithm="supermimic", task=task,
            cuda_graph_mode=args.cuda_graph_mode,
            training_env_config=training_env_config,
        )
        print(f"Task={args.task}, reset_type={args.reset_type}, mode={args.mode}, seed={args.seed}")
        if args.mode == "viser":
            env.reset(seed=args.seed)
            print("Open http://localhost:8080 in your LOCAL browser (use SSH -L tunnel).")
            ViserPlayViewer(env, policy, frame_rate=30).run()
        elif args.mode == "native":
            env.reset(seed=args.seed)
            NativeMujocoViewer(env, policy, frame_rate=30).run()
        else:
            from mjlab.utils.wrappers import VideoRecorder
            # Record only the first complete episode, from the chosen reset category.
            recorder = VideoRecorder(
                env,
                video_folder=args.video_dir,
                episode_trigger=lambda episode_number: episode_number == 0,
                name_prefix=f"{args.task}_type{args.reset_type}_seed{args.seed}",
                video_length=None,
            )
            try:
                observations, _ = recorder.reset(seed=args.seed)
                with torch.inference_mode():
                    for _ in range(env.max_episode_length):
                        action = policy(observations)
                        observations, reward, terminated, truncated, extras = recorder.step(action)
                        if bool((terminated | truncated)[0].item()):
                            break
            finally:
                recorder.close()
            if video_path.is_file():
                print(f"Saved video: {video_path}")
            else:
                print(f"Video file not found at expected path: {video_path}")
    finally:
        # The video wrapper closes the underlying env itself; double-close is avoided.
        if args.mode != "video":
            env.close()


if __name__ == "__main__":
    main()
