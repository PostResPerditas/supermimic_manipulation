# SuperMimic for robot manipulation

This repository adapts [SuperMimic](https://arxiv.org/abs/2610.28480), policy optimization by
backpropagating the task return through a learned differentiable world model, from humanoid
motion tracking to state-based robot manipulation. Episodes start from
[OmniReset](https://arxiv.org/abs/2603.15789)-style reset distributions. Everything runs in
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) with a simulated ARX5 arm.

中文说明：[README.zh-CN.md](README.zh-CN.md)

<table>
  <tr>
    <th>Lemon: put the lemon on the plate</th>
    <th>BoxInBin: put the box in the bin</th>
    <th>MugHang: hang the mug by its handle</th>
  </tr>
  <tr>
    <td><img src="docs/media/lemon.gif" width="100%"></td>
    <td><img src="docs/media/box_in_bin.gif" width="100%"></td>
    <td><img src="docs/media/mug_hang.gif" width="100%"></td>
  </tr>
</table>

Each clip shows two episodes of the trained policy. The arm starts at its home pose (all joints
at zero) and the object and target come from a held-out test bank. The policy only sees
low-dimensional state; the camera is for viewing.

## Results

Every number below is native task success (the simulator's own success test, held for
0.5 s) from complete episodes with the deterministic actor. No episode was filtered.

**Held-out reset bank.** 256 episodes per task, 64 from each reset category, on a bank
collected with a different seed and split than the training bank. Every task uses its
checkpoint at 4,000 iterations.

| Task | Iterations | Training time | Overall | Per category 0 / 1 / 2 / 3 |
| --- | --- | --- | --- | --- |
| `lemon` | 4,000 | 1.8 h | **100.0%** | 100 / 100 / 100 / 100 |
| `box_in_bin` | 4,000 | 1.9 h | **94.9%** | 97 / 91 / 95 / 97 |
| `mug_hang` | 4,000 | 2.8 h | **96.9%** | 94 / 98 / 95 / 100 |

Training time is wall-clock time from launch to the checkpoint, including the first 1,000
world-model-only iterations, on one GPU (see [Hardware](#hardware)). Runs shared their GPU
with another training job, so the times are upper bounds. The MugHang run was interrupted at
iteration 1,000 and resumed from that checkpoint; its time includes both parts.

Category 0 is the full task (object on the table, gripper free); categories 1-3 start
part-way through it with the object already grasped (see [docs/tasks.md](docs/tasks.md)).

**Data.** A 4,000-iteration run collects 41 M simulated transitions (4,096 parallel worlds,
about 24 days of simulated time at 20 Hz). The policy is trained only through the world
model: no policy-gradient (PPO) update is used.

## Hardware

Every run uses one GPU of a single server:

| Component | Configuration |
| --- | --- |
| GPU | 1 × NVIDIA A100-SXM4-80GB per run (driver 535.230, CUDA 12.2); about 27 GB of GPU memory per run |
| CPU | Intel Xeon Platinum 8336C (112 threads), shared by all jobs on the server |
| Software | PyTorch 2.9.1 (CUDA 12.8 build), mjlab 1.5.3, MuJoCo 3.10, MuJoCo Warp, Warp 1.17 |
| Simulation | 4,096 parallel worlds, plain CUDA graphs (`--cuda-graph-mode plain`, needed with drivers older than CUDA 12.4) |
| Precision | BF16 autocast for the MLPs, FP32 parameters, TF32 matmuls |

One iteration (collection, world-model updates and one actor/critic update) takes about
1.4-2.8 s depending on the task and on GPU sharing.

## Approach

SuperMimic was introduced for humanoid motion tracking, where a dense tracking reward is
available at every step. Manipulation is different: the decisive reward is a sparse success
test that depends on contacts (the object is supported, released and at rest), and tasks are
long compared with the 16 steps the policy can look ahead through the model. The adaptation
has four parts.

1. **Start states that cut the horizon (OmniReset).** Following OmniReset, each task has a
   reset bank of physically validated start states in four categories: the full task (object
   on the table, gripper free), the object grasped on the table, the object grasped in the
   air, and the object grasped near the goal. Training samples the categories uniformly.
   Later categories put the success test within reach of the short imagined rollouts, and
   the value learned there propagates back to the full task. States are proposed
   procedurally (inverse kinematics for grasped states) and kept only if they stay valid
   under the task's own physics and controller.

2. **A world model of the physical state only.** The arm is driven by an operational-space
   controller (OSC) that re-anchors at the current end-effector pose on every action, so it
   has no internal memory. The world model predicts increments of the arm joints and the
   object pose and velocity (28 numbers) from the executed controller command, conditioned on
   the episode's randomized physical parameters (every input and output dimension is listed in
   [docs/supermimic.md](docs/supermimic.md#world-model)). Actor observations in imagination are
   rebuilt from the predicted state with differentiable forward kinematics; on real states
   they match the simulator's observations to about 1e-6.

3. **Analytic rewards and a differentiable success test.** Rewards are not learned. They
   are computed from the imagined state with the task's own weights. Contact-based success
   tests are replaced by geometric surrogates (for example, both fingers opened past a
   threshold instead of "no finger contact"). They are hard in the forward pass and use
   sigmoid straight-through gradients backward, so the critic sees the true +10 success
   value while the actor gets a usable gradient, scaled by 0.02 so that it does not dominate.
   The 0.5 s hold is carried as an analytic counter.

4. **Imagination-only shaping where the native reward has no gradient.** Once the object is
   placed, reaching stops paying and opening the fingers is rewarded, so holding on is never
   preferred to letting go. For MugHang the native threading term vanishes while the handle
   is misaligned with the bar, so imagination adds a long-range pull of the handle toward the
   hanging point. These terms exist only inside imagination; data collection, reported
   rewards and every evaluation use the unmodified native task.

Each iteration collects real transitions with the current actor plus Gaussian noise, fits
the world model on 8-step open-loop windows, imagines 16 steps from replayed states,
maximizes the imagined return plus a TD(λ) critic value by backpropagation through time,
and fits the critic. The first 1,000 iterations train only the world model. Details:
[docs/supermimic.md](docs/supermimic.md).

## Install

Requirements: Linux, an NVIDIA GPU, [uv](https://docs.astral.sh/uv/) and Git LFS.

```bash
git lfs pull
uv sync --locked --extra dev
source .venv/bin/activate        # or prefix every clab command with `uv run`
clab --help
```

`--cuda-graph-mode plain` in the commands below is needed on NVIDIA drivers older than
CUDA 12.4; newer drivers can drop it. Commands never overwrite existing outputs.

## Released checkpoints and reset banks

| Hugging Face repository | Content |
| --- | --- |
| [emopointer/supermimic-manipulation](https://huggingface.co/emopointer/supermimic-manipulation) | The three checkpoints of the results table: `<task>/model_004000.pt` (world model, actor, critic and optimizer states) with the training settings in `<task>/params/` |
| [emopointer/supermimic-manipulation-reset-banks](https://huggingface.co/datasets/emopointer/supermimic-manipulation-reset-banks) | Training banks `<task>_train.pt` (10,000 states per category) and held-out banks `<task>_test.pt` (256 per category) |

`<task>` is `lemon`, `box_in_bin` or `mug_hang`. Download them with the Hugging Face CLI
(run through `uvx`, so nothing is added to the environment):

```bash
uvx --from huggingface_hub hf download emopointer/supermimic-manipulation \
  --local-dir checkpoints
uvx --from huggingface_hub hf download emopointer/supermimic-manipulation-reset-banks \
  --repo-type dataset --local-dir banks
```

## Option A: start from the trained checkpoints

Download both repositories as above, then evaluate a checkpoint on its held-out bank. This
reproduces the results table (256 complete episodes, 64 per reset category):

```bash
TASK=box_in_bin        # lemon | box_in_bin | mug_hang
clab eval policy --task $TASK --algorithm supermimic \
  --checkpoint checkpoints/$TASK/model_004000.pt \
  --reset-bank banks/${TASK}_test.pt --episodes-per-type 64 --num-envs 64 \
  --seeds 100000 --split-label held_out --cuda-graph-mode plain \
  --output reports/${TASK}_heldout.json
```

The report holds the overall and per-category success, episode lengths and returns; keep
`checkpoints/$TASK/params/` next to the checkpoint, since evaluation checks the saved
training environment against the current one.

To keep training a released agent, resume from it. `--iterations` counts the additional
iterations, so 6,000 more reach the default 10,000 in total. The world model, actor, critic,
optimizers and iteration count are restored; the replay buffer is not saved, so it is
refilled first:

```bash
clab train supermimic --task $TASK --reset-bank banks/${TASK}_train.pt \
  --resume checkpoints/$TASK/model_004000.pt --iterations 6000 \
  --output logs/supermimic/${TASK}_resumed --cuda-graph-mode plain
```

## Option B: train from scratch

1. **Reset banks.** Collect a training bank and a held-out bank with a different seed and
   split (or download the released banks as above and skip this step):

   ```bash
   TASK=box_in_bin        # lemon | box_in_bin | mug_hang
   clab resets collect --task $TASK --output banks/${TASK}_train.pt --split train
   clab resets collect --task $TASK --output banks/${TASK}_test.pt \
     --split test --seed 7 --per-type 256
   ```

2. **Train.** The default budget is 10,000 iterations with a checkpoint every 1,000; the
   results table uses `model_004000.pt`. The first 1,000 iterations fit only the world model.

   ```bash
   clab train supermimic --task $TASK --reset-bank banks/${TASK}_train.pt \
     --output logs/supermimic/$TASK --seed 1 --cuda-graph-mode plain
   ```

   Progress is logged to TensorBoard (`tensorboard --logdir logs/supermimic`), including
   the per-category success of the training episodes.

3. **Evaluate** on the held-out bank:

   ```bash
   clab eval policy --task $TASK --algorithm supermimic \
     --checkpoint logs/supermimic/$TASK/model_004000.pt \
     --reset-bank banks/${TASK}_test.pt --episodes-per-type 64 --num-envs 64 \
     --seeds 100000 --split-label held_out --cuda-graph-mode plain \
     --output reports/${TASK}_heldout.json
   ```

A PPO baseline on the same environments is available as `clab train ppo`
(see [docs/workflow.md](docs/workflow.md)).

## Tasks

| Task | Goal | Horizon | Actor / critic inputs | Reset categories |
| --- | --- | --- | --- | --- |
| `lemon` | Put a lemon on a plate and release it | 32 s | 195 / 153 | 4 |
| `box_in_bin` | Put a box in a bin, release it and let it rest | 16 s | 195 / 164 | 4 |
| `mug_hang` | Hang a mug on a tree by its handle | 16 s | 210 / 240 | 4 |

All tasks use the same 7-D action (a relative end-effector pose command plus a binary
gripper command) at 20 Hz, with 480 Hz physics.

## Documentation

| Topic | Document |
| --- | --- |
| Method, task adapters and options | [docs/supermimic.md](docs/supermimic.md) |
| Tasks, success conditions and reset categories | [docs/tasks.md](docs/tasks.md) |
| Reset banks, training and evaluation | [docs/workflow.md](docs/workflow.md) |

## Layout

```text
src/clab/
├── algorithms/supermimic/  # world model, actor/critic, imagination, runner, training entry
├── algorithms/ppo/         # PPO baseline on mjlab / RSL-RL
├── tasks/supermimic.py     # shared SuperMimic adapter: state capture, integration, observations
├── tasks/<task>/           # config, MDP terms, props, resets, collector, SuperMimic adapter
├── robots/arx5/            # robot model and assets, OSC preset, binding
├── controllers/            # OSC action, pose helpers, differentiable kinematics
├── evaluation/             # complete-episode evaluation
└── cli/                    # the clab command
```

## References

This is an independent adaptation; it is not affiliated with the authors of either paper and
does not reproduce their experiments.

- **SuperMimic.** Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel and K. Sreenath.
  *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model.*
  [arXiv:2610.28480](https://arxiv.org/abs/2610.28480), 2026.
  The implementation in `src/clab/algorithms/supermimic/` adapts the open-source
  reimplementation [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
  (MIT; its license is kept in `src/clab/algorithms/supermimic/REFERENCE_LICENSE`).
- **OmniReset.** P. Yin, T. Westenbroek, Z. Zhang, J. Tran, I. Dagnino, E. Shilamkar,
  N. Mbiziwo-Tiapo, S. Bagaria, X. Liu, G. Mullins, A. Kolobov and A. Gupta.
  *Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning.* ICLR 2026.
  [arXiv:2603.15789](https://arxiv.org/abs/2603.15789),
  [project page](https://weirdlabuw.github.io/omnireset/),
  code: [UW-Lab/UWLab](https://github.com/UW-Lab/UWLab).
  The reset categories here follow its reset distributions; the collectors are our own.

```bibtex
@article{liao2026supermimic,
  title   = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author  = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and
             Horowitz, Roberto and Abbeel, Pieter and Sreenath, Koushil},
  journal = {arXiv preprint arXiv:2610.28480},
  year    = {2026}
}

@inproceedings{yin2026emergent,
  title     = {Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning},
  author    = {Yin, Patrick and Westenbroek, Tyler and Zhang, Zhengyu and Tran, Joshua and
               Dagnino, Ignacio and Shilamkar, Eeshani and Mbiziwo-Tiapo, Numfor and
               Bagaria, Simran and Liu, Xinlei and Mullins, Galen and Kolobov, Andrey and
               Gupta, Abhishek},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.15789}
}
```

Task props follow [robosuite](https://github.com/ARISE-Initiative/robosuite) object models;
their license notices are kept next to the assets.

## License

The code is released under the [MIT License](LICENSE). Third-party code and assets keep their
own licenses: the adapted supermimic_vibed code (MIT,
`src/clab/algorithms/supermimic/REFERENCE_LICENSE`), the robosuite-derived props (MIT,
`src/clab/tasks/lemon/assets/ROBOSUITE_LICENSE`), and the ShapeNet mug model used by MugHang,
which is subject to the [ShapeNet terms of use](https://shapenet.org/terms) (non-commercial
research and education).
