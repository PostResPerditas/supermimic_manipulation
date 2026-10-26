# SuperMimic

SuperMimic trains a policy by backpropagating the task return through a learned,
differentiable world model instead of using policy-gradient estimates. The implementation
in `src/clab/algorithms/supermimic/` adapts the open-source
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) code (MIT license kept in
`REFERENCE_LICENSE`) to state-based manipulation; it does not claim to reproduce the
original paper's experiments.

The algorithm is task-agnostic. Everything task-specific lives in a task adapter:
`src/clab/tasks/supermimic.py` holds the shared rigid-object adapter, and each task adds
`src/clab/tasks/<task>/supermimic.py`.

## Algorithm

Each outer iteration:

1. **Collect.** The deterministic actor plus Gaussian noise (std 0.1) acts in the real
   simulator; transitions enter a time × world replay that keeps native terminal states.
2. **Fit the world model.** Windows of 8 steps are predicted open-loop and trained with a
   normalized delta loss; steps after a termination are masked. By default the number of
   batches is sized so that each new transition is seen in about 4 windows
   (`--world-model-schedule epochs --world-model-epochs 4`).
3. **Improve the policy.** From replayed states and actor histories, the actor imagines
   16 steps through the frozen world model and critic (parameters frozen, input gradients
   kept) and maximizes the discounted imagined return plus the terminal value.
4. **Fit the critic** to TD(λ) targets of the same imagined rollouts.

The first 1,000 iterations train only the world model. The default budget is 10,000
iterations. Optimizers are Adam (2e-4) with a cosine decay to 3% over
`--learning-rate-decay-iterations` (default 10,000, the full budget); γ = 0.99, λ = 0.95,
gradient-norm clip 1. MLPs run under BF16 autocast with FP32 parameters, optimizer state,
normalizers, integration and rewards; TF32 matmuls are on by default. The world-model loss
and one imagined transition are compiled with `torch.compile`.

## Rollout state

The adapter captures the physical state after each control step (before an automatic
reset) through a read-only hook; it never replaces the simulator or the controller.

| Part | Content |
| --- | --- |
| Physical state (predicted) | Arm and finger joint positions and velocities, object position, rotation (first two matrix columns) and linear/angular velocity |
| Analytic state (not predicted) | Previous action, episode clock and success counter |
| Context (constant per episode) | Base and target poses, gravity, randomized masses, inertias, friction and joint parameters |

A world whose physics diverged is terminated by the task; its transition is stored with the
finite pre-step state, zero reward and an invalid flag, and world-model windows mask it
(`diverged_transitions` in the completion report counts them). A contact sensor that
matches more contacts than its storage only drops the extra matches, so such world-steps
are counted (`Train/contact_match_overflow_world_steps`). A physics capacity overflow drops
contacts or constraints: that world's transitions are excluded like diverged ones until it
resets (`Train/physics_overflow_world_steps`), and the run stops if more than 1% of the
worlds overflow at once.

## World model

The operational-space controller re-anchors at the current end-effector pose on every
action, so it has no internal memory to model: the world model only needs the physical state
of the arm and the object.

**Input** (180 for BoxInBin, 183 for Lemon, 267 for MugHang) is the concatenation of three
parts.

| Part | Dims | Meaning |
| --- | --- | --- |
| `robot_q` | 8 | Joint positions: six arm joints (rad) and two finger joints (m) |
| `robot_dq` | 8 | Joint velocities of the same joints |
| `<object>_position` | 3 | Object position (m) in world axes, relative to the environment origin |
| `<object>_rotation` | 6 | Object orientation as the first two columns of its rotation matrix (continuous, unlike quaternions or Euler angles) |
| `<object>_velocity` | 6 | Object linear (m/s) and angular (rad/s) velocity in world axes |
| Executed command | 7 | What the controller actually executes for the raw action: position offset (3, m) and rotation vector (3, rad) after the controller's clip, scaling and per-step norm bound, plus the commanded finger position (1, m: 0.002 closed, 0.044 open) |
| Context | 142 / 145 / 229 | Constants of the episode, listed below |

| Context field | Dims | Meaning |
| --- | --- | --- |
| `base_position`, `base_rotation` | 3 + 6 | Robot base pose |
| `<target>_position`, `<target>_rotation` | 3 + 6 | Pose of the fixed target (bin, plate or tree) |
| `origin_z` | 1 | Height of the environment origin |
| `gravity` | 3 | Gravity vector |
| `body_mass` | 12 | Randomized masses of the 9 robot bodies, the object, the target and the table |
| `body_inertia` | 36 | Their principal inertias (3 each) |
| `geom_friction` | 48 / 51 / 135 | Sliding, torsional and rolling friction (3 each) of the 16 / 17 / 45 collision geoms of robot, object, target and table (MugHang's mug has 32 collision pieces) |
| `dof_armature`, `dof_damping`, `dof_frictionloss` | 8 each | Joint armature, damping and dry friction of the 8 robot joints |

Feeding the actual randomized physical parameters lets one model cover the domain
randomization of training without having to infer it from history.

**Output** (28) is the normalized increment of the physical state over one 0.05 s control
step: `robot_q` 8, `robot_dq` 8, object position 3, object rotation 3 and object velocity 6.
The rotation increment is a world-axis rotation vector (the SO(3) tangent), `log(R_next
R_prev⁻¹)`, integrated as `exp(δ) R_prev`; all other parts are added. Inputs and outputs are
normalized with running means and standard deviations.

**Analytic state** (9, not predicted) completes the 40-D rollout state: the previous raw
action (7), the episode clock (1) and the success counter (1). They are advanced by rules,
not learned.

| Network | Architecture | Parameters (BoxInBin) |
| --- | --- | --- |
| World model | MLP 5 × 2048, LayerNorm and SiLU, zero-initialized output layer | 17.2 M |
| Actor | MLP 2048-1024-512-256, ELU; observation normalizer; no output squashing | 3.2 M |
| Critic | MLP 256-128, ELU; input = rollout state + context + actor observation (377 / 380 / 479) | 0.13 M |

**Training loss.** Windows of 8 consecutive steps are rolled out open-loop: each prediction
is fed back as the next input. At every step the error between the predicted and the real
state (as a 28-D increment, with the rotation error as a rotation vector) is divided by the
output standard deviation, floored at 1e-3 (1e-2 for joint positions, joint velocities and
object velocity), squared and averaged. Steps after an episode end, and transitions from a
world whose physics diverged, are masked.

The actor observation (195 = 39 × 5 for BoxInBin and Lemon, 210 = 42 × 5 for MugHang) holds
five frames of: object pose in the target frame (6), previous action (7), joint positions
(8), end-effector pose in the base frame (6), object pose in the end-effector frame (6) and
target pose in the end-effector frame (6), poses as position plus axis-angle; MugHang adds
the handle-to-hanging-point vector (3). In imagination these frames are rebuilt from the
predicted state with differentiable forward kinematics and match the native observation to
about 1e-6 on real states. The world model is a reduced surrogate, not an exact MuJoCo state.

## Imagined reward and success

The world model predicts states, not rewards. Imagined rewards use the native reward
weights and are computed analytically from imagined states (reaching from forward
kinematics plus the grasp offset, task guidance from object poses, joint-velocity cost from
predicted joint speeds). Contact, support-force and release tests have no exact analogue in
the reduced state, so each task replaces them with geometric surrogates:

| Task | Imagined success |
| --- | --- |
| `box_in_bin` | Native placement geometry; speeds below 0.03 m/s and 0.5 rad/s (native 0.01 and 0.1); both fingers opened past 0.028 m |
| `lemon` | Native centre XY and height thresholds; height at most `contact_margin` (0.01 m) above the resting height as the contact proxy; both fingers opened past 0.023 m |
| `mug_hang` | Native tree-frame envelope and handle-on-bar geometry; looser speed limit; both fingers opened past 0.015 m |

Common to all tasks:

- the native 0.5 s hold is kept (success counter carried as analytic state);
- the success term's gradient to the actor is scaled by `success_gradient_scale` (0.02),
  keeping its forward value (+10) for the critic;
- geometry, speed and success gates are hard in the forward pass and use sigmoid
  straight-through gradients backward.

Two imagination-only shaping terms cover places where the native reward has no useful
gradient:

- **Release.** Once the object is placed, reaching stops paying and `release_weight` (0.2) ×
  the finger opening fraction from the held width to the release opening is added, so holding
  on is never preferred to letting go.
- **Carry (MugHang).** The native threading term vanishes while the handle is misaligned
  with the bar, which is how every carry starts, so imagination also pulls the handle toward
  the hanging target, `carry_weight` (0.2) × exp(−distance / 1 m).

These settings are recorded in `params/mdp.json` and can be overridden with
`--adapter-option NAME=VALUE` (`stable_linear_speed`, `stable_angular_speed`,
`release_opening`, `release_weight`, `success_gradient_scale`; Lemon also `contact_margin`,
MugHang also `carry_weight`). The task model also accepts `action_bound_weight` (default 0),
a squared imagination cost on raw commands beyond ±`gripper_bound` (gripper) and the
controller's input limit (pose), where the binary gripper's sigmoid and the clip give no
derivative. Real collection and evaluation always use the native reward and success.

## Options and diagnostics

| Option | Effect |
| --- | --- |
| `--iterations N`, `--learning-rate-decay-iterations N` | Budget and cosine-decay length; keep them equal |
| `--world-model-schedule {batches,epochs}` | Fixed `--world-model-batches` per iteration, or `--world-model-epochs` passes over new transitions |
| `--validation-fraction F` | Hold the last worlds out of world-model fitting and log validation losses (default 0.05) |
| `--pin-replay` | Freeze the replay collected before policy updates start and stream new data into a second ring |
| `--exploration-fraction F` | In the first worlds, occasionally flip the gripper command for several steps |
| `--wm-ensemble-size K --uncertainty-penalty P` | Train K independent world models and penalize their disagreement in imagination |
| `--policy-update-interval K` | Update actor/critic every K iterations after the world-model-only phase |
| `--reset-start-fraction F` | Draw a fraction F of imagination starts from episode-start (reset) rows instead of uniformly over replay |
| `--no-compile-world-model`, `--no-compile-policy-update` | Disable `torch.compile` (slower; useful for debugging) |
| `--smoke` | Tiny networks and budgets; checks saving, reloading and one more update |

Key logged values: `Model/wm_step_loss` and `Model/wm_val_step_loss` (world-model
accuracy), `Imagination/imagined_reward`, and `Model/candidate_{surrogate,native,
surrogate_only,native_only}_rate`, which compare the imagined success predicate with the
native one on real states. A high `surrogate_only` rate means the policy can exploit the
surrogate; a high `native_only` rate means the surrogate is too strict.

## Adding a task

1. Subclass `RigidObjectMDP`: set `task`, `object_name`, `target_name`, `held_width`,
   `release_opening`, optional `process_fields` and `actor_widths`, and implement the
   success surrogate, the reward and any extra observations.
2. Subclass `RigidObjectAdapter`: entity names, object/target contact geoms, the actor
   observation terms, the native reward terms it reproduces and the native success term.
3. Point the task binding's `supermimic_adapter` at the adapter class.
4. Check it with `clab train supermimic --smoke`: the completion report lists
   `native_observation_max_error` and the candidate agreement rates.
