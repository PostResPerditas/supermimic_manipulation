# SuperMimic Manipulation — BoxInBin 已验证基线（交给 Codex）

## 1. 项目与当前任务

- 仓库：`https://github.com/emoPointer/supermimic_manipulation`
- 项目名称：**SuperMimic Manipulation**，请勿简称为原版 SuperMimic。
- 服务器工程：`/home/lqy3090_1/PROJECT/supermimic_manipulation`
- GPU：单张 RTX 3090 24GB（`CUDA_VISIBLE_DEVICES=0`）。另一张卡没有用于这次训练。
- `uv` 使用项目已有的 `server_uv.sh`，其安装路径是 `/home/lqy3090_1/PROJECT/_tools/uv/bin/uv`；不要使用系统 `uv` 或更改全局环境。
- 任务：`box_in_bin`，从零训练，不加载专家策略和预训练 World Model。
- 使用官方发布的 `banks/box_in_bin_train.pt`；测试使用 `banks/box_in_bin_test.pt`。

## 2. 最终采用的正式训练命令（历史记录）

> 这是已经成功运行完毕的**原始命令**。相同的 `--output` 路径已有文件；再次运行应设置新目录，避免覆盖。

```bash
cd /home/lqy3090_1/PROJECT/supermimic_manipulation

CUDA_VISIBLE_DEVICES=0 \
bash server_uv.sh uv run --python 3.12 --no-sync \
clab train supermimic \
  --task box_in_bin \
  --reset-bank banks/box_in_bin_train.pt \
  --output logs/supermimic/box_in_bin_official_3072_s1 \
  --seed 1 \
  --num-envs 3072 \
  --iterations 10000 \
  --world-model-only-iterations 1000 \
  --wm-batch-size 2048 \
  --policy-batch-size 4096 \
  --buffer-steps 1000 \
  --warmup-steps 64 \
  --learning-rate-decay-iterations 10000 \
  --save-interval 1000 \
  --log-interval 50 \
  --compile-world-model \
  --compile-policy-update \
  --cuda-graph-mode plain
```

`clab train supermimic` 是本仓库 CLI 的真实命令；讨论时称整个项目为 SuperMimic Manipulation。`--world-model-only-iterations 1000` 表示前 1000 次只更新 WM，后续仍持续更新 WM，并同时训练策略；**不是 WM 只训练 1000 次**。

## 3. 训练结果与保留的检查点

- 训练：10,000 iterations，76,996,608 transitions，142,480 次 World Model 更新，11,382.70 秒；无物理发散和容量溢出。
- 最佳 held-out checkpoint：**`model_005000.pt`**（第 5,000 次）。
- 评估：256 episodes（Type 0–3 各 64），seed 100000，held-out bank。
- 最佳：Overall **89.45% = 229/256**；Type 0 **87.50%**；Type 1 **87.50%**；Type 2 **90.62%**；Type 3 **92.19%**。
- 原作者提供的 checkpoint 在同一服务器评估：Overall **91.41% = 234/256**。
- 本地最终 10,000 次 checkpoint 的 Overall **58.59%**，说明后期存在性能退化。该数值可保留在文档和 TensorBoard 中，无需为了清盘继续存储每个 checkpoint。
- 完成报告中 `candidate_surrogate_only_rate=0.319136` 为**累计候选状态差异比例**，不是完成回合的成功率；不能直接当作 Reward Hacking 定论。

## 4. 最小保留集（清理前核实）

**必须保留：**

- `src/`、`scripts/`、相关资产及项目依赖文件、`uv.lock`、`pyproject.toml`。
- `server_uv.sh`、`sync_manipulation.sh`、`.venv/`（不打算重装依赖时）。
- `banks/box_in_bin_train.pt`、`banks/box_in_bin_test.pt`。
- `checkpoints/box_in_bin/`（作者权重及配置，用作对照）。
- `checkpoints/local_best/model_005000.pt`（本地最佳自训权重；清理脚本会先从训练输出复制）。
- `logs/supermimic/box_in_bin_official_3072_s1/params/`、`completion.json`、`tensorboard/`（复现实验元数据和训练曲线）。
- `reports/box_in_bin_official_3072_s1_005000_heldout.json` 及配套 `.env.yaml`；`reports/box_in_bin_heldout.json`（若存在，官方模型基线结果）。

**可按需保留：** `model_010000.pt`（若后续需要直接可视化最终退化策略）；其它阶段 checkpoint 可删除，因为已有相应指标记录。`banks/` 中其他任务的数据及 `checkpoints/` 中其他任务的官方权重可以留待之后做新任务，不在清理脚本里删除。

## 5. 建议的最佳 checkpoint 评估命令

```bash
cd /home/lqy3090_1/PROJECT/supermimic_manipulation
CUDA_VISIBLE_DEVICES=0 \
bash server_uv.sh uv run --python 3.12 --no-sync \
clab eval policy \
  --task box_in_bin \
  --algorithm supermimic \
  --checkpoint checkpoints/local_best/model_005000.pt \
  --reset-bank banks/box_in_bin_test.pt \
  --episodes-per-type 64 \
  --num-envs 64 \
  --seeds 100000 \
  --split-label held_out \
  --cuda-graph-mode plain \
  --output reports/box_in_bin_best_recheck.json
```

注意 `--output` 对应的 JSON 或 `.env.yaml` 已存在时，评估会拒绝覆盖，需换一个未使用过的文件名。

## 6. 研发方向约束

基础复现已完成，不优先再优化 BoxInBin 的 GPU 并行数；后续优先将任务适配器扩展到关节物体操作，保持训练主干，逐步引入机器人新构型。保留对 World Model 想象奖励与原生物理成功判据偏差的后续分析可能性。
