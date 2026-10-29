#!/usr/bin/env bash
# Run on the SERVER from any directory after copying into project/scripts/.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NAME="box_in_bin_probe_2048_100"
RUN_DIR="logs/supermimic/${NAME}"
MON_DIR="logs/monitor/${NAME}"
if [[ -e "$RUN_DIR" || -e "$MON_DIR" ]]; then
  echo "Refusing to overwrite previous experiment: $RUN_DIR or $MON_DIR" >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }
mkdir -p "$MON_DIR"
git rev-parse HEAD > "$MON_DIR/git_commit.txt" 2>/dev/null || true
printf '%s\n' 'date,memory_used_mib,memory_total_mib,gpu_util_percent,memory_util_percent,power_watt' > "$MON_DIR/gpu.csv"
nvidia-smi -i 0 \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader,nounits -l 1 >> "$MON_DIR/gpu.csv" 2> "$MON_DIR/nvidia_smi.err" &
MON_PID=$!
cleanup() {
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
}
trap cleanup EXIT
printf 'Experiment: %s\nMonitoring: %s/gpu.csv\n' "$RUN_DIR" "$MON_DIR"
# Do not resume: a clean from-scratch comparison to the earlier 1024-world run.
set +e
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
bash server_uv.sh uv run --python 3.12 --no-sync \
clab train supermimic \
  --task box_in_bin \
  --reset-bank banks/box_in_bin_train.pt \
  --output "$RUN_DIR" \
  --seed 1 \
  --num-envs 2048 \
  --iterations 100 \
  --world-model-only-iterations 40 \
  --wm-batch-size 512 \
  --policy-batch-size 512 \
  --buffer-steps 512 \
  --warmup-steps 64 \
  --save-interval 50 \
  --log-interval 10 \
  --no-compile-world-model \
  --no-compile-policy-update \
  --cuda-graph-mode plain 2>&1 | tee "$MON_DIR/train.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$MON_DIR/exit_code.txt"
printf 'Exit code: %s\n' "$STATUS"
exit "$STATUS"
