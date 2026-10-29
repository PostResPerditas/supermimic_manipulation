SuperMimic Manipulation: local editing, remote training
======================================================

LOCAL (Ubuntu):
  cd /mnt/ssd/Bennkyou/PROJECT
  git clone https://github.com/emoPointer/supermimic_manipulation.git
  cd supermimic_manipulation
  git lfs install
  git lfs pull
  # Copy sync_manipulation.sh and server_uv.sh into this project directory.
  chmod +x sync_manipulation.sh server_uv.sh
  ./sync_manipulation.sh dry-run
  ./sync_manipulation.sh push

SERVER:
  ssh server
  cd /home/lqy3090_1/PROJECT/supermimic_manipulation
  bash server_uv.sh status
  # Only if uv (or its Python 3.12) is not yet installed:
  bash server_uv.sh install
  bash server_uv.sh uv sync --python 3.12 --locked --extra dev
  bash server_uv.sh uv run --python 3.12 --no-sync clab --help
  # Model/resource files are synced from the local Git LFS checkout.

SERVER download reset banks (optional):
  bash server_uv.sh uv tool run --from huggingface_hub hf download \
    emopointer/supermimic-manipulation-reset-banks \
    --repo-type dataset --local-dir banks

SERVER smoke example (requires downloaded banks):
  CUDA_VISIBLE_DEVICES=0 bash server_uv.sh uv run --python 3.12 --no-sync \
    clab train supermimic --task box_in_bin \
    --reset-bank banks/box_in_bin_train.pt \
    --output logs/supermimic/box_in_bin_smoke \
    --num-envs 32 --smoke --cuda-graph-mode plain

LOCAL retrieve:
  ./sync_manipulation.sh pull reports
  ./sync_manipulation.sh pull logs
  # Receives into server_outputs/reports and server_outputs/logs,
  # not local reports or logs. No --delete is ever used.

NOTE:
  Project .python-version defaults to 3.13. All server uv commands here
  explicitly select Python 3.12, supported by pyproject.toml.
  When running commands directly without server_uv.sh, the custom path/cache
  environment is not automatically applied. Prefer the wrapper on server.
