#!/usr/bin/env bash
# Run on LOCAL Ubuntu. Sync source to your ~/.ssh/config Host 'server'.
# Server-generated outputs are never pushed; pulled outputs are stored under server_outputs/.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LOCAL_DIR="${MM_LOCAL_DIR:-$SCRIPT_DIR}"
SSH_HOST="${MM_SSH_HOST:-server}"
REMOTE_DIR="${MM_REMOTE_DIR:-/home/lqy3090_1/PROJECT/supermimic_manipulation}"
SERVER_OUTPUTS_DIR="${MM_SERVER_OUTPUTS_DIR:-$LOCAL_DIR/server_outputs}"

usage() {
  cat <<'HELP'
Usage (run on LOCAL Ubuntu, from the project directory):
  ./sync_manipulation.sh dry-run                  Preview code sync
  ./sync_manipulation.sh push                     Push source + models (default)
  ./sync_manipulation.sh push-banks               Explicitly upload banks/ (may be large)
  ./sync_manipulation.sh push-data                Explicitly upload data/ (may be large)
  ./sync_manipulation.sh pull logs                Download server logs + checkpoints
  ./sync_manipulation.sh pull reports             Download server evaluation reports
  ./sync_manipulation.sh pull logs/supermimic/box_in_bin
  ./sync_manipulation.sh pull checkpoints         Download released checkpoints (if present)

Pull destination: <local-project>/server_outputs/<requested-remote-path>/
No rsync --delete is used. Neither venv nor generated outputs are pushed.
Environment overrides: MM_LOCAL_DIR, MM_SSH_HOST, MM_REMOTE_DIR, MM_SERVER_OUTPUTS_DIR.
HELP
}

[[ -d "$LOCAL_DIR" ]] || { echo "Local project not found: $LOCAL_DIR" >&2; exit 1; }
command -v rsync >/dev/null 2>&1 || { echo 'Missing rsync; install locally: sudo apt install rsync' >&2; exit 1; }
[[ "$SSH_HOST" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'Unsafe SSH_HOST' >&2; exit 2; }
[[ "$REMOTE_DIR" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Unsafe REMOTE_DIR' >&2; exit 2; }

BASE_OPTS=(-az --partial --info=progress2)
CODE_EXCLUDES=(
  --exclude='.git/'
  --exclude='.venv/' --exclude='venv/' --exclude='.python-venv/'
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pyo'
  --exclude='.pytest_cache/' --exclude='.ruff_cache/' --exclude='.mypy_cache/'
  --exclude='.idea/' --exclude='.vscode/' --exclude='.tox/'
  --exclude='*.egg-info/' --exclude='build/' --exclude='dist/'
  --exclude='data/' --exclude='datasets/' --exclude='banks/' --exclude='reset_datasets/'
  --exclude='output/' --exclude='outputs/' --exclude='results/'
  --exclude='logs/' --exclude='reports/' --exclude='checkpoints/'
  --exclude='runs/' --exclude='wandb/' --exclude='server_outputs/'
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
  --exclude='*.safetensors' --exclude='*.onnx' --exclude='*.npz' --exclude='*.npy'
  --exclude='*.hdf5' --exclude='*.h5' --exclude='*.mp4'
  --exclude='*.mjb' --exclude='*.log' --exclude='*.tmp'
  --include='.env.example'
  --exclude='.env' --exclude='.env.*' --exclude='*.pem' --exclude='id_rsa'
  --exclude='.ssh/' --exclude='.netrc'
)

remote_mkdir() {
  ssh -o BatchMode=yes "$SSH_HOST" "mkdir -p '$REMOTE_DIR'"
}

upload_directory() {
  local dirname="$1"
  [[ -d "$LOCAL_DIR/$dirname" ]] || {
    echo "Local directory not found: $LOCAL_DIR/$dirname" >&2
    exit 1
  }
  remote_mkdir
  ssh -o BatchMode=yes "$SSH_HOST" "mkdir -p '$REMOTE_DIR/$dirname'"
  echo "[push-$dirname] $LOCAL_DIR/$dirname/ -> $SSH_HOST:$REMOTE_DIR/$dirname/"
  rsync "${BASE_OPTS[@]}" -- "$LOCAL_DIR/$dirname/" "$SSH_HOST:$REMOTE_DIR/$dirname/"
}

mode="${1:-push}"
case "$mode" in
  push|dry-run)
    remote_mkdir
    args=("${BASE_OPTS[@]}" "${CODE_EXCLUDES[@]}")
    if [[ "$mode" == 'dry-run' ]]; then
      args+=(--dry-run --itemize-changes)
    fi
    echo "[$mode] $LOCAL_DIR/ -> $SSH_HOST:$REMOTE_DIR/"
    rsync "${args[@]}" -- "$LOCAL_DIR/" "$SSH_HOST:$REMOTE_DIR/"
    ;;
  push-banks)
    upload_directory banks
    ;;
  push-data)
    upload_directory data
    ;;
  pull)
    rel="${2:-}"
    rel="${rel%/}"
    if [[ -z "$rel" || "$rel" == /* || "$rel" =~ (^|/)\.\.(/|$) || \
          ! "$rel" =~ ^(logs|reports|checkpoints|outputs|output|results|runs|wandb)(/[a-zA-Z0-9_.-]+)*$ ]]; then
      echo 'Usage: ./sync_manipulation.sh pull logs|reports|checkpoints|outputs|logs/<run>' >&2
      exit 2
    fi
    if ! ssh -o BatchMode=yes "$SSH_HOST" "test -d '$REMOTE_DIR/$rel'"; then
      echo "Remote directory does not exist: $REMOTE_DIR/$rel" >&2
      exit 1
    fi
    mkdir -p "$SERVER_OUTPUTS_DIR/$rel"
    echo "[pull] $SSH_HOST:$REMOTE_DIR/$rel/ -> $SERVER_OUTPUTS_DIR/$rel/"
    rsync "${BASE_OPTS[@]}" -- "$SSH_HOST:$REMOTE_DIR/$rel/" "$SERVER_OUTPUTS_DIR/$rel/"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
