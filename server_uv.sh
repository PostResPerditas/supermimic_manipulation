#!/usr/bin/env bash
# Run on SERVER; isolated uv + Python/cache paths under /home/lqy3090_1/PROJECT/_tools/uv.
# Place this file in /home/lqy3090_1/PROJECT/supermimic_manipulation/server_uv.sh.
# Shared uv binaries/caches with SuperMimic-Reproduced; project .venv is separate.

MM_SERVER_BASE="${MM_SERVER_BASE:-/home/lqy3090_1/PROJECT}"
MM_PROJECT_DIR="${MM_PROJECT_DIR:-$MM_SERVER_BASE/supermimic_manipulation}"
MM_UV_ROOT="${MM_UV_ROOT:-$MM_SERVER_BASE/_tools/uv}"

mm_activate_uv() {
  export MM_SERVER_BASE MM_PROJECT_DIR MM_UV_ROOT
  export UV_INSTALL_DIR="$MM_UV_ROOT/bin"
  export UV_CACHE_DIR="$MM_UV_ROOT/cache"
  export UV_PYTHON_INSTALL_DIR="$MM_UV_ROOT/python"
  export UV_PYTHON_BIN_DIR="$MM_UV_ROOT/python-bin"
  export UV_TOOL_DIR="$MM_UV_ROOT/tools"
  export UV_TOOL_BIN_DIR="$MM_UV_ROOT/tool-bin"
  export UV_PYTHON_PREFERENCE="only-managed"
  # Upstream .python-version pins 3.13; use 3.12 explicitly for this project.
  export UV_PYTHON="3.12"
  export UV_PROJECT_ENVIRONMENT="$MM_PROJECT_DIR/.venv"

  export XDG_CACHE_HOME="$MM_UV_ROOT/xdg-cache"
  export XDG_DATA_HOME="$MM_UV_ROOT/xdg-data"
  export XDG_CONFIG_HOME="$MM_UV_ROOT/xdg-config"
  export HF_HOME="$MM_UV_ROOT/huggingface"
  export TORCH_HOME="$MM_UV_ROOT/torch"
  export PIP_CACHE_DIR="$MM_UV_ROOT/pip-cache"
  export TMPDIR="$MM_UV_ROOT/tmp"
  export MPLCONFIGDIR="$MM_UV_ROOT/matplotlib"
  export TRITON_CACHE_DIR="$MM_UV_ROOT/triton-cache"
  export CUDA_CACHE_PATH="$MM_UV_ROOT/cuda-cache"
  export PATH="$MM_UV_ROOT/bin:$MM_UV_ROOT/tool-bin:$PATH"

  mkdir -p \
    "$MM_UV_ROOT/bin" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
    "$UV_PYTHON_BIN_DIR" "$UV_TOOL_DIR" "$UV_TOOL_BIN_DIR" \
    "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" \
    "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
    "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" || return 1

  # Optional SSH reverse tunnel from 'server-tunnel' / 'server-proxy'.
  # Compatibility: the old SM_USE_TUNNEL_PROXY variable is accepted, too.
  if [[ "${MM_USE_TUNNEL_PROXY:-${SM_USE_TUNNEL_PROXY:-0}}" == "1" ]]; then
    export http_proxy="http://127.0.0.1:17897"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
    export no_proxy="localhost,127.0.0.1,::1"
    export NO_PROXY="$no_proxy"
    unset ALL_PROXY all_proxy
  fi
}

mm_uv_status() {
  echo "Project:     $MM_PROJECT_DIR"
  echo "uv binary:   $MM_UV_ROOT/bin/uv"
  if [[ -x "$MM_UV_ROOT/bin/uv" ]]; then
    "$MM_UV_ROOT/bin/uv" --version
    echo "uv cache:    $("$MM_UV_ROOT/bin/uv" cache dir)"
    echo "Python dir:  $("$MM_UV_ROOT/bin/uv" python dir)"
    "$MM_UV_ROOT/bin/uv" python find 3.12 2>/dev/null || true
  else
    echo 'uv not installed. Run: bash server_uv.sh install'
  fi
  echo "venv:        $UV_PROJECT_ENVIRONMENT"
  echo "proxy:       ${https_proxy:-${HTTPS_PROXY:-(not set)}}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  fi
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  mm_activate_uv
  return $?
fi

set -euo pipefail
mode="${1:-help}"
if (( $# > 0 )); then shift; fi

case "$mode" in
  install)
    mm_activate_uv
    if [[ ! -x "$MM_UV_ROOT/bin/uv" ]]; then
      command -v curl >/dev/null 2>&1 || { echo 'Missing curl' >&2; exit 1; }
      curl -LsSf https://astral.sh/uv/install.sh | env \
        UV_INSTALL_DIR="$MM_UV_ROOT/bin" UV_NO_MODIFY_PATH=1 sh
    fi
    "$MM_UV_ROOT/bin/uv" python install 3.12
    mm_uv_status
    ;;
  uv)
    mm_activate_uv
    [[ -x "$MM_UV_ROOT/bin/uv" ]] || {
      echo "uv not found at $MM_UV_ROOT/bin/uv. Run: bash server_uv.sh install" >&2
      exit 1
    }
    [[ -d "$MM_PROJECT_DIR" ]] || { echo "Project not found: $MM_PROJECT_DIR" >&2; exit 1; }
    cd "$MM_PROJECT_DIR"
    exec env -u PYTHONPATH "$MM_UV_ROOT/bin/uv" "$@"
    ;;
  status)
    mm_activate_uv
    mm_uv_status
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage (run on SERVER):
  bash server_uv.sh status
  bash server_uv.sh install                       # reuse/install custom uv + managed Python 3.12
  bash server_uv.sh uv sync --python 3.12 --locked --extra dev
  bash server_uv.sh uv run --python 3.12 --no-sync clab --help
  source server_uv.sh                            # optional, exports paths in current shell

Optional proxy via SSH reverse tunnel:
  MM_USE_TUNNEL_PROXY=1 bash server_uv.sh uv sync --python 3.12 --locked --extra dev
HELP
    ;;
  *)
    echo "Unknown action: $mode" >&2
    exit 2
    ;;
esac
