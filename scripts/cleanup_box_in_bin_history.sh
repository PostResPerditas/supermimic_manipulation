#!/usr/bin/env bash
# Run only on the training server, from this project's scripts/ folder.
# Dry run by default. Explicit --apply + confirmation is required for deletion.
set -euo pipefail
shopt -s nullglob

MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--apply" ]]; then
  echo "Usage: bash scripts/cleanup_box_in_bin_history.sh [--dry-run|--apply]" >&2
  exit 2
fi

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXPECTED="/home/lqy3090_1/PROJECT/supermimic_manipulation"
if [[ "$PROJECT" != "$EXPECTED" ]]; then
  echo "Refusing: this cleanup is only for server project: $EXPECTED" >&2
  echo "Detected: $PROJECT" >&2
  exit 2
fi
cd "$PROJECT"

RUN="logs/supermimic/box_in_bin_official_3072_s1"
SOURCE="$RUN/model_005000.pt"
TARGET="checkpoints/local_best/model_005000.pt"

if [[ ! -f "$SOURCE" && ! -f "$TARGET" ]]; then
  echo "ERROR: best checkpoint absent at both $SOURCE and $TARGET; nothing deleted" >&2
  exit 1
fi
if [[ -f "$SOURCE" && -f "$TARGET" ]] && ! cmp -s -- "$SOURCE" "$TARGET"; then
  echo "ERROR: source and archived best checkpoints differ; inspect before cleaning" >&2
  exit 1
fi

# Strictly enumerated categories of expendable historical experiments.
CANDIDATES=(
  logs/supermimic/box_in_bin_probe_*
  logs/supermimic/box_in_bin_compile_*
  logs/supermimic/box_in_bin_formal_4096
  logs/supermimic/box_in_bin_official_cap_*
  logs/supermimic/box_in_bin_smoke_*
  logs/monitor/box_in_bin_probe_*
  logs/monitor/box_in_bin_compile_*
  logs/monitor/box_in_bin_formal_4096
  logs/monitor/box_in_bin_official_cap_*
  reports/box_in_bin_probe_*
  reports/box_in_bin_formal_*
  reports/box_in_bin_train_bank_*
)

# Evaluate intermediate checkpoints from the retained 10k-iteration run;
# preserve the best (5000) report and its env YAML.
for p in reports/box_in_bin_official_3072_s1_*_heldout.json \
         reports/box_in_bin_official_3072_s1_*_heldout.env.yaml; do
  case "$p" in
    reports/box_in_bin_official_3072_s1_005000_heldout.json|\
    reports/box_in_bin_official_3072_s1_005000_heldout.env.yaml) ;;
    *) CANDIDATES+=("$p") ;;
  esac
done

# All run checkpoints are redundant once best checkpoint is safely copied.
for p in "$RUN"/model_*.pt; do
  CANDIDATES+=("$p")
done

EXISTING=()
for p in "${CANDIDATES[@]}"; do
  [[ -e "$p" || -L "$p" ]] && EXISTING+=("$p")
done

printf '\nProject: %s\n' "$PROJECT"
printf 'Mode: %s\n' "$MODE"
printf 'Best checkpoint: %s\n' "$TARGET"
if [[ ! -f "$TARGET" ]]; then
  echo "Best checkpoint must first be copied from: $SOURCE"
fi
printf '\nPotential deletions (%s paths):\n' "${#EXISTING[@]}"
for p in "${EXISTING[@]}"; do
  du -sh -- "$p" 2>/dev/null || printf '  %s\n' "$p"
done
printf '\nPreserved: banks/, checkpoints/box_in_bin/, %s, %s/{params,tensorboard,completion.json}, best 5000 evaluation report\n' "$TARGET" "$RUN"

if [[ "$MODE" == "--dry-run" ]]; then
  echo "\nPREVIEW ONLY. To apply: bash scripts/cleanup_box_in_bin_history.sh --apply"
  exit 0
fi

read -r -p 'Type DELETE EXPERIMENT RUNS to confirm: ' CONFIRM
if [[ "$CONFIRM" != 'DELETE EXPERIMENT RUNS' ]]; then
  echo 'Cancelled; no files deleted.'
  exit 1
fi
mkdir -p -- "$(dirname -- "$TARGET")"
if [[ ! -f "$TARGET" ]]; then
  cp -- "$SOURCE" "$TARGET"
fi
cmp -s -- "$SOURCE" "$TARGET" 2>/dev/null || {
  if [[ -f "$SOURCE" ]]; then
    echo 'ERROR: cannot verify backed-up best checkpoint; stop.' >&2
    exit 1
  fi
}

for p in "${EXISTING[@]}"; do
  echo "Deleting: $p"
  rm -rf -- "$p"
done

echo 'Cleanup complete.'
du -sh logs reports checkpoints banks 2>/dev/null || true
