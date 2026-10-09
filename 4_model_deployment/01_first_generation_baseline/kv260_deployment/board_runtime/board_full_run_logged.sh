#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BASE=${1:-0xA0000000}
APP=${2:-wangdao_pl_flat}
LABEL=${3:-train_dev10}
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/${LABEL}_$(date -u +%Y%m%dT%H%M%SZ).log"

if sh "$ROOT/run_measured_full.sh" "$BASE" "$APP" > "$LOG" 2>&1 && sh "$ROOT/verify_full.sh" >> "$LOG" 2>&1; then
    echo WANGDAO_TRAIN_DEV10_DEPLOY_PASS >> "$LOG"
    echo WANGDAO_TRAIN_DEV10_BIT_EXACT_PASS >> "$LOG"
else
    status=$?
    cat "$LOG"
    exit "$status"
fi
cat "$LOG"
sh "$ROOT/package_board_results.sh" "$LABEL"
