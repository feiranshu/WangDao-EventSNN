#!/bin/sh
# Default formal KV260 entry point: measure idle/active SOM power around one full run.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNTIME_DIR="$ROOT/runtime"
MEASURER="$ROOT/measure_kv260_power.py"
BASE_ADDRESS="${1:-0xA0000000}"
APP_NAME="${2:-wangdao_pl_flat}"

command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }
[ -f "$MEASURER" ] || { echo "missing power tool: $MEASURER" >&2; exit 1; }

set -- "$ROOT/sequences"/*.inputs.bin
[ -e "$1" ] || { echo "no sequence input files" >&2; exit 1; }
task_count=$#
patch_count=0
for input in "$@"; do
    bytes=$(wc -c < "$input")
    [ $((bytes % 512)) -eq 0 ] || { echo "invalid input size: $input" >&2; exit 1; }
    patch_count=$((patch_count + bytes / 512))
done

mkdir -p "$RUNTIME_DIR"
set -- python3 "$MEASURER" \
    --sample-rate-hz "${WANGDAO_POWER_SAMPLE_RATE_HZ:-10}" \
    --idle-seconds "${WANGDAO_POWER_IDLE_SECONDS:-10}"
if [ -n "${WANGDAO_POWER_NODE:-}" ]; then
    set -- "$@" --power-node "$WANGDAO_POWER_NODE"
fi
set -- "$@" \
    --task-count "$task_count" \
    --patch-count "$patch_count" \
    --csv "$RUNTIME_DIR/power_samples.csv" \
    --summary "$RUNTIME_DIR/power_summary.json" \
    -- sh "$ROOT/board_full_run.sh" "$BASE_ADDRESS" "$APP_NAME"
exec "$@"
