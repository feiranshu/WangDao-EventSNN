#!/bin/sh
# Run the complete PS+PL Proposal-SNN + H2 pipeline on the KV260 board.
# Usage: sh board_full_run.sh [base-address] [application-name]
set -eu

BASE_ADDRESS="${1:-0xA0000000}"
APP_NAME="${2:-wangdao_pl_flat}"
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INPUT_DIR="$ROOT/sequences"
EVENT_DIR="$ROOT/events"
CAPTURE_DIR="$ROOT/captures"
PREDICTION_DIR="$ROOT/predictions"
RUNTIME_DIR="$ROOT/runtime"
AXI_DRIVER="/tmp/wangdao_axi_batch.$$"
H2_DRIVER="/tmp/wangdao_ps_h2.$$"

for file in "$ROOT/wangdao_axi_batch.c" "$ROOT/wangdao_ps_h2.c" "$ROOT/h2_weights.bin" "$ROOT/ps_manifest.json"; do
    if [ ! -f "$file" ]; then
        echo "missing package file: $file" >&2
        exit 1
    fi
done

set -- "$INPUT_DIR"/*.inputs.bin
if [ ! -e "$1" ]; then
    echo "no sequence input files found under $INPUT_DIR" >&2
    exit 1
fi

command -v xmutil >/dev/null 2>&1 || { echo "xmutil is required" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required for monotonic full-pipeline timing" >&2; exit 1; }
if command -v cc >/dev/null 2>&1; then
    CC=cc
elif command -v gcc >/dev/null 2>&1; then
    CC=gcc
else
    echo "cc or gcc is required on the board" >&2
    exit 1
fi

sudo -v
if ! sudo xmutil listapps | grep -E "^[[:space:]]*$APP_NAME[[:space:]].*->[[:digit:]]+" >/dev/null; then
    echo "TARGET_APPLICATION_NOT_ACTIVE: $APP_NAME" >&2
    echo "Activate the prepared wangdao_pl_flat application, then rerun this script." >&2
    exit 31
fi

trap 'rm -f "$AXI_DRIVER" "$H2_DRIVER"' EXIT
"$CC" -O2 -std=c11 -Wall -Wextra -o "$AXI_DRIVER" "$ROOT/wangdao_axi_batch.c"
"$CC" -O2 -std=c11 -Wall -Wextra -o "$H2_DRIVER" "$ROOT/wangdao_ps_h2.c" -lm
mkdir -p "$CAPTURE_DIR" "$PREDICTION_DIR" "$RUNTIME_DIR"

sequence=0
for input in "$@"; do
    sequence=$((sequence + 1))
    name=$(basename "$input" .inputs.bin)
    event_file="$EVENT_DIR/$name.events.bin"
    capture="$CAPTURE_DIR/$name.capture.bin"
    prediction="$PREDICTION_DIR/$name.pred.bin"
    pl_runtime="$RUNTIME_DIR/$name.pl.runtime.json"
    h2_runtime="$RUNTIME_DIR/$name.h2.runtime.json"
    full_runtime="$RUNTIME_DIR/$name.full.runtime.json"
    if [ ! -f "$event_file" ]; then
        echo "missing event file: $event_file" >&2
        exit 1
    fi
    sequence_started=$(python3 -c 'import time; print(time.monotonic_ns())')
    echo "[$sequence] PL Proposal-SNN: $name"
    sudo "$AXI_DRIVER" "$BASE_ADDRESS" "$input" "$capture" "$pl_runtime"
    sudo chown "$(id -u):$(id -g)" "$capture"
    sudo chown "$(id -u):$(id -g)" "$pl_runtime"
    echo "[$sequence] PS H2 Refiner: $name"
    "$H2_DRIVER" "$event_file" "$capture" "$ROOT/h2_weights.bin" "$prediction" "$h2_runtime"
    sequence_finished=$(python3 -c 'import time; print(time.monotonic_ns())')
    sequence_elapsed=$((sequence_finished - sequence_started))
    printf '{\n  "schema_version": 1,\n  "status": "ok",\n  "sequence": "%s",\n  "clock": "Python time.monotonic_ns",\n  "timing_units": "nanoseconds",\n  "full_model_total_ns": %s,\n  "includes": ["PL driver process", "capture fsync", "PS H2 process", "prediction fsync"],\n  "excludes": ["driver compilation", "SSH/SCP", "power sampling"]\n}\n' \
        "$name" "$sequence_elapsed" > "$full_runtime"
done

sync

echo "WANGDAO_FULL_PS_PL_PASS sequences=$sequence"
