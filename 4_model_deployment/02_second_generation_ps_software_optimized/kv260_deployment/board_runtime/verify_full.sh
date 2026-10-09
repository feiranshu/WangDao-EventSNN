#!/bin/sh
# Verify both the PL Proposal-SNN captures and the PS H2 predictions.
# Run on the board after board_full_run.sh.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INPUT_DIR="$ROOT/sequences"
CAPTURE_DIR="$ROOT/captures"
EVENT_DIR="$ROOT/events"
PREDICTION_DIR="$ROOT/predictions"
GOLDEN_DIR="$ROOT/h2_golden"

set -- "$INPUT_DIR"/*.inputs.bin
if [ ! -e "$1" ]; then
    echo "no sequence input files found under $INPUT_DIR" >&2
    exit 1
fi

sequence=0
for input in "$@"; do
    sequence=$((sequence + 1))
    name=$(basename "$input" .inputs.bin)
    s1_golden="$INPUT_DIR/$name.golden.bin"
    event_file="$EVENT_DIR/$name.events.bin"
    capture="$CAPTURE_DIR/$name.capture.bin"
    h2_golden="$GOLDEN_DIR/$name.pred.bin"
    prediction="$PREDICTION_DIR/$name.pred.bin"
    for file in "$s1_golden" "$event_file" "$capture" "$h2_golden" "$prediction"; do
        if [ ! -f "$file" ]; then
            echo "missing verification file: $file" >&2
            exit 1
        fi
    done
    if ! cmp -s "$s1_golden" "$capture"; then
        echo "PL output mismatch for $name" >&2
        exit 1
    fi
    if ! cmp -s "$h2_golden" "$prediction"; then
        echo "PS H2 output mismatch for $name" >&2
        exit 1
    fi
    echo "[$sequence] PASS $name (PL + PS H2)"
done

echo "WANGDAO_FULL_VERIFY_PASS sequences=$sequence"
