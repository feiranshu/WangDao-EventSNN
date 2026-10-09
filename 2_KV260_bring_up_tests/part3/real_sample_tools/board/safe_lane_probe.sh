#!/bin/sh
# Load the safe app, run one discriminating word vector, and identify byte order.

set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/wangdao_safe_lib.sh"
trap restore_default EXIT

PROBE_DIR=${1:-$PACKAGE_DIR/lane_probe}
[ -f "$PROBE_DIR/lane_probe_00_words.txt" ] || {
    echo "missing lane probe: $PROBE_DIR/lane_probe_00_words.txt" >&2
    exit 1
}

install_and_load_wangdao

probe_count=0
for input_words in "$PROBE_DIR"/lane_probe_*_words.txt; do
    [ -f "$input_words" ] || continue
    name=${input_words##*/}
    stem=${name%_words.txt}
    actual="$PROBE_DIR/${stem}_actual.txt"
    sudo sh "$SCRIPT_DIR/wangdao_axi_capture.sh" "$BASE_ADDRESS" \
        "$input_words" "$actual"
    probe_count=$((probe_count + 1))
done
[ "$probe_count" -gt 0 ] || { echo 'no probe inputs found' >&2; exit 1; }

matches=0
matched_order=''
for candidate in "$PROBE_DIR"/lane_probe_00_expected_order_*.txt; do
    [ -f "$candidate" ] || continue
    name=${candidate##*/}
    order=${name#lane_probe_00_expected_order_}
    order=${order%.txt}
    all_equal=1
    for input_words in "$PROBE_DIR"/lane_probe_*_words.txt; do
        name=${input_words##*/}
        stem=${name%_words.txt}
        if ! cmp -s "$PROBE_DIR/${stem}_actual.txt" \
            "$PROBE_DIR/${stem}_expected_order_${order}.txt"; then
            all_equal=0
            break
        fi
    done
    if [ "$all_equal" -eq 1 ]; then
        matches=$((matches + 1))
        matched_order=$order
    fi
done

if [ "$matches" -ne 1 ]; then
    echo "LANE_PROBE_FAIL matches=$matches probes=$probe_count" >&2
    echo 'Copy all lane_probe_*_actual.txt files back to the PC for diagnosis.' >&2
    exit 1
fi

echo "LANE_PROBE_PASS time_to_lane=$matched_order probes=$probe_count"
