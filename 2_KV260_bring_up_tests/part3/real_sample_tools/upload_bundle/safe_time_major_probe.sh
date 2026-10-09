#!/bin/sh
# Identify the exact four-block/time and byte/feature layout.

set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/wangdao_safe_lib.sh"
trap restore_default EXIT

PROBE_DIR=${1:-$PACKAGE_DIR/time_major_probe}
[ -f "$PROBE_DIR/time_major_probe_00_words.txt" ] || {
    echo "missing time-major probe: $PROBE_DIR/time_major_probe_00_words.txt" >&2
    exit 1
}

install_and_load_wangdao

probe_count=0
for input_words in "$PROBE_DIR"/time_major_probe_*_words.txt; do
    [ -f "$input_words" ] || continue
    name=${input_words##*/}
    stem=${name%_words.txt}
    actual="$PROBE_DIR/${stem}_actual.txt"
    sudo sh "$SCRIPT_DIR/wangdao_axi_capture.sh" "$BASE_ADDRESS" \
        "$input_words" "$actual"
    probe_count=$((probe_count + 1))
done
[ "$probe_count" -eq 2 ] || {
    echo "expected 2 time-major probes, found $probe_count" >&2
    exit 1
}

matches=0
matched_layout=''
for candidate in "$PROBE_DIR"/time_major_probe_00_expected_*.txt; do
    [ -f "$candidate" ] || continue
    name=${candidate##*/}
    layout=${name#time_major_probe_00_expected_}
    layout=${layout%.txt}
    all_equal=1
    for input_words in "$PROBE_DIR"/time_major_probe_*_words.txt; do
        name=${input_words##*/}
        stem=${name%_words.txt}
        if ! cmp -s "$PROBE_DIR/${stem}_actual.txt" \
            "$PROBE_DIR/${stem}_expected_${layout}.txt"; then
            all_equal=0
            break
        fi
    done
    if [ "$all_equal" -eq 1 ]; then
        matches=$((matches + 1))
        matched_layout=$layout
    fi
done

if [ "$matches" -ne 1 ]; then
    echo "TIME_MAJOR_PROBE_FAIL matches=$matches probes=$probe_count" >&2
    exit 1
fi

block_for_time=${matched_layout#b}
block_for_time=${block_for_time%%_l*}
lane_for_feature_offset=${matched_layout##*_l}
echo "TIME_MAJOR_PROBE_PASS block_for_time=$block_for_time lane_for_feature_offset=$lane_for_feature_offset probes=$probe_count"

