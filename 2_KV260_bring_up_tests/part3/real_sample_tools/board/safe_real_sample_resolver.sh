#!/bin/sh
# Resolve the three orders left by probe 00 using the real sample_000 itself.

set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/wangdao_safe_lib.sh"
trap restore_default EXIT

RESOLVER_DIR=${1:-$PACKAGE_DIR/real_sample_resolver}
EXPECTED="$RESOLVER_DIR/sample_000_expected_logits.txt"
SMOKE="$PACKAGE_DIR/wangdao_axi_smoke.sh"

[ -f "$EXPECTED" ] || { echo "missing golden file: $EXPECTED" >&2; exit 1; }
[ -f "$SMOKE" ] || { echo "missing smoke tool: $SMOKE" >&2; exit 1; }

install_and_load_wangdao

matches=0
matched_order=''
for order in 1032 1302 1320; do
    input="$RESOLVER_DIR/sample_000_input_words_order_${order}.txt"
    [ -f "$input" ] || { echo "missing candidate input: $input" >&2; exit 1; }
    echo "===== REAL SAMPLE CANDIDATE order=$order ====="
    if sudo sh "$SMOKE" "$BASE_ADDRESS" "$input" "$EXPECTED"; then
        matches=$((matches + 1))
        matched_order=$order
        echo "REAL_SAMPLE_CANDIDATE_PASS time_to_lane=$order"
    else
        echo "REAL_SAMPLE_CANDIDATE_REJECTED time_to_lane=$order"
    fi
done

if [ "$matches" -ne 1 ]; then
    echo "REAL_SAMPLE_MAPPING_FAIL matches=$matches" >&2
    exit 1
fi

echo "REAL_SAMPLE_MAPPING_PASS time_to_lane=$matched_order inputs=128 logits=16"

