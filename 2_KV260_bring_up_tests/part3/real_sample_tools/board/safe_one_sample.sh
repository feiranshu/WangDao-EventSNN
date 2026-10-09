#!/bin/sh
# Final exact comparison for sample_000 (or another 128-word/16-logit sample).

set -eu
if [ "$#" -ne 2 ]; then
    echo 'usage: safe_one_sample.sh <input-words> <expected-logits>' >&2
    exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/wangdao_safe_lib.sh"
trap restore_default EXIT

INPUT_WORDS=$1
EXPECTED_LOGITS=$2
[ -f "$INPUT_WORDS" ] || { echo "input not found: $INPUT_WORDS" >&2; exit 1; }
[ -f "$EXPECTED_LOGITS" ] || { echo "golden not found: $EXPECTED_LOGITS" >&2; exit 1; }
[ -f "$PACKAGE_DIR/wangdao_axi_smoke.sh" ] || {
    echo "missing $PACKAGE_DIR/wangdao_axi_smoke.sh" >&2
    exit 1
}

install_and_load_wangdao
sudo sh "$PACKAGE_DIR/wangdao_axi_smoke.sh" "$BASE_ADDRESS" \
    "$INPUT_WORDS" "$EXPECTED_LOGITS"

echo "REAL_SAMPLE_PASS input=$INPUT_WORDS expected=$EXPECTED_LOGITS"
