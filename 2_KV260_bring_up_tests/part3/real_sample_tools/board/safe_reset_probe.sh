#!/bin/sh
# Prove that every START begins from a clean SNN state: full -> zero -> full.

set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/wangdao_safe_lib.sh"
trap restore_default EXIT

install_and_load_wangdao

SMOKE="$PACKAGE_DIR/wangdao_axi_smoke.sh"
for required in "$SMOKE" "$PACKAGE_DIR/patch0_words.txt" \
    "$PACKAGE_DIR/patch0_expected.txt" "$PACKAGE_DIR/patch1_words.txt" \
    "$PACKAGE_DIR/patch1_expected.txt"; do
    [ -f "$required" ] || { echo "missing reset-probe file: $required" >&2; exit 1; }
done

echo '===== RESET PROBE: FULL ====='
sudo sh "$SMOKE" "$BASE_ADDRESS" \
    "$PACKAGE_DIR/patch1_words.txt" "$PACKAGE_DIR/patch1_expected.txt"
echo '===== RESET PROBE: ZERO ====='
sudo sh "$SMOKE" "$BASE_ADDRESS" \
    "$PACKAGE_DIR/patch0_words.txt" "$PACKAGE_DIR/patch0_expected.txt"
echo '===== RESET PROBE: FULL AGAIN ====='
sudo sh "$SMOKE" "$BASE_ADDRESS" \
    "$PACKAGE_DIR/patch1_words.txt" "$PACKAGE_DIR/patch1_expected.txt"

echo 'RESET_PROBE_PASS sequence=patch1,patch0,patch1'
