#!/bin/sh
# Install the board-validated package, run two smoke vectors, then restore default.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ARTIFACT_DIR=${WANGDAO_ARTIFACT_DIR:-"$ROOT/../firmware/../firmware/../firmware/../firmware/../firmware/../firmware/../firmware/../firmware/artifacts"}
DEFAULT_APP=${WANGDAO_DEFAULT_APP:-k26-starter-kits}
CUSTOM_APP=${WANGDAO_CUSTOM_APP:-wangdao_pl_flat}
BASE=${WANGDAO_BASE_ADDRESS:-0xA0000000}

is_active() { xmutil listapps | awk -v app="$1" '$1==app && $2=="0" && $3=="0" {found=1} END {exit !found}'; }
restore_default() {
    set +e
    is_active "$CUSTOM_APP" && sudo xmutil unloadapp
    if ! is_active "$DEFAULT_APP"; then sudo xmutil loadapp "$DEFAULT_APP" >/dev/null 2>&1 || true; fi
}
trap restore_default EXIT HUP INT TERM

sh "$ROOT/artifact_preflight.sh" "$ARTIFACT_DIR"
sudo mkdir -p "/lib/firmware/xilinx/$CUSTOM_APP"
sudo cp "$ARTIFACT_DIR/wangdao_pl_flat.bit.bin" "/lib/firmware/xilinx/$CUSTOM_APP/"
sudo cp "$ARTIFACT_DIR/wangdao_pl_flat.dtbo" "/lib/firmware/xilinx/$CUSTOM_APP/"
sudo cp "$ARTIFACT_DIR/shell.json" "/lib/firmware/xilinx/$CUSTOM_APP/"

active=$(xmutil listapps | awk '$2=="0" && $3=="0" {print $1; exit}')
case "$active" in
    "") ;;
    "$DEFAULT_APP"|"$CUSTOM_APP") sudo xmutil unloadapp ;;
    *) echo "refusing to unload unexpected active app: $active" >&2; exit 1 ;;
esac
sudo xmutil loadapp "$CUSTOM_APP" >/dev/null 2>&1 || true
is_active "$CUSTOM_APP" || { echo "custom app did not become active" >&2; exit 1; }
[ "$(cat /sys/class/fpga_manager/fpga0/state)" = operating ] || { echo "FPGA is not operating" >&2; exit 1; }

if [ "${WANGDAO_SKIP_SMOKE:-0}" != 1 ]; then
    sudo sh "$ROOT/wangdao_axi_smoke.sh" "$BASE" "$ROOT/../smoke_tests/patch0_words.txt" "$ROOT/../smoke_tests/patch0_expected.txt"
    sudo sh "$ROOT/wangdao_axi_smoke.sh" "$BASE" "$ROOT/../smoke_tests/patch1_words.txt" "$ROOT/../smoke_tests/patch1_expected.txt"
fi
echo WANGDAO_DEPLOY_OK
