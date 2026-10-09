#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEFAULT_APP='k26-starter-kits'
CUSTOM_APP='wangdao_pl_flat'

is_active() {
    sudo xmutil listapps | awk -v app="$1" \
        '$1 == app && $0 ~ /0->0/ { found=1 } END { exit !found }'
}

restore_default() {
    echo
    echo '===== RESTORE DEFAULT PL ====='

    set +e
    if is_active "$CUSTOM_APP"; then
        sudo xmutil unloadapp || true
    elif is_active "$DEFAULT_APP"; then
        :
    else
        active=$(sudo xmutil listapps | awk '$0 ~ /0->0/ { print $1; exit }')
        if [ -n "$active" ]; then
            echo "WARNING: refusing to unload unexpected active app: $active" >&2
            return
        fi
    fi
    if ! is_active "$DEFAULT_APP"; then
        sudo xmutil loadapp "$DEFAULT_APP" || true
    fi

    sleep 2

    echo
    echo '===== DEFAULT APP AFTER RESTORE ====='
    sudo xmutil listapps || true

    echo
    echo '===== FPGA STATE AFTER RESTORE ====='
    cat /sys/class/fpga_manager/fpga0/state || true

    echo
    echo '===== TEMPERATURE AFTER RESTORE ====='
    find /sys/class/thermal -maxdepth 2 -type f -name temp \
      -print -exec cat {} \; 2>/dev/null || true
}

trap restore_default EXIT
trap 'exit 130' HUP INT TERM

echo '===== INSTALL CUSTOM APP ====='

sh "$SCRIPT_DIR/artifact_preflight.sh" "$SCRIPT_DIR"

sudo install -d -m 0755 \
    "/lib/firmware/xilinx/$CUSTOM_APP"

# 确保目录里只有一个 bitstream
sudo rm -f \
    "/lib/firmware/xilinx/$CUSTOM_APP/wangdao.bit.bin"

sudo install -m 0644 \
    "$SCRIPT_DIR/wangdao_pl_flat.bit.bin" \
    "$SCRIPT_DIR/wangdao_pl_flat.dtbo" \
    "$SCRIPT_DIR/shell.json" \
    "/lib/firmware/xilinx/$CUSTOM_APP/"

echo
echo '===== PACKAGE CONTENT ====='
sudo find "/lib/firmware/xilinx/$CUSTOM_APP" \
    -maxdepth 1 -type f \
    -printf '%f  %s bytes\n' | sort

echo
echo '===== BEFORE SWITCH ====='
sudo xmutil listapps

echo
echo '===== UNLOAD DEFAULT PL ====='
APP_STATUS="$(sudo xmutil listapps)"
ACTIVE_APP="$(printf '%s\n' "$APP_STATUS" | awk '$0 ~ /0->0/ { print $1; exit }')"
case "$ACTIVE_APP" in
    '') ;;
    "$DEFAULT_APP"|"$CUSTOM_APP") sudo xmutil unloadapp ;;
    *)
        echo "ERROR: refusing to unload unexpected active app: $ACTIVE_APP" >&2
        exit 1
        ;;
esac

echo
echo '===== LOAD WANGDAO PL ====='
sudo xmutil loadapp "$CUSTOM_APP"

sleep 2

echo
echo '===== CUSTOM APP STATUS ====='
APP_STATUS="$(sudo xmutil listapps)"
echo "$APP_STATUS"

echo "$APP_STATUS" | grep "$CUSTOM_APP" | grep -q '0->0' || {
    echo 'ERROR: wangdao_pl_flat did not acquire slot 0'
    exit 1
}

echo
echo '===== FPGA STATE ====='
FPGA_STATE="$(cat /sys/class/fpga_manager/fpga0/state)"
echo "$FPGA_STATE"

test "$FPGA_STATE" = "operating"

echo
echo '===== TEMPERATURE AFTER LOAD ====='
find /sys/class/thermal -maxdepth 2 -type f -name temp \
  -print -exec cat {} \; 2>/dev/null || true

echo
echo '===== PATCH 0 ====='
sudo sh "$SCRIPT_DIR/wangdao_axi_smoke.sh" \
    0xA0000000 \
    "$SCRIPT_DIR/patch0_words.txt" \
    "$SCRIPT_DIR/patch0_expected.txt"

echo
echo '===== TEMPERATURE AFTER PATCH 0 ====='
find /sys/class/thermal -maxdepth 2 -type f -name temp \
  -print -exec cat {} \; 2>/dev/null || true

echo
echo '===== PATCH 1 ====='
sudo sh "$SCRIPT_DIR/wangdao_axi_smoke.sh" \
    0xA0000000 \
    "$SCRIPT_DIR/patch1_words.txt" \
    "$SCRIPT_DIR/patch1_expected.txt"

echo
echo '===== TEMPERATURE AFTER PATCH 1 ====='
find /sys/class/thermal -maxdepth 2 -type f -name temp \
  -print -exec cat {} \; 2>/dev/null || true

echo
echo '== DEPLOY_OK =='

echo
echo '===== RELEVANT DMESG ====='
sudo dmesg -T | grep -Ei \
  'fpga|overlay|firmware|seerr|axi' | tail -n 200 || true
