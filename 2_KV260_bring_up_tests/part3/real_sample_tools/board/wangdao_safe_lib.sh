#!/bin/sh
# Shared safe loader for the frozen Wangdao PL_FLAT application.

DEFAULT_APP='k26-starter-kits'
CUSTOM_APP='wangdao_pl_flat'
BASE_ADDRESS='0xA0000000'

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PACKAGE_DIR=${WANGDAO_PACKAGE_DIR:-$SCRIPT_DIR}

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
    sudo xmutil listapps || true
    printf 'FPGA_STATE='
    cat /sys/class/fpga_manager/fpga0/state || true
}

install_and_load_wangdao() {
    if [ ! -f "$SCRIPT_DIR/artifact_preflight.sh" ]; then
        echo "missing artifact preflight: $SCRIPT_DIR/artifact_preflight.sh" >&2
        return 1
    fi
    sh "$SCRIPT_DIR/artifact_preflight.sh" "$PACKAGE_DIR"

    for required in wangdao_pl_flat.bit.bin wangdao_pl_flat.dtbo shell.json; do
        if [ ! -f "$PACKAGE_DIR/$required" ]; then
            echo "missing package file: $PACKAGE_DIR/$required" >&2
            return 1
        fi
    done

    sudo install -d -m 0755 "/lib/firmware/xilinx/$CUSTOM_APP"
    sudo rm -f "/lib/firmware/xilinx/$CUSTOM_APP/wangdao.bit.bin"
    sudo install -m 0644 \
        "$PACKAGE_DIR/wangdao_pl_flat.bit.bin" \
        "$PACKAGE_DIR/wangdao_pl_flat.dtbo" \
        "$PACKAGE_DIR/shell.json" \
        "/lib/firmware/xilinx/$CUSTOM_APP/"

    app_status=$(sudo xmutil listapps)
    active=$(printf '%s\n' "$app_status" | awk '$0 ~ /0->0/ { print $1; exit }')
    case "$active" in
        '') ;;
        "$DEFAULT_APP"|"$CUSTOM_APP") sudo xmutil unloadapp ;;
        *)
            echo "refusing to unload unexpected active app: $active" >&2
            return 1
            ;;
    esac
    sudo xmutil loadapp "$CUSTOM_APP"
    sleep 2

    app_status=$(sudo xmutil listapps)
    echo "$app_status"
    echo "$app_status" | grep "$CUSTOM_APP" | grep -q '0->0' || {
        echo 'wangdao_pl_flat did not acquire slot 0' >&2
        return 1
    }

    fpga_state=$(cat /sys/class/fpga_manager/fpga0/state)
    echo "FPGA_STATE=$fpga_state"
    [ "$fpga_state" = 'operating' ] || return 1
}

