#!/bin/sh
# Refuse to install a board package unless every validated artifact matches.
set -eu

ARTIFACT_DIR=${1:-"$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"}

check_file() {
    name=$1
    expected_bytes=$2
    expected_hash=$3
    path="$ARTIFACT_DIR/$name"

    [ -f "$path" ] || { echo "missing artifact: $path" >&2; exit 1; }
    actual_bytes=$(wc -c < "$path" | tr -d '[:space:]')
    actual_hash=$(sha256sum "$path" | awk '{print $1}')
    [ "$actual_bytes" = "$expected_bytes" ] || {
        echo "size mismatch: $name ($actual_bytes)" >&2
        exit 1
    }
    [ "$actual_hash" = "$expected_hash" ] || {
        echo "sha256 mismatch: $name ($actual_hash)" >&2
        exit 1
    }
    echo "ARTIFACT_OK $name bytes=$actual_bytes sha256=$actual_hash"
}

check_file wangdao_pl_flat.bit.bin 7797692 3bf7473f26f5ddcd1157c2513706d179982eb131e8cb38880601d30e105dee23
check_file wangdao_pl_flat.dtbo 1986 0a1c9eb846b6929129b315b5e3ff510a06e332a74744dc048a9bc46c47ca2706
check_file shell.json 50 0377daaf2050b88df14de9a79d120144cf49381d00fa552402395a322cb7b1a3
grep -q '"shell_type"[[:space:]]*:[[:space:]]*"PL_FLAT"' "$ARTIFACT_DIR/shell.json"
grep -q '"num_slots"[[:space:]]*:[[:space:]]*"1"' "$ARTIFACT_DIR/shell.json"
echo WANGDAO_ARTIFACT_PREFLIGHT_PASS
