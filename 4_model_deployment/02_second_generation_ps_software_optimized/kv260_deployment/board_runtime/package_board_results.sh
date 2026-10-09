#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LABEL=${1:-train_dev10}
case "$LABEL" in *[!A-Za-z0-9_.-]*) echo "unsafe result label: $LABEL" >&2; exit 2;; esac
HASH_FILE="$ROOT/artifact_hashes.txt"
ARCHIVE="$ROOT/${LABEL}_board_results.tar.gz"

: > "$HASH_FILE"
for path in "$ROOT/ps_manifest.json" "$ROOT/s1_manifest.json" "$ROOT/h2_weights.bin" \
            "$ROOT/hardware_resource_manifest.json"; do
    [ ! -f "$path" ] || sha256sum "$path" >> "$HASH_FILE"
done
for dir in captures predictions runtime; do
    if [ -d "$ROOT/$dir" ]; then
        find "$ROOT/$dir" -type f -print0 | sort -z | xargs -0 -r sha256sum >> "$HASH_FILE"
    fi
done
rm -f "$ARCHIVE"
set -- ps_manifest.json s1_manifest.json hardware_resource_manifest.json artifact_hashes.txt captures predictions
[ ! -d "$ROOT/runtime" ] || set -- "$@" runtime
[ ! -d "$ROOT/logs" ] || set -- "$@" logs
(cd "$ROOT" && tar -czf "$ARCHIVE" "$@")
echo "WANGDAO_RESULT_ARCHIVE=$ARCHIVE"
