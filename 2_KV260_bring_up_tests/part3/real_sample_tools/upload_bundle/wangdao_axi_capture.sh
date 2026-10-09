#!/bin/sh
# Run one 128-word input and capture all 16 output words without comparing.
# Usage: wangdao_axi_capture.sh <base-address> <input-words> <output-words>

set -eu

if [ "$#" -ne 3 ]; then
    echo 'usage: wangdao_axi_capture.sh <base-address> <input-words> <output-words>' >&2
    exit 2
fi

BASE=$1
INPUT_WORDS=$2
OUTPUT_WORDS=$3

if command -v devmem >/dev/null 2>&1; then
    read_dev_mem() { devmem "$1" 32; }
    write_dev_mem() { devmem "$1" 32 "$2" >/dev/null; }
elif command -v busybox >/dev/null 2>&1 && busybox --list | grep -qx devmem; then
    read_dev_mem() { busybox devmem "$1" 32; }
    write_dev_mem() { busybox devmem "$1" 32 "$2" >/dev/null; }
else
    echo 'Neither devmem nor BusyBox devmem is available' >&2
    exit 1
fi

[ -f "$INPUT_WORDS" ] || { echo "input file not found: $INPUT_WORDS" >&2; exit 1; }
input_count=$(awk 'NF && $1 !~ /^#/ { count++ } END { print count+0 }' "$INPUT_WORDS")
[ "$input_count" -eq 128 ] || {
    echo "expected 128 input words, got $input_count" >&2
    exit 1
}

base_num=$((BASE))
control_addr=$(printf '0x%X' $((base_num + 0x000)))
status_addr=$(printf '0x%X' $((base_num + 0x004)))
config_addr=$(printf '0x%X' $((base_num + 0x008)))

read_reg() {
    raw=$(read_dev_mem "$1") || return 1
    printf '%s' "$raw" | tr -d '[:space:]'
}

normalize_hex() {
    value=$(printf '%s' "$1" | tr -d '[:space:]' | tr '[:lower:]' '[:upper:]')
    case "$value" in 0X*) value=${value#0X} ;; esac
    while [ "${#value}" -lt 8 ]; do value="0${value}"; done
    printf '%s' "$value"
}

echo "WANGDAO_CONFIG $(read_reg "$config_addr")"

index=0
while IFS= read -r value || [ -n "$value" ]; do
    value=$(printf '%s' "$value" | tr -d '[:space:]')
    [ -z "$value" ] && continue
    case "$value" in \#*) continue ;; esac
    address=$(printf '0x%X' $((base_num + 0x100 + index * 4)))
    write_dev_mem "$address" "$value"
    index=$((index + 1))
done < "$INPUT_WORDS"

write_dev_mem "$control_addr" 0x00000001

# Do not accept the sticky done bit from the preceding transaction.
# This conservative guard delay is outside any future latency benchmark.
sleep 1

poll_count=0
done_seen=0
last_status=0x00000000
while [ "$poll_count" -lt 6000 ]; do
    last_status=$(read_reg "$status_addr")
    status_num=$((last_status))
    if [ $((status_num & 2)) -ne 0 ] && [ $((status_num & 1)) -eq 0 ]; then
        done_seen=1
        break
    fi
    poll_count=$((poll_count + 1))
    sleep 0.01
done

[ "$done_seen" -eq 1 ] || {
    echo "timeout waiting for done; last status=$last_status" >&2
    exit 1
}
status_num=$((last_status))
[ $((status_num & 4)) -eq 0 ] || {
    echo "hardware error flag set; status=$last_status" >&2
    exit 1
}

output_tmp="${OUTPUT_WORDS}.tmp.$$"
trap 'rm -f "$output_tmp"' HUP INT TERM EXIT
: > "$output_tmp"
index=0
while [ "$index" -lt 16 ]; do
    address=$(printf '0x%X' $((base_num + 0x900 + index * 4)))
    actual=$(normalize_hex "$(read_reg "$address")")
    echo "0x$actual" >> "$output_tmp"
    echo "WANGDAO_LOGIT index=$index value=0x$actual"
    index=$((index + 1))
done
mv "$output_tmp" "$OUTPUT_WORDS"
chmod 0644 "$OUTPUT_WORDS" || true
if [ -n "${SUDO_USER:-}" ]; then
    chown "$SUDO_USER" "$OUTPUT_WORDS" || true
fi
trap - HUP INT TERM EXIT

echo "WANGDAO_CAPTURE_PASS inputs=128 logits=16 status=$last_status polls=$poll_count output=$OUTPUT_WORDS"
