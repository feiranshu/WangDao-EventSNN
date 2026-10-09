#!/bin/sh
# Run one frozen Proposal patch through the AXI-Lite register block.
# Arguments: base-address input-words-file expected-words-file

set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: wangdao_axi_smoke.sh <base> <input-words> <expected-words>" >&2
    exit 2
fi

BASE="$1"
INPUT_WORDS="$2"
EXPECTED_WORDS="$3"
if command -v devmem >/dev/null 2>&1; then
    read_dev_mem() { devmem "$1" 32; }
    write_dev_mem() { devmem "$1" 32 "$2" >/dev/null; }
elif command -v busybox >/dev/null 2>&1 && busybox --list | grep -qx devmem; then
    read_dev_mem() { busybox devmem "$1" 32; }
    write_dev_mem() { busybox devmem "$1" 32 "$2" >/dev/null; }
else
    echo "Neither devmem nor BusyBox devmem is available" >&2
    exit 1
fi

base_num=$((BASE))
control_addr=$(printf '0x%X' $((base_num + 0x000)))
status_addr=$(printf '0x%X' $((base_num + 0x004)))
config_addr=$(printf '0x%X' $((base_num + 0x008)))
input_base=$(printf '0x%X' $((base_num + 0x100)))
output_base=$(printf '0x%X' $((base_num + 0x900)))

read_reg() {
    read_dev_mem "$1" | tr -d '[:space:]'
}

write_reg() {
    write_dev_mem "$1" "$2"
}

normalize_hex() {
    value=$(printf '%s' "$1" | tr -d '[:space:]' | tr '[:lower:]' '[:upper:]')
    case "$value" in
        0X*) value=${value#0X} ;;
    esac
    while [ "${#value}" -lt 8 ]; do
        value="0${value}"
    done
    printf '%s' "$value"
}

echo "WANGDAO_CONFIG $(read_reg "$config_addr")"

input_count=0
while IFS= read -r value || [ -n "$value" ]; do
    value=$(printf '%s' "$value" | tr -d '[:space:]')
    [ -z "$value" ] && continue
    case "$value" in
        \#*) continue ;;
    esac
    address=$(printf '0x%X' $((base_num + 0x100 + input_count * 4)))
    write_reg "$address" "$value"
    input_count=$((input_count + 1))
done < "$INPUT_WORDS"

if [ "$input_count" -ne 128 ]; then
    echo "expected 128 input words, got $input_count" >&2
    exit 1
fi

write_reg "$control_addr" 0x00000001

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

if [ "$done_seen" -ne 1 ]; then
    echo "timeout waiting for done; last status=$last_status" >&2
    exit 1
fi

status_num=$((last_status))
if [ $((status_num & 4)) -ne 0 ]; then
    echo "hardware error flag set; status=$last_status" >&2
    exit 1
fi

expected_count=0
while IFS= read -r expected || [ -n "$expected" ]; do
    expected=$(printf '%s' "$expected" | tr -d '[:space:]')
    [ -z "$expected" ] && continue
    case "$expected" in
        \#*) continue ;;
    esac
    address=$(printf '0x%X' $((base_num + 0x900 + expected_count * 4)))
    actual=$(read_reg "$address")
    actual_norm=$(normalize_hex "$actual")
    expected_norm=$(normalize_hex "$expected")
    if [ "$actual_norm" != "$expected_norm" ]; then
        echo "logit mismatch index=$expected_count address=$address actual=0x$actual_norm expected=0x$expected_norm" >&2
        exit 1
    fi
    echo "WANGDAO_LOGIT index=$expected_count value=0x$actual_norm"
    expected_count=$((expected_count + 1))
done < "$EXPECTED_WORDS"

if [ "$expected_count" -ne 16 ]; then
    echo "expected 16 output words, got $expected_count" >&2
    exit 1
fi

echo "WANGDAO_PASS inputs=128 logits=16 status=$last_status polls=$poll_count"
