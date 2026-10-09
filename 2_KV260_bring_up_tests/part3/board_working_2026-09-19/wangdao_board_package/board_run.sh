#!/bin/sh
# Wangdao KV260: run this ON the board console (serial or ssh), with the
# package files in the current directory. No host-side scp/ssh is used.
# Usage: cd <package-dir> && sh board_run.sh
set -eu

for f in wangdao_pl_flat.bit.bin wangdao_pl_flat.dtbo shell.json wangdao_axi_smoke.sh patch0_words.txt patch0_expected.txt patch1_words.txt patch1_expected.txt; do
    if [ ! -f "$f" ]; then
        echo "missing file: $f" >&2
        exit 1
    fi
done

sudo -v

sudo install -d -m 0755 /lib/firmware/xilinx/wangdao_pl_flat
sudo install -m 0644 wangdao_pl_flat.bit.bin wangdao_pl_flat.dtbo shell.json /lib/firmware/xilinx/wangdao_pl_flat/
# Overlay firmware-name is wangdao_pl_flat.bit.bin; keep the legacy alias too.
sudo cp -f /lib/firmware/xilinx/wangdao_pl_flat/wangdao_pl_flat.bit.bin /lib/firmware/xilinx/wangdao_pl_flat/wangdao.bit.bin

echo '--- APPS BEFORE ---'
sudo xmutil listapps
sudo xmutil unloadapp || true
sudo xmutil loadapp wangdao_pl_flat
sleep 2
echo '--- APPS AFTER ---'
sudo xmutil listapps
sudo xmutil listapps | grep -E 'wangdao_pl_flat.*->[0-9]' >/dev/null || { echo LOAD_VERIFY_FAILED >&2; exit 1; }
echo '--- PATCH 0 ---'
sudo sh wangdao_axi_smoke.sh 0xA0000000 patch0_words.txt patch0_expected.txt
echo '--- PATCH 1 ---'
sudo sh wangdao_axi_smoke.sh 0xA0000000 patch1_words.txt patch1_expected.txt
echo '== DEPLOY_OK =='
