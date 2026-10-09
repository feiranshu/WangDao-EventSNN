#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-step2_formal_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p runs
python run_step2.py preflight --output "runs/${RUN_NAME}.preflight.json" --require-cuda
python -m unittest discover -s tests -v
python run_step2.py run-all \
  --output "runs/${RUN_NAME}" \
  --device cuda \
  --run

python collect_step2_results.py "runs/${RUN_NAME}" \
  --output "runs/${RUN_NAME}.results.zip"

echo "Finished: runs/${RUN_NAME}"
echo "Return archive: runs/${RUN_NAME}.results.zip"
