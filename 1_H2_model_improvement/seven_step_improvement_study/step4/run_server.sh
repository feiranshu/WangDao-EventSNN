#!/usr/bin/env bash
set -euo pipefail
python -m unittest discover -s tests -v
python run_step4.py preflight --output runs/preflight.json --require-cuda --require-gpus 4
python run_step4.py run-all --output runs/step4_formal_20261004 --device cuda --gpus 0,1,2,3 --run
