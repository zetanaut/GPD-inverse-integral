#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${PYTHON:-python3}"
"$PYTHON" verify_manifest.py
"$PYTHON" -m unittest -v test_closeout
"$PYTHON" closeout.py --out replayed
