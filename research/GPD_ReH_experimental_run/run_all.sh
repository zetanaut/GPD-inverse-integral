#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" prepare_run.py
# This command is verification/preparation ONLY. Missing experimental maps are
# recorded; it never builds kernels, fits a GPD, or launches a replica batch.
