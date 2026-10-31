#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m unittest -v test_direct_error
python control.py
if [[ -n "${GPD_REVIEW_ROOT:-}" ]]; then
  python verify_runtime.py
  python -m unittest -v test_nlo_adapter
  for id in d000 d001; do
    if [[ -f "results/development_${id}/RESULT.json" ]]; then
      python replay_result.py --input "results/inputs/${id}/input.npz" \
        --result "results/development_${id}" --out "results/REPLAY_${id}_local.json"
    fi
  done
fi
# No training is performed by this verification command.
