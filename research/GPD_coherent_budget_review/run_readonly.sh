#!/usr/bin/env bash
set -euo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=${1:-/Users/dustin/work/GPDs/PRC-version/project}
python3 "$HERE/audit.py" \
  --packet "$ROOT/completion_runs/ReH_coherent_budget_check" \
  --project "$ROOT" \
  --prior "$ROOT/completion_runs/ReH_coherent_GPD_pilot" \
  --cohort "$ROOT/completion_runs/ReH_coherent_surface_pilot/coherent_surfaces.npz" \
  --out "$HERE/local_replay_results" --moments
(cd "$HERE" && python3 -m unittest -v test_review)
