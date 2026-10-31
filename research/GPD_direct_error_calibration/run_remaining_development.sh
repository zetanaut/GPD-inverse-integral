#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${GPD_REVIEW_ROOT:?Set GPD_REVIEW_ROOT to the existing project root}"
python verify_runtime.py
for id in d000 d001; do
  test -f "results/development_${id}/RESULT.json"
  python - "$id" <<'PY'
import json,sys
from pathlib import Path
r=json.loads((Path('results')/('development_'+sys.argv[1])/'RESULT.json').read_text())
if r['status']!='finite_budget_feasible':raise SystemExit('Initial two development cases did not pass; do not expand')
PY
done
# Fixed 2 additional datasets, exactly two fresh starts each. Stop after a
# failure; do not change settings or regenerate a registered observation.
for id in d002 d003; do
  python prepare_development.py --id "$id" --out "results/inputs/$id"
  if [[ ! -f "results/development_${id}/RESULT.json" ]]; then
    python fit_nlo.py --input "results/inputs/${id}/input.npz" \
      --out "results/development_${id}" > "results/development_${id}.log" 2>&1
  fi
  python replay_result.py --input "results/inputs/${id}/input.npz" \
    --result "results/development_${id}" --out "results/REPLAY_${id}_local.json"
  python - "$id" <<'PY'
import json,sys
from pathlib import Path
r=json.loads((Path('results')/('development_'+sys.argv[1])/'RESULT.json').read_text())
if r['status']!='finite_budget_feasible':raise SystemExit('Preserved abstention; stop expansion and package evidence')
PY
done
