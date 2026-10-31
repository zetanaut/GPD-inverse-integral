#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests
python - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('.')
manifest=json.loads((root/'MANIFEST.json').read_text())
for item in manifest['files']:
    p=root/item['path']
    assert p.is_file(),str(p)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],str(p)
print('Source/input/result hashes verified:',len(manifest['files']))
PY
