"""Verify the content of the imported source/artifact distribution."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text())
failures=[]
for item in manifest['files']:
    p=root/item['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        failures.append(item['path'])
if failures:
    raise SystemExit('Missing or changed imported files: '+', '.join(failures))
print(f"Verified {len(manifest['files'])} imported source and artifact hashes.")
