#!/usr/bin/env python3
"""Verify the shipped package without writing source files or executing model code."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
records=json.loads((root/'MANIFEST.json').read_text())['files']
for r in records:
 p=root/r['path']
 if not p.is_file() or p.stat().st_size!=r['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
  raise SystemExit('Changed/missing package member: '+r['path'])
print(f'Package members verified: {len(records)}')
