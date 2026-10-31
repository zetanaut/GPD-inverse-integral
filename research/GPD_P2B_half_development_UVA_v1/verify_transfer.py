"""Standard-library strict verification of the immutable transfer directory."""
import argparse
import hashlib
import json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def verify(root,expected=None):
    root=Path(root).resolve();manifest=root/'MANIFEST.json'
    if expected and digest(manifest)!=expected:raise RuntimeError('Transfer manifest identity changed')
    rows=json.loads(manifest.read_text())['files'];names=set()
    for row in rows:
        rel=Path(row['path']);p=root/rel
        if (rel.is_absolute() or '..' in rel.parts or row['path'] in names or p.is_symlink()
                or not p.is_file() or root not in p.resolve().parents
                or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']):
            raise RuntimeError('Transfer mismatch: '+str(rel))
        names.add(row['path'])
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual!=names|{'MANIFEST.json'}:raise RuntimeError('Missing/unmanifested transfer entries: '+str(sorted(actual.symmetric_difference(names|{'MANIFEST.json'}))))
    return dict(status='PASS',files=len(rows),manifest_sha256=digest(manifest))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--expected-manifest')
    a=p.parse_args();print(json.dumps(verify(a.root,a.expected_manifest)))
