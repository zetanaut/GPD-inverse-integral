"""Read-only payload verification; standard library only."""
from pathlib import Path
import hashlib
import json


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    for r in manifest['files']:
        p = (root/r['path']).resolve()
        if root not in p.parents or not p.is_file():
            raise RuntimeError('Invalid/missing payload: '+r['path'])
        data = p.read_bytes()
        if len(data) != r['bytes'] or hashlib.sha256(data).hexdigest() != r['sha256']:
            raise RuntimeError('Payload identity mismatch: '+r['path'])
    return len(manifest['files'])


if __name__ == '__main__':
    print(json.dumps(dict(status='PASS', payload_files=verify(Path(__file__).resolve().parent))))
