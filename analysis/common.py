"""Paths and integrity checks shared by the public analysis commands."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = HERE / 'data'
DEFAULT_OUT = REPO / 'outputs' / 'analysis'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_inputs():
    records = json.loads((HERE / 'INPUT_MANIFEST.json').read_text())['files']
    for record in records:
        name = Path(record['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Invalid input path: ' + str(name))
        path = HERE / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Missing or changed analysis input: ' + str(name))
    return len(records)
