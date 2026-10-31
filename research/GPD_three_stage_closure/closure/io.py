from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=raw: raise FileExistsError(f'Conflicting immutable result: {path}')
        return
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_bytes(raw); os.replace(tmp,path)

def atomic_npz(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:np.asarray(v) for k,v in arrays.items()}
    if any(a.dtype.hasobject for a in arrays.values()): raise ValueError('Object arrays forbidden')
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(arrays) or any(not np.array_equal(z[k],v) for k,v in arrays.items()):
                raise FileExistsError(f'Conflicting immutable arrays: {path}')
        return
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.partial')
    try:
        with os.fdopen(fd,'wb') as f: np.savez_compressed(f,**arrays)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def coordinate_key(xi,q2):
    xi,q2=float(xi),float(q2)
    if not np.isfinite([xi,q2]).all() or not 0<xi<1 or q2<=0:
        raise ValueError('Expected 0<xi<1 and Q2>0')
    return xi.hex()+'|'+q2.hex()
