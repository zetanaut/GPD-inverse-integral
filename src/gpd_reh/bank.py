"""Exact-coordinate, source-keyed NLO rows. Never substitute nearest xi or Q2."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .inputs import coordinate_key,digest,atomic_json

CHANNELS=['Sigma','T_rescaled','Hg=xg']
PHYSICS={'order':'NLO','scheme':'MSbar','nf':3,'Q02':2.0,'alpha0':0.35691594851514724,
         'muF_over_Q':1.0,'muR_over_Q':1.0,'running':'two-loop','input_channels':CHANNELS,
         'output_components':['Re_internal','Im_internal'],'charges_included':True}

class MissingMaps(RuntimeError):pass

def read_index(directory):
    d=Path(directory);p=d/'index.json'
    if not p.is_file():return {'physics':PHYSICS,'entries':[]}
    index=json.loads(p.read_text())
    for key,value in PHYSICS.items():
        if index.get('physics',{}).get(key)!=value:raise ValueError(f'NLO physics contract mismatch: {key}')
    seen=set()
    for e in index['entries']:
        key=coordinate_key(e['xi'],e['Q2'])
        if key in seen:raise ValueError('Duplicate coordinate entries')
        seen.add(key)
        rel=Path(e['file'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe bank relative path')
        p=d/rel
        if not p.is_file() or digest(p)!=e['sha256']:raise ValueError('Missing/changed NLO bank file')
        if not e.get('source_key') or not e.get('validation_file'):raise ValueError('Bank row requires source and diagnostic provenance')
        vr=Path(e['validation_file'])
        if vr.is_absolute() or '..' in vr.parts:raise ValueError('Unsafe validation relative path')
        vp=d/vr
        if not vp.is_file() or digest(vp)!=e.get('validation_sha256'):raise ValueError('Missing/changed map diagnostics')
    return index

def preflight(prepared,directory):
    prepared=Path(prepared);directory=Path(directory);index=read_index(directory)
    have={coordinate_key(e['xi'],e['Q2']):e for e in index['entries']}
    with (prepared/'map_requests.csv').open(newline='') as f:requests=list(csv.DictReader(f))
    missing=[r for r in requests if coordinate_key(r['xi'],r['Q2_GeV2']) not in have]
    return {'n_requested_pairs':len(requests),'n_exact_pairs_present':len(requests)-len(missing),
        'n_missing_pairs':len(missing),'n_missing_xi_families':len(set(r['xi_hex'] for r in missing)),
        'missing_map_ids':[r['map_id'] for r in missing],
        'status':'READY_FOR_CENTRAL_FIT' if not missing else 'EXACT_MAPS_REQUIRED',
        'note':'No fit runs or nearest-coordinate substitution in preflight.'}

def load_rows(prepared,directory,method):
    if method not in ('nominal','reference'):raise ValueError('method must be nominal/reference')
    prepared=Path(prepared);directory=Path(directory);index=read_index(directory)
    entries={coordinate_key(e['xi'],e['Q2']):e for e in index['entries']}
    with (prepared/'observations.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    missing=[];out=[]
    for obs in rows:
        key=coordinate_key(obs['xi'],obs['Q2_GeV2'])
        if key not in entries:missing.append(obs['map_id']);continue
        e=entries[key]
        with np.load(directory/e['file'],allow_pickle=False) as a:
            x=np.asarray(a[method+'_x'],float);weights=np.asarray(a[method+'_weights'],float)
            if float(a['xi'])!=float(obs['xi']) or float(a['Q2'])!=float(obs['Q2_GeV2']):
                raise ValueError('Bank payload coordinate mismatch')
        if x.ndim!=1 or weights.shape!=(2,3,len(x)) or not np.isfinite(x).all() or not np.isfinite(weights).all():
            raise ValueError('Invalid bank: expect x[N], weights[Re/Im,3,N]')
        if np.any(np.diff(x)<=0) or np.any(np.abs(x)>1):raise ValueError('Invalid x nodes')
        out.append((obs,x,weights))
    if missing:raise MissingMaps('Exact NLO rows missing for '+','.join(sorted(set(missing))))
    return out

class ExperimentalForward(nn.Module):
    """Observations may share an evolution map, but retain their individual t."""
    def __init__(self,prepared,directory,method='nominal'):
        super().__init__();records=load_rows(prepared,directory,method)
        # Pad only zero-weight slots; no numerical interpolation between coordinates.
        nmax=max(len(x) for _,x,_ in records);B=len(records)
        X=np.zeros((B,nmax));W=np.zeros((B,2,3,nmax))
        for i,(r,x,w) in enumerate(records):X[i,:len(x)]=x;W[i,:,:,:len(x)]=w
        self.register_buffer('x',torch.tensor(X,dtype=torch.float64))
        self.register_buffer('weights',torch.tensor(W,dtype=torch.float64))
        self.register_buffer('xi',torch.tensor([float(r['xi']) for r,_,_ in records],dtype=torch.float64))
        self.register_buffer('t',torch.tensor([float(r['t_GeV2']) for r,_,_ in records],dtype=torch.float64))
        self.row_ids=tuple(r['row_id'] for r,_,_ in records)
    def forward(self,model):
        field=model(self.x,self.xi,self.t)
        return torch.einsum('bocn,bcn->bo',self.weights,field)
