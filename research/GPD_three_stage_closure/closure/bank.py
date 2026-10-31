"""Read-only exact-coordinate NLO rows, with no nearest-coordinate substitution."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from .io import coordinate_key,sha256

PHYSICS={'order':'NLO','scheme':'MSbar','nf':3,'Q02':2.,'alpha0':.35691594851514724,
 'muF_over_Q':1.,'muR_over_Q':1.,'running':'two-loop',
 'input_channels':['Sigma','T_rescaled','Hg=xg'],'output_components':['Re_internal','Im_internal'],'charges_included':True}

class RowBank:
    def __init__(self,*directories):
        self.records={};self.loaded={};self.index_hashes={}
        for directory in directories:
            directory=Path(directory)
            if not (directory/'index.json').exists():continue
            index=json.loads((directory/'index.json').read_text())
            for k,v in PHYSICS.items():
                if index.get('physics',{}).get(k)!=v:raise ValueError(f'Physics mismatch {k}')
            self.index_hashes[str(directory)]=sha256(directory/'index.json')
            for entry in index['entries']:
                k=coordinate_key(entry['xi'],entry['Q2'])
                for name,key in [('file','sha256'),('validation_file','validation_sha256')]:
                    p=Path(entry[name])
                    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe bank member')
                    if not (directory/p).is_file() or sha256(directory/p)!=entry[key]:raise ValueError('Bank integrity failure')
                if k in self.records:
                    old=self.records[k][1]
                    if old['sha256']!=entry['sha256']:raise ValueError('Conflicting exact-coordinate row')
                    continue
                self.records[k]=(directory,entry)
    def has(self,xi,q2):return coordinate_key(xi,q2) in self.records
    def get(self,xi,q2,method):
        if method not in ('nominal','reference'):raise ValueError('Specify nominal or reference')
        key=coordinate_key(xi,q2)
        if key not in self.records:raise KeyError(f'Missing exact NLO row {key}')
        if (key,method) not in self.loaded:
            directory,e=self.records[key]
            with np.load(directory/e['file'],allow_pickle=False) as z:
                if coordinate_key(z['xi'],z['Q2'])!=key:raise ValueError('Coordinate payload mismatch')
                x=z[method+'_x'].copy();w=z[method+'_weights'].copy()
            if x.ndim!=1 or w.shape!=(2,3,len(x)) or not np.isfinite(x).all() or not np.isfinite(w).all():raise ValueError('Bad row shape/values')
            if np.any(np.diff(x)<=0) or np.max(abs(x))>1:raise ValueError('Invalid nodes')
            self.loaded[key,method]=(x,w)
        return self.loaded[key,method]
    def field_cff(self,rows,truth,method):
        result=[]
        for r in rows:
            x,w=self.get(r['xi'],r['Q2_GeV2'],method)
            h=truth.values(x,r['xi'],r['t_GeV2']).T
            result.append(np.einsum('ocn,cn->o',w,h))
        return np.array(result)
