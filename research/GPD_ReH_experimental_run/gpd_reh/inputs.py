"""Prepare only ReH from the author-identified, unmerged publication snapshot."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

SOURCE_SHA256 = '8d7274db9dd7e4fcc3ab84c0230701242274e42868f9c3e66e3d61a1a6838696'
INPUT_FIELDS = ('row_id','source_line','source_set','method','k_GeV','Q2_GeV2','xB','xi',
                't_GeV2','ReH_source','ReH_internal','sigma_reported','map_id',
                'paper_cut_pass','cut_reason','leading_power_boundary_flag')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n'
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)

def write_csv(path,rows,fields):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def coordinate_key(xi,Q2):
    xi,Q2=float(xi),float(Q2)
    if not math.isfinite(xi+Q2) or not 0<xi<1 or Q2<=0:
        raise ValueError('require finite 0<xi<1 and Q2>0')
    return xi.hex()+'|'+Q2.hex()

def read_source(path, expected_sha256=SOURCE_SHA256):
    path=Path(path)
    if expected_sha256 is not None and digest(path)!=expected_sha256:
        raise ValueError('Source changed: explicitly create a new source/version contract')
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        required={'Set #','DNN','k','QQ','x_b','t','Pred ReH','Pred ReH Std'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing ReH/kinematic columns')
        raw=list(reader)
    records=[];seen=set()
    for i,r in enumerate(raw):
        k,q,xb,t,reh,sigma=(float(r[c]) for c in ('k','QQ','x_b','t','Pred ReH','Pred ReH Std'))
        if not all(map(math.isfinite,(k,q,xb,t,reh,sigma))) or min(k,q,sigma)<=0 or t>=0 or not 0<xb<1:
            raise ValueError(f'Invalid supplied value or uncertainty on line {i+2}')
        key=(k,q,xb,t)
        if key in seen:raise ValueError('Duplicate source kinematics: do not silently merge')
        seen.add(key);xi=xb/(2-xb)
        reasons=[]
        if q<=1.5:reasons.append('Q2<=1.5')
        if abs(t)/q>0.25:reasons.append('|t|/Q2>0.25')
        limit=-4*0.9382720813**2*xi**2/(1-xi**2)
        records.append(dict(zip(INPUT_FIELDS,[f'reh:{i:04d}',i+2,r['Set #'],r['DNN'],k,q,xb,xi,t,
            reh,-reh,sigma,'',int(not reasons),';'.join(reasons),int(t>limit+1e-12)])))
    return records

def coordinate_only_t_nodes(rows,count=7):
    """Nearest observed t to evenly spaced empirical quantiles, no CFF dependence."""
    ts=np.unique([float(r['t_GeV2']) for r in rows]);idx=np.rint(np.linspace(0,len(ts)-1,min(count,len(ts)))).astype(int)
    return ts[idx].tolist()

def anchor_indices(rows,count=6):
    """Deterministic farthest-point geometry sampling; never uses ReH or its errors."""
    a=np.array([[r['xi'],math.log(r['Q2_GeV2']),r['t_GeV2']] for r in rows])
    scales=np.std(a,axis=0);scales[scales==0]=1;a=(a-np.mean(a,axis=0))/scales
    selected=[0];distance=np.full(len(a),np.inf)
    for _ in range(min(count,len(a))-1):
        distance=np.minimum(distance,np.sum((a-a[selected[-1]])**2,axis=1));distance[selected]=-np.inf
        selected.append(int(np.argmax(distance)))
    return selected

def prepare(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    allrows=read_source(source);rows=[r for r in allrows if r['paper_cut_pass']]
    groups={};requests=[]
    for r in rows:
        key=coordinate_key(r['xi'],r['Q2_GeV2'])
        if key not in groups:
            mid=f'm{len(groups):03d}';groups[key]=mid
            requests.append({'map_id':mid,'xi':r['xi'],'Q2_GeV2':r['Q2_GeV2'],
                'xi_hex':r['xi'].hex(),'Q2_hex':r['Q2_GeV2'].hex(),'Q0_squared_GeV2':2.0,
                'direction':'backward' if r['Q2_GeV2']<2 else 'forward' if r['Q2_GeV2']>2 else 'identity'})
        r['map_id']=groups[key]
    t=np.array([r['t_GeV2'] for r in rows]);tmean=float(t.mean());tstd=float(t.std())
    families=defaultdict(list)
    for r in requests:families[r['xi_hex']].append(r)
    write_csv(out/'source_rows_all.csv',allrows,INPUT_FIELDS)
    write_csv(out/'observations.csv',rows,INPUT_FIELDS)
    write_csv(out/'map_requests.csv',requests,list(requests[0]))
    write_csv(out/'plot_anchors.csv',[rows[i] for i in anchor_indices(rows)],INPUT_FIELDS)
    atomic_json(out/'kernel_families.json',{'families':[{'xi':float.fromhex(k),'xi_hex':k,'map_ids':[v['map_id'] for v in vals],
                'Q2_GeV2':[v['Q2_GeV2'] for v in vals]} for k,vals in sorted(families.items(),key=lambda z:float.fromhex(z[0]))]})
    np.savez_compressed(out/'observations.npz',
        row_id=np.array([r['row_id'] for r in rows]),map_id=np.array([r['map_id'] for r in rows]),
        xi=np.array([r['xi'] for r in rows]),Q2=np.array([r['Q2_GeV2'] for r in rows]),t=t,
        y=np.array([r['ReH_internal'] for r in rows]),sigma=np.array([r['sigma_reported'] for r in rows]))
    summary={'source_file':source.name,'source_sha256':digest(source),'source_paper':'2504.15458v4',
        'component':'ReH only','target':'Hcharge_plus=(2/9)*(Sigma+T_rescaled)',
        'n_original':len(allrows),'n_selected':len(rows),'n_excluded':len(allrows)-len(rows),
        'n_exact_evolution_families':len(families),'n_exact_xi_Q2_pairs':len(requests),
        'n_t_coordinates':len(set(t)),'n_Q2_below_Q02_rows':sum(r['Q2_GeV2']<2 for r in rows),
        'n_backward_map_requests':sum(r['direction']=='backward' for r in requests),
        'n_boundary_flags_retained':sum(r['leading_power_boundary_flag'] for r in rows),
        't_mean':tmean,'t_std_population':tstd,'moment_t_nodes':coordinate_only_t_nodes(rows),
        'ranges':{col:[min(r[col] for r in rows),max(r[col] for r in rows)] for col in ('xi','Q2_GeV2','t_GeV2','sigma_reported')},
        'uncertainties':'quoted source uncertainties, including propagated statistical/systematic content; not added again',
        'central_weighting':'diagonal reported-error weighted objective; no cross-row independence theorem asserted',
        'scope':'central experimental run preparation; no fit or uncertainty interval generated'}
    atomic_json(out/'INPUT_SUMMARY.json',summary)
    return summary
