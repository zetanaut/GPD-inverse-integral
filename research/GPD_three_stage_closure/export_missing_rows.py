#!/usr/bin/env python3
"""Export NEW adjoint rows from EXISTING exact-xi generators. Never build kernels.

Run on the established source-generator machine. Default --limit 1 is a measured
preflight. Set --limit 0 only after inspecting that timing. Original banks and
PIPELINE files are read-only. New rows and their diagnostics use a separate index.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_name,'1')
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
from closure.io import sha256,atomic_json,atomic_npz,coordinate_key
from closure.bank import PHYSICS,RowBank
ROOT=Path(__file__).resolve().parent

def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def build(project,out,limit=1):
    project=Path(project).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    NLO=project/'GPD_NLO_completion_package';pilot=project/'GPD_NLO_resolution_review/pilot'
    families=project/'GPD_ReH_experimental_run/families'
    required=[pilot/'preflight.py',pilot/'common.py',NLO/'candidate/forward.py',NLO/'src/ordered_evolution.py']
    if any(not p.is_file() for p in required):raise FileNotFoundError('Existing NLO source runtime is required; no installation or kernel fallback')
    os.environ['GPD_REVIEW_ROOT']=str(project)
    sys.path[:0]=[str(pilot),str(NLO),str(NLO/'candidate'),str(NLO/'frozen/python')]
    pf=_module('closure_existing_preflight',pilot/'preflight.py')
    # All NEW small row caches go here, not into the accepted historical bank.
    pf.WORK=out/'row_work';pf.WORK.mkdir(exist_ok=True)
    if Path(pf.NLO).resolve()!=NLO:raise RuntimeError('NLO project resolved elsewhere')
    sources={str(p.relative_to(project)):sha256(p) for p in required}
    for folder in [NLO/'candidate',NLO/'src',NLO/'frozen/python']:
        for p in sorted(folder.glob('*.py')):sources[str(p.relative_to(project))]=sha256(p)
    atomic_json(out/'SOURCE_LOCK.json',sources)
    have=RowBank(ROOT/'inputs/accepted_experimental_banks',out)
    req=json.loads((ROOT/'prepared/MAP_REQUESTS.json').read_text())['requests']
    selected=[r for r in req if not have.has(r['xi'],r['Q2'])]
    if limit>0:selected=selected[:limit]
    completed=[]
    for r in selected:
        xi,q2=r['xi'],r['Q2'];token=hashlib.sha256(float(xi).hex().encode()).hexdigest()[:16]
        state_path=families/token/'PIPELINE.json'
        if not state_path.is_file():raise FileNotFoundError(f'Existing family unavailable: {state_path}')
        state=json.loads(state_path.read_text())
        if state['xi_hex']!=float(xi).hex():raise ValueError('Family coordinate mismatch')
        begin=time.perf_counter();arrays={'xi':xi,'Q2':q2};diags={}
        for method,label in [('weak10','nominal'),('strong10','reference')]:
            directory=state['directories'][method]
            opdir=NLO/directory
            if not (opdir/'operators.npz').exists():raise FileNotFoundError(f'Copy existing generator arrays: {opdir}; do not build kernels')
            b,k0,k1,key=pf.op(directory)
            if float(b.xi).hex()!=float(xi).hex():raise ValueError('Generator coordinate mismatch')
            row=pf.cached_row(directory,q2)
            if row.shape!=(2,3,len(b.nodes)) or not np.isfinite(row).all():raise ValueError('Unexpected existing adjoint-row ABI')
            # Hard quadrature and ordered time checks are recorded separately.
            h192=pf.hard_row(b,q2,192)[0];h384=pf.hard_row(b,q2,384)[0]
            nominal,nd=pf.evolve(h192,k0,k1,q2,rtol=1e-12,atol=1e-14,adjoint=True)
            tight,td=pf.evolve(h384,k0,k1,q2,rtol=1e-13,atol=1e-15,adjoint=True)
            arrays[label+'_x']=np.array(b.nodes);arrays[label+'_weights']=np.array(row)
            diags[method]={'basis':key['basis'],'KEY_sha256':sha256(opdir/'KEY.json'),
                'operators_sha256':sha256(opdir/'operators.npz'),'directory':directory,
                'hard_192_384_row_max_abs':float(np.max(abs(h384-h192))),
                'combined_hard_time_adjoint_row_max_abs':float(np.max(abs(tight-nominal))),
                'ordered_diagnostics':nd,'tight_diagnostics':td,
                'note':'operator-row diagnostics; truth/learned-field contraction and full refinement still required'}
        k=coordinate_key(xi,q2);rowtoken=hashlib.sha256(k.encode()).hexdigest()[:20]
        filename=f'row_{rowtoken}.npz';prov=f'row_{rowtoken}_provenance.json'
        atomic_npz(out/filename,**arrays)
        record={'coordinate':r,'source_lock_sha256':sha256(out/'SOURCE_LOCK.json'),
            'parent_pipeline_sha256':sha256(state_path),'methods':diags,'seconds':time.perf_counter()-begin,
            'status':'EXPORTED_WITH_ROW_DIAGNOSTICS_NOT_A_FULL_FIELD_CERTIFICATE'}
        # Timing is not regenerated for an already committed row.
        if not (out/prov).exists():atomic_json(out/prov,record)
        prior=json.loads((out/'index.json').read_text()) if (out/'index.json').exists() else {'physics':PHYSICS,'entries':[]}
        if not any(coordinate_key(e['xi'],e['Q2'])==k for e in prior['entries']):
            prior['entries'].append({'xi':xi,'Q2':q2,'file':filename,'sha256':sha256(out/filename),
                'validation_file':prov,'validation_sha256':sha256(out/prov),
                'source_key':sha256(out/'SOURCE_LOCK.json')})
            text=json.dumps(prior,indent=2,sort_keys=True,allow_nan=False)+'\n';tmp=out/'index.json.partial';tmp.write_text(text);tmp.replace(out/'index.json')
        completed.append({'xi':xi,'Q2':q2,'seconds':record['seconds']})
        print(json.dumps(completed[-1]),flush=True)
    final_bank=RowBank(ROOT/'inputs/accepted_experimental_banks',out)
    return {'completed_this_invocation':completed,'remaining':sum(not final_bank.has(r['xi'],r['Q2']) for r in req)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'scale_rows');p.add_argument('--limit',type=int,default=1)
    a=p.parse_args();print(json.dumps(build(a.project,a.out,a.limit),indent=2))
