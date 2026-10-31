#!/usr/bin/env python3
"""Read-only constraint curvature along the LAST RECORDED accepted step.

No optimizer, random draw, kernel construction, or candidate checkpoint is written.
The full locally cached numerical domain is required. Only the generic numerical
helper is unit-tested in this package; native-project execution is a proposed next check.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np

CASES=('ensemble_mean','replica_000','replica_001')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def directional_profile(evaluate,alphas=(0.,.25,.5,1.,2.),steps=(1e-4,1e-5)):
    """evaluate(alpha) returns the unchanged constraint vector at theta+alpha*d."""
    cache={}
    def val(a):
        a=float(a)
        if a not in cache:
            x=np.asarray(evaluate(a),dtype=float)
            if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Nonfinite/invalid constraint evaluation at alpha='+str(a))
            cache[a]=x.copy()
        return cache[a]
    c0=val(0.);derivatives=[]
    for h in steps:
        if not h>0:raise ValueError('Positive derivative steps required')
        derivatives.append((val(h)-val(-h))/(2*h))
    slope=derivatives[-1]
    rows=[dict(alpha=float(a),actual=val(a),affine=c0+float(a)*slope,
               nonlinear_minus_affine=val(a)-(c0+float(a)*slope)) for a in alphas]
    return rows,np.max(np.abs(derivatives[-1]-derivatives[0])),cache

def run(project,out):
    import torch
    project=Path(project).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Use a fresh diagnostic output directory; existing results stay immutable')
    src=project/'completion_runs/ReH_coherent_budget_check'
    lock=json.loads((src/'RUN_LOCK.json').read_text())
    # Exact original sources/inputs, not a nearby file or another publication ensemble.
    for rel,h in lock['sources'].items():
        p=project/rel
        if not p.is_file() or sha(p)!=h:raise ValueError('Missing/changed bound source: '+str(p))
    spec=importlib.util.spec_from_file_location('budget_probe_original_binding',src/'run_budget.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    out.mkdir(parents=True)
    frozen={'run_lock_sha256':sha(src/'RUN_LOCK.json'),'probe_sha256':sha(__file__),
            'purpose':'Read-only recorded-step curvature, not solver updates or physical intervals.',
            'alphas':[0,.25,.5,1,2],'central_difference_steps':[1e-4,1e-5]}
    (out/'PLAN.json').write_text(json.dumps(frozen,indent=2)+'\n')
    for case in CASES:
        t0=time.monotonic();obsfile=project/'completion_runs/ReH_coherent_GPD_pilot/prepared'/f'{case}.npz'
        with np.load(obsfile,allow_pickle=False) as z:inputs={k:z[k].copy() for k in z.files}
        p=mod.problem(inputs);d=mod.BranchDomain('enriched')
        a=torch.load(src/case/'STAGE_19.pt',weights_only=True,map_location='cpu')
        b=torch.load(src/case/'STAGE_20.pt',weights_only=True,map_location='cpu')
        if a['stage']!=19 or b['stage']!=20:raise ValueError('Incorrect saved stage')
        x=a['flat'].detach().clone();step=b['flat'].detach()-x
        if x.numel()!=3867 or not torch.isfinite(step).all():raise ValueError('Unexpected network')
        journal=json.loads((src/case/'SOLVER.json').read_text());last=journal['history'][-1]
        recorded=np.asarray(last['constraints'],dtype=float)
        def evaluate(alpha):
            with torch.no_grad():return d.constraints(p,x+float(alpha)*step).cpu().numpy()
        profiles,derivative_change,cache=directional_profile(evaluate)
        max_replay=float(np.max(np.abs(cache[0.]-recorded)))
        # This is a source-arithmetic check; do not reinterpret it as a fit-quality cut.
        if max_replay>1e-6:raise ValueError('Recorded starting constraints do not replay: '+str(max_replay))
        indices=set(last['working_indices'])
        for h in journal['history']:
            for trial in h['trials']:
                if not trial['accepted']:indices.add(int(np.argmax(trial['constraints'])))
        inds=sorted(indices);rows=[]
        for r in profiles:
            alpha=r['alpha']
            with torch.no_grad():parts=p.components(x+alpha*step).cpu().numpy()
            rows.append(dict(alpha=alpha,J=float(sum(parts)),data_RMS=float(np.sqrt(parts[0])),
                             moment_constraint=float(r['actual'][0]),maximum_constraint=float(np.max(r['actual'])),
                             selected=[dict(index=i,name=d.names[i],actual=float(r['actual'][i]),affine=float(r['affine'][i]),
                                            nonlinear_minus_affine=float(r['nonlinear_minus_affine'][i])) for i in inds]))
        result=dict(case=case,stage19_sha256=sha(src/case/'STAGE_19.pt'),stage20_sha256=sha(src/case/'STAGE_20.pt'),
                    step_norm=float(torch.linalg.vector_norm(step)),starting_constraint_replay_max=max_replay,
                    directional_derivative_refinement_max=float(derivative_change),profiles=rows,
                    n_constraint_evaluations=len(cache),elapsed_seconds=time.monotonic()-t0,
                    scope='Evaluations along a saved direction only; alpha=2 is NOT an accepted fitted state or a full-forward certificate.')
        (out/f'{case}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(case,'diagnostic complete',round(result['elapsed_seconds'],2),'s',flush=True)
        del p,d

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.project,args.out)
