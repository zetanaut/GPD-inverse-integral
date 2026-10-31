import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path(os.environ.get('GPD_REVIEW_ROOT',str(HERE.parent)))
sys.path.insert(0,str(ROOT/'completion_runs/NLO_constrained_solver'))
from problem import Oracle,tensor,torch
from residual_subproblem import solve_model
seed=int(sys.argv[1]);rtime=time.monotonic();o=Oracle(seed)
z=np.load(HERE/f'results/LOCAL_MODEL_{seed}.npz');x=z['x'];c=z['c'];ids=z['ids'];Q=z['Q'];other=ids!=0
v=o.values(x);records=[]
for radius in (.0025,.01):
 t=time.monotonic();step,info=solve_model(z['g'],z['R'],Q,z['A'][other],c[ids[other]],z['e'],z['E'],radius)
 info.update(seconds=time.monotonic()-t,requested_radius=radius)
 trial=x+step;values=o.values(trial)
 info['before_correction']={'J':values['J'],'decrease':v['J']-values['J'],'max_constraint':float(max(values['c'])),'moment_constraint':float(values['c'][0])}
 corrections=[]
 # Same bounded normal correction policy as the submitted solver.
 for k in range(3):
  if max(values['c'])<=0:break
  ci=np.flatnonzero(values['c']>-1e-9);B=o.constraint_rows(trial,ci);norm=np.linalg.norm(B,axis=1)
  if np.any(norm==0):break
  Bn=B/norm[:,None];rhs=-(values['c'][ci]+1e-9)/norm
  ds=Bn.T@np.linalg.lstsq(Bn@Bn.T,rhs,rcond=1e-12)[0]
  if np.linalg.norm(ds)>.5*np.linalg.norm(step) or np.linalg.norm(trial+ds-x)>1.25*radius:break
  trial=trial+ds;values=o.values(trial);corrections.append({'norm':float(np.linalg.norm(ds)),'maximum_constraint':float(max(values['c']))})
 info['after_correction']={'J':values['J'],'decrease':v['J']-values['J'],'max_constraint':float(max(values['c'])),'strict_feasible':bool(np.all(values['c']<=0)),'total_step_norm':float(np.linalg.norm(trial-x)),'SOC':corrections}
 records.append(info);torch.save({'flat':tensor(trial),'diagnostic_only':True},HERE/f'results/TRIAL_{seed}_{radius}.pt')
 print(json.dumps(info,indent=2),flush=True)
(HERE/f'results/LOCAL_STEPS_{seed}.json').write_text(json.dumps({'seed':seed,'base_J':v['J'],'records':records,'seconds':time.monotonic()-rtime,'scope':'Two local convex model probes, not an optimization campaign or accepted estimator.'},indent=2)+'\n')
