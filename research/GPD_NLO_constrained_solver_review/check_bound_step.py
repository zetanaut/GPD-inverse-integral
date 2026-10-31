"""Exactly one outer-loop integration check per saved start, NOT a full fit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,importlib.util,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path(os.environ.get('GPD_REVIEW_ROOT',str(HERE.parent)))
T=ROOT/'completion_runs/NLO_constrained_solver';sys.path.insert(0,str(T))
from problem import Oracle,torch,kkt
from bind_solver import bind
label=int(sys.argv[1]);o=Oracle(label)
x=torch.load(T/f'start{label}/final.pt',weights_only=True)['flat'].numpy();o.x0=x.copy()
spec=importlib.util.spec_from_file_location('isolated_GN',T/'solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
o,stats=bind(m,o);records=[]
settings={**m.SETTINGS,'initial_radius':.0125,'max_iterations':1}
x1,rs=m.solve(o,kkt,lambda a,r,t:records.append(r),settings)
v0=o.values(x);v1=o.values(x1)
result={'label':label,'outer_steps':1,'initial_J':v0['J'],'final_J':v1['J'],
 'strict_feasible':bool(np.all(v1['c']<=0)), 'decrease':v0['J']-v1['J'],
 'parameter_step':float(np.linalg.norm(x1-x)), 'termination':rs['status'],
 'stats':stats,'pass':bool(np.all(v1['c']<=0) and v1['J']<v0['J']),
 'scope':'One outer-loop step integration check only. No stationarity or inference acceptance.'}
(HERE/f'results/BOUND_STEP_{label}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
