import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path(os.environ.get('GPD_REVIEW_ROOT',str(HERE.parent)))
T=ROOT/'completion_runs/NLO_constrained_solver';sys.path.insert(0,str(T))
from problem import Oracle,tensor,torch,original
seed=int(sys.argv[1]);kind=sys.argv[2]
if kind not in ('saved','trial'):raise ValueError(kind)
state=T/f'start{seed}/final.pt' if kind=='saved' else HERE/f'results/TRIAL_{seed}_0.01.pt'
o=Oracle(seed);x=torch.load(state,weights_only=True)['flat'];out=HERE/f'results/full_{seed}_{kind}';out.mkdir(exist_ok=True)
started=time.monotonic();o.p.set_model(x);records=[];new=0
for point in original.common.operator_directories():
 for q in original.Q2S:
  filename=f'xi{point["xi"]}_Q2{q}.json';target=out/filename
  if target.exists():d=json.loads(target.read_text())
  else:
   if new>=5:
    print(json.dumps({'stage':'partial','new_points':new,'remaining':'resume same command'}));raise SystemExit(0)
   d=original.preflight.validate_point(point,q,'d000_feasible_domain',o.p.model)
   original.write(target,d);new+=1
  rec={k:d[k] for k in ('xi','Q2','pass','gates','CFF_estimated_total','CFF_limits')}
  rec.update(field_totals={k:v['estimated_total'] for k,v in d['fields'].items()},file=filename,sha256=original.sha(target));records.append(rec)
f={'pass':all(r['pass'] for r in records),'records':records,'all_15_points':len(records)==15}
m=original.moment_validation(o.p,x)
error=0.
if kind=='saved':
 orig=json.loads((T/f'start{seed}/FULL_FORWARD.json').read_text())
 for a,b in zip(f['records'],orig['records']):
  error=max(error,float(np.max(abs(np.array(a['CFF_estimated_total'])-np.array(b['CFF_estimated_total'])))))
r={'seed':seed,'kind':kind,'full_forward':f,'moments':m,'saved_CFF_budget_max_diff':error,'seconds':time.monotonic()-started,
 'scope':'Fresh original full-forward and off-node moment validation; not training.'}
(HERE/f'results/FULL_{seed}_{kind}.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'seed':seed,'kind':kind,'all_forward_pass':f['pass'],'num_points':len(f['records']),'moment_pass':m['pass'],'moment_RMS':m['grids']['independent160']['quadrature_RMS']['384'],'error':error,'seconds':r['seconds']},indent=2))
