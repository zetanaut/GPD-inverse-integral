import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('GPD_REVIEW_ROOT',str(HERE.parent)))
TASK=ROOT/'completion_runs/NLO_constrained_solver';sys.path.insert(0,str(TASK))
from problem import Oracle,tensor,torch,kkt
from solver import model_coordinates,trust_step,SETTINGS
from moment_linearization import residual_and_jacobian,norm_model
label=int(sys.argv[1]);started=time.monotonic();o=Oracle(label)
x=torch.load(TASK/f'start{label}/final.pt',weights_only=True)['flat'].numpy();v=o.values(x)
ids=np.flatnonzero(v['c']>=-.01);g,R,A=o.derivatives(x,ids);model=model_coordinates(R,g,A)
t=time.monotonic();e,E=residual_and_jacobian(o.p,o.domain,tensor(x));print('jac_seconds',time.monotonic()-t,E.shape,flush=True)
idx0=int(np.flatnonzero(ids==0)[0]);er=abs(np.linalg.norm(e)-1-v['c'][0]);ger=np.max(abs(e@E/np.linalg.norm(e)-A[idx0]));print('residual_identity',er,'gradient_identity',ger,flush=True)
checks=[]
for radius in (.003125,.0125):
 step,qp=trust_step(model,v['c'][ids],.8*radius,SETTINGS)
 if step is None:checks.append({'radius':radius,'subproblem_failed':True});continue
 w=o.values(x+step)
 q={'radius':radius,'step_norm':float(np.linalg.norm(step)),'old_scalar_linear':float(v['c'][0]+A[idx0]@step),
    'residual_norm_model':norm_model(e,E,step)[0],'actual_moment_constraint':float(w['c'][0]),
    'actual_max_constraint':float(w['c'].max()),'J':w['J'],'base_J':v['J']}
 checks.append(q);print(q,flush=True)
np.savez_compressed(HERE/f'results/LOCAL_MODEL_{label}.npz',x=x,e=e,E=E,g=g,R=R,A=A,ids=ids,c=v['c'],Q=model['Q'])
# Independent central-difference scan for new residual Jacobian, not fitting.
d=np.sin(.173*np.arange(len(x))+.5);d/=np.linalg.norm(d);ad=E@d;scan=[]
for h in (1e-4,1e-5,1e-6):
 with torch.no_grad():
  ep=o.domain.mom.residual(o.p,tensor(x+h*d)).numpy().ravel()/(1e-3*np.sqrt(960))
  em=o.domain.mom.residual(o.p,tensor(x-h*d)).numpy().ravel()/(1e-3*np.sqrt(960))
 fd=(ep-em)/(2*h)
 scan.append({'h':h,'normalized_max_error':float(np.max(abs(fd-ad))/max(1,float(np.max(abs(ad))))),'relative_L2':float(np.linalg.norm(fd-ad)/max(1,np.linalg.norm(ad)))})
out={'label':label,'norm_value_error':er,'norm_gradient_max_error':ger,'trials':checks,'derivative_scan':scan,'seconds':time.monotonic()-started,'scope':'Original final fields, derivatives and local trial predictions only; no accepted neural continuation.'}
(HERE/f'results/CURVATURE_{label}.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
