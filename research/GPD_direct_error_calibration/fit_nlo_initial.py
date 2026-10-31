"""Bounded DEVELOPMENT estimator. No calibration data, truth or warm checkpoint is read.
Imports immutable original FiLM/objective/constraints and residual-norm solver.
Usage: GPD_REVIEW_ROOT=... python fit_nlo.py --input ... --out ...
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, math, json, time, hashlib, argparse, importlib.util
from pathlib import Path
ROOT=Path(os.environ['GPD_REVIEW_ROOT']).resolve()
sys.path.insert(0,str(ROOT/'completion_runs/NLO_constrained_solver'))
import problem as old
from problem import np,torch,tensor,common
sys.path.insert(0,str(ROOT/'GPD_NLO_constrained_solver_review'))
import bind_solver
from domain import Domain, moment_validation, full_forward, write, save

def array_hash(a):
 a=np.asarray(a,dtype='<f8');return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()

class Problem(old.Problem):
 def __init__(self,y,cov,seed):
  # Deliberately DO NOT invoke old constructor (it reads a handcrafted checkpoint).
  with torch.random.fork_rng():
   torch.manual_seed(seed);self.model=common.ReferenceNetwork('film').double()
  self.spec=[(n,p.shape,p.numel()) for n,p in self.model.named_parameters()]
  self.x0=torch.cat([p.detach().flatten() for p in self.model.parameters()]);assert self.x0.numel()==3867
  self.y=tensor(y).flatten();self.cov=tensor(cov)
  if self.y.numel()!=18 or self.cov.shape!=(18,18): raise ValueError('This DEVELOPMENT adapter requires the original 18-scalar geometry')
  if not torch.isfinite(self.y).all() or not torch.isfinite(self.cov).all() or not torch.equal(self.cov,self.cov.T):raise ValueError('Invalid data/covariance')
  self.chol=torch.linalg.cholesky(self.cov)
  with np.load(ROOT/'completion_runs/NLO_feasible_domain/inputs/fit_bank.npz',allow_pickle=False) as z:
   self.bank={k:tensor(z[k]) for k in ('x','xi','rows')}
  self.p=common.projector(48);self.complements=[]
  for _,N,_ in self.p.spec:
   P=common.poly.design(self.p.xi.numpy(),N);Q=np.linalg.qr(P,mode='complete')[0]
   self.complements.append(tensor(Q[:,P.shape[1]:]))

class Oracle(old.Oracle):
 def __init__(self,p,d,x0):
  self.p=p;self.domain=d;self.n=p.x0.numel();self.names=d.names;self.x0=x0.copy();self.cached=None
  self.counts={k:0 for k in ('evaluations','objective_gradients','residual_jacobians','constraint_jacobian_builds','constraint_jacobian_rows')}
  self.timers={k:0. for k in ('evaluation_seconds','residual_jacobian_seconds','constraint_jacobian_seconds')}

def targets(p,x,order):
 z,w=np.polynomial.legendre.leggauss(order);res=[]
 for lo,hi,moment in ((.2,.4,False),(0.,1.,True)):
  xx=(lo+hi)/2+(hi-lo)*z/2;ww=w*(hi-lo)/2
  h=p.field(x,tensor(xx)[None],tensor([.275]))[0];charge=(2/9)*(h[0]+h[1])
  res.append(float((charge*tensor(ww*(xx if moment else np.full(len(xx),5.)))).sum()))
 return res

def restore_head(p,d,x):
 # A predeclared amplitude reduction creates a feasible start only; not a
 # confidence-bound prior or a change of feasible-set thresholds.
 with torch.no_grad():
  c=d.constraints(p,x).numpy();r=float(c[0]+1)
  if not np.isfinite(c).all():return None,{'reason':'nonfinite'}
  if np.any(c[1:]>0):return None,{'reason':'nonmoment_constraint','max':float(c[1:].max())}
  s=min(1.,.8/max(r,1e-300));out=x.clone();i=0
  for n,shape,size in p.spec:
   if n in ('output.weight','output.bias'):out[i:i+size]*=s
   i+=size
  new=d.constraints(p,out).numpy()
  if not np.isfinite(new).all() or np.any(new>0):return None,{'reason':'restoration_failed','max':float(new.max())}
  return out,{'factor':s,'original_moment_ratio':r,'max_constraint':float(new.max())}

def run(y,cov,out):
 out=Path(out)
 if (out/'RESULT.json').exists():raise FileExistsError('Immutable result already exists')
 out.mkdir(parents=True,exist_ok=True);torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 d=Domain();start=time.monotonic();candidates=[];logs=[]
 for seed in (31,43):
  p=Problem(y,cov,seed);random=p.x0.clone();x=random.clone().requires_grad_(True)
  opt=torch.optim.Adam([x],lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0)
  for step in range(400):
   opt.zero_grad(set_to_none=True);loss=p.objective(x)
   if not torch.isfinite(loss):raise FloatingPointError('Adam nonfinite')
   loss.backward();opt.step()
  prepared=[];rest=[]
  for name,state in [('random',random),('Adam400',x.detach())]:
   r,info=restore_head(p,d,state);rest.append({'name':name,**info})
   if r is not None:
    with torch.no_grad():v=float(p.objective(r))
    prepared.append((v,name,r))
  if not prepared:
   logs.append({'seed':seed,'status':'no_feasible_start','restoration':rest});continue
  prepared.sort(key=lambda item:(item[0],item[1]));v,name,x0=prepared[0]
  candidates.append({'seed':seed,'stage':0,'x':x0.clone(),'J':v,'p':p})
  spec=importlib.util.spec_from_file_location('bounded_outer_'+str(seed),ROOT/'completion_runs/NLO_constrained_solver/solver.py')
  solver=importlib.util.module_from_spec(spec);spec.loader.exec_module(solver)
  oracle=Oracle(p,d,x0.numpy());oracle,stats=bind_solver.bind(solver,oracle)
  settings=dict(solver.SETTINGS,max_iterations=60,max_evaluations=12000,optimization_seconds=float('inf'),initial_radius=.01)
  def record(a,rec,trial):
   k=rec['iteration']+1
   if k in (20,40,60):
    with torch.no_grad():j=float(p.objective(tensor(a)))
    candidates.append({'seed':seed,'stage':k,'x':tensor(a).clone(),'J':j,'p':p})
   print(json.dumps({'seed':seed,'iteration':k,'J':rec['J'],'accepted':rec.get('accepted'),'seconds':time.monotonic()-start}),flush=True)
  last,diag=solver.solve(oracle,old.kkt,record,settings)
  # A deterministic early termination is a legitimate finite-budget candidate,
  # not a claim of first-order stationarity. No time-dependent partial result.
  if not any(c['seed']==seed and np.array_equal(c['x'].numpy(),last) for c in candidates):
   candidates.append({'seed':seed,'stage':len(diag['history']),'x':tensor(last),'J':float(p.objective(tensor(last))),'p':p})
  write(out/f'seed{seed}_solver.json',diag)
  logs.append({'seed':seed,'status':diag['status'],'restoration':rest,'initial_candidate':name,'initial_hash':array_hash(random.numpy()),'stats':stats})
 # Validate candidates in deterministic J/seed/stage order; bounded to four.
 candidates.sort(key=lambda c:(c['J'],c['seed'],c['stage']))
 validations=[];selected=None
 for index,c in enumerate(candidates[:4]):
  dest=out/f'candidate{index}';dest.mkdir(exist_ok=True);p,x=c['p'],c['x']
  with torch.no_grad():
   constraints=d.constraints(p,x).numpy();m=moment_validation(p,x);ta=targets(p,x,192);tb=targets(p,x,384)
  save(dest/'checkpoint.pt',{'state_dict':{n:v.detach().clone() for n,v in p.parameters(x).items()},'flat':x.clone(),'seed':c['seed'],'stage':c['stage']})
  write(dest/'MOMENTS.json',m)
  gate=bool(np.isfinite(constraints).all() and np.all(constraints<=0) and m['pass'] and max(abs(np.array(ta)-tb))<=1e-7)
  f={'pass':False,'not_run':'precheck_failed'}
  if gate:f=full_forward(p,x,dest/'full_forward')
  with torch.enable_grad():
   a=x.detach().requires_grad_(True);g=torch.autograd.grad(p.objective(a),a)[0]
  item={'seed':c['seed'],'stage':c['stage'],'J':c['J'],'max_constraint':float(constraints.max()),'moment_pass':m['pass'],
        'forward_pass':f['pass'],'targets':tb,'target_refinement_max':float(max(abs(np.array(ta)-tb))),
        'raw_gradient_L2':float(torch.linalg.vector_norm(g)),'checkpoint':str((dest/'checkpoint.pt').relative_to(out)),
        'pass':bool(gate and f['pass'])}
  write(dest/'FORWARD.json',f);write(dest/'CHECK.json',item);validations.append(item)
  if item['pass']:selected=item;break
 result={'status':'finite_budget_feasible' if selected else 'abstain','selected':selected,'validations':validations,'starts':logs,
         'data_sha256':array_hash(y),'covariance_sha256':array_hash(cov),'seconds':time.monotonic()-start,
         'stationarity_required':False,'scope':'development finite-budget estimator; not global optimization or calibrated GPD coverage'}
 write(out/'RESULT.json',result);return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
 with np.load(args.input,allow_pickle=False) as data:y=data['y'].copy();cov=data['covariance'].copy()
 print(json.dumps(run(y,cov,args.out),indent=2),flush=True)
