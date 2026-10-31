"""Bounded preallocation checks for the exact first registered task; no fit."""
import json,math,sys,time,resource,os,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from closure.solver_binding import problem_and_domain,oracle_at,isolated_solver
from closure.io import sha256

torch.set_num_threads(1)
p,d=problem_and_domain('ideal_L0_Z1_s31')
assert p.x0.numel()==3867 and p.seed==31 and len(d.entries)==45 and len(d.names)==496
assert all(v.requires_grad for v in p.model.parameters())
assert 'truths' not in p.study and 'observation_rule' not in p.study
start=time.monotonic()
with torch.no_grad():
 r=p.residual(p.x0);J=float(p.objective(p.x0));c=d.constraints(p,p.x0)
 pred=p.forward(lambda x,xi,t:p.field(p.x0,x,xi,t)).index_select(1,p.components_index)
 direct=float(((((pred-p.y)/p.sigma)*torch.sqrt(p.mass[:,None]))**2).sum()+p.p(lambda x,xi,t:p.field(p.x0,x,xi,t)))
assert abs(J-float(r.square().sum()))<1e-10
assert abs(J-direct)<1e-10
assert len(c)==len(d.names) and torch.isfinite(c).all()
constraints_seconds=time.monotonic()-start
x=p.x0.detach().requires_grad_(True)
t=time.monotonic();g=torch.autograd.grad(p.objective(x),x)[0].detach();R=p.jacobian(p.x0);Rsec=time.monotonic()-t
assert R.shape[1]==3867 and R.shape[0]==len(r)
error=float((g-2*R.T@r).abs().max());assert error<1e-7
rng=np.random.default_rng(20261007);direction=torch.tensor(rng.normal(size=3867),dtype=torch.float64);direction/=direction.norm()
fd=float((p.objective((p.x0+1e-5*direction).detach())-p.objective((p.x0-1e-5*direction).detach()))/(2e-5))
ad=float(g@direction);assert abs(fd-ad)<1e-7*max(1,abs(ad))
t=time.monotonic();e,E=d.mom.residual_and_jacobian(p,p.x0);Msec=time.monotonic()-t
assert E.shape==(len(e),3867)
assert abs(np.linalg.norm(e)-1-float(c[0]))<1e-9
coeff=p.p.coefficients(lambda xx,xi,t:p.field(p.x0,xx,xi,t))
assert len({item['t'] for item in coeff})==13 and len(coeff)==13*len(p.p.spec)
assert all(len(item['coefficients'])==len(item['powers']) for item in coeff)
seed43,_=problem_and_domain('ideal_L0_Z1_s43')
assert not torch.equal(seed43.x0,p.x0)
seed31_again,_=problem_and_domain('ideal_L0_Z1_s31')
assert torch.equal(seed31_again.x0,p.x0)
o=oracle_at(p,d,p.x0);solver,o,stats,settings=isolated_solver(o)
v=o.values(p.x0.numpy())
assert abs(v['J']-J)<1e-9 and np.array_equal(v['c'],c.detach().numpy())
rec={'status':'BOUNDED_BINDING_CHECKED_NO_FIT','task':'ideal_L0_Z1_s31','parameter_count':3867,'residual_count':len(r),'constraint_count':len(c),'J':J,'direct_weighted_J':direct,'data_J':float(p.components(p.x0)[0]),'moment_J':float(p.components(p.x0)[1]),'constraint_max':float(c.max()),'constraint_seconds':constraints_seconds,'residual_jacobian_seconds':Rsec,'moment_predictive_jacobian_seconds':Msec,'gradient_reconstruction_max_abs':error,'directional_AD':ad,'directional_FD':fd,'moment_predictive_residual_count':len(e),'free_moment_coefficient_rows':len(coeff),'seed31_initial_sha256':hashlib.sha256(p.x0.numpy().tobytes()).hexdigest(),'seed43_distinct':True,'same_seed_reproducible':True,'oracle_J':v['J'],'solver_max_iterations':settings['max_iterations'],'solver_initial_radius':settings['initial_radius'],'observed_sha256':sha256(ROOT/'prepared/observed/ideal_L0_Z1.npz'),'new_scientific_fits':0,'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'readiness';out.mkdir(exist_ok=True)
(out/'LOCAL_BINDING_CHECK.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec),flush=True)
