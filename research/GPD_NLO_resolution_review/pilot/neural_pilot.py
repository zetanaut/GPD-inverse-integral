"""Original FiLM/free-moment inverse with saved NLO rows and task-04 solver."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from common import *
import math, time, types
from scipy.linalg import cho_factor, cho_solve
from preflight import op, cached_row, validate_point

# Preserve the tested optimizer and exact residual/Jacobian algorithms. Only its
# legacy import is redirected; no old bank, observation or checkpoint is loaded.
solver_source=(TASK04/'solver.py').read_text()
old='from src.core import ReferenceNetwork, Forward, load_bank, projector, atomic_json, digest, model_diagnostics'
new='from common import ReferenceNetwork, projector, digest, write as atomic_json'
assert solver_source.count(old)==1
solver=types.ModuleType('accepted_task04_solver');solver.__file__=str(TASK04/'solver.py')
sys.modules[solver.__name__]=solver
exec(compile(solver_source.replace(old,new),str(TASK04/'solver.py'),'exec'),solver.__dict__)

class NLOForward(torch.nn.Module):
    def __init__(self, q2s, method='weak10'):
        super().__init__();points=operator_directories();xx=[];rr=[]
        for point in points:
            d=point['directories'][method];b,*_=op(d)
            xx.append(b.nodes);rr.append(np.stack([cached_row(d,q) for q in q2s]))
        self.register_buffer('x',torch.tensor(np.stack(xx)))
        self.register_buffer('xi',torch.tensor([p['xi'] for p in points]))
        self.register_buffer('rows',torch.tensor(np.stack(rr)))
    def forward(self,model):
        return torch.einsum('sqrcn,scn->sqr',self.rows,model(self.x,self.xi))

def targets(truth,q2s):
    return torch.tensor([[json.loads((WORK/'preflight'/f'{truth}_xi{p["xi"]}_Q2{q}.json').read_text())['checks']['strong10']['CFF']
                          for q in q2s] for p in operator_directories()])

class Problem(solver.ResidualProblem):
    def __init__(self,truth,seed):
        self.key=f'{truth}_seed{seed}';self.ckpath=WORK/f'initial_seed{seed}.pt'
        self.checkpoint=torch.load(self.ckpath,weights_only=True)
        self.model=ReferenceNetwork('film').double();self.model.load_state_dict(self.checkpoint['state_dict'])
        self.spec=[(n,p.shape,p.numel()) for n,p in self.model.named_parameters()]
        self.x0=torch.cat([p.detach().flatten() for p in self.model.parameters()])
        self.forward=NLOForward([2.3,4.7,9.3]);self.y=targets(truth,[2.3,4.7,9.3]).flatten()
        # This factor is numerical normalization, NOT an invented covariance.
        self.chol=.04*torch.eye(len(self.y));self.p=projector();self.lam=1.
        self.complements=[]
        for _,N,_ in self.p.spec:
            P=poly.design(self.p.xi.numpy(),N);Q=np.linalg.qr(P,mode='complete')[0]
            self.complements.append(torch.tensor(Q[:,P.shape[1]:]))
        self.npoly=self.p.xi.numel()*len(self.p.spec)

def moment_diagnostics(model):
    xf,xd=helpers.moment_grids();pf=projector(96,xf);pd=projector(96,xd);pc=projector(48,xd)
    with torch.no_grad(): mf=pf.moments(model).numpy();md=pd.moments(model).numpy();mc=pc.moments(model).numpy()
    entries=[]
    for i,(ch,n,_) in enumerate(pf.spec):
        a=np.linalg.lstsq(poly.design(xf,n),mf[:,i],rcond=None)[0]
        r=(md[:,i]-poly.design(xd,n)@a)/pf.scales[i].item()
        entries.append({'channel':ch,'spin':n,'coefficients':a.tolist(),'rms':float(np.sqrt(np.mean(r*r))),
            'max_abs':float(max(abs(r))),'quadrature_rms':float(np.sqrt(np.mean(((mc[:,i]-md[:,i])/pf.scales[i].item())**2)))})
    rms=float(np.sqrt(np.mean([v['rms']**2 for v in entries])))
    return {'entries':entries,'combined_rms':rms,'pass':rms<=1e-3}

def truth_values(x,xi,dterm=False):
    # Exactly the two frozen task03 functions; diagnostics only after fitting.
    out=np.array([2*x*(1-x*x)**2*(1+.2*xi*xi),.5*x*(1-x*x)**2*(1-.1*xi*xi),.8*(1-x*x)**2*(1+.1*xi*xi)])
    if dterm:
        z=x/xi;e=np.where(abs(z)<1,(1-z*z)**2,0.)
        out+=np.array([.09*z*e,-.02*z*e,.03*xi*e])
    return out

def truth_diagnostics(model,truth):
    errors=[]
    for xi in (.135,.275,.455):
        x,w=poly.make_grid([xi],128);x=x[0];w=w[0]
        with torch.no_grad():h=model(torch.tensor(x[None]),torch.tensor([xi]))[0].numpy()
        ref=truth_values(x,xi,truth=='Dterm_added')
        rel=np.sqrt((h-ref)**2@w)/np.maximum(1e-15,np.sqrt(ref**2@w))
        c=2/9*(h[0]+h[1]);cr=2/9*(ref[0]+ref[1])
        errors.append({'xi':xi,'component_relative_L2':rel.tolist(),'charge_relative_L2':float(np.sqrt((c-cr)**2@w)/np.sqrt(cr**2@w))})
    return {'meaning':'after-fit recovery diagnostics only; never optimizer selection or tuning','Q02':2.,'points':errors}

def warm_start(p,out):
    theta=torch.nn.Parameter(p.x0.clone());t=time.monotonic();history=[]
    opt=torch.optim.Adam([theta],lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0)
    for i in range(400):
        opt.zero_grad(set_to_none=True);f=p.original_objective(theta);f.backward();opt.step()
        if i%100==0:history.append({'step':i,'J_before':float(f.detach())})
    opt=torch.optim.LBFGS([theta],max_iter=400,max_eval=500,history_size=40,tolerance_grad=1e-10,tolerance_change=1e-13,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(set_to_none=True);f=p.original_objective(theta);f.backward();return f
    opt.step(closure);opt.zero_grad(set_to_none=True);f=closure()
    state=opt.state[theta];p.x0=theta.detach().clone()
    torch.save({'flat':p.x0,'initial_sha256':digest(p.ckpath)},out/'warm_start.pt')
    write(out/'warm_start.json',{'Adam_steps':400,'Adam_history':history,'LBFGS_iterations':int(state['n_iter']),
       'LBFGS_evaluations':int(state['func_evals']),'J':float(f.detach()),'gradient_l2':float(theta.grad.norm()),'seconds':time.monotonic()-t})

def finish(p,x,out):
    t=time.monotonic();hist=[]
    # Verbatim numerical schedule of task04/stationarity.py, redirected outputs.
    for it in range(5):
        xx=x.detach().requires_grad_(True);value=p.original_objective(xx);g=torch.autograd.grad(value,xx)[0]
        gn=float(g.norm().detach());old=float(value.detach())
        if gn<=1e-6:break
        r=p.residual(x).detach().numpy();J=p.jacobian(x).detach().numpy();K=J@J.T
        mu=1e-4*max(np.max(np.diag(K)),1e-30)
        d=-J.T@cho_solve(cho_factor(K+mu*np.eye(len(r)),lower=True,check_finite=False),r,check_finite=False)
        y=x+torch.tensor(d);new=float(p.original_objective(y).detach());accepted=bool(np.isfinite(new) and new<old)
        hist.append({'step':it,'J_before':old,'J_after':new,'gradient_before':gn,'mu':float(mu),'accepted':accepted})
        if not accepted:break
        x=y
    xx=x.detach().requires_grad_(True);value=p.original_objective(xx);g=torch.autograd.grad(value,xx)[0]
    p.set_model(x);torch.save({'state_dict':p.model.state_dict(),'flat':x.detach(),'key':p.key,'initial_sha256':digest(p.ckpath)},out/'final.pt')
    r=p.residual(x).detach()
    result={'J':float(value.detach()),'gradient_l2':float(g.norm()),'optimizer_pass':bool(float(g.norm())<=1e-6),
      'data_loss':float(r[:p.y.numel()].square().sum()),'poly_loss':float(r[p.y.numel():].square().sum()),
      'history':hist,'seconds':time.monotonic()-t}
    write(out/'stationarity.json',result)
    return result

def interface_checks(p):
    # Equivalence of original projector and QR-complement residual, and exact
    # full Jacobian consistency. No truth recovery is examined here.
    theta=p.x0.detach().requires_grad_(True);r=p.residual(theta);J=p.jacobian(theta)
    orig=p.original_objective(theta);g=torch.autograd.grad(orig,theta)[0]
    out={'objective_absolute_difference':abs(float(orig.detach()-r.square().sum().detach())),
         'gradient_max_abs_difference':float((g-2*J.T@r.detach()).abs().max())}
    out['pass']=out['objective_absolute_difference']<=1e-10*max(1,float(orig.detach())) and out['gradient_max_abs_difference']<=1e-8
    return out

def run():
    plan=prepare()
    for name in ('TRUTH_PREFLIGHT.json','INITIAL_PREFLIGHT.json'):
        assert json.loads((WORK/name).read_text())['pass'], 'Preflight failure: no optimization allowed'
    for truth in plan['truths']:
        for seed in plan['seeds']:
            key=f'{truth}_seed{seed}';out=WORK/'fits'/key;out.mkdir(parents=True,exist_ok=True)
            if (out/'METRICS.json').exists():continue
            if (out/'warm_start.pt').exists():raise RuntimeError('Partial fit exists; inspect before any resume; no automatic extra budget')
            p=Problem(truth,seed);check=interface_checks(p);write(out/'INTERFACE_CHECKS.json',check);assert check['pass']
            print('START',key,flush=True);start=time.monotonic();warm_start(p,out)
            optimizer,x=solver.solve(p,solver.Settings(),out)
            result=finish(p,x,out);result.update(key=key,truth=truth,seed=seed)
            result['moments']=moment_diagnostics(p.model);result['truth_diagnostics']=truth_diagnostics(p.model,truth)
            rows=[]
            for point in operator_directories():
                for q2 in plan['training_Q2']+plan['heldout_Q2']:
                    nr=validate_point(point,q2,key,p.model);write(out/f'numerical_xi{point["xi"]}_Q2{q2}.json',nr)
                    target=json.loads((WORK/'preflight'/f'{truth}_xi{point["xi"]}_Q2{q2}.json').read_text())['checks']['strong10']['CFF']
                    nominal=np.array(nr['checks']['weak10']['CFF']);ref=np.array(nr['checks']['strong10']['independent_hard']['reference'])
                    rows.append({'xi':point['xi'],'Q2':q2,'split':'train' if q2 in plan['training_Q2'] else 'heldout_scale',
                        'target':target,'nominal_prediction':nominal.tolist(),'reference_prediction':ref.tolist(),
                        'nominal_residual':(nominal-target).tolist(),'reference_residual':(ref-target).tolist(),'numerical_pass':nr['pass']})
            result['CFF_metrics']=rows;result['gates']={'solver':result['optimizer_pass'],'polynomiality':result['moments']['pass'],'numerics':all(r['numerical_pass'] for r in rows)}
            result['pass']=all(result['gates'].values());result['seconds_total']=time.monotonic()-start
            write(out/'METRICS.json',result);print('COMPLETE',key,result['gates'],flush=True)
    results=[json.loads((WORK/'fits'/f'{truth}_seed{seed}'/'METRICS.json').read_text()) for truth in plan['truths'] for seed in plan['seeds']]
    write(WORK/'FIT_SUMMARY.json',{'runs':results,'pass':all(r['pass'] for r in results),'symbolic_selection':'OFF'})

if __name__=='__main__':run()
