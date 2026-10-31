"""One prospectively bounded Powell-Hestenes-Rockafellar inequality AL.

Internal penalties and multipliers are solver mechanics. The reported objective
is always the original J. Neither an inner-solver success nor a multiplier fit
is an acceptance test. No old profile/scoring wrapper is imported.
"""
import time,json
import numpy as np
from scipy.optimize import minimize
from kkt_check import check_kkt

SETTINGS=dict(outer_iterations=24,inner_iterations=80,inner_evaluations=160,
              maxcor=30,maxls=20,inner_gtol=1e-9,inner_ftol=2.22e-15,
              initial_penalty=1.,penalty_factor=5.,penalty_max=1e8,
              required_violation_reduction=.5,wall_seconds=6600,
              gradient_tolerance=1e-6,active_tolerance=1e-6,
              complementarity_tolerance=1e-8)

class BudgetStop(Exception):pass

def run_al(x0,oracle,record,settings=SETTINGS):
    """oracle(x, weights) -> J,c,gJ,gradient(J+weights*c).
    oracle(x,None) returns J,c and the gradient of the AL supplied by
    oracle.set_augmented(lam,rho). All derivatives use original x coordinates.
    """
    x=np.array(x0,float,copy=True);start=time.monotonic();total=0;history=[]
    j,c,g,_=oracle(x,np.zeros(oracle.n_constraints))
    lam=np.zeros_like(c);rho=settings['initial_penalty'];previous=np.inf
    status='outer_iteration_budget';last_failed=None
    for outer in range(settings['outer_iterations']):
        if time.monotonic()-start>=settings['wall_seconds']:
            status='wall_budget';break
        oracle.set_augmented(lam,rho);calls=0;last=x.copy();failed=None
        def f(z):
            nonlocal calls,total,failed
            if time.monotonic()-start>=settings['wall_seconds']:
                failed=z.copy();raise BudgetStop('wall_budget')
            if calls>=settings['inner_evaluations']:
                failed=z.copy();raise BudgetStop('inner_evaluation_budget')
            calls+=1;total+=1
            val,der=oracle.augmented(z)
            if not np.isfinite(val) or not np.isfinite(der).all():
                failed=z.copy();raise FloatingPointError('nonfinite AL value or derivative')
            return val,der
        def callback(z):
            nonlocal last
            last=z.copy()
        reason=None
        try:
            result=minimize(f,x,jac=True,method='L-BFGS-B',callback=callback,
                           options={'maxiter':settings['inner_iterations'],
                                    'maxfun':settings['inner_evaluations'],
                                    'maxcor':settings['maxcor'],'maxls':settings['maxls'],
                                    'gtol':settings['inner_gtol'],'ftol':settings['inner_ftol']})
            x=result.x.copy()
            inner={'success':bool(result.success),'message':str(result.message),
                   'nit':int(result.nit),'nfev':int(result.nfev)}
        except (BudgetStop,FloatingPointError) as e:
            x=last;reason=str(e);inner={'success':False,'message':reason,'nfev':calls}
        # Fresh, independent-of-LBFGS objective gradient at returned iterate.
        j,c,g,_=oracle(x,np.zeros(oracle.n_constraints))
        lam=np.maximum(0.,lam+rho*c)
        _,_,_,lag=oracle(x,lam)
        violation=float(np.max(np.maximum(c,0),initial=0.))
        complement=float(np.max(np.abs(lam*c),initial=0.))
        rec={'outer':outer,'J':float(j),'constraint_values':c.tolist(),
             'raw_gradient_L2':float(np.linalg.norm(g)),
             'AL_multipliers':lam.tolist(),'AL_stationarity_L2':float(np.linalg.norm(lag)),
             'complementarity_max':complement,'maximum_excess':violation,
             'penalty':rho,'inner':inner,'evaluations':calls,'total_evaluations':total,
             'seconds':time.monotonic()-start,'strictly_feasible':bool(np.all(c<=0.))}
        history.append(rec);record(x,rec,failed)
        # Strict primal feasibility is mandatory; no tolerance is added to c<=0.
        if np.all(c<=0.) and np.linalg.norm(lag)<=settings['gradient_tolerance'] and complement<=settings['complementarity_tolerance']:
            status='necessary_AL_first_order_candidate';break
        if reason=='wall_budget':status='wall_budget';break
        if reason and reason.startswith('nonfinite'):status='nonfinite';break
        if violation>settings['required_violation_reduction']*previous:
            rho=min(settings['penalty_max'],settings['penalty_factor']*rho)
        previous=violation
    return x,{'status':status,'seconds':time.monotonic()-start,
              'total_optimization_evaluations':total,'history':history,
              'multipliers':lam.tolist()}

def necessary_kkt(g,c,A,indices,names):
    """Only rows with c>=-active_tol are needed in multiplier fit.
    Inactive rows have exactly zero fitted multipliers; never fabricate their
    unevaluated Jacobians. Full c is separately checked for primal feasibility.
    """
    out=check_kkt(g,c[indices],A,gradient_tolerance=SETTINGS['gradient_tolerance'],
                  active_tolerance=SETTINGS['active_tolerance'],
                  complementarity_tolerance=SETTINGS['complementarity_tolerance'])
    out['full_primal_feasible']=bool(np.all(c<=0.))
    out['full_maximum_excess']=float(np.max(np.maximum(c,0),initial=0.))
    out['evaluated_jacobian_indices']=list(map(int,indices))
    out['evaluated_constraint_names']=[names[i] for i in indices]
    sv=np.linalg.svd(A,compute_uv=False) if len(A) else np.empty(0)
    threshold=1e-10*max(float(np.max(sv,initial=0.)),1e-30)
    rank=int(np.sum(sv>threshold));licq=rank==len(A)
    out['constraint_qualification']={'active_rows':len(A),'rank':rank,
                                   'rank_relative_threshold':1e-10,'singular_values':sv.tolist(),
                                   'LICQ_for_active_filter_rows':licq,
                                   'scope':'Rows within normalized active tolerance, not a proof of global regularity'}
    out['pass']=bool(out['pass'] and out['full_primal_feasible'] and licq)
    return out
