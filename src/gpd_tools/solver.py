"""Feasible trust-region SQP with a Gauss-Newton objective model.

No AL, added objective penalty, statistical regularizer, fixed GPD basis or
parameter freezing. Temporary orthogonal coordinates are rebuilt from full
neural derivatives at every iterate; final KKT stays in original coordinates.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import time
import numpy as np
from scipy.linalg import qr,eigh

SETTINGS={
 'max_iterations':600,'max_evaluations':12000,'optimization_seconds':6600,
 'initial_radius':.05,'maximum_radius':1.,'minimum_radius':1e-12,'step_radius_fraction':.8,
 'working_window':.01,'fraction_to_boundary':.995,
 'maximum_trials':16,'maximum_SOC':3,'SOC_target':1e-9,'SOC_relative_limit':.5,
 'accept_ratio':1e-4,'shrink_ratio':.25,'grow_ratio':.75,'radius_shrink':.25,
 'radius_growth':2.,'unchanged_termination_limit':3,
 'span_rank_tolerance':1e-12,'QP_rank_tolerance':1e-12,'QP_iterations':300,
 'QP_stationarity':1e-11,'QP_feasibility':1e-11,
 'minimum_damping_relative':1e-14,'damping_bisections':32,
 'KKT_L2':1e-6,'active_tolerance':1e-6,'complementarity':1e-8}

def positive_qp(diagonal,g,A,b,settings=SETTINGS):
    """Primal feasible active-set convex QP in temporary coordinates.
    min g.s+1/2 s.D.s, A.s<=b; zero is a feasible starting point.
    Constraint rows are normalized before calling this routine.
    """
    D=np.asarray(diagonal);n=len(g);s=np.zeros(n);W=[];linear_solves=0
    if np.any(b<0) or np.any(D<=0):raise ValueError('Positive D and zero-feasible bounds required')
    tiny=settings['QP_stationarity']*max(1.,np.linalg.norm(g))
    for it in range(settings['QP_iterations']):
        grad=g+D*s;p0=-grad/D;lam=np.empty(0)
        if W:
            B=A[W];S=(B/D)@B.T
            lam=np.linalg.lstsq(S,B@p0,rcond=settings['QP_rank_tolerance'])[0]
            linear_solves+=1;p=p0-(B.T@lam)/D
        else:p=p0
        # Projected-gradient norm, not the small Newton-step norm, stops inner QP.
        residual=grad+(A[W].T@lam if W else 0.)
        if np.linalg.norm(residual)<=tiny:
            if not W or min(lam)>=-tiny:
                full=np.zeros(len(b));full[W]=np.maximum(lam,0.)
                feasible=bool(np.max(A@s-b,initial=-np.inf)<=settings['QP_feasibility'])
                return s,full,{'success':feasible,'iterations':it+1,'linear_solves':linear_solves,
                    'working':W,'stationarity':float(np.linalg.norm(grad+A.T@full)),
                    'maximum_excess':float(np.max(A@s-b,initial=-np.inf)) if len(b) else 0.}
            del W[int(np.argmin(lam))];continue
        alpha=1.;block=None;direction=A@p;slack=b-A@s
        for j in range(len(b)):
            if j not in W and direction[j]>0:
                step=max(0.,float(slack[j]/direction[j]))
                if step<alpha:alpha=step;block=j
        s=s+alpha*p
        if block is not None and alpha<1.:W.append(block)
        if not np.isfinite(s).all():break
    return s,np.zeros(len(b)),{'success':False,'iterations':it+1,'linear_solves':linear_solves,
                               'reason':'QP active-set iteration/finite failure','working':W}

def model_coordinates(R,g,A,settings=SETTINGS):
    pieces=[R,g[None,:],A];V=np.concatenate(pieces,0).T
    norms=np.linalg.norm(V,axis=0);V=V[:,norms>0]/norms[norms>0]
    Q,T,piv=qr(V,mode='economic',pivoting=True,check_finite=True)
    diag=abs(np.diag(T));rank=int(np.sum(diag>settings['span_rank_tolerance']*max(np.max(diag,initial=0.),1e-30)))
    Q=Q[:,:rank]
    B=R@Q;H=2*B.T@B;ev,U=eigh(H,check_finite=True)
    # Only roundoff-negative eigenvalues of a Gram matrix are clipped.
    if min(ev,default=0)<-1e-10*max(1.,max(ev,default=0)):raise FloatingPointError('GN Gram materially indefinite')
    ev=np.maximum(ev,0.);Q=Q@U
    return {'Q':Q,'eigenvalues':ev,'gradient':Q.T@g,'A':A@Q,'rank':rank,
            'gradient_span_error':float(np.linalg.norm(g-Q@(Q.T@g)))}

def trust_step(model,c,radius,settings=SETTINGS):
    e=model['eigenvalues'];g=model['gradient'];A=model['A'];norms=np.linalg.norm(A,axis=1)
    usable=norms>0
    if np.any((~usable)&(c>0)):return None,{'success':False,'reason':'constant infeasible row'}
    B=A[usable]/norms[usable,None];b=-settings['fraction_to_boundary']*c[usable]/norms[usable]
    if np.any(b<0):return None,{'success':False,'reason':'infeasible base'}
    floor=settings['minimum_damping_relative']*max(1.,max(e,default=0.));calls=[]
    def evaluate(mu):
        s,lam,info=positive_qp(e+mu,g,B,b,settings);calls.append(info)
        return s,lam,info
    low=floor;s,lam,info=evaluate(low)
    if not info['success']:return None,{'success':False,'reason':'QP failure','QP_calls':calls}
    chosen=low
    if np.linalg.norm(s)>radius:
        high=max(low*10,np.linalg.norm(g)/max(radius,1e-30))
        for _ in range(32):
            sh,lh,ih=evaluate(high)
            if not ih['success']:return None,{'success':False,'reason':'QP failure','QP_calls':calls}
            if np.linalg.norm(sh)<=radius:break
            high*=10
        else:return None,{'success':False,'reason':'cannot bracket trust radius','QP_calls':calls}
        s,lam,info=sh,lh,ih;chosen=high
        for _ in range(settings['damping_bisections']):
            mid=np.sqrt(low*high);sm,lm,im=evaluate(mid)
            if not im['success']:return None,{'success':False,'reason':'QP failure','QP_calls':calls}
            if np.linalg.norm(sm)>radius:low=mid
            else:high=mid;s,lam,info=sm,lm,im;chosen=mid
            if high<=low*(1+1e-5):break
    mapped=np.zeros(len(c));mapped[usable]=lam/norms[usable]
    return model['Q']@s,{'success':True,'damping':chosen,'subspace_rank':model['rank'],
            'gradient_span_error':model['gradient_span_error'],'multipliers_original_rows':mapped.tolist(),
            'step_norm':float(np.linalg.norm(s)),'QP_calls':calls}

def solve(oracle,kkt_check,record,settings=SETTINGS):
    x=oracle.x0.copy();start=time.monotonic();radius=settings['initial_radius'];stagnation=0
    values=oracle.values(x);history=[];counts={'linear_model_builds':0,'QP_calls':0,'linear_solves':0,'SOC_linear_solves':0,'trial_rejections':0,'accepted_steps':0}
    linear_seconds=0.;status='iteration_budget';extra=set()
    if not np.isfinite(values['c']).all() or np.any(values['c']>0):
        return x,{'status':'infeasible_initial','history':[],'counts':counts,'seconds':time.monotonic()-start}
    for iteration in range(settings['max_iterations']):
        if time.monotonic()-start>=settings['optimization_seconds']:status='wall_budget';break
        if oracle.counts['evaluations']>=settings['max_evaluations']:status='evaluation_budget';break
        c=values['c'];working=np.array(sorted(set(np.flatnonzero(c>=-settings['working_window']))|extra),int)
        g,R,A=oracle.derivatives(x,working);check=kkt_check(g,c,A,working,oracle.names)
        rec={'iteration':iteration,'J':values['J'],'J_components':values['components'].tolist(),
             'raw_gradient_L2':float(np.linalg.norm(g)),'KKT':check,'constraints':c.tolist(),
             'working_indices':working.tolist(),'radius_start':radius,'trials':[]}
        if check['pass']:
            status='original_coordinate_KKT_pass';rec['accepted']=False;rec['termination']=status
            history.append(rec);record(x,rec,None);break
        accepted=False;last_trial=None
        for trial in range(settings['maximum_trials']):
            if time.monotonic()-start>=settings['optimization_seconds'] or oracle.counts['evaluations']>=settings['max_evaluations']:
                status='wall_or_evaluation_budget';break
            t=time.monotonic();model=model_coordinates(R,g,A,settings);counts['linear_model_builds']+=1
            step,qp=trust_step(model,c[working],settings['step_radius_fraction']*radius,settings);linear_seconds+=time.monotonic()-t
            counts['QP_calls']+=len(qp.get('QP_calls',[]));counts['linear_solves']+=sum(q.get('linear_solves',0) for q in qp.get('QP_calls',[]))
            detail={'trial':trial,'radius':radius,'subproblem':qp,'accepted':False,'SOC':[]}
            if step is None:
                radius*=settings['radius_shrink'];detail['reason']='subproblem_failure';rec['trials'].append(detail)
                counts['trial_rejections']+=1
                if radius<settings['minimum_radius']:break
                continue
            candidate=x+step;v=oracle.values(candidate);base_length=np.linalg.norm(step)
            # Nonlinear normal corrections prevent a purely tangent SQP step
            # from being rejected solely for its second-order outward curvature.
            for correction in range(settings['maximum_SOC']):
                if time.monotonic()-start>=settings['optimization_seconds'] or oracle.counts['evaluations']>=settings['max_evaluations']:break
                if np.all(v['c']<=0.):break
                rows=np.flatnonzero(v['c']>-settings['SOC_target'])
                if not len(rows):break
                extra.update(map(int,rows));B=oracle.constraint_rows(candidate,rows)
                norm=np.linalg.norm(B,axis=1);good=norm>0
                if not np.all(good):break
                Bn=B/norm[:,None];rhs=-(v['c'][rows]+settings['SOC_target'])/norm
                t=time.monotonic();z=np.linalg.lstsq(Bn@Bn.T,rhs,rcond=settings['QP_rank_tolerance'])[0]
                ds=Bn.T@z;linear_seconds+=time.monotonic()-t;counts['SOC_linear_solves']+=1
                info={'indices':rows.tolist(),'norm':float(np.linalg.norm(ds)),
                      'linear_residual_L2':float(np.linalg.norm(Bn@ds-rhs))}
                detail['SOC'].append(info)
                if np.linalg.norm(ds)>settings['SOC_relative_limit']*max(base_length,1e-30):info['used']=False;break
                if np.linalg.norm(candidate+ds-x)>radius*(1+1e-10):info['used']=False;break
                candidate+=ds;info['used']=True;v=oracle.values(candidate)
            last_trial=candidate.copy();actual_step=candidate-x
            # The SOC is a feasibility restoration, not a fresh quadratic step.
            # Compare against the original SQP model decrease; including SOC in
            # that denominator can reward nearly reflecting along a curved wall.
            predicted=-float(g@step+np.linalg.norm(R@step)**2)
            actual=values['J']-v['J'];ratio=actual/predicted if predicted>0 else -1.
            feasible=bool(np.isfinite(v['J']) and np.isfinite(v['c']).all() and np.all(v['c']<=0))
            detail.update(J=v['J'],constraints=v['c'].tolist(),predicted_decrease=predicted,
                          corrected_step_model_decrease=-float(g@actual_step+np.linalg.norm(R@actual_step)**2),
                          actual_decrease=actual,ratio=ratio,feasible=feasible,
                          step_norm=float(np.linalg.norm(actual_step)))
            if feasible and predicted>0 and actual>0 and ratio>=settings['accept_ratio']:
                detail['accepted']=True;accepted=True;x=candidate;values=v;counts['accepted_steps']+=1;stagnation=0
                if ratio<settings['shrink_ratio']:radius*=settings['radius_shrink']
                elif ratio>settings['grow_ratio'] and np.linalg.norm(actual_step)>.7*radius:
                    radius=min(settings['maximum_radius'],radius*settings['radius_growth'])
                extra=set();rec['trials'].append(detail);break
            detail['reason']='nonlinear_infeasible' if not feasible else 'insufficient_actual_decrease'
            counts['trial_rejections']+=1;rec['trials'].append(detail)
            new=set(map(int,np.flatnonzero(v['c']>0)))|extra
            if not new.issubset(set(working)):
                working=np.array(sorted(set(working)|new),int);g,R,A=oracle.derivatives(x,working)
            radius*=settings['radius_shrink']
            if radius<settings['minimum_radius']:break
        rec.update(accepted=accepted,radius_end=radius,seconds=time.monotonic()-start,
                   counts=dict(counts),oracle_counts=dict(oracle.counts),linear_algebra_seconds=linear_seconds)
        if not accepted:stagnation+=1
        rec['unchanged_state_terminations']=stagnation;history.append(rec);record(x,rec,last_trial)
        if status=='wall_or_evaluation_budget':break
        if stagnation>=settings['unchanged_termination_limit']:status='three_unchanged_state_terminations';break
        if radius<settings['minimum_radius']:status='trust_radius_floor';break
    return x,{'status':status,'history':history,'counts':counts,'oracle_counts':dict(oracle.counts),
               'oracle_timers':dict(getattr(oracle,'timers',{})),'linear_algebra_seconds':linear_seconds,
               'seconds':time.monotonic()-start,'final_radius':radius}
