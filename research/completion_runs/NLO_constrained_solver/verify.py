"""Mechanical checks and supplied fixed-state witnesses. No neural optimization."""
from problem import *
from first_order import tangent_direction,normal_residual
from solver import model_coordinates,trust_step,SETTINGS

def main():
    result=[];review=ROOT/'GPD_NLO_feasible_domain_review'
    for label in (31,43):
        o=Oracle(label);x=o.x0;v=o.values(x)
        with np.load(review/f'results/FIRST_ORDER_EXTENDED_{label}.npz') as z:expected={k:z[k] for k in z.files}
        ids=expected['indices'];g,R,A=o.derivatives(x,ids)
        direction,td=tangent_direction(g,A)
        lam,normal,ok=normal_residual(g,A)
        start_linear=time.monotonic();model=model_coordinates(R,g,A)
        proposal,qp=trust_step(model,v['c'][ids],SETTINGS['step_radius_fraction']*SETTINGS['initial_radius'])
        qp['seconds']=time.monotonic()-start_linear
        if proposal is not None:
            qp['linearized_max_c']=float(np.max(v['c'][ids]+A@proposal))
            qp['original_model_decrease']=-float(g@proposal+np.linalg.norm(R@proposal)**2)
            qp['radius_pass']=bool(np.linalg.norm(proposal)<=SETTINGS['initial_radius'])
        witnesses=[]
        for sign in (1.,):
            q=o.values(x+sign*1e-4*expected['direction'])
            witnesses.append({'step':1e-4,'J':q['J'],'decrease':v['J']-q['J'],'max_c':float(q['c'].max()),
                              'pass':bool(q['J']<v['J'] and np.all(q['c']<=0))})
        errors={'objective_residual_identity':abs(v['J']-v['residual']@v['residual']),
                'gradient_residual_identity_L2':float(np.linalg.norm(g-2*R.T@v['residual'])),
                'review_gradient_max_abs':float(np.max(abs(g-expected['g']))),
                'review_constraints_max_abs':float(np.max(abs(v['c']-expected['c']))),
                'review_row_relative_max':float(np.max(abs(A-expected['A'])/np.maximum(1,abs(expected['A'])))),
                'review_tangent_max_abs':float(np.max(abs(direction-expected['direction'])))}
        derivatives=[]
        for k,d in enumerate((expected['direction'],np.cos(.173*np.arange(len(x))+.31))):
            d=d/np.linalg.norm(d);ad=R@d;ad_c=A@d
            probes=[]
            for eps in (1e-4,1e-5):
                vp=o.values(x+eps*d);vm=o.values(x-eps*d)
                fd=(vp['residual']-vm['residual'])/(2*eps);fc=(vp['c'][ids]-vm['c'][ids])/(2*eps)
                er=float(np.max(abs(fd-ad)/np.maximum(1,abs(ad))))
                ec=float(np.max(abs(fc-ad_c)/np.maximum(1,abs(ad_c))))
                ej=abs((vp['J']-vm['J'])/(2*eps)-g@d)/max(1,abs(g@d))
                probes.append({'eps':eps,'residual_error_scaled':er,'constraint_error_scaled':ec,
                               'objective_error_scaled':float(ej),'pass':bool(er<=1e-6 and ec<=1e-3 and ej<=1e-6)})
            derivatives.append({'direction':k,'probes':probes,'pass':all(p['pass'] for p in probes)})
        checks={'residual_identity':errors['objective_residual_identity']<1e-11,
                'gradient_identity':errors['gradient_residual_identity_L2']<1e-8,
                'review_gradient':errors['review_gradient_max_abs']<1e-8,
                'review_constraints':errors['review_constraints_max_abs']<1e-6,
                'review_rows':errors['review_row_relative_max']<1e-5,
                'review_direction':errors['review_tangent_max_abs']<1e-6,
                'witnesses':all(w['pass'] for w in witnesses),
                'derivatives':all(d['pass'] for d in derivatives),'normal_solve':bool(ok)}
        checks['linear_subproblem']=bool(proposal is not None and qp['success'] and qp['radius_pass'] and qp['linearized_max_c']<=1e-8 and qp['original_model_decrease']>0)
        checks={k:bool(v) for k,v in checks.items()}
        out={'label':label,'J':v['J'],'errors':errors,'witnesses':witnesses,'derivatives':derivatives,
             'tangent':td,'near_normal_residual':float(np.linalg.norm(normal)),'subproblem_only_no_neural_update':qp,
             'counts':o.counts,'timers':o.timers,'checks':checks,'pass':all(checks.values())}
        write(HERE/f'MECHANICAL_{label}.json',out);result.append(out)
        print(label,out['pass'],errors,flush=True)
    write(HERE/'MECHANICAL.json',{'pass':all(r['pass'] for r in result),'starts':[31,43],
          'scope':'Saved feasible warm-state mechanical checks, local probes and residual structure only; no neural continuation or historical full-forward replay'})
    if not all(r['pass'] for r in result):raise SystemExit(1)

if __name__=='__main__':main()
