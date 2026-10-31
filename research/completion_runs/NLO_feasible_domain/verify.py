"""Mechanical, derivative and independent fixed-state checks. No fits."""
from domain import *
from torch.func import jvp

def directions(n):
    a=np.arange(n,dtype=float)
    out=[np.cos(.173*a+.31),np.sin(.113*a+.71)]
    return [tensor(d/np.linalg.norm(d)) for d in out]

def independent_moment(p,x,order=384):
    xf,xd=common.helpers.moment_grids();pf=common.projector(order,xf);pd=common.projector(order,xd)
    with torch.no_grad():
        mf=pf.moments(lambda a,b:p.field(x,a,b)).numpy()
        md=pd.moments(lambda a,b:p.field(x,a,b)).numpy()
    cols=[]
    for i,(_,N,_) in enumerate(pf.spec):
        coeff=np.linalg.lstsq(common.poly.design(xf,N),mf[:,i],rcond=None)[0]
        cols.append((md[:,i]-common.poly.design(xd,N)@coeff)/float(pf.scales[i]))
    return float(np.sqrt(np.mean(np.column_stack(cols)**2)))

def proxy_from_full(d,margin=.95):
    cff=sum(np.array(v) for k,v in d['CFF_terms'].items() if k not in ('ordered_time','independent_hard'))
    ref=np.array(d['checks']['strong10']['CFF']);lim=1e-4*np.maximum(1,abs(ref))
    field=np.array([d['fields'][k]['estimated_total'] for k in ('Sigma','T_rescaled','Hg')])
    mc=np.concatenate([d['checks'][k]['moments']['relative_errors'] for k in ('weak10','strong10')])
    return np.r_[cff/(margin*lim)-1,field/(margin*1e-3)-1,mc/(margin*1e-4)-1]

def derivative_record(p,domain,x,independent=False):
    records=[];start=time.monotonic()
    for di,d in enumerate(directions(len(x))):
        xx=x.detach().requires_grad_(True)
        J=p.objective(xx);g=torch.autograd.grad(J,xx)[0]
        ad_j=float(g@d)
        _,ad=jvp(lambda z:domain.constraints(p,z),(x,),(d,));ad=ad.detach().numpy()
        probes=[]
        for eps in (1e-4,1e-5):
            with torch.no_grad():
                jp=float(p.independent_objective(x+eps*d));jm=float(p.independent_objective(x-eps*d))
                cp=domain.constraints(p,x+eps*d).numpy();cm=domain.constraints(p,x-eps*d).numpy()
            fd=(cp-cm)/(2*eps);jd=(jp-jm)/(2*eps)
            jerr=abs(jd-ad_j)/max(1,abs(ad_j));cerr=abs(fd-ad)/np.maximum(1,abs(ad))
            probes.append({'step':eps,'objective_FD':jd,'objective_error_scaled':jerr,
                           'constraint_FD':fd.tolist(),'constraint_error_scaled':cerr.tolist(),
                           'maximum_constraint_error_scaled':float(max(cerr)),
                           'pass':bool(jerr<=1e-6 and max(cerr)<=1e-3)})
        item={'direction':di,'objective_AD':ad_j,'constraint_AD':ad.tolist(),'probes':probes,
              'pass':all(z['pass'] for z in probes)}
        if independent and di==0:
            eps=1e-4;point=common.operator_directories()[1];q=4.7
            full=[]
            for sign in (-1,1):
                p.set_model(x+sign*eps*d)
                r=preflight.validate_point(point,q,'mechanical_direction_only',p.model)
                full.append(proxy_from_full(r,domain.margin))
            off=1+5*11+1*11;fd=(full[1]-full[0])/(2*eps)
            err=abs(fd-ad[off:off+11])/np.maximum(1,abs(ad[off:off+11]))
            mfd=(independent_moment(p,x+eps*d)-independent_moment(p,x-eps*d))/(2*eps*1e-3)
            merror=abs(mfd-ad[0])/max(1,abs(ad[0]))
            item['independent_direct_evolution']={'xi':point['xi'],'Q2':q,'FD':fd.tolist(),
                    'AD':ad[off:off+11].tolist(),'errors_scaled':err.tolist(),
                    'moment_384_FD':mfd,'moment_error_scaled':merror,
                    'pass':bool(max(err)<=1e-3 and merror<=1e-6)}
            item['pass']=bool(item['pass'] and item['independent_direct_evolution']['pass'])
        records.append(item)
    return {'records':records,'pass':all(d['pass'] for d in records),
            'derivative_tolerances':{'objective_scaled':1e-6,'normalized_constraint_scaled':1e-3,
                'meaning':'AD-vs-FD checks; no change to any primal physics/numerical tolerance. Inactive near-zero transport differences are cancellation limited.'},
            'seconds':time.monotonic()-start,'peak_RSS_bytes':rss()}

def main():
    domain=Domain();results=[];t=time.monotonic()
    for label in (31,43):
        p=Problem(label);x=p.x0
        with torch.no_grad():
            comp=p.components(x).numpy();c=domain.constraints(p,x).numpy()
            direct=float(p.independent_objective(x))
        # Reuse accepted original full-gate records, no re-fitting/audit campaign.
        comparisons=[]
        old=ROOT/'completion_runs/NLO_noisy_feasibility'/f'datasets/d000/start{label}/initial_numerics'
        for pi,point in enumerate(common.operator_directories()):
            for qi,q in enumerate(Q2S):
                path=old/f'xi{point["xi"]}_Q2{q}.json';d=json.loads(path.read_text())
                expected=proxy_from_full(d,domain.margin);off=1+(pi*5+qi)*11
                err=float(np.max(abs(expected-c[off:off+11])))
                comparisons.append({'source':str(path.relative_to(ROOT)),'sha256':sha(path),
                                    'maximum_normalized_value_error':err,'pass':err<=1e-4})
        with torch.no_grad():
            m192=float(domain.mom.value(p,x));m384=independent_moment(p,x)
        dr=derivative_record(p,domain,x,independent=True)
        result={'label':label,'J':float(comp.sum()),'J_components':comp.tolist(),
                'objective_independent_difference':abs(float(comp.sum())-direct),
                'constraints':dict(zip(domain.names,c.tolist())),
                'moment_192':m192,'moment_384':m384,'moment_refinement_difference':abs(m192-m384),
                'existing_full_comparisons':comparisons,'derivatives':dr,
                'pass':bool(abs(float(comp.sum())-direct)<=1e-10 and abs(m192-m384)<=1e-7 and all(r['pass'] for r in comparisons) and dr['pass'])}
        results.append(result);write(HERE/f'MECHANICAL_start{label}.json',result)
        print('mechanical',label,'pass',result['pass'],'max constraint',max(c),'seconds',time.monotonic()-t,flush=True)
    write(HERE/'MECHANICAL.json',{'pass':all(r['pass'] for r in results),'starts':[r['label'] for r in results],
                                'seconds':time.monotonic()-t,'peak_RSS_bytes':rss()})
    if not all(r['pass'] for r in results):raise SystemExit(1)

if __name__=='__main__':main()
