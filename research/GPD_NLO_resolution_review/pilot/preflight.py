"""Bounded saved-operator validation. Never builds or writes an operator."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from common import *
import time, argparse
from scipy.integrate import solve_ivp
sys.path.insert(0,str(NLO/'candidate'))
from forward import load, initial, evaluate_field, hard_row, jacobian
from src.ordered_evolution import evolve
from frozen.python.nlo_forward import evolve_field
from hard_reference import reference
from point_suite import compare, components

OPERATORS={}; HARD={}
def op(path):
    if path not in OPERATORS: OPERATORS[path]=load(NLO/path)
    return OPERATORS[path]

def hard(b,q2,order):
    key=(b.xi,b.degree,q2,order)
    if key not in HARD:HARD[key]=hard_row(b,q2,order)[0]
    return HARD[key]

def moments(b,v):
    x,w=b.outputs(96);f=evaluate_field(b,v,x)
    return np.array([(w*x)@f[0],w@f[2],(w*x)@f[1]])

def moment_control(b,v,h,q2):
    before=moments(b,v); after=moments(b,h)
    m0=np.array([[-16/9,1],[16/9,-1]])
    m1=np.array([[-2000/243,611/108],[2000/243,-611/108]])
    def rhs(t,v):
        z=v[0]
        return np.r_[-9*z*z-64*z**3,(2*z*m0+4*z*z*m1)@v[1:3],(2*z*(-16/9)+4*z*z*(-2360/243))*v[3]]
    sol=solve_ivp(rhs,[0,np.log(q2/2)],np.r_[.35691594851514724/(4*np.pi),before],method='DOP853',rtol=1e-12,atol=1e-14)
    ref=sol.y[1:,-1]; err=abs(after-ref)/np.maximum(1e-15,abs(ref))
    return {'initial':before.tolist(),'evolved':after.tolist(),'independent_reference':ref.tolist(),'relative_errors':err.tolist(),'pass':bool(max(err)<=1e-4)}

def cached_row(directory,q2):
    b,k0,k1,key=op(directory)
    sources={p:digest(NLO/p) for p in ['candidate/forward.py','candidate/enriched.py','candidate/spectral.py','candidate/basis_factory.py','src/ordered_evolution.py','frozen/python/nlo_hard.py','frozen/python/nlo_forward.py']}
    meta={'directory':directory,'operators_sha256':digest(NLO/directory/'operators.npz'),'KEY_sha256':digest(NLO/directory/'KEY.json'),
          'basis':key['basis'],'xi':b.xi,'Q02':2.,'Q2':q2,'nf':3,'scheme':'MSbar','hard_order':192,'sources':sources,
          'adjoint_rtol':1e-12,'adjoint_atol':1e-14,'sign':'internal','charges':'included exactly once'}
    token=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    path=WORK/'rows'/f'{token}.npz';path.parent.mkdir(exist_ok=True)
    if path.exists():
        with np.load(path) as z:
            assert json.loads(str(z['metadata_json']))==meta
            return z['dense'].copy()
    J,_=jacobian(NLO/directory,q2,192);dense=dense_row(b,J)
    np.savez_compressed(path,dense=dense,x=b.nodes,metadata_json=json.dumps(meta,sort_keys=True))
    return dense

def validate_point(point,q2,field_name,model=None):
    start=time.monotonic();data={};checks={}
    for label,d in point['directories'].items():
        b,k0,k1,key=op(d)
        v=initial(b,field_name=='Dterm_added') if model is None else packing(b,sample(model,b))
        h,diag=evolve(v,k0,k1,q2,rtol=1e-11,atol=1e-13)
        row=hard(b,q2,192); c=row@h;data[label]=(b,h,c)
        checks[label]={'CFF':components(c)}
        if label in ('weak10','strong10'):
            tight,_=evolve(v,k0,k1,q2,rtol=1e-13,atol=1e-15)
            rk=evolve_field(v,k0,k1,q2,steps=128)
            ref,rd=reference(b,h,q2,1e-9)
            checks[label].update({
              'hard_192_384_change':np.abs(components((hard(b,q2,384)-row)@h)).tolist(),
              'time_CFF_change':np.maximum(np.abs(components(row@(h-tight))),np.abs(components(row@(h-rk)))).tolist(),
              'time_field_max_abs':float(max(np.max(abs(h-tight)),np.max(abs(h-rk)))),
              'independent_hard':{'reference':components(ref),'difference':components(c-ref),**rd},
              'moments':moment_control(b,v,h,q2),
              'adjoint_forward_absolute':float(abs(np.einsum('rcn,nc->r',cached_row(d,q2),sample(model,b) if model is not None else unpack(b,v))[0]-c.real))})
            if model is None and field_name=='Dterm_added':
                dv=v-initial(b,False);dh,_=evolve(dv,k0,k1,q2,rtol=1e-12,atol=1e-14)
                z,w=np.polynomial.legendre.leggauss(96);x=(1+b.xi)/2+(1-b.xi)/2*z
                support=np.max(abs(evaluate_field(b,dh,x)),axis=1)
                checks[label]['Dterm_support']={'DGLAP_max_abs':support.tolist(),'Im_CFF':float((row@dh).imag),
                        'pass':bool(max(support)<=1e-6 and abs((row@dh).imag)<=1e-6)}
    pairs=[('weak6','weak8'),('weak8','weak10'),('strong6','strong8'),('strong8','strong10'),('weak10','strong10')]
    pairs += [(m,kind+'_'+m) for kind in ('inner72','output48') for m in ('weak10','strong10')]
    comp={a+'__'+b:compare(data[a],data[b],q2) for a,b in pairs}
    def cdelta(a,b):return np.abs(comp[a+'__'+b]['CFF_difference'])
    reserve=1e-6*max(1,np.log(q2/2)/np.log(2.3/2))
    terms={'candidate_representation':2*cdelta('weak8','weak10'),
           'reference_representation':2*cdelta('strong8','strong10'),
           'candidate_reference_discrepancy':cdelta('weak10','strong10'),
           'kernel_quadrature':2*sum(cdelta(m,'inner72_'+m) for m in ('weak10','strong10')),
           'output_quadrature':2*sum(cdelta(m,'output48_'+m) for m in ('weak10','strong10')),
           'hard_quadrature':2*sum(np.array(checks[m]['hard_192_384_change']) for m in ('weak10','strong10')),
           'ordered_time':2*sum(np.array(checks[m]['time_CFF_change']) for m in ('weak10','strong10')),
           'independent_hard':2*sum(np.abs(checks[m]['independent_hard']['difference'])+np.array(checks[m]['independent_hard']['reference_quad_error_estimate']) for m in ('weak10','strong10')),
           'source_arithmetic_allowance':np.full(2,reserve)}
    ctotal=sum(terms.values()); limit=1e-4*np.maximum(1,np.abs(checks['strong10']['independent_hard']['reference']))
    fields={}
    for ch in ('Sigma','T_rescaled','Hg'):
        def f(a,b):return comp[a+'__'+b]['field_relative_L2'][ch]
        ft={'candidate_representation':2*f('weak8','weak10'),'reference_representation':2*f('strong8','strong10'),
            'candidate_reference_discrepancy':f('weak10','strong10'),
            'kernel_quadrature':2*sum(f(m,'inner72_'+m) for m in ('weak10','strong10')),
            'output_quadrature':2*sum(f(m,'output48_'+m) for m in ('weak10','strong10')),
            'time_and_arithmetic_allowance':reserve}
        fields[ch]={'terms':ft,'estimated_total':sum(ft.values()),'pass':bool(sum(ft.values())<=1e-3)}
    gates={'CFF':bool(np.all(ctotal<=limit)),'fields':all(v['pass'] for v in fields.values()),
           'moments':all(checks[m]['moments']['pass'] for m in ('weak10','strong10')),
           'Dterm_support':all(checks[m].get('Dterm_support',{'pass':True})['pass'] for m in ('weak10','strong10')),
           'hard_no_warnings':not any(checks[m]['independent_hard']['warnings'] for m in ('weak10','strong10'))}
    result={'xi':point['xi'],'Q2':q2,'field':field_name,'checks':checks,'comparisons':comp,
      'CFF_terms':{k:v.tolist() for k,v in terms.items()},'CFF_estimated_total':ctotal.tolist(),'CFF_limits':limit.tolist(),
      'fields':fields,'gates':gates,'pass':all(gates.values()),'seconds':time.monotonic()-start}
    return result

def unpack(b,v):
    h=np.zeros((len(b.nodes),3));nq=len(b.qids)
    h[b.qids,0]=v[:nq];h[b.qids,1]=v[nq:2*nq];h[b.gids,2]=v[2*nq:]
    return h

def gradient_checks():
    from torch.func import functional_call
    point=operator_directories()[1];d=point['directories']['weak10'];b,k0,k1,_=op(d);q2=4.7
    dense=torch.tensor(cached_row(d,q2));row=hard(b,q2,192);results=[]
    for seed in (31,43):
        model=ReferenceNetwork('film').double();model.load_state_dict(torch.load(WORK/f'initial_seed{seed}.pt',weights_only=True)['state_dict'])
        spec=[(n,p.shape,p.numel()) for n,p in model.named_parameters()]
        flat=torch.cat([p.detach().flatten() for p in model.parameters()]).requires_grad_(True)
        def values(theta):
            pars={};offset=0
            for n,s,k in spec:pars[n]=theta[offset:offset+k].view(s);offset+=k
            return functional_call(model,pars,(torch.tensor(b.nodes[None]),torch.tensor([b.xi])))[0]
        torch.manual_seed(1703);direction=torch.randn_like(flat);direction/=direction.norm()
        out=torch.einsum('rcn,cn->r',dense,values(flat))
        ad=torch.stack([torch.autograd.grad(v,flat,retain_graph=True)[0]@direction for v in out]).detach().numpy()
        probes=[]
        for eps in (1e-4,1e-5,1e-6):
            c=[]
            for sign in (-1,1):
                v=packing(b,values(flat+sign*eps*direction).detach().numpy().T)
                h,_=evolve(v,k0,k1,q2,rtol=1e-13,atol=1e-15);c.append(row@h)
            fd=np.array(components((c[1]-c[0])/(2*eps)));err=abs(fd-ad)
            probes.append({'eps':eps,'direct_evolved_FD':fd.tolist(),'error':err.tolist(),'pass':bool(np.all(err<=1e-6*np.maximum(1,abs(ad))))})
        results.append({'seed':seed,'input':'original non-polynomial FiLM random field','parameters':len(flat),'AD':ad.tolist(),'probes':probes,'pass':all(p['pass'] for p in probes)})
    write(WORK/'GRADIENT_CHECKS.json',results)
    return all(r['pass'] for r in results)

def main():
    plan=prepare();start=time.monotonic();summary=[]
    for field in plan['truths']:
        for point in operator_directories():
            for q2 in plan['training_Q2']+plan['heldout_Q2']:
                path=WORK/'preflight'/f'{field}_xi{point["xi"]}_Q2{q2}.json'
                if path.exists():r=json.loads(path.read_text())
                else:r=validate_point(point,q2,field);write(path,r)
                summary.append({k:r[k] for k in ('xi','Q2','field','pass','gates','CFF_estimated_total')})
                print(json.dumps(summary[-1]),flush=True)
        # All declared grid checks are retained, including failures; never fit an affected truth.
    write(WORK/'TRUTH_PREFLIGHT.json',{'points':summary,'pass':all(r['pass'] for r in summary),'seconds':time.monotonic()-start})
    if not all(r['pass'] for r in summary):
        print('SCIENTIFIC_BLOCKER: truth numerical preflight failed; no optimization',flush=True);return
    gp=gradient_checks();initial_results=[]
    for seed in plan['seeds']:
        model=ReferenceNetwork('film').double();model.load_state_dict(torch.load(WORK/f'initial_seed{seed}.pt',weights_only=True)['state_dict'])
        for point in operator_directories():
            for q2 in plan['training_Q2']+plan['heldout_Q2']:
                path=WORK/'preflight'/f'initial{seed}_xi{point["xi"]}_Q2{q2}.json'
                if path.exists():r=json.loads(path.read_text())
                else:r=validate_point(point,q2,f'initial{seed}',model);write(path,r)
                initial_results.append({k:r[k] for k in ('xi','Q2','field','pass','gates','CFF_estimated_total')})
                print(json.dumps(initial_results[-1]),flush=True)
    write(WORK/'INITIAL_PREFLIGHT.json',{'points':initial_results,'gradient_pass':gp,'pass':gp and all(r['pass'] for r in initial_results),'seconds':time.monotonic()-start})

if __name__=='__main__':main()
