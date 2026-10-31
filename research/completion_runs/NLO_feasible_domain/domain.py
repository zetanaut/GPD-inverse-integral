"""Prospective constrained FiLM domain. No kernel construction or truth access.

The differentiable numerical filter includes evolved fields, not just input
projection. The unchanged independent forward routine remains authoritative.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys,json,hashlib,time,resource
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'GPD_NLO_resolution_review/pilot'))
import common,preflight
from src.ordered_evolution import coupling
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
Q2S=[2.3,4.7,9.3,3.1,6.1]
LABELS=['weak8','weak10','strong8','strong10','inner72_weak10',
        'inner72_strong10','output48_weak10','output48_strong10']
PAIRS=[('weak8','weak10',2.),('strong8','strong10',2.),
       ('weak10','strong10',1.),('weak10','inner72_weak10',2.),
       ('strong10','inner72_strong10',2.),('weak10','output48_weak10',2.),
       ('strong10','output48_strong10',2.)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');torch.save(obj,t);t.replace(p)
def rss():
    n=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform=='darwin' else n*1024)
def tensor(a):return torch.as_tensor(a,dtype=torch.float64)

ROW_INDEX=None
def existing_row(directory,q2):
    global ROW_INDEX
    if ROW_INDEX is None:
        ROW_INDEX={}
        for p in (common.WORK/'rows').glob('*.npz'):
            with np.load(p,allow_pickle=False) as z:m=json.loads(str(z['metadata_json']))
            key=(m['directory'],float(m['Q2']))
            if key in ROW_INDEX:raise ValueError('ambiguous existing row')
            ROW_INDEX[key]=(p,m)
    path,m=ROW_INDEX[(directory,float(q2))] # miss is an error, never build
    assert sha(common.NLO/directory/'operators.npz')==m['operators_sha256']
    assert sha(common.NLO/directory/'KEY.json')==m['KEY_sha256']
    for rel,h in m['sources'].items():assert sha(common.NLO/rel)==h
    assert m['Q02']==2 and m['nf']==3 and m['scheme']=='MSbar' and m['hard_order']==192
    with np.load(path,allow_pickle=False) as z:return z['dense'].copy()
preflight.cached_row=existing_row

class Problem:
    def __init__(self,label):
        if label not in (31,43):raise ValueError('Only original d000 starts')
        self.label=label
        with torch.random.fork_rng():
            torch.manual_seed(0) # construction only; every parameter replaced below
            self.model=common.ReferenceNetwork('film').double()
        state=torch.load(HERE/f'inputs/initial{label}.pt',weights_only=True)
        self.model.load_state_dict(state['state_dict'])
        self.spec=[(n,p.shape,p.numel()) for n,p in self.model.named_parameters()]
        self.x0=torch.cat([p.detach().flatten() for p in self.model.parameters()])
        assert torch.equal(self.x0,state['flat']) and self.x0.numel()==3867
        with np.load(HERE/'inputs/d000.npz',allow_pickle=False) as z:
            self.y=tensor(z['y']).flatten();self.cov=tensor(z['covariance'])
        assert self.y.numel()==18 and torch.equal(self.cov,self.cov.T)
        self.chol=torch.linalg.cholesky(self.cov)
        with np.load(HERE/'inputs/fit_bank.npz',allow_pickle=False) as z:
            self.bank={k:tensor(z[k]) for k in ('x','xi','rows')}
        self.p=common.projector(48)
    def parameters(self,x):
        d={};i=0
        for name,shape,n in self.spec:d[name]=x[i:i+n].view(shape);i+=n
        return d
    def field(self,x,xx,xi):return functional_call(self.model,self.parameters(x),(xx,xi))
    def components(self,x):
        h=self.field(x,self.bank['x'],self.bank['xi'])
        r=torch.einsum('sqrcn,scn->sqr',self.bank['rows'],h).flatten()-self.y
        rw=torch.linalg.solve_triangular(self.chol,r[:,None],upper=False).flatten()
        ld=rw.square().sum()/18
        lp=self.p(lambda a,b:self.field(x,a,b)) # 492 denominator, all coefficients free
        return torch.stack((ld,lp))
    def objective(self,x):return self.components(x).sum()
    def independent_objective(self,x):
        pred=torch.einsum('sqrcn,scn->sqr',self.bank['rows'],self.field(x,self.bank['x'],self.bank['xi'])).flatten()
        r=pred-self.y;data=r@torch.linalg.solve(self.cov,r)/18
        M=self.p.moments(lambda a,b:self.field(x,a,b));pen=0.
        for i,(_,N,_) in enumerate(self.p.spec):
            P=tensor(common.poly.design(self.p.xi.numpy(),N))
            a=torch.linalg.lstsq(P,M[:,i,None],driver='gelsd').solution[:,0]
            residual=(M[:,i]-P@a)/self.p.scales[i];pen=pen+residual@residual
        return data+pen/492
    def set_model(self,x):self.model.load_state_dict({k:v.detach() for k,v in self.parameters(x).items()});return self.model

class Moments:
    def __init__(self,order=192,diagnostic=None):
        self.xf,self.xd=map(np.asarray,common.helpers.moment_grids())
        if diagnostic is not None:self.xd=np.asarray(diagnostic)
        self.pf=common.projector(order,self.xf);self.pd=common.projector(order,self.xd)
        self.predict=[]
        for _,N,_ in self.pf.spec:
            P=common.poly.design(self.xf,N)
            self.predict.append(tensor(common.poly.design(self.xd,N)@np.linalg.pinv(P)))
    def residual(self,p,x):
        mf=self.pf.moments(lambda a,b:p.field(x,a,b));md=self.pd.moments(lambda a,b:p.field(x,a,b))
        return torch.stack([(md[:,i]-P@mf[:,i])/self.pf.scales[i] for i,P in enumerate(self.predict)],1)
    def value(self,p,x):return torch.linalg.vector_norm(self.residual(p,x))/np.sqrt(len(self.xd)*12)

def ordered_maps(k0,k1):
    """Propagate identities with GIVEN generators; no evolution kernel integrals.
    One solve samples all five Q2 values, preserving noncommuting ordering.
    """
    a,end,_=coupling(max(Q2S));n=len(k0)
    times=np.log(np.array(sorted(Q2S))/2.)
    def rhs(t,v):
        u=v.reshape(n,n);z=2*float(a(t));return (z*(k0@u)+z*z*(k1@u)).ravel()
    s=solve_ivp(rhs,(0.,end),np.eye(n).ravel(),method='DOP853',t_eval=times,rtol=1e-13,atol=1e-15)
    if not s.success or not np.isfinite(s.y).all():raise RuntimeError('Given-matrix propagation failed')
    return {q:s.y[:,i].reshape(n,n).copy() for i,q in enumerate(sorted(Q2S))},s.nfev

def spin2_map(q2):
    m0=np.array([[-16/9,1,0],[16/9,-1,0],[0,0,-16/9]])
    m1=np.array([[-2000/243,611/108,0],[2000/243,-611/108,0],[0,0,-2360/243]])
    def rhs(t,v):
        a=v[0];u=v[1:].reshape(3,3)
        return np.r_[-9*a*a-64*a**3,((2*a*m0+4*a*a*m1)@u).ravel()]
    s=solve_ivp(rhs,[0,np.log(q2/2)],np.r_[.35691594851514724/(4*np.pi),np.eye(3).ravel()],method='DOP853',rtol=1e-13,atol=1e-15)
    assert s.success
    return s.y[1:,-1].reshape(3,3)

def build_contractions():
    path=HERE/'inputs/contractions.pt'
    if path.exists():raise FileExistsError('Contractions already exist; do not overwrite')
    results=[];records=[];t=time.monotonic()
    for point in common.operator_directories():
        bs={};ds={};indices={};maps={}
        for label in LABELS:
            d=point['directories'][label];b,k0,k1,_=preflight.op(d);bs[label]=b
            U,nfev=ordered_maps(k0,k1);maps[label]=U
            nq=len(b.qids);n=len(k0);xx,ww=b.outputs(96)
            M=np.zeros((3,n));M[0,:nq]=(ww*xx)@b.basis(xx,'odd')
            M[1,2*nq:]=ww@b.basis(xx,'even');M[2,nq:2*nq]=(ww*xx)@b.basis(xx,'odd')
            ds[label]={'nodes':tensor(b.nodes),'qids':torch.tensor(b.qids),'gids':torch.tensor(b.gids),
                       'U':{str(q):tensor(U[q]) for q in Q2S},'nq':nq,'M':tensor(M),
                       'hard':{str(q):tensor(np.stack((preflight.hard(b,q,192).real,preflight.hard(b,q,192).imag))) for q in Q2S},
                       'hard_delta':{str(q):tensor(np.stack(((preflight.hard(b,q,384)-preflight.hard(b,q,192)).real,(preflight.hard(b,q,384)-preflight.hard(b,q,192)).imag))) for q in Q2S}}
            records.append({'directory':d,'operators_sha256':sha(common.NLO/d/'operators.npz'),
                            'KEY_sha256':sha(common.NLO/d/'KEY.json'),'nfev':nfev,'dimension':n})
        pairs=[]
        for l,r,wgt in PAIRS:
            b,bb=bs[l],bs[r];edges=np.unique(np.r_[b.edges,bb.edges]);z,w=np.polynomial.legendre.leggauss(128)
            x=np.concatenate([(a+b)/2+(b-a)*z/2 for a,b in zip(edges[:-1],edges[1:])]);ww=np.concatenate([(b-a)*w/2 for a,b in zip(edges[:-1],edges[1:])])
            pairs.append({'left':l,'right':r,'weight':wgt,'w':tensor(ww),
                          'lq':tensor(b.basis(x,'odd')),'lg':tensor(b.basis(x,'even')),
                          'rq':tensor(bb.basis(x,'odd')),'rg':tensor(bb.basis(x,'even'))})
        results.append({'xi':point['xi'],'data':ds,'pairs':pairs})
        print('contractions ready xi',point['xi'],'seconds',time.monotonic()-t,flush=True)
    data={'points':results,'spin2':{str(q):tensor(spin2_map(q)) for q in Q2S}}
    save(path,data)
    write(HERE/'CONTRACTION_BUILD.json',{'scope':'Linear propagations/contractions of supplied generators; no kernel integration, no fitting, no target data',
          'rtol':1e-13,'atol':1e-15,'method':'DOP853 ordered matrix identities','sources':records,
          'seconds':time.monotonic()-t,'peak_RSS_bytes':rss(),'sha256':sha(path)})

class Domain:
    def __init__(self,margin=.95,order=192):
        self.data=torch.load(HERE/'inputs/contractions.pt',weights_only=True)
        self.mom=Moments(order);self.margin=margin
        self.names=['moment_predictive_RMS_80_over_1e-3']
        for point in self.data['points']:
            for q in Q2S:
                prefix=f'xi{point["xi"]}_Q2{q}'
                self.names += [prefix+'_CFF_'+s for s in ('Re','Im')]
                self.names += [prefix+'_field_'+s for s in ('Sigma','T','Hg')]
                self.names += [prefix+'_transport_'+m+'_'+s for m in ('weak10','strong10') for s in ('Sigma','Hg','T')]
    def constraints(self,p,x):
        vals=[(self.mom.value(p,x)/1e-3-1).reshape(1)]
        for point in self.data['points']:
            starts={}
            for label,d in point['data'].items():
                h=p.field(x,d['nodes'][None],tensor([point['xi']]))[0]
                starts[label]=torch.cat((h[0,d['qids']],h[1,d['qids']],h[2,d['gids']]))
            for q in Q2S:
                key=str(q);ds=point['data'];hs={k:d['U'][key]@starts[k] for k,d in ds.items()}
                cs={k:d['hard'][key]@hs[k] for k,d in ds.items()}
                reserve=1e-6*max(1,np.log(q/2)/np.log(2.3/2))
                cb=x.new_full((2,),reserve);fb=x.new_full((3,),reserve)
                for pair in point['pairs']:
                    l,r=pair['left'],pair['right'];nl,nr=ds[l]['nq'],ds[r]['nq']
                    fl=torch.stack((pair['lq']@hs[l][:nl],pair['lq']@hs[l][nl:2*nl],pair['lg']@hs[l][2*nl:]))
                    fr=torch.stack((pair['rq']@hs[r][:nr],pair['rq']@hs[r][nr:2*nr],pair['rg']@hs[r][2*nr:]))
                    err=torch.linalg.vector_norm((fl-fr)*torch.sqrt(pair['w']),dim=1)
                    den=torch.linalg.vector_norm(fr*torch.sqrt(pair['w']),dim=1).clamp_min(1e-15)
                    fb=fb+pair['weight']*err/den
                    cb=cb+pair['weight']*torch.abs(cs[l]-cs[r])
                for m in ('weak10','strong10'):cb=cb+2*torch.abs(ds[m]['hard_delta'][key]@hs[m])
                limit=1e-4*torch.abs(cs['strong10']).clamp_min(1.)
                vals.extend((cb/(self.margin*limit)-1,fb/(self.margin*1e-3)-1))
                for m in ('weak10','strong10'):
                    d=ds[m];ref=self.data['spin2'][key]@(d['M']@starts[m]);act=d['M']@hs[m]
                    vals.append(torch.abs(act-ref)/(self.margin*1e-4*torch.abs(ref).clamp_min(1e-15))-1)
        return torch.cat(vals)

def full_forward(p,x,out):
    p.set_model(x);out=Path(out);records=[]
    for point in common.operator_directories():
        for q in Q2S:
            d=preflight.validate_point(point,q,'d000_feasible_domain',p.model)
            filename=f'xi{point["xi"]}_Q2{q}.json';write(out/filename,d)
            records.append({k:d[k] for k in ('xi','Q2','pass','gates','CFF_estimated_total','CFF_limits')})
            records[-1].update(field_totals={k:v['estimated_total'] for k,v in d['fields'].items()},file=filename,sha256=sha(out/filename))
    return {'pass':all(r['pass'] for r in records),'records':records,'all_15_points':True}

def moment_validation(p,x):
    xf,xd=map(np.asarray,common.helpers.moment_grids());new=.02+.96*(np.arange(160)+.25)/160
    assert np.min(abs(new[:,None]-xf))>1e-12 and np.min(abs(new[:,None]-xd))>1e-12
    result={'new_grid':new.tolist(),'disjoint':True,'grids':{}}
    with torch.no_grad():
        for name,grid in [('enforced80',xd),('independent160',new)]:
            rr={order:Moments(order,grid).residual(p,x).numpy() for order in (96,192,384)}
            d={'quadrature_RMS':{str(k):float(np.sqrt(np.mean(v*v))) for k,v in rr.items()},
               'residual_refinement_RMS_192_384':float(np.sqrt(np.mean((rr[384]-rr[192])**2))),
               'entries':[{'channel':ch,'spin':N,'rms':float(np.sqrt(np.mean(rr[384][:,i]**2))),
                           'max_abs':float(np.max(abs(rr[384][:,i])))} for i,(ch,N,_) in enumerate(common.poly.labels())]}
            d['pass']=d['quadrature_RMS']['384']<=1e-3 and d['residual_refinement_RMS_192_384']<=1e-7
            result['grids'][name]=d
        pp=common.poly.MomentProjector(new,spins=(10,12),order=384,scales=[common.helpers.moment_scale(c,n) for c in ('Sigma','T','Hg') for n in (10,12)])
        result['spins10_12_diagnostic_RMS']=float(pp(lambda a,b:p.field(x,a,b)).sqrt())
    result['pass']=all(d['pass'] for d in result['grids'].values())
    result['scope']='Finite developmental moment grids; not all-order/global polynomiality'
    return result

if __name__=='__main__':build_contractions()
