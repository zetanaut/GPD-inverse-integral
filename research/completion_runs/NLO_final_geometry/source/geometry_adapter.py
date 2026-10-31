"""Explicitly indexed geometry; unchanged reference FiLM and moment objective."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,math,json
ROOT=Path(os.environ.get('GPD_REVIEW_ROOT',Path(__file__).resolve().parents[3]))
sys.path.insert(0,str(ROOT/'GPD_direct_error_calibration'))
import fit_nlo as smoke
from fit_nlo import np,torch,tensor,common
from torch.func import jacrev,vmap

def row_id(xi,q,component):return f'xi={float(xi).hex()}|Q2={float(q).hex()}|{component}'

class Bank:
    def __init__(self,points):
        self.points=tuple(points)
        if not self.points:raise ValueError('empty geometry')
        keys=[(float(p['xi']),float(p['Q2'])) for p in points]
        if len(set(keys))!=len(keys):raise ValueError('duplicate complex point')
        self.ids=tuple(row_id(x,q,c) for x,q in keys for c in ('Re','Im'))
        self.x=tensor(np.stack([p['x'] for p in points]))
        self.xi=tensor([p['xi'] for p in points])
        self.rows=tensor(np.stack([p['rows'] for p in points]))
        if self.rows.shape!=(len(points),2,3,self.x.shape[1]):raise ValueError('row dimensions')
        if not all(torch.isfinite(z).all() for z in (self.x,self.xi,self.rows)):raise ValueError('nonfinite bank')
        self.n_scalar=len(self.ids)
    def subset(self,indices):return Bank([self.points[i] for i in indices])

def original_bank():
    with np.load(ROOT/'completion_runs/NLO_feasible_domain/inputs/fit_bank.npz',allow_pickle=False) as z:
        return Bank([{'xi':float(z['xi'][i]),'Q2':q,'x':z['x'][i].copy(),'rows':z['rows'][i,j].copy(),
                      'source':'immutable existing fit_bank.npz','role':'likelihood'}
                     for i in range(len(z['xi'])) for j,q in enumerate((2.3,4.7,9.3))])

class Problem(smoke.Problem):
    def __init__(self,y,cov,seed,bank,input_row_ids):
        if seed not in (31,43):raise ValueError('fixed starts31/43 only')
        ids=tuple(input_row_ids)
        if len(ids)!=len(set(ids)) or set(ids)!=set(bank.ids):raise ValueError('explicit row IDs must match geometry exactly')
        perm=[ids.index(k) for k in bank.ids]
        y=np.asarray(y,dtype=float).reshape(-1);cov=np.asarray(cov,dtype=float)
        if len(y)!=bank.n_scalar or cov.shape!=(bank.n_scalar,bank.n_scalar):raise ValueError('data/covariance dimensions')
        y=y[perm];cov=cov[np.ix_(perm,perm)]
        if not np.isfinite(y).all() or not np.isfinite(cov).all() or not np.array_equal(cov,cov.T):raise ValueError('invalid covariance/data')
        with torch.random.fork_rng():
            torch.manual_seed(seed);self.model=common.ReferenceNetwork('film').double()
        self.spec=[(n,p.shape,p.numel()) for n,p in self.model.named_parameters()]
        self.x0=torch.cat([p.detach().flatten() for p in self.model.parameters()])
        assert self.x0.numel()==3867
        self.y=tensor(y);self.cov=tensor(cov);self.chol=torch.linalg.cholesky(self.cov)
        self.geometry_bank=bank;self.n_scalar=bank.n_scalar;self.p=common.projector(48)
        self.complements=[]
        for _,N,_ in self.p.spec:
            P=common.poly.design(self.p.xi.numpy(),N);Q=np.linalg.qr(P,mode='complete')[0]
            self.complements.append(tensor(Q[:,P.shape[1]:]))
    def prediction(self,x):
        b=self.geometry_bank;h=self.field(x,b.x,b.xi)
        return torch.einsum('prcn,pcn->pr',b.rows,h).flatten()
    def components(self,x):
        r=torch.linalg.solve_triangular(self.chol,(self.prediction(x)-self.y)[:,None],upper=False).flatten()
        return torch.stack((r.square().sum()/self.n_scalar,self.p(lambda a,b:self.field(x,a,b))))
    def residual(self,x):
        r=torch.linalg.solve_triangular(self.chol,(self.prediction(x)-self.y)[:,None],upper=False).flatten()/math.sqrt(self.n_scalar)
        h=self.field(x,self.p.x,self.p.xi)
        M=[(h[:,ch]*getattr(self.p,f'xw{i}')).sum(-1) for i,(ch,_,_) in enumerate(self.p.spec)]
        poly=torch.cat([q.T@m/self.p.scales[i] for i,(q,m) in enumerate(zip(self.complements,M))])/math.sqrt(492)
        return torch.cat((r,poly))
    def residual_jacobian(self,x):
        b=self.geometry_bank
        def cff(theta,xx,xi,rows):
            h=self.field(theta,xx[None,:],xi[None])[0]
            return torch.einsum('rcn,cn->r',rows,h)
        jd=vmap(jacrev(cff),in_dims=(None,0,0,0))(x,b.x,b.xi,b.rows).reshape(self.n_scalar,-1)
        jd=torch.linalg.solve_triangular(self.chol,jd,upper=False)/math.sqrt(self.n_scalar)
        ch=torch.tensor([s[0] for s in self.p.spec]);xw=torch.stack([getattr(self.p,f'xw{i}') for i in range(12)],1)
        def moments(theta,xx,xi,weights):
            h=self.field(theta,xx[None,:],xi[None])[0];return (h[ch]*weights).sum(-1)
        jm=vmap(jacrev(moments),in_dims=(None,0,0,0))(x,self.p.x,self.p.xi,xw)
        jp=torch.cat([q.T@jm[:,i,:]/self.p.scales[i] for i,q in enumerate(self.complements)],0)/math.sqrt(492)
        return torch.cat((jd,jp))
    def independent_objective(self,x):
        r=self.prediction(x)-self.y;data=r@torch.linalg.solve(self.cov,r)/self.n_scalar
        M=self.p.moments(lambda a,b:self.field(x,a,b));pen=0.
        for i,(_,N,_) in enumerate(self.p.spec):
            P=tensor(common.poly.design(self.p.xi.numpy(),N))
            a=torch.linalg.lstsq(P,M[:,i,None],driver='gelsd').solution[:,0]
            residual=(M[:,i]-P@a)/self.p.scales[i];pen=pen+residual@residual
        return data+pen/492

class Domain:
    """One moment norm and11 original numerical inequalities per explicit point."""
    def __init__(self,points,spin2,margin=.95,order=192):
        self.points=points;self.spin2=spin2;self.margin=margin
        from domain import Moments
        self.mom=Moments(order);self.names=['moment_predictive_RMS_80_over_1e-3']
        for point in points:
            prefix=f'xi{point["xi"]}_Q2{point["Q2"]}'
            self.names += [prefix+'_CFF_'+s for s in ('Re','Im')]
            self.names += [prefix+'_field_'+s for s in ('Sigma','T','Hg')]
            self.names += [prefix+'_transport_'+m+'_'+s for m in ('weak10','strong10') for s in ('Sigma','Hg','T')]
    def constraints(self,p,x):
        vals=[(self.mom.value(p,x)/1e-3-1).reshape(1)]
        for point in self.points:
            q=point['Q2'];key=str(q);ds=point['data'];starts={}
            for label,d in ds.items():
                h=p.field(x,d['nodes'][None],tensor([point['xi']]))[0]
                starts[label]=torch.cat((h[0,d['qids']],h[1,d['qids']],h[2,d['gids']]))
            hs={k:d['U'][key]@starts[k] for k,d in ds.items()}
            cs={k:d['hard'][key]@hs[k] for k,d in ds.items()}
            reserve=1e-6*max(1,np.log(q/2)/np.log(2.3/2))
            cb=x.new_full((2,),reserve);fb=x.new_full((3,),reserve)
            for pair in point['pairs']:
                l,r=pair['left'],pair['right'];nl,nr=ds[l]['nq'],ds[r]['nq']
                fl=torch.stack((pair['lq']@hs[l][:nl],pair['lq']@hs[l][nl:2*nl],pair['lg']@hs[l][2*nl:]))
                fr=torch.stack((pair['rq']@hs[r][:nr],pair['rq']@hs[r][nr:2*nr],pair['rg']@hs[r][2*nr:]))
                err=torch.linalg.vector_norm((fl-fr)*torch.sqrt(pair['w']),dim=1)
                den=torch.linalg.vector_norm(fr*torch.sqrt(pair['w']),dim=1).clamp_min(1e-15)
                fb=fb+pair['weight']*err/den;cb=cb+pair['weight']*torch.abs(cs[l]-cs[r])
            for m in ('weak10','strong10'):cb=cb+2*torch.abs(ds[m]['hard_delta'][key]@hs[m])
            limit=1e-4*torch.abs(cs['strong10']).clamp_min(1.)
            vals.extend((cb/(self.margin*limit)-1,fb/(self.margin*1e-3)-1))
            for m in ('weak10','strong10'):
                d=ds[m];ref=self.spin2[key]@(d['M']@starts[m]);act=d['M']@hs[m]
                vals.append(torch.abs(act-ref)/(self.margin*1e-4*torch.abs(ref).clamp_min(1e-15))-1)
        return torch.cat(vals)

def original_domain():
    data=torch.load(ROOT/'completion_runs/NLO_feasible_domain/inputs/contractions.pt',weights_only=True)
    return Domain([dict(p,Q2=q) for p in data['points'] for q in (2.3,4.7,9.3,3.1,6.1)],data['spin2'])
