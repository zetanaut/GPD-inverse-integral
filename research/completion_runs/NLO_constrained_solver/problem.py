"""Warm-start-only adapter to the immutable feasible-domain packet.

All functions defining the model, J, inequalities and authoritative validation
are imported unchanged. Only residual/Jacobian plumbing is new.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,math,time
from pathlib import Path
import numpy as np
import torch
from torch.func import jacrev,vmap
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=HERE.parent/'NLO_feasible_domain'
sys.path.insert(0,str(OLD))
import domain as original
from domain import tensor,write,save,sha,rss,common
from optimizer import necessary_kkt

class Problem(original.Problem):
    def __init__(self,label):
        # This constructor reads original data/model definitions; no optimization.
        super().__init__(label)
        state=torch.load(HERE/f'inputs/initial{label}.pt',weights_only=True)
        self.x0=state['flat'].detach().clone();self.set_model(self.x0)
        if 'state_dict' in state:
            assert all(torch.equal(self.model.state_dict()[k],v) for k,v in state['state_dict'].items())
        self.complements=[]
        for _,N,_ in self.p.spec:
            P=common.poly.design(self.p.xi.numpy(),N);Q=np.linalg.qr(P,mode='complete')[0]
            self.complements.append(tensor(Q[:,P.shape[1]:]))
    def residual(self,x):
        h=self.field(x,self.bank['x'],self.bank['xi'])
        pred=torch.einsum('sqrcn,scn->sqr',self.bank['rows'],h).flatten()
        r=torch.linalg.solve_triangular(self.chol,(pred-self.y)[:,None],upper=False).flatten()/math.sqrt(18)
        h=self.field(x,self.p.x,self.p.xi)
        M=[(h[:,ch]*getattr(self.p,f'xw{i}')).sum(-1) for i,(ch,_,_) in enumerate(self.p.spec)]
        poly=torch.cat([q.T@m/self.p.scales[i] for i,(q,m) in enumerate(zip(self.complements,M))])/math.sqrt(492)
        return torch.cat((r,poly))
    def residual_jacobian(self,x):
        # Same local-chain algorithm as accepted task04 solver.py, with bank buffers.
        def cff(theta,xx,xi,rows):
            h=self.field(theta,xx[None,:],xi[None])[0]
            return torch.einsum('qrcn,cn->qr',rows,h).flatten()
        jd=vmap(jacrev(cff),in_dims=(None,0,0,0))(x,self.bank['x'],self.bank['xi'],self.bank['rows'])
        jd=torch.linalg.solve_triangular(self.chol,jd.reshape(18,-1),upper=False)/math.sqrt(18)
        ch=torch.tensor([s[0] for s in self.p.spec]);xw=torch.stack([getattr(self.p,f'xw{i}') for i in range(12)],1)
        def moments(theta,xx,xi,weights):
            h=self.field(theta,xx[None,:],xi[None])[0]
            return (h[ch]*weights).sum(-1)
        jm=vmap(jacrev(moments),in_dims=(None,0,0,0))(x,self.p.x,self.p.xi,xw)
        jp=torch.cat([q.T@jm[:,i,:]/self.p.scales[i] for i,q in enumerate(self.complements)],0)/math.sqrt(492)
        return torch.cat((jd,jp))

class Oracle:
    def __init__(self,label):
        self.p=Problem(label);self.domain=original.Domain();self.n=self.p.x0.numel()
        self.names=self.domain.names;self.x0=self.p.x0.numpy().copy();self.cached=None
        self.counts={'evaluations':0,'objective_gradients':0,'residual_jacobians':0,
                     'constraint_jacobian_builds':0,'constraint_jacobian_rows':0}
        self.timers={'evaluation_seconds':0.,'residual_jacobian_seconds':0.,'constraint_jacobian_seconds':0.}
    def values(self,a):
        t=time.monotonic()
        with torch.no_grad():
            x=tensor(a);r=self.p.residual(x).numpy();comp=self.p.components(x).numpy();c=self.domain.constraints(self.p,x).numpy()
        self.counts['evaluations']+=1;self.timers['evaluation_seconds']+=time.monotonic()-t
        return {'J':float(comp.sum()),'components':comp,'residual':r,'c':c}
    def derivatives(self,a,indices):
        if self.cached is None or not np.array_equal(a,self.cached['x']):
            t=time.monotonic();x=tensor(a).detach().requires_grad_(True)
            j=self.p.objective(x);g=torch.autograd.grad(j,x)[0].detach().numpy()
            self.counts['objective_gradients']+=1
            R=self.p.residual_jacobian(x.detach()).detach().numpy()
            self.counts['residual_jacobians']+=1;self.timers['residual_jacobian_seconds']+=time.monotonic()-t
            t=time.monotonic();c=self.domain.constraints(self.p,x)
            self.cached={'x':np.array(a,copy=True),'tensor':x,'g':g,'R':R,'c':c,'rows':{}}
            self.counts['constraint_jacobian_builds']+=1
            self.timers['constraint_jacobian_seconds']+=time.monotonic()-t
        t=time.monotonic();cache=self.cached
        for i in indices:
            if int(i) not in cache['rows']:
                cache['rows'][int(i)]=torch.autograd.grad(cache['c'][int(i)],cache['tensor'],retain_graph=True)[0].detach().numpy()
                self.counts['constraint_jacobian_rows']+=1
        A=np.stack([cache['rows'][int(i)] for i in indices]) if len(indices) else np.empty((0,self.n))
        self.timers['constraint_jacobian_seconds']+=time.monotonic()-t
        return cache['g'],cache['R'],A
    def constraint_rows(self,a,indices):
        # Trial-point normals for second-order feasibility correction only.
        t=time.monotonic();x=tensor(a).detach().requires_grad_(True);c=self.domain.constraints(self.p,x)
        A=np.stack([torch.autograd.grad(c[int(i)],x,retain_graph=True)[0].detach().numpy() for i in indices]) if len(indices) else np.empty((0,self.n))
        self.counts['constraint_jacobian_builds']+=1;self.counts['constraint_jacobian_rows']+=len(indices)
        self.timers['constraint_jacobian_seconds']+=time.monotonic()-t
        return A

def kkt(g,c,A,indices,names):
    # Exactly the old first-order gate, independent of working-set window.
    chosen=np.flatnonzero(c[indices]>=-1e-6)
    active=indices[chosen]
    return necessary_kkt(g,c,A[chosen],active,names)
