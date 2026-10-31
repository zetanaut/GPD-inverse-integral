"""Generalized component-selective, actual-t original FiLM residual adapter.

No truth functions are imported or available to this object. The independent
reference targets are provided by the preparation controller, not this fitter.
This module DOES NOT replace the established optimizer or numerical-domain code.
"""
from __future__ import annotations
import math,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call,jacrev
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from gpd_reh.model import ExperimentalFiLM
from gpd_reh.moments import FixedTMoments
from .bank import RowBank

class GroupedForward:
    """Evaluate H once per (xi,t,node set), then reuse it for every Q2 row."""
    def __init__(self,geometry,bank,method='nominal'):
        groups={};self.n=len(geometry)
        import hashlib
        for i,r in enumerate(geometry):
            x,w=bank.get(r['xi'],r['Q2_GeV2'],method)
            key=(float(r['xi']).hex(),float(r['t_GeV2']).hex(),hashlib.sha256(x.tobytes()).hexdigest())
            if key not in groups:groups[key]={'x':x,'xi':r['xi'],'t':r['t_GeV2'],'ids':[],'weights':[]}
            groups[key]['ids'].append(i);groups[key]['weights'].append(w)
        self.groups=[]
        for g in groups.values():
            self.groups.append({'x':torch.tensor(g['x'][None,:],dtype=torch.float64),
                'xi':torch.tensor([g['xi']],dtype=torch.float64),'t':torch.tensor([g['t']],dtype=torch.float64),
                'ids':torch.tensor(g['ids'],dtype=torch.long),'weights':torch.tensor(np.array(g['weights']),dtype=torch.float64)})
    def __call__(self,model):
        out=None
        for g in self.groups:
            h=model(g['x'],g['xi'],g['t'])[0]
            v=torch.einsum('bocn,cn->bo',g['weights'],h)
            if out is None:out=v.new_zeros(self.n,2)
            out=out.index_copy(0,g['ids'],v)
        return out

class ClosureProblem:
    def __init__(self,observations,bank,study,seed=31,moment_order=48):
        # Store only fit settings: generator coefficients never enter the estimator.
        import copy
        self.study={k:copy.deepcopy(study[k]) for k in ("network","polynomiality","solver")}
        with np.load(observations,allow_pickle=False) as z:
            self.components_index=torch.tensor(z['components'],dtype=torch.long)
            self.y=torch.tensor(z['y'].copy(),dtype=torch.float64)
            self.sigma=torch.tensor(z['sigma'].copy(),dtype=torch.float64)
            self.mass=torch.tensor(z['mass'].copy(),dtype=torch.float64)
            self.row_ids=z['row_id'].tolist()
            self.geometry=[{'xi':float(xi),'Q2_GeV2':float(q),'t_GeV2':float(t)} for xi,q,t in zip(z['xi'],z['Q2'],z['t'])]
        if self.components_index.tolist() not in ([0],[0,1]):raise ValueError('Wrong observation components')
        if self.y.shape!=(len(self.geometry),len(self.components_index)) or self.sigma.shape!=self.y.shape:raise ValueError('Data shape mismatch')
        if not torch.isfinite(self.y).all() or not torch.isfinite(self.sigma).all() or not (self.sigma>0).all():raise ValueError('Invalid observed values/weights')
        if not torch.isfinite(self.mass).all() or not (self.mass>0).all() or abs(float(self.mass.sum())-1)>1e-12:raise ValueError('Positive normalized observation mass required')
        self.forward=GroupedForward(self.geometry,bank,'nominal')
        net=study['network'];self.seed=int(seed)
        with torch.random.fork_rng():
            torch.manual_seed(self.seed);self.model=ExperimentalFiLM(net['t_mean'],net['t_std']).double()
        self.spec=[(n,p.shape,p.numel()) for n,p in self.model.named_parameters()]
        self.x0=torch.cat([p.detach().flatten() for p in self.model.parameters()])
        if self.x0.numel()!=3867:raise ValueError('Architecture drift')
        pol=study['polynomiality'];self.p=FixedTMoments(pol['t_nodes'],np.array(pol['xi_fit']),order=moment_order,spins=tuple(pol['spins']))
        self.lam=float(pol['lambda']);self.n_scalar=self.y.numel()
    def parameters(self,flat):
        if flat.shape!=self.x0.shape or flat.dtype!=torch.float64:raise ValueError('Wrong flat state')
        p={};i=0
        for name,shape,n in self.spec:p[name]=flat[i:i+n].view(shape);i+=n
        return p
    def field(self,flat,x,xi,t):return functional_call(self.model,self.parameters(flat),(x,xi,t))
    def residual(self,flat):
        model=lambda x,xi,t:self.field(flat,x,xi,t)
        pred=self.forward(model).index_select(1,self.components_index)
        data=((pred-self.y)/self.sigma)*torch.sqrt(self.mass[:,None])
        return torch.cat((data.flatten(),math.sqrt(self.lam)*self.p.residual(model)))
    def objective(self,flat):return self.residual(flat).square().sum()
    def original_objective(self,flat):return self.objective(flat)
    def components(self,flat):
        r=self.residual(flat);return torch.stack((r[:self.n_scalar].square().sum(),r[self.n_scalar:].square().sum()))
    def jacobian(self,flat,chunk_size=8):
        """Block AD AFTER row/moment integration; no entire-field Jacobian."""
        flat=flat.detach();data=[None]*len(self.geometry)
        for g in self.forward.groups:
            ci=self.components_index
            factor=torch.sqrt(self.mass[g['ids']])[:,None]/self.sigma[g['ids']]
            def local(v):
                h=self.field(v,g['x'],g['xi'],g['t'])[0]
                return torch.einsum('bocn,cn->bo',g['weights'],h)[:,ci]*factor
            blocks=jacrev(local,chunk_size=chunk_size)(flat).detach()
            for i,k in enumerate(g['ids'].tolist()):data[k]=blocks[i]
        data=torch.cat(data)
        poly=[];channels=torch.tensor([c for c,_,_ in self.p.spec],dtype=torch.long)
        for t in self.p.t_nodes:
            jm=[]
            for j in range(len(self.p.xi)):
                def local_m(v):
                    h=self.field(v,self.p.x[j:j+1],self.p.xi[j:j+1],t.reshape(1))[0]
                    return (h[channels]*self.p.xw[j]).sum(-1)
                jm.append(jacrev(local_m,chunk_size=chunk_size)(flat).detach())
            jm=torch.stack(jm)
            for i in range(len(self.p.spec)):
                poly.append(getattr(self.p,f'complement{i}').T@jm[:,i,:]/self.p.scales[i])
        return torch.cat((data,math.sqrt(self.lam)*torch.cat(poly)/math.sqrt(self.p.normalization)))
    def residual_jacobian(self,flat):return self.jacobian(flat)
    def set_model(self,flat):
        self.model.load_state_dict({k:v.detach() for k,v in self.parameters(flat).items()});return self.model
