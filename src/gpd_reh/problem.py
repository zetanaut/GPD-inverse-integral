"""Residual/Jacobian adapter for the existing finite-budget optimizer.

This module does not select/replace an optimizer or run experimental fitting.
It exposes the /N real-only data loss and fixed-t polynomiality consistently.
"""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call,jacrev
from .model import ExperimentalFiLM
from .moments import FixedTMoments
from .bank import ExperimentalForward

class ExperimentalProblem:
    def __init__(self,prepared,bank_directory,seed=31,moment_order=48):
        prepared=Path(prepared);meta=json.loads((prepared/'INPUT_SUMMARY.json').read_text())
        with np.load(prepared/'observations.npz',allow_pickle=False) as z:
            self.y=torch.tensor(z['y'],dtype=torch.float64);self.sigma=torch.tensor(z['sigma'],dtype=torch.float64)
            row_ids=tuple(z['row_id'].tolist())
        if self.y.ndim!=1 or not torch.isfinite(self.y).all() or not torch.isfinite(self.sigma).all() or not (self.sigma>0).all():
            raise ValueError('Invalid ReH observations or quoted uncertainties')
        self.forward=ExperimentalForward(prepared,bank_directory)
        if row_ids!=self.forward.row_ids:raise ValueError('Observation order changed')
        with torch.random.fork_rng():
            torch.manual_seed(seed);self.model=ExperimentalFiLM(meta['t_mean'],meta['t_std_population']).double()
        self.spec=[(name,p.shape,p.numel()) for name,p in self.model.named_parameters()]
        self.x0=torch.cat([p.detach().flatten() for p in self.model.parameters()]);self.seed=int(seed)
        self.p=FixedTMoments(meta['moment_t_nodes'],order=moment_order);self.lam=1.0
    def parameters(self,flat):
        if flat.ndim!=1 or flat.numel()!=self.x0.numel():raise ValueError('Wrong parameter vector')
        params={};i=0
        for name,shape,n in self.spec:params[name]=flat[i:i+n].view(shape);i+=n
        return params
    def field(self,flat,x,xi,t):
        return functional_call(self.model,self.parameters(flat),(x,xi,t))
    def residual(self,flat):
        model=lambda x,xi,t:self.field(flat,x,xi,t)
        prediction=self.forward(model)[:,0]   # Im predicted but never in the objective.
        data=(prediction-self.y)/self.sigma/math.sqrt(self.y.numel())
        return torch.cat((data,math.sqrt(self.lam)*self.p.residual(model)))
    def objective(self,flat):return self.residual(flat).square().sum()
    def independent_objective(self,flat):
        model=lambda x,xi,t:self.field(flat,x,xi,t)
        re=self.forward(model)[:,0]
        return ((re-self.y)/self.sigma).square().mean()+self.lam*self.p(model)
    def jacobian(self,flat,chunk_size=8):
        """Exact first derivative, integrating each field slice BEFORE AD.

        Avoid a batched reverse sweep through the entire t-xi field. Only a
        12-by-parameter block is differentiated at a time. The returned matrix
        is first-order solver data; higher derivatives of this method are not
        part of the interface. chunk_size is retained for call compatibility.
        """
        theta=flat.detach();data=[];polys=[]
        with torch.no_grad():
            for i in range(self.y.numel()):
                def datum(v):
                    h=self.field(v,self.forward.x[i:i+1],self.forward.xi[i:i+1],self.forward.t[i:i+1])[0]
                    return (self.forward.weights[i,0]*h).sum()/self.sigma[i]/math.sqrt(self.y.numel())
                data.append(jacrev(datum)(theta).detach())
            if self.lam:
                channels=torch.tensor([ch for ch,_,_ in self.p.spec],device=theta.device)
                for t in self.p.t_nodes:
                    blocks=[]
                    for i in range(len(self.p.xi)):
                        def local_moments(v):
                            h=self.field(v,self.p.x[i:i+1],self.p.xi[i:i+1],t.reshape(1))[0]
                            return (h[channels]*self.p.xw[i]).sum(-1)
                        blocks.append(jacrev(local_moments)(theta).detach())
                    Jm=torch.stack(blocks)
                    for i in range(len(self.p.spec)):
                        polys.append(getattr(self.p,f'complement{i}').T@Jm[:,i,:]/self.p.scales[i])
        Jd=torch.stack(data)
        if not self.lam:
            npoly=sum(getattr(self.p,f"complement{i}").shape[1] for i in range(len(self.p.spec)))*len(self.p.t_nodes)
            return torch.cat((Jd,torch.zeros(npoly,len(theta),dtype=theta.dtype,device=theta.device)))
        return torch.cat((Jd,math.sqrt(self.lam)*torch.cat(polys)/math.sqrt(self.p.normalization)))
    def set_model(self,flat):
        self.model.load_state_dict({k:v.detach() for k,v in self.parameters(flat).items()})
        return self.model
