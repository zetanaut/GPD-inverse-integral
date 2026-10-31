"""All-allowed moments at fixed t; never fit one polynomial through different t."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from .polynomiality_base import make_grid,labels,design

SPINS=(2,4,6,8)

def scale(channel,N):
    return 16./((N+1)*(N+3)*(N+5)) if channel<2 else 16./((N-1)*(N+1)*(N+3))

class FixedTMoments(nn.Module):
    def __init__(self,t_nodes,xi_nodes=None,order=48,spins=SPINS):
        super().__init__();ts=np.asarray(t_nodes,float)
        xi=np.linspace(.02,.98,41) if xi_nodes is None else np.asarray(xi_nodes,float)
        if ts.ndim!=1 or not len(ts) or not np.isfinite(ts).all() or len(np.unique(ts))!=len(ts):
            raise ValueError('Unique finite fixed-t nodes required')
        if any(N<2 or N%2 for N in spins) or len(xi)<=max(spins)//2+1:
            raise ValueError('Invalid spins or insufficient xi collocation')
        x,w=make_grid(xi,order);self.spec=labels(spins);self.normalization=len(ts)*len(xi)*len(self.spec)
        self.register_buffer('t_nodes',torch.tensor(ts,dtype=torch.float64))
        self.register_buffer('xi',torch.tensor(xi,dtype=torch.float64))
        self.register_buffer('x',torch.tensor(x,dtype=torch.float64))
        self.register_buffer('xw',torch.tensor(np.stack([2*w*x**m for _,_,m in self.spec],axis=1),dtype=torch.float64))
        for i,(ch,N,_) in enumerate(self.spec):
            P=design(xi,N);Q=np.linalg.qr(P,mode='complete')[0]
            self.register_buffer(f'complement{i}',torch.tensor(Q[:,P.shape[1]:],dtype=torch.float64))
        self.register_buffer('scales',torch.tensor([scale(ch,N) for ch,N,_ in self.spec],dtype=torch.float64))
    def raw_moments(self,model,t):
        h=model(self.x,self.xi,torch.ones_like(self.xi)*t)
        return torch.stack([(h[:,ch]*self.xw[:,i]).sum(-1) for i,(ch,_,_) in enumerate(self.spec)],dim=1)
    def residual(self,model):
        out=[]
        for t in self.t_nodes:
            M=self.raw_moments(model,t)
            for i in range(len(self.spec)):
                out.append(getattr(self,f'complement{i}').T@M[:,i]/self.scales[i])
        return torch.cat(out)/math.sqrt(self.normalization)
    def forward(self,model):
        return self.residual(model).square().sum()
    def coefficients(self,model):
        """Diagnostics only. Physical-xi monomial coefficients, t varies freely."""
        result=[]
        with torch.no_grad():
            for t in self.t_nodes:
                M=self.raw_moments(model,t).cpu().numpy()
                for i,(ch,N,_) in enumerate(self.spec):
                    P=np.stack([self.xi.cpu().numpy()**(2*k) for k in range(N//2+1)],axis=1)
                    a=np.linalg.lstsq(P,M[:,i],rcond=None)[0]
                    result.append({'t':float(t),'channel':ch,'spin':N,'powers':list(range(0,N+1,2)),'coefficients':a.tolist()})
        return result
