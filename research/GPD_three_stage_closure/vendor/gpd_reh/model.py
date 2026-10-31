"""Same 3867-parameter FiLM; t activates its existing second condition input."""
from __future__ import annotations
import math
import torch
from .reference_model import ReferenceNetwork

class ExperimentalFiLM(ReferenceNetwork):
    def __init__(self,t_mean,t_std):
        if not math.isfinite(t_mean) or not math.isfinite(t_std) or t_std<=0:
            raise ValueError('Positive finite coordinate-only t standard deviation required')
        super().__init__('film');self.t_mean=float(t_mean);self.t_std=float(t_std)
    def raw_at_t(self,x,xi,t):
        c=torch.stack(((xi.square()-.18)/.18,(t-self.t_mean)/self.t_std),dim=-1)
        h=torch.tanh(self.input(x[...,None]))
        for block,condition in zip(self.blocks,self.conditioners):
            gamma,beta=condition(c).chunk(2,dim=-1)
            h=h+torch.tanh((1+gamma[:,None,:])*block(h)+beta[:,None,:])/2
        return self.output(h)
    def forward(self,x,xi,t):
        # Values/domains are checked once by the observation/bank adapter, outside
        # torch.func transforms. This forward remains differentiable and vmap-safe.
        if x.ndim!=2 or xi.shape!=(x.shape[0],) or t.shape!=xi.shape:
            raise ValueError('Expected x[B,N], xi[B], t[B]')
        if x.dtype!=torch.float64 or xi.dtype!=x.dtype or t.dtype!=x.dtype or xi.device!=x.device or t.device!=x.device:
            raise ValueError('Use float64 coordinates on the model device')
        p=self.raw_at_t(x,xi,t);m=self.raw_at_t(-x,xi,t)
        raw=torch.cat(((p[...,:2]-m[...,:2])/2,(p[...,2:]+m[...,2:])/2),dim=-1)
        return (raw*torch.clamp(1-x.square(),min=0).square()[...,None]).transpose(-1,-2)
