"""Differentiable moment-space variable projection with unknown coefficients.

Spin N: quark power m=N-1; gluon power m=N-2 for Hg=xg.
Allowed skewness powers: 0,2,...,N. This does not constrain all moments.
No derivative penalty is imposed. Fixed numerical normalization is not a likelihood.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from numpy.polynomial.legendre import leggauss
import torch
from torch import nn

CHANNELS=('Sigma','T','G')

def make_grid(xi,order=48):
 xi=np.asarray(xi,dtype=float)
 if xi.ndim!=1 or np.any(xi<=0) or np.any(xi>=1):raise ValueError('Require one-dimensional 0<xi<1')
 u,w=leggauss(order);u=(u+1)/2;w=w/2
 x=np.concatenate((xi[:,None]*u,xi[:,None]+(1-xi[:,None])*u),axis=1)
 weights=np.concatenate((xi[:,None]*w,(1-xi[:,None])*w),axis=1)
 return x,weights

def labels(spins=(2,4,6,8)):
 return [(ch,N,N-1 if ch<2 else N-2) for ch in range(3) for N in spins]

def design(xi,N,powers=None,xi_scale=.6):
 if powers is None:powers=tuple(range(N//2+1))
 z=(np.asarray(xi)/xi_scale)**2
 return np.stack([z**k for k in powers],axis=1) if len(powers) else np.zeros((len(z),0))

class MomentProjector(nn.Module):
 def __init__(self,xi,spins=(2,4,6,8),order=48,scales=None,supports=None):
  super().__init__();xi=np.asarray(xi,dtype=float);x,w=make_grid(xi,order)
  self.spec=labels(spins);self.supports=supports or [list(range(N//2+1)) for ch,N,m in self.spec]
  self.register_buffer('xi',torch.tensor(xi,dtype=torch.float64));self.register_buffer('x',torch.tensor(x,dtype=torch.float64))
  self.register_buffer('weights',torch.tensor(w,dtype=torch.float64))
  self.register_buffer('scales',torch.ones(len(self.spec),dtype=torch.float64) if scales is None else torch.as_tensor(scales,dtype=torch.float64).clone())
  for i,((ch,N,m),support) in enumerate(zip(self.spec,self.supports)):
   P=design(xi,N,support);Q=np.linalg.qr(P,mode='reduced')[0] if P.shape[1] else P
   self.register_buffer(f'Q{i}',torch.tensor(Q,dtype=torch.float64))
   self.register_buffer(f'xw{i}',torch.tensor(2*w*x**m,dtype=torch.float64))
 def moments(self,model):
  h=model(self.x,self.xi)
  return torch.stack([(h[:,ch]*getattr(self,f'xw{i}')).sum(dim=-1) for i,(ch,N,m) in enumerate(self.spec)],dim=1)
 def residual(self,M):
  out=[]
  for i in range(len(self.spec)):
   q=getattr(self,f'Q{i}');v=M[:,i];out.append((v-q@(q.T@v))/self.scales[i])
  return torch.stack(out,dim=1)
 def forward(self,model):return self.residual(self.moments(model)).square().mean()
 def set_scales(self,M,floor):
  with torch.no_grad():self.scales.copy_(M.square().mean(0).sqrt().clamp_min(floor))

def fit_coefficients(xi,M,spec,supports=None):
 coeff=[]
 for i,(_,N,_) in enumerate(spec):
  ss=list(range(N//2+1)) if supports is None else supports[i]
  P=design(xi,N,ss);a=np.linalg.lstsq(P,M[:,i],rcond=1e-12)[0]
  full=np.zeros(N//2+1);full[ss]=a;coeff.append(full)
 return coeff

def evaluate_coefficients(xi,coeff,spec):
 return np.column_stack([design(xi,N)@a for a,(_,N,_) in zip(coeff,spec)])

def best_subset(xi,y,N,tolerance):
 """Exhaustive deterministic polynomial support search. A compression rule, not significance testing."""
 candidates=[];den=max(float(np.linalg.norm(y)),1e-30)
 for size in range(N//2+2):
  for support in combinations(range(N//2+1),size):
   P=design(xi,N,support)
   a=np.linalg.lstsq(P,y,rcond=1e-12)[0] if size else np.zeros(0)
   err=float(np.linalg.norm(P@a-y)/den)
   candidates.append((err,support,a))
  valid=[v for v in candidates if len(v[1])==size and v[0]<=tolerance]
  if valid:
   err,support,a=min(valid,key=lambda v:(v[0],v[1]));break
 else:err,support,a=min(candidates,key=lambda v:v[0])
 full=np.zeros(N//2+1);full[list(support)]=a
 return {'support':list(support),'coefficient_scaled':full.tolist(),'compression_rel':err}

def symbolic_candidates(xi,M,spec,tolerance):
 # Search only the all-allowed polynomial projection; raw nonpolynomial leakage is recorded separately.
 full=fit_coefficients(xi,M,spec);smooth=evaluate_coefficients(xi,full,spec)
 return [best_subset(xi,smooth[:,i],N,tolerance) for i,(_,N,_) in enumerate(spec)]
