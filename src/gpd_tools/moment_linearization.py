"""Derivative of the existing 80-node moment residual; no change of constraints.

Imports no project modules. The caller supplies the unchanged problem and domain.
The returned vector e obeys ||e||_2 - 1 == domain.constraints(problem,theta)[0].
"""
import math
import numpy as np
import torch
from torch.func import jacrev, vmap

def residual_and_jacobian(problem, domain, theta, block=8):
    if block < 1: raise ValueError('block must be positive')
    mom=domain.mom
    den=1e-3*math.sqrt(len(mom.xd)*12)
    def raw(proj):
        channel=torch.tensor([s[0] for s in proj.spec],dtype=torch.long)
        weights=torch.stack([getattr(proj,f'xw{i}') for i in range(12)],dim=1)
        def one(t,xx,xi,ww):
            h=problem.field(t,xx[None,:],xi[None])[0]
            return (h[channel]*ww).sum(-1)
        jac=[]
        for lo in range(0,len(proj.xi),block):
            j=vmap(jacrev(one),in_dims=(None,0,0,0))(theta,proj.x[lo:lo+block],proj.xi[lo:lo+block],weights[lo:lo+block])
            jac.append(j.detach())
        return torch.cat(jac,dim=0)
    jf=raw(mom.pf);jd=raw(mom.pd)
    jj=torch.stack([(jd[:,i,:]-P@jf[:,i,:])/mom.pf.scales[i] for i,P in enumerate(mom.predict)],dim=1)
    with torch.no_grad():e=mom.residual(problem,theta)/den
    return e.flatten().numpy(),(jj/den).reshape(-1,theta.numel()).numpy()

def norm_model(e,E,step):
    e=np.asarray(e);E=np.asarray(E);s=np.asarray(step)
    if e.ndim!=1 or E.shape!=(len(e),len(s)):raise ValueError('incompatible shapes')
    if not all(np.isfinite(v).all() for v in (e,E,s)):raise ValueError('nonfinite input')
    v=e+E@s;n=float(np.linalg.norm(v))
    g=(v@E)/n if n else np.zeros_like(s)
    return n-1,g
