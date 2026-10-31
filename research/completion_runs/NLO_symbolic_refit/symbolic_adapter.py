"""Moment-subspace adapter; leaves GPD network, data, normalization and NLO rows intact.
Supports contain powers of z=(xi/0.6)^2, NOT powers of xi.
Coefficients remain variable-projected, not fixed to their proposal values.
Call only in a separately declared symbolic hypothesis branch.
"""
from __future__ import annotations
import importlib
import numpy as np
import torch

def install_supports(problem, supports):
    """Replace ONLY the allowed moment subspaces and matching residual complements."""
    poly=importlib.import_module(type(problem.p).__module__)
    old=problem.p
    if len(supports)!=len(old.spec):raise ValueError('One support per existing moment is required')
    clean=[]
    for ss,(_,spin,_) in zip(supports,old.spec):
        if any(isinstance(k,(bool,np.bool_)) or int(k)!=k for k in ss):raise ValueError('Integer z exponents required')
        ss=list(map(int,ss))
        if ss!=sorted(set(ss)) or any(k<0 or k>spin//2 for k in ss):raise ValueError('Support must be sorted, unique and physically allowed')
        clean.append(ss)
    xi=old.xi.detach().cpu().numpy();spins=list(dict.fromkeys(s for _,s,_ in old.spec))
    order=old.x.shape[1]//2
    p=poly.MomentProjector(xi,spins=spins,order=order,scales=old.scales.detach().cpu().numpy(),supports=clean).to(device=old.x.device,dtype=old.x.dtype)
    complements=[]
    for ss,(_,spin,_) in zip(clean,p.spec):
        P=poly.design(xi,spin,ss);Q=np.linalg.qr(P,mode='complete')[0]
        complements.append(torch.as_tensor(Q[:,len(ss):],dtype=old.x.dtype,device=old.x.device))
    problem.p=p;problem.complements=complements
    # Keep original normalization J * number of moments, NOT residual rank.
    problem.npoly=p.xi.numel()*len(p.spec)
    return problem

def propose_supports(problem,tolerance=0.001):
    """One fixed full-space compression hypothesis, from the fitted network only."""
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Positive finite tolerance required')
    poly=importlib.import_module(type(problem.p).__module__);old=problem.p
    xi=old.xi.detach().cpu().numpy();spins=list(dict.fromkeys(s for _,s,_ in old.spec))
    p=poly.MomentProjector(xi,spins=spins,order=96,scales=old.scales.detach().cpu().numpy()).to(device=old.x.device,dtype=old.x.dtype)
    with torch.no_grad():mom=p.moments(problem.model).cpu().numpy()
    if not np.isfinite(mom).all():raise ValueError('Nonfinite fitted moments')
    records=poly.symbolic_candidates(xi,mom,p.spec,tolerance)
    for rec,(channel,spin,_) in zip(records,p.spec):
        rec.update(channel_index=channel,spin=spin,variable='z=(xi/0.6)^2',tolerance=tolerance,
                   coefficient_role='proposal diagnostic only; never frozen in GPD refit')
        rec['xi_powers']=[2*k for k in rec['support']]
        rec['coefficient_xi']=(np.asarray(rec['coefficient_scaled'])/(.6**(2*np.arange(spin//2+1)))).tolist()
    return records
