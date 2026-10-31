"""Efficient ordered evolution of GIVEN matrices; does not certify the kernels.

The scalar two-loop coupling is solved once with dense output. No assumption
that K0 and K1 commute. No reuse of Lambda_LO as Lambda_NLO. Complex hard-row
adjoints use transpose without complex conjugation for the linear CFF map.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

@dataclass(frozen=True)
class Prescription:
    q02:float=2.0
    alpha0:float=.35691594851514724
    nf:int=3
    scheme:str='MSbar'
    def __post_init__(self):
        if self.nf!=3 or self.scheme!='MSbar' or self.q02<=0 or not 0<self.alpha0<1:raise ValueError('Only the frozen fixed-nf3 MSbar contract is supported')

def coupling(q2,p=Prescription()):
    if q2<=0 or not np.isfinite(q2):raise ValueError('Invalid Q2')
    end=float(np.log(q2/p.q02));a0=p.alpha0/(4*np.pi)
    if end==0:return lambda t:np.full_like(np.asarray(t,dtype=float),a0),0,0
    sol=solve_ivp(lambda t,a:-9*a*a-64*a*a*a,(0.,end),[a0],method='DOP853',dense_output=True,rtol=2e-13,atol=1e-15)
    if not sol.success or sol.y[0,-1]<=0:raise RuntimeError('Two-loop running failed; no fallback')
    return lambda t:sol.sol(t)[0],end,sol.nfev

def matrices(k0,k1):
    k0=np.asarray(k0,float);k1=np.asarray(k1,float)
    if k0.ndim!=2 or k0.shape[0]!=k0.shape[1] or k1.shape!=k0.shape or not np.isfinite(k0).all() or not np.isfinite(k1).all():raise ValueError('Finite equal square K0/K1 required')
    return k0,k1

def evolve(field,k0,k1,q2,p=Prescription(),rtol=1e-10,atol=1e-12,adjoint=False):
    k0,k1=matrices(k0,k1);v=np.asarray(field)
    if v.ndim!=1 or len(v)!=len(k0) or not np.isfinite(v).all():raise ValueError('Finite vector must match matrices')
    if rtol<=0 or atol<=0:raise ValueError('Positive integration tolerances required')
    a_s,end,nalpha=coupling(q2,p)
    if end==0:return v.copy(),dict(nfev=0,coupling_nfev=0,method='identity')
    span=(end,0.) if adjoint else (0.,end)
    def rhs(t,h):
        a=2*float(a_s(t)) # alpha/(2pi)
        if adjoint:return -(a*(k0.T@h)+a*a*(k1.T@h))
        return a*(k0@h)+a*a*(k1@h)
    sol=solve_ivp(rhs,span,v.astype(complex if np.iscomplexobj(v) else float),method='DOP853',rtol=rtol,atol=atol)
    if not sol.success or not np.isfinite(sol.y[:,-1]).all():raise RuntimeError('Ordered evolution failed; no last-iterate acceptance')
    return sol.y[:,-1],dict(nfev=sol.nfev,coupling_nfev=nalpha,method='DOP853',rtol=rtol,atol=atol)

def assemble_raw(raw):
    """Raw six-sector order: NS+,NS-,QQsinglet,QG,GQ,GG; state Sigma,T,Hg."""
    out=[]
    for name in ['K0','K1']:
        k=np.asarray(raw[name],float);n=k.shape[0]
        if k.shape!=(n,n,6):raise ValueError('Expected six square sectors')
        z=np.zeros((n,n));out.append(np.block([[k[:,:,2],z,k[:,:,3]],[z,k[:,:,0],z],[k[:,:,4],z,k[:,:,5]]]))
    return out
