"""Independent, t-dependent, all-spin-polynomial structural truth functions.

These are GENERATORS only, not the inverse parameterization. No experimental
CFF central value or learned checkpoint sets a coefficient. Channels are
Sigma,T_rescaled,Hg=xg. The reported quark combination is (2/9)*(Sigma+T).
Both truth functions have nontrivial ERBL and DGLAP support. Positivity,
external PDFs, individual flavors and sum-rule values are NOT certified.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import eval_jacobi, jacobi, beta as beta_function

# (radial Jacobi order i, skewness power k of xi**(2*k), amplitude, t-slope / GeV^2)
_REGISTRY={
    'Z1': {'beta':2.0,'terms':[
        [(0,0,6.,.65),(0,1,.90,.9),(1,0,-.30,1.05),(1,1,.20,.7),(1,2,.05,.8),(2,0,.08,1.2)],
        [(0,0,2.4,.85),(0,1,-.35,.6),(1,0,.12,1.1),(1,1,-.08,.9),(2,1,.025,1.3)],
        [(0,0,1.5,.55),(0,1,.22,.75),(1,0,-.045,.95),(1,2,.022,.6),(2,0,.02,1.1)]],
        'D':[(0.,0.,.9),(0.,0.,1.),(0.,0.,.8)]},
    'Z2': {'beta':1.6,'terms':[
        [(0,0,4.8,.7),(0,1,.55,.5),(1,0,.42,1.1),(1,1,-.31,.9),(1,2,.12,.75),(2,0,-.14,1.35),(2,2,.06,.8)],
        [(0,0,1.9,.6),(0,1,-.26,1.15),(1,0,-.19,.85),(1,2,.08,.6),(2,1,.05,1.4)],
        [(0,0,1.8,.75),(0,1,.31,.5),(1,0,.12,1.05),(1,1,-.07,.7),(2,0,-.025,1.2)]],
        # q: z*(1-z^2)^2*(d0+d2*z^2); g: |xi|*(1-z^2)^2*(d0+d2*z^2)
        'D':[(.70,-.30,.9),(-.20,.08,1.15),(.12,-.05,.65)]}
}

@dataclass(frozen=True)
class Truth:
    name:str
    def __post_init__(self):
        if self.name not in _REGISTRY: raise ValueError('Truth must be Z1 or Z2')
    @property
    def spec(self): return _REGISTRY[self.name]
    def values(self,x,xi,t):
        """Return shape broadcast(x,xi,t).shape + (3,), at Q0^2=2."""
        x,xi,t=np.broadcast_arrays(np.asarray(x,float),np.asarray(xi,float),np.asarray(t,float))
        if not np.isfinite(x).all() or not np.isfinite(xi).all() or not np.isfinite(t).all():
            raise ValueError('Nonfinite truth coordinates')
        if np.any(abs(xi)>=1) or np.any(t>0): raise ValueError('Use |xi|<1, t<=0')
        u=x*x; beta=self.spec['beta']; env=np.clip(1-u,0,None)**beta
        val=np.zeros(x.shape+(3,))
        for c,terms in enumerate(self.spec['terms']):
            b=.5 if c<2 else -.5; pref=x if c<2 else np.ones_like(x)
            for i,k,a,slope in terms:
                if k>i+1: raise ValueError('Invalid polynomial truth term')
                val[...,c]+=a*np.exp(slope*t)*pref*env*eval_jacobi(i,beta,b,2*u-1)*xi**(2*k)
            d0,d2,slope=self.spec['D'][c]
            ax=abs(xi); z=np.divide(x,ax,out=np.zeros_like(x),where=ax>0)
            inside=(ax>0)&(abs(x)<ax)
            D=np.where(inside,np.clip(1-z*z,0,None)**2*(d0+d2*z*z),0.)
            D=D*z if c<2 else D*ax
            val[...,c]+=np.exp(slope*t)*D
        return np.where((abs(x)<=1)[...,None],val,0.)
    def field(self,x,xi,t):
        x=np.asarray(x,float); xi=np.asarray(xi,float);t=np.asarray(t,float)
        if x.ndim!=2 or xi.shape!=(len(x),) or t.shape!=xi.shape: raise ValueError('x[B,N],xi[B],t[B] required')
        return np.moveaxis(self.values(x,xi[:,None],t[:,None]),-1,1)
    def moment_coefficients(self,channel,N,t):
        """Full [-1,1] spin-N moment, in physical powers xi^0,xi^2,...,xi^N."""
        if channel not in (0,1,2) or N<2 or N%2: raise ValueError('channel=0..2, even spin>=2')
        ts=np.asarray(t,float); out=np.zeros(ts.shape+(N//2+1,));beta=self.spec['beta'];j=N//2-1
        b=.5 if channel<2 else -.5
        # 2*int_0^1 x^(N-1)*x ... dx for q, x^(N-2) ... dx for g.
        a_beta=(N+1)/2 if channel<2 else (N-1)/2
        from numpy.polynomial import Polynomial
        for i,k,a,slope in self.spec['terms'][channel]:
            if i>j: continue  # Exact Jacobi orthogonality, not a fitted zero.
            raw=np.asarray(jacobi(i,beta,b).c)[::-1]
            poly=Polynomial(raw)(Polynomial([-1.,2.]))
            integ=sum(float(c)*beta_function(a_beta+m,beta+1) for m,c in enumerate(poly.coef))
            out[...,k]+=a*np.exp(slope*ts)*integ
        d0,d2,slope=self.spec['D'][channel]
        aD=(N+1)/2 if channel<2 else (N-1)/2
        out[...,-1]+=np.exp(slope*ts)*(d0*beta_function(aD,3)+d2*beta_function(aD+1,3))
        return out
    def moment(self,channel,N,xi,t):
        xi,t=np.broadcast_arrays(np.asarray(xi,float),np.asarray(t,float))
        coeff=self.moment_coefficients(channel,N,t)
        return np.sum(coeff*xi[...,None]**np.arange(0,N+1,2),axis=-1)

def names():return tuple(_REGISTRY)
def specifications():
    import copy
    return copy.deepcopy(_REGISTRY)
