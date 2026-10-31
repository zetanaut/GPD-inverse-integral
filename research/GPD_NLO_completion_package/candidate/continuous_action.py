"""Paired finite-part quadrature on physical-x analytic functions/basis rows.
The source part values are shared with the frozen implementation; only numerical
integration and input representation are independent. No kernel formula changes.
"""
from pathlib import Path
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
sys.path.insert(0,str(Path(__file__).resolve().parent))
from action_probe import panels

def geometry(x,xi,n,extra=(),pair_power=2):
    """Endpoint-square Gauss panels; dyadic shells resolve crossover layers."""
    z,w=leggauss(n);u=(z+1)/2;wu=w/2
    reg,pairs=panels(x,xi);ys=[];ws=[]
    scale=.25*min(abs(x-xi),abs(x+xi),abs(x),1-abs(x))
    for a,b in reg:
        cuts=sorted(set([a,b]+[t for t in extra if a<t<b]));parts=[]
        for c,d in zip(cuts[:-1],cuts[1:]):
            singular=np.array([-xi,xi,-abs(x),abs(x)])
            left=np.min(abs(singular-c))<=2.01*scale
            right=np.min(abs(singular-d))<=2.01*scale
            if not left and not right:
                parts.append((c,d));continue
            mid=(c+d)/2
            for edge,direction,do_grade in [(c,1,left),(d,-1,right)]:
                delta=(d-c)/2
                if not do_grade:
                    parts.append((min(edge,mid),max(edge,mid)));continue
                while delta>scale:
                    p1=edge+direction*delta/2;p2=edge+direction*delta
                    parts.append((min(p1,p2),max(p1,p2)));delta/=2
                p1=edge+direction*delta;parts.append((min(edge,p1),max(edge,p1)))
        for c,d in parts:
            mid=(c+d)/2
            # Cover the whole panel with a linear map when squaring would
            # put nodes inside the source ABI exclusion distance. No masking.
            power=1 if (mid-c)*u[0]**2<=2e-14 else 2
            ys.extend([c+(mid-c)*u**power,d-(d-mid)*u**power]);ws.extend([wu*power*(mid-c)*u**(power-1),wu*power*(d-mid)*u**(power-1)])
    for p,r in pairs:
        # y=+/-x is paired with the same physical exclusion coordinate.
        tcuts=sorted(set([0.,r]+[abs(v-p) for v in extra if 0<abs(v-p)<r]))
        for a,b in zip(tcuts[:-1],tcuts[1:]):
            power=1 if a==0 and (b-a)*u[0]**pair_power<=2e-14 else pair_power
            t=a+(b-a)*u**power;wt=wu*power*(b-a)*u**(power-1)
            ys.extend([p+t,p-t]);ws.extend([wt,wt])
    return np.concatenate(ys),np.concatenate(ws)

def source_local(p,order,s,x,xi):
    return p.evaluate(order,s,x,xi,[.813579 if abs(x-.813579)>1e-5 else .713579],True)[2]

def action(p,order,s,x,xi,n,f):
    y,w=geometry(x,xi,n);r,k,_=p.evaluate(order,s,x,xi,y)
    fx=f(np.array([x]))[0];fy=f(y)
    return np.einsum('i,i...->...',w,r.reshape((-1,)+(1,)*(fy.ndim-1))*fy+k.reshape((-1,)+(1,)*(fy.ndim-1))*(fy-fx))+source_local(p,order,s,x,xi)*fx

def adaptive_action(p,order,s,x,xi,f,eps=2e-8):
    fx=float(f(np.array([x]))[0]);reg,pairs=panels(x,xi);total=0.;err=0.
    def fn(y):
        r,k,_=p.evaluate(order,s,x,xi,[y]);return float(r[0]*f(np.array([y]))[0]+k[0]*(f(np.array([y]))[0]-fx))
    for a,b in reg:
        v,e=quad(fn,a,b,epsabs=eps/20,epsrel=eps/20,limit=150)
        total+=v;err+=e
    for pole,radius in pairs:
        v,e=quad(lambda t:fn(pole+t)+fn(pole-t),0,radius,epsabs=eps/20,epsrel=eps/20,limit=150)
        total+=v;err+=e
    return total+source_local(p,order,s,x,xi)*fx,err
