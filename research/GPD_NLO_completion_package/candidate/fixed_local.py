"""Direct Eq. (79) local GG finite part; independent paired quadrature.
The delta coefficient is fixed by BFM Eqs. (175)-(176). No fitted term.
"""
import numpy as np
from continuous_action import geometry

def gg_local(p,x,xi,n):
 y,w=geometry(x,xi,n,pair_power=1)
 _,k,_=p.evaluate(1,'GG',x,xi,y)
 kr=np.array([p.evaluate(1,'GG',float(z),xi,[x])[1][0] for z in y])
 return float(w@(k-kr)-3/108*(35*3+74*4/3))
