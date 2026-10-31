"""Analytic endpoint subtraction for BFM's integrable QG log-squared term.
At reduced X=Y, F+Fbar=-1/[X(1-X)] and F+Fbar+1/[Y(1-Y)]=0.
Thus the leading singularity is TF*(CF-CA)*(F+Fbar)*log^2(|Y-X|/|X|).
Apply both halves, physical Jacobian1/(4xi^2), nf3, and inherited rho support.
Subtract times f(x) numerically and add its exact integral; no contact term.
"""
import numpy as np
from action_probe import panels

def subtraction(y,x,xi):
 if not 0<x<1 or x==xi:raise ValueError('positive non-crossover output required')
 radius=dict(panels(x,xi)[1])[x];delta=y-x;sub=np.zeros_like(y);total=0.
 coefficient=2.5/(xi*xi-x*x)
 def add(mask,sign,scale):
  nonlocal total
  sub[mask]+=sign*coefficient*np.log(abs(delta[mask])/scale)**2
  lr=np.log(radius/scale);total+=sign*coefficient*radius*(lr*lr-2*lr+2)
 positive=(delta>0)&(delta<radius)
 add(positive,1.,xi+x)
 if x>xi:add(positive,1.,x-xi)
 else:add((delta<0)&(delta>-radius),-1.,xi-x)
 return sub,total
