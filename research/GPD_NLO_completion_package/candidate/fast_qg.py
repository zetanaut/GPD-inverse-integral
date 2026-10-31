"""Arithmetic-equivalent MPFR evaluator for the existing QG guard region.
All local callbacks and non-QG functions use the unchanged source bridge.
"""
from pathlib import Path
import ctypes,numpy as np
class FastQG:
 def __init__(self,source,library):
  self.source=source;self.lib=ctypes.CDLL(str(Path(library).resolve()));d=ctypes.POINTER(ctypes.c_double);self.fn=self.lib.mp_qg_vector;self.fn.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_double,d,d];self.fn.restype=None;self.ptr=d
 def evaluate(self,order,sector,x,xi,y,include_local=False):
  if order!=1 or sector!='QG':return self.source.evaluate(order,sector,x,xi,y,include_local)
  y=np.ascontiguousarray(np.atleast_1d(y),dtype=np.float64)
  if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or np.any(abs(y)>=1):raise ValueError('Queries must lie strictly inside (-1,1)')
  if not 0<xi<1 or not -1<x<1 or abs(abs(x)-xi)<1e-14:raise ValueError('Unsupported exact crossover/endpoint; use justified one-sided limits')
  if np.any(np.minimum.reduce([abs(y-x),abs(y+x),abs(y-xi),abs(y+xi)])<1e-14):raise ValueError('Do not evaluate singular source points; integrate regularized limits')
  mask=np.minimum(abs(y-xi),abs(y+xi))<.02*xi;r=np.empty_like(y)
  if np.any(~mask):r[~mask]=self.source.evaluate(order,sector,x,xi,y[~mask],False)[0]
  if np.any(mask):
   yy=np.ascontiguousarray(y[mask]);v=np.empty_like(yy);self.fn(len(yy),x,xi,yy.ctypes.data_as(self.ptr),v.ctypes.data_as(self.ptr));r[mask]=3*v
  if not np.isfinite(r).all():raise FloatingPointError('Nonfinite QG; no replacement')
  return r,np.zeros_like(r),0. if include_local else None
