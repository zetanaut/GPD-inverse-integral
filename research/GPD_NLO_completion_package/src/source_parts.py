"""Regular/plus/local source adapter; NOT a convolution or physics certificate."""
from pathlib import Path
import ctypes
import numpy as np
SECTORS=['NS_plus','NS_minus','QQ_singlet','QG','GQ','GG']
class SourceParts:
    def __init__(self,library):
        self.lib=ctypes.CDLL(str(Path(library).resolve()))
        d=ctypes.POINTER(ctypes.c_double);i=ctypes.c_int;c=ctypes.c_double
        self.fn=self.lib.source_parts;self.fn.argtypes=[i,i,c,c,i,d,d,d,d,i];self.fn.restype=None
        self.old=self.lib.point_k1;self.old.argtypes=[c,c,c,d];self.old.restype=None
    def evaluate(self,order,sector,x,xi,y,include_local=False):
        if order not in [0,1] or sector not in SECTORS:raise ValueError('Invalid order/sector')
        y=np.ascontiguousarray(np.atleast_1d(y),dtype=np.float64)
        if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or np.any(abs(y)>=1):raise ValueError('Queries must lie strictly inside (-1,1)')
        if not 0<xi<1 or not -1<x<1 or abs(abs(x)-xi)<1e-14:raise ValueError('Unsupported exact crossover/endpoint; use justified one-sided limits')
        if np.any(np.minimum.reduce([abs(y-x),abs(y+x),abs(y-xi),abs(y+xi)])<1e-14):raise ValueError('Do not evaluate singular source points; integrate regularized limits')
        r=np.empty_like(y);p=np.empty_like(y);l=ctypes.c_double();d=ctypes.POINTER(ctypes.c_double)
        self.fn(order,SECTORS.index(sector),x,xi,len(y),y.ctypes.data_as(d),r.ctypes.data_as(d),p.ctypes.data_as(d),ctypes.byref(l),int(include_local))
        if not np.isfinite(r).all() or not np.isfinite(p).all() or (include_local and not np.isfinite(l.value)):raise FloatingPointError('Source term is nonfinite; no replacement applied')
        return r,p,l.value if include_local else None
    def previous_sum(self,x,y,xi):
        z=np.empty(6);self.old(x,y,xi,z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)));return z
