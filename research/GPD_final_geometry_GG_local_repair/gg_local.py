"""Strictly conditional GG-local fallback. Does not modify regular/plus kernels.

Only intercepts an NLO GG local failure after the original regular/plus call is
verified finite. Every previously successful result is returned unchanged.
"""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import math
import numpy as np
from numpy.polynomial.legendre import leggauss

NODES = (24, 48, 96)
TOLERANCE = 1e-9  # stricter than captured 1e-8 max(1, |integral|) local criterion
CPP_SHA256 = '339e3631f01c0dca271dc596bd46234707cb668b6e53dc218466cbe540f7c895'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_domain(x, xi):
    if not (math.isfinite(x) and math.isfinite(xi) and 0 < xi < 1 and 0 < abs(x) < 1):
        raise ValueError('Unsupported local integration domain')
    if min(abs(x), abs(abs(x)-xi), 1-abs(x)) <= 64*np.finfo(float).eps:
        raise ValueError('Exact/roundoff crossover, midpoint or endpoint is unsupported')

def panels(x, xi):
    check_domain(x, xi)
    cuts=sorted(set([-1., -abs(x), -xi, 0., xi, abs(x), 1.]))
    poles=[-abs(x), abs(x)]
    radii=[.45*min(p-cuts[cuts.index(p)-1], cuts[cuts.index(p)+1]-p) for p in poles]
    cuts=sorted(set(cuts+[p+sign*r for p,r in zip(poles,radii) for sign in (-1,1)]))
    regular=[(a,b) for a,b in zip(cuts[:-1],cuts[1:])
             if not any(abs((a+b)/2-p)<r for p,r in zip(poles,radii))]
    scale=.25*min(abs(x-xi),abs(x+xi),abs(x),1-abs(x))
    split=[]
    for a,b in regular:
        delta=(b-a)/2
        while delta>scale:
            split.extend([(a+.5*delta,a+delta),(b-delta,b-.5*delta)])
            delta *= .5
        split.extend([(a,a+delta),(b-delta,b)])
    return split,list(zip(poles,radii))

def explicit_per_flavor():
    # Preserve the captured Fortran explicit term's actual arithmetic. Its
    # -1./108. constant is evaluated at default REAL kind before promotion.
    # This is NOT a fitted delta/contact correction. The intended exact value
    # differs from this by only roundoff, recorded separately in the report.
    return float(np.float32(-1.) / np.float32(108.))*(35.*3.+74.*(4./3.))

class Reference:
    def __init__(self, library):
        self.path=Path(library).resolve()
        self.lib=ct.CDLL(str(self.path))
        self.fn=self.lib.gg_local_reference
        self.ptr=ct.POINTER(ct.c_double)
        self.fn.argtypes=[ct.c_double,ct.c_double,ct.c_int,self.ptr,
                          ct.c_double,ct.c_int,self.ptr]
        self.fn.restype=None

    def values(self,x,xi,y,pole=0.,paired=False):
        check_domain(float(x),float(xi))
        y=np.ascontiguousarray(y,dtype=np.float64)
        if y.ndim!=1 or not len(y) or not np.isfinite(y).all():
            raise ValueError('Finite vector coordinates required')
        if paired:
            if pole not in (-abs(x), abs(x)) or np.any(y<=0):
                raise ValueError('Positive paired offsets and a valid pole required')
        else:
            if np.any(abs(y)>=1) or any(np.any(y==p) for p in (-xi,xi,-x,x)):
                raise ValueError('Singular/unphysical direct query')
        out=np.empty((len(y),2))
        self.fn(x,xi,len(y),y.ctypes.data_as(self.ptr),pole,int(paired),out.ctypes.data_as(self.ptr))
        if not np.isfinite(out).all():
            raise FloatingPointError('Nonfinite high-precision reference: not replaced')
        return out

    def integral(self,x,xi,n):
        z,w=leggauss(n);u=(z+1)/2;w=w/2
        reg,pairs=panels(x,xi);ys=[];ws=[]
        for a,b in reg:
            mid=(a+b)/2
            power=2 if (mid-a)*u[0]**2>1e-15 else 1
            ys.extend([a+(mid-a)*u**power, b-(b-mid)*u**power])
            ws.extend([w*(mid-a)*power*u**(power-1), w*(b-mid)*power*u**(power-1)])
        y=np.concatenate(ys);weight=np.concatenate(ws)
        s=np.einsum('i,ij->j',weight,self.values(x,xi,y))
        for p,r in pairs:
            t=r*u**2;wt=2*r*w*u
            s+=np.einsum('i,ij->j',wt,self.values(x,xi,t,p,True))
        return s

    def certify(self,x,xi):
        check_domain(x,xi)
        values=np.array([self.integral(x,xi,n) for n in NODES])
        errors=np.max(abs(np.diff(values,axis=0)),axis=0)
        limits=TOLERANCE*np.maximum(1.,abs(values[-1]))
        if not np.isfinite(values).all() or not np.all(errors<=limits):
            raise FloatingPointError('GG local reference refinement failed; no cap escalation')
        local=float(values[-1,0]+3.*(values[-1,1]+explicit_per_flavor()))
        return {'x':x,'xi':xi,'nf':3,'node_counts':list(NODES),
                'integrals':values.tolist(),'max_successive_change':errors.tolist(),
                'limits':limits.tolist(),'explicit_per_flavor':explicit_per_flavor(),
                'local':local,'estimate':float(errors@[1.,3.]),
                'empirical_not_rigorous':True,'method':'same captured GG expression; MPFR192 paired subtraction; split Gaussian refinement'}

class FallbackSource:
    def __init__(self,source,reference,identity,cache_dir):
        self.source=source;self.reference=reference;self.identity=identity
        self.cache=Path(cache_dir);self.cache.mkdir(exist_ok=True,parents=True)

    def __getattr__(self,name):
        return getattr(self.source,name)

    def _path(self,x,xi):
        key={'recipe':self.identity,'x_hex':float(x).hex(),'xi_hex':float(xi).hex(),'nf':3}
        name=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
        return self.cache/(name+'.json'),key

    def evaluate(self,order,sector,x,xi,y,include_local=False):
        eligible=(order==1 and sector=='GG' and include_local)
        if not eligible:return self.source.evaluate(order,sector,x,xi,y,include_local)
        path,key=self._path(x,xi)
        if path.exists():
            result=json.loads(path.read_text())
            if result['key']!=key:raise ValueError('Fallback cache identity mismatch')
            r,p,_=self.source.evaluate(order,sector,x,xi,y,False)
            if not np.isfinite(r).all() or not np.isfinite(p).all() or not math.isfinite(result['value']):
                raise FloatingPointError('Nonfinite regular/plus or cached local value')
            return r,p,result['value']
        try:
            return self.source.evaluate(order,sector,x,xi,y,True)
        except FloatingPointError as original:
            # This separate check excludes a regular/plus failure. Such failures
            # are never masked by a replacement local term.
            r,p,_=self.source.evaluate(order,sector,x,xi,y,False)
            if not np.isfinite(r).all() or not np.isfinite(p).all():raise
            record=self.reference.certify(float(x),float(xi))
            if not math.isfinite(record['local']):raise FloatingPointError('Invalid local result')
            result={'key':key,'value':record['local'],'original_error':repr(original),'reference':record}
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(path)
            return r,p,result['value']
