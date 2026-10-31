"""Coordinate-safe GG local evaluator; identical formulas and certification gate."""
import ctypes as ct
import numpy as np
from numpy.polynomial.legendre import leggauss
from parent_policy import Reference as ParentReference, check_domain

class Reference(ParentReference):
    def __init__(self, library):
        super().__init__(library)
        self.integrate_native = self.lib.gg_local_integral_mapped
        self.integrate_native.argtypes=[ct.c_double,ct.c_double,ct.c_int,self.ptr,self.ptr,ct.c_int,self.ptr]
        self.integrate_native.restype=ct.c_int
        self.count_native=self.lib.gg_local_panel_count
        self.count_native.argtypes=[ct.c_double,ct.c_double];self.count_native.restype=ct.c_int
        self.panel_native=self.lib.gg_local_panel_values
        self.panel_native.argtypes=[ct.c_double,ct.c_double,ct.c_int,ct.c_int,self.ptr,self.ptr]
        self.panel_native.restype=ct.c_int

    def integral(self,x,xi,n):
        check_domain(x,xi)
        if not isinstance(n,(int,np.integer)) or n<2:
            raise ValueError('Quadrature count must be an integer >=2')
        z,w=leggauss(int(n));u=np.ascontiguousarray((z+1)/2);w=np.ascontiguousarray(w/2)
        out=np.empty(2)
        rc=self.integrate_native(x,xi,n,u.ctypes.data_as(self.ptr),w.ctypes.data_as(self.ptr),0,out.ctypes.data_as(self.ptr))
        if rc or not np.isfinite(out).all():raise FloatingPointError(f'Coordinate-preserving native integral rejected: {rc}')
        return out

    def certify(self,x,xi):
        # Same fixed 24/48/96 sequence, two successive differences, 1e-9 bound,
        # and explicit per-flavor delta contribution as the parent.
        record=super().certify(x,xi)
        record['method']='unchanged GG expression; MPFR192 physical-coordinate maps, weighted integrands and panel accumulation'
        record['panels']=self.count_native(x,xi)
        return record

    def mapped_panel(self,x,xi,panel,u):
        check_domain(x,xi);u=np.ascontiguousarray(u,dtype=np.float64)
        if u.ndim!=1 or not len(u) or not np.isfinite(u).all() or np.any(u<=0) or np.any(u>=1):
            raise ValueError('Panel coordinates must lie strictly between zero and one')
        out=np.empty((len(u),2))
        rc=self.panel_native(x,xi,int(panel),len(u),u.ctypes.data_as(self.ptr),out.ctypes.data_as(self.ptr))
        if rc or not np.isfinite(out).all():raise FloatingPointError(f'Panel evaluation rejected: {rc}')
        return out
