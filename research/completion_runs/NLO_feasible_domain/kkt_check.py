"""Necessary first-order checks for a prospective constrained neural estimator.

Does NOT implement a neural solver, modify the current estimator, or certify a
minimum. Inputs are derivatives of the ORIGINAL objective and explicit
constraints at one point. Inequalities use c(theta)<=0. Their hard acceptance
limits must be defined by the physics/numerics contract, not by this helper.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.optimize import lsq_linear

def check_kkt(gradient, inequality_values, inequality_jacobian, *,
              equality_values=None, equality_jacobian=None,
              gradient_tolerance: float=1e-6, equality_tolerance: float=1e-6,
              active_tolerance: float=1e-6,
              complementarity_tolerance: float=1e-8) -> dict:
    g=np.asarray(gradient,dtype=float)
    c=np.asarray(inequality_values,dtype=float)
    A=np.asarray(inequality_jacobian,dtype=float)
    if g.ndim!=1 or c.ndim!=1 or A.shape!=(len(c),len(g)):
        raise ValueError('Require gradient (p,), c (m,), Dc (m,p)')
    if (equality_values is None)!=(equality_jacobian is None):
        raise ValueError('Supply both equality values and Jacobian')
    e=np.empty(0) if equality_values is None else np.asarray(equality_values,dtype=float)
    E=np.empty((0,len(g))) if equality_jacobian is None else np.asarray(equality_jacobian,dtype=float)
    if e.ndim!=1 or E.shape!=(len(e),len(g)):
        raise ValueError('Require equality values (k,), Jacobian (k,p)')
    tolerances=(gradient_tolerance,equality_tolerance,active_tolerance,complementarity_tolerance)
    if any(not np.isfinite(t) or t<0 for t in tolerances):
        raise ValueError('Tolerances must be finite and nonnegative')
    if not all(np.isfinite(a).all() for a in (g,c,A,e,E)):
        return {'pass':False,'finite':False,'reason':'nonfinite value or derivative'}
    active=np.flatnonzero(c>=-active_tolerance)
    C=np.vstack((E,A[active]))
    multipliers=np.zeros(len(C));fit_success=True
    if len(C):
        # Normalize only the multiplier solve's columns, NOT the objective or
        # primal constraints. Map multipliers back before stationarity/complementarity.
        norms=np.linalg.norm(C,axis=1);usable=norms>0
        if np.any(usable):
            flags=np.r_[np.zeros(len(e),dtype=bool),np.ones(len(active),dtype=bool)][usable]
            low=np.where(flags,0.,-np.inf)
            result=lsq_linear((C[usable]/norms[usable,None]).T,-g,
                             bounds=(low,np.full(len(low),np.inf)),
                             method='bvls',tol=1e-12,max_iter=1000)
            multipliers[usable]=result.x/norms[usable]
            fit_success=bool(result.success)
    nu=multipliers[:len(e)];lam=np.zeros(len(c));lam[active]=multipliers[len(e):]
    stationarity=g+E.T@nu+A.T@lam
    s=float(np.linalg.norm(stationarity))
    complement=float(np.max(abs(lam*c),initial=0.))
    # No hidden relaxation of c<=0: keep boundary errors explicit.
    primal=bool(np.all(c<=0.))
    equality=bool(np.all(abs(e)<=equality_tolerance))
    dual=bool(np.all(lam>=0.))
    gates={'finite':True,'inequality_feasible':primal,'equality_feasible':equality,
           'dual_feasible':dual,'stationarity':s<=gradient_tolerance,
           'complementarity':complement<=complementarity_tolerance,
           'multiplier_solve':fit_success}
    return {'pass':all(gates.values()),'gates':gates,
            'raw_gradient_L2':float(np.linalg.norm(g)),
            'stationarity_L2':s,'complementarity_max':complement,
            'maximum_inequality_excess':float(max(0.,np.max(c,initial=-np.inf))),
            'equality_max_abs':float(np.max(abs(e),initial=0.)),
            'active_indices':active.tolist(),'inequality_multipliers':lam.tolist(),
            'equality_multipliers':nu.tolist(),
            'qualification':'Necessary local KKT diagnostics only; not sufficiency, constraint qualification, global optimality, or coverage.'}
