"""Local convex GN/trust-region model retaining a residual-norm constraint.

This returns a PROPOSAL, not a neural optimum. All nonlinear inequalities and
original-coordinate acceptance checks must be reevaluated by the caller.
The orthonormal Q is only a temporary step coordinate system.
No statistical regularizer, polynomial coefficient, or kernel is introduced.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, lsq_linear

def solve_model(gradient, residual_jacobian, Q, linear_rows, linear_values,
                moment_residual, moment_jacobian, radius,
                maxiter=160, ftol=1e-11):
    g=np.asarray(gradient,float);R=np.asarray(residual_jacobian,float)
    Q=np.asarray(Q,float);A=np.asarray(linear_rows,float);c=np.asarray(linear_values,float)
    e=np.asarray(moment_residual,float);E=np.asarray(moment_jacobian,float)
    n=len(g)
    if Q.ndim!=2:raise ValueError("Q must be a matrix")
    k=Q.shape[1]
    if Q.ndim!=2 or Q.shape[0]!=n or R.ndim!=2 or R.shape[1]!=n or A.shape!=(len(c),n) or E.shape!=(len(e),n):
        raise ValueError('Inconsistent model dimensions')
    if not all(np.isfinite(v).all() for v in (g,R,Q,A,c,e,E)) or not np.isfinite(radius) or radius<=0:
        raise ValueError('Nonfinite model or nonpositive radius')
    if np.max(abs(Q.T@Q-np.eye(k)),initial=0)>1e-8:raise ValueError('Q must be orthonormal')
    if np.max(c,initial=-1)>0 or np.linalg.norm(e)>1+1e-12:raise ValueError('Infeasible model base')
    B=(R@Q)*radius;h=(Q.T@g)*radius;H=2*B.T@B
    M=(A@Q)*radius;F=(E@Q)*radius
    cost_scale=max(np.linalg.norm(h),1e-16)
    linear_scale=np.maximum(np.linalg.norm(M,axis=1),1e-12)
    norm_scale=max(np.linalg.norm(F.T@e),1e-6)
    def fun(u):return (h@u+.5*u@H@u)/cost_scale
    def jac(u):return (h+H@u)/cost_scale
    def constraints(u):
        v=e+F@u
        return np.r_[-(c+M@u)/linear_scale,(1-np.linalg.norm(v))/norm_scale,1-u@u]
    def constraint_jac(u):
        v=e+F@u;nv=np.linalg.norm(v)
        dn=v@F/nv if nv else np.zeros(k)
        return np.vstack((-M/linear_scale[:,None],-dn[None,:]/norm_scale,-2*u[None,:]))
    result=minimize(fun,np.zeros(k),jac=jac,method='SLSQP',
        constraints=[{'type':'ineq','fun':constraints,'jac':constraint_jac}],
        options={'ftol':ftol,'maxiter':int(maxiter),'disp':False})
    u=result.x;d=radius*(Q@u);cv=constraints(u);rows=constraint_jac(u)
    ids=np.flatnonzero(cv<=1e-6)
    # Multipliers for normalized local constraints, not physical/KKT acceptance.
    multipliers=np.zeros(len(cv));stat=jac(u).copy()
    if len(ids):
        mul=lsq_linear(-rows[ids].T,-jac(u),bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=2000)
        multipliers[ids]=mul.x;stat=jac(u)-rows.T@multipliers
    return d,{'optimizer_success':bool(result.success),'message':str(result.message),
        'iterations':int(result.nit),'function_evaluations':int(result.nfev),
        'model_decrease':float(-fun(u)*cost_scale),'step_norm':float(np.linalg.norm(d)),
        'linear_max_constraint':float(np.max(c+A@d,initial=-1)),
        'moment_norm_constraint':float(np.linalg.norm(e+E@d)-1),
        'trust_excess':float(np.linalg.norm(d)-radius),
        'scaled_local_stationarity_L2':float(np.linalg.norm(stat)),
        'scaled_local_complementarity':float(np.max(abs(multipliers*cv),initial=0)),
        'scope':'Proposal only; reevaluate original nonlinear constraints and original-coordinate KKT.'}
