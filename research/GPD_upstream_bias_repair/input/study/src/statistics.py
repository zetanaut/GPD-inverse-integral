"""Paired study metrics and deterministic linear CFF extraction controls."""
from __future__ import annotations
import numpy as np


def solve_cff(A, b, y, sigma, rcond=1e-12):
    """Column-scaled WLS; rank/null diagnostics retained, no hidden ridge.

    In the reduced source the observable is affine in four local CFF outputs.
    The solution is a numerical CONTROL, not reproduction of CDNN/FQDNN training.
    In a rank-deficient problem the returned minimum-scaled-norm convention does
    not make ReH identified; its null projection is reported explicitly.
    """
    A,b,y,sigma = [np.asarray(x,dtype=float) for x in (A,b,y,sigma)]
    if A.ndim != 2 or A.shape[1] != 4 or any(x.shape!=(len(A),) for x in (b,y,sigma)):
        raise ValueError('Incompatible CFF control arrays')
    if not all(np.all(np.isfinite(x)) for x in (A,b,y,sigma)) or np.any(sigma<=0):
        raise ValueError('Invalid data or uncertainty')
    D=A/sigma[:,None]; c=(y-b)/sigma
    norms=np.linalg.norm(D,axis=0); safe=np.where(norms>0,norms,1.)
    Q=D/safe
    U,s,Vt=np.linalg.svd(Q,full_matrices=False)
    mask=s>(rcond*s[0]) if len(s) and s[0]>0 else np.zeros_like(s,dtype=bool)
    inv=np.zeros_like(s); inv[mask]=1/s[mask]
    z=Vt.T@(inv*(U.T@c)); theta=z/safe
    coef=(Vt.T*inv)@U.T
    # Uncertainty below is an algebraic raw-data control. The same sigma remains
    # a fixed matching metric for surrogate inputs, NOT a surrogate likelihood.
    covariance=(coef@coef.T)/safe[:,None]/safe[None,:]
    rowspace=Vt[mask].T@Vt[mask]
    reh_null=np.linalg.norm((np.eye(4)-rowspace)[:,0])
    return {'theta':theta,'singular_values_scaled':s,'rank':int(mask.sum()),
            'condition_scaled':float(s[0]/s[-1]) if len(s) and s[-1]>0 else float('inf'),
            'reh_null_projection_scaled':float(reh_null), 'raw_wls_covariance':covariance,
            'weighted_residual_rms':float(np.sqrt(np.mean((D@theta-c)**2)))}


def known_nuisance_reh(A,b,y,sigma,truth_other):
    """Oracle diagnostic only; nuisance values supplied by the synthetic generator."""
    w=1/np.asarray(sigma)**2; a=np.asarray(A)[:,0]
    r=np.asarray(y)-b-np.asarray(A)[:,1:]@np.asarray(truth_other)
    den=np.dot(w*a,a)
    if den<=0: raise ValueError('No ReH sensitivity')
    return float(np.dot(w*a,r)/den)


def cohort_decomposition(estimates,truth):
    """Estimates [noise,seed,kinematics]; deterministic clean draw excluded.

    Reports a finite-cohort identity: MSE = bias² + between-noise variance of
    seed means + within-noise seed variance, all with population denominators.
    This identity is not a claim that 2 seeds fully isolate algorithmic variance.
    """
    e=np.asarray(estimates,dtype=float);truth=np.asarray(truth,dtype=float)
    if e.ndim!=3 or e.shape[-1]!=len(truth) or not np.all(np.isfinite(e)):
        raise ValueError('Need a complete finite paired cohort')
    means=e.mean(axis=1);grand=means.mean(axis=0)
    bias2=float(np.mean((grand-truth)**2))
    between=float(np.mean((means-grand)**2))
    within=float(np.mean((e-means[:,None,:])**2))
    mse=float(np.mean((e-truth)**2))
    return {'mse':mse,'rmse':mse**.5,'bias_squared':bias2,
            'between_noise_seedmean_variance':between,'within_noise_seed_variance':within,
            'identity_error':float(abs(mse-bias2-between-within))}


def symmetric_pool(A,b,y,sigma,orbit):
    """Pool only actually supplied reflection partners, no mirrored fake data."""
    out=[]
    for key in np.unique(orbit):
        idx=np.flatnonzero(orbit==key);w=1/sigma[idx]**2
        if not np.allclose(A[idx],A[idx[0]],rtol=1e-8,atol=1e-10) or not np.allclose(b[idx],b[idx[0]],rtol=1e-8,atol=1e-10):
            raise ValueError('Orbit does not have equal observable design')
        out.append((A[idx[0]], b[idx[0]],np.sum(w*y[idx])/w.sum(),1/np.sqrt(w.sum())))
    return tuple(np.array([a[j] for a in out]) for j in range(4))
