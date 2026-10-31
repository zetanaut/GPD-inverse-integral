"""Small first-order diagnostics, not a constrained optimizer or acceptance gate."""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear

def _inputs(gradient, rows):
    g=np.asarray(gradient,dtype=float); a=np.asarray(rows,dtype=float)
    if g.ndim!=1 or a.ndim!=2 or a.shape[1]!=g.size:
        raise ValueError('Expected gradient (p,) and Jacobian rows (m,p)')
    if not np.isfinite(g).all() or not np.isfinite(a).all():
        raise ValueError('Nonfinite derivative')
    return g,a

def tangent_direction(gradient, rows, relative_rank_tolerance=1e-10):
    """Unit downhill direction tangent to the supplied row span, or zero.

    Row selection is a diagnostic choice. Nonlinear feasibility must be checked
    separately; this does not relax an active-set or stationarity tolerance.
    """
    g,a=_inputs(gradient,rows)
    if not np.isfinite(relative_rank_tolerance) or not 0<relative_rank_tolerance<1:
        raise ValueError('Rank tolerance must lie in (0,1)')
    norms=np.linalg.norm(a,axis=1); nz=norms>0
    normalized=a[nz]/norms[nz,None]
    if len(normalized):
        _,s,v=np.linalg.svd(normalized,full_matrices=False)
        rank=int(np.count_nonzero(s>relative_rank_tolerance*s[0]))
        d=-g+v[:rank].T@(v[:rank]@g)
    else:
        s=np.empty(0);rank=0;d=-g.copy()
    length=float(np.linalg.norm(d))
    if length<=100*np.finfo(float).eps*max(1,float(np.linalg.norm(g))):
        d=np.zeros_like(g)
    else:
        d=d/length
    return d,{'rank':rank,'singular_values':s.tolist(),
              'projected_gradient_L2':length,'directional_derivative':float(g@d)}

def normal_residual(gradient, rows):
    """Best nonnegative cancellation using these normals; NOT full KKT."""
    g,a=_inputs(gradient,rows);norms=np.linalg.norm(a,axis=1);nz=norms>0
    lam=np.zeros(len(a));success=True
    if np.any(nz):
        result=lsq_linear((a[nz]/norms[nz,None]).T,-g,
                         bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=1000)
        lam[nz]=result.x/norms[nz];success=bool(result.success)
    return lam,g+a.T@lam,success
