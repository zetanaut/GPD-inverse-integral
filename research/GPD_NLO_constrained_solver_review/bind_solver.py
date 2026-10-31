"""Attach the tested residual-norm step to an ISOLATED solver-module copy.

The supplied worker's outer trust-region loop, nonlinear checks, second-order
correction, objective, and final KKT stay unchanged. Never modify production
source files through this adapter. Use a new run directory and freeze hashes.
"""
from __future__ import annotations
import time
import numpy as np
from moment_linearization import residual_and_jacobian
from residual_subproblem import solve_model

def bind(solver_module, oracle):
    """Return (same oracle, stats); caller invokes isolated solver_module.solve.

    `solver_module` must be a separately loaded copy, not a shared global module.
    Only its step function is replaced. The current outer loop supplies full
    derivatives through the bound oracle; the binding never touches the data.
    """
    old_derivatives=oracle.derivatives
    cache={}
    stats={'moment_jacobian_builds':0,'moment_jacobian_seconds':0.,'norm_subproblem_calls':0,
           'norm_subproblem_seconds':0.,'norm_subproblem_iterations':0}
    def derivatives(a,indices):
        g,R,A=old_derivatives(a,indices)
        if 'x' not in cache or not np.array_equal(a,cache['x']):
            import torch
            t=time.monotonic()
            e,E=residual_and_jacobian(oracle.p,oracle.domain,torch.as_tensor(a,dtype=torch.float64))
            stats['moment_jacobian_seconds']+=time.monotonic()-t
            stats['moment_jacobian_builds']+=1
            cache.update(x=np.array(a,copy=True),e=e,E=E)
        cache.update(g=g,R=R,A=A,indices=np.array(indices,copy=True))
        return g,R,A
    oracle.derivatives=derivatives
    def step(model,c,radius,settings):
        if 'x' not in cache:raise RuntimeError('Must evaluate base derivatives first')
        ids=cache['indices'];other=ids!=0
        if len(c)!=len(ids):raise ValueError('Working-set dimension mismatch')
        t=time.monotonic()
        d,info=solve_model(cache['g'],cache['R'],model['Q'],cache['A'][other],np.asarray(c)[other],
                          cache['e'],cache['E'],radius)
        stats['norm_subproblem_seconds']+=time.monotonic()-t
        stats['norm_subproblem_calls']+=1;stats['norm_subproblem_iterations']+=info['iterations']
        # These are proposal tolerances ONLY. The outer loop reevaluates all
        # actual inequalities and enforces the original strict primal rule.
        ok=info['optimizer_success'] and np.isfinite(d).all() and info['linear_max_constraint']<=1e-8
        ok=ok and info['moment_norm_constraint']<=1e-8 and info['trust_excess']<=1e-8*max(radius,1e-12)
        info.update(success=bool(ok),subspace_rank=int(model['rank']),
                    gradient_span_error=float(model['gradient_span_error']),
                    QP_calls=[],solver_stats=dict(stats),engine='residual-norm local convex model')
        return (d if ok else None),info
    solver_module.trust_step=step
    return oracle,stats
