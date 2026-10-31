"""Diagnostics-first development helpers; no global campaign gate or truth-based selection."""
from __future__ import annotations
import ast,copy,inspect,math,hashlib
from pathlib import Path
import numpy as np

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def weighted_problem(base, lam):
    """Change only the soft polynomiality objective and its exact residual/Jacobian.
    The architecture, allowed coefficients and numerical fitting constraints are unchanged.
    This is a DEVELOPMENT estimator variation, not a deployment patch.
    """
    lam=float(lam)
    if not math.isfinite(lam) or lam<=0:raise ValueError('Positive finite lambda required')
    class Weighted(base):
        polynomiality_weight=lam
        def components(self,x):
            c=super().components(x)
            return c*c.new_tensor([1.,lam])
        def residual(self,x):
            r=super().residual(x)
            return self._scale_residual(r)
        def _scale_residual(self,r):
            import torch
            return torch.cat((r[:self.n_scalar],r[self.n_scalar:]*math.sqrt(lam)),dim=0)
        def residual_jacobian(self,x):
            r=super().residual_jacobian(x)
            import torch
            return torch.cat((r[:self.n_scalar],r[self.n_scalar:]*math.sqrt(lam)),dim=0)
        def independent_objective(self,x):
            # Parent independent objective is J1, with a separate least-squares moment calculation.
            j1=super().independent_objective(x)
            unweighted=super().components(x)
            return j1+(lam-1)*unweighted[1]
        def original_components(self,x):return super().components(x)
    return Weighted

def training_prefix(original_run, replacements:dict):
    """Execute EXACT old optimization/ranking code, returning before old binary validation.

    Reject an unrecognized source layout. No solver step or setting is rewritten.
    The new selection/diagnostics are outside the old implementation.
    """
    source=inspect.getsource(original_run)
    node=ast.parse(source).body[0]
    if not isinstance(node,ast.FunctionDef):raise ValueError('Expected run function')
    cuts=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='validations' for t in n.targets)]
    if len(cuts)!=1:raise ValueError('Unrecognized validation boundary')
    cut=cuts[0];prefix=node.body[:cut]
    dump=ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False)
    if 'CANDIDATE_ROSTER.json' not in source or 'ranked/candidate' not in source:
        raise ValueError('Missing immutable ranked roster')
    node=copy.deepcopy(node);node.body=prefix+ast.parse("return {'roster':roster,'starts':logs,'optimization_seconds':time.monotonic()-start,'status':'optimization_complete_not_validated'}").body
    env=dict(original_run.__globals__);env.update(replacements)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),inspect.getsourcefile(original_run) or '<frozen-run>','exec'),env)
    return env[node.name],hashlib.sha256(dump.encode()).hexdigest()

def continuous_summary(moments:dict, forward:dict, constraints, target_refinement:float) -> dict:
    """No quality threshold changes the decision. Missing/nonfinite arithmetic remains explicit."""
    c=np.asarray(constraints,dtype=float)
    if c.ndim!=1 or not c.size:raise ValueError('Missing constraints')
    grid=moments['grids']['independent160'];rms=float(grid['quadrature_RMS']['384'])
    entries=grid['entries']
    rec=forward['records']
    if len(rec)!=24:raise ValueError('Incomplete final-geometry forward diagnostics')
    budgets=np.asarray([r['CFF_estimated_total'] for r in rec],dtype=float)
    limits=np.asarray([r['CFF_limits'] for r in rec],dtype=float)
    fields=np.asarray([list(r['field_totals'].values()) for r in rec],dtype=float)
    if budgets.shape!=(24,2) or limits.shape!=(24,2) or fields.shape!=(24,3):raise ValueError('Malformed forward records')
    values=np.r_[c,rms,target_refinement,budgets.ravel(),limits.ravel(),fields.ravel()]
    finite=bool(np.isfinite(values).all() and np.all(limits>0) and np.all(budgets>=0))
    flags=[]
    if rms>1e-3:flags.append('moment_RMS_exceeds_legacy_quality_target')
    if np.any(c>0):flags.append('fit_constraint_quality_targets_exceeded')
    if np.any(budgets>limits):flags.append('CFF_quality_targets_exceeded')
    if np.any(fields>1e-3):flags.append('field_quality_targets_exceeded')
    if target_refinement>1e-7:flags.append('target_quadrature_quality_target_exceeded')
    warning_points=[{'xi':r['xi'],'Q2':r['Q2']} for r in rec if not r.get('gates',{}).get('hard_no_warnings',True)]
    if warning_points:flags.append('independent_hard_integration_warnings')
    if not finite:flags.append('nonfinite_or_invalid_arithmetic')
    return {'computationally_evaluable':finite,'quality_flags':flags,'moment_RMS':rms,
            'moment_relative_excess_vs_legacy':rms/1e-3-1,
            'moment_channel_spin_entries':entries,
            'moment_quadrature_change':grid['residual_refinement_RMS_192_384'],
            'target_quadrature_change':float(target_refinement),
            'max_fit_constraint':float(c.max()),'n_legacy_fit_violations':int((c>0).sum()),
            'CFF_component_max_estimated_error':budgets.max(axis=0).tolist(),
            'max_CFF_budget_over_legacy_limit':float((budgets/limits).max()),
            'max_estimated_field_error':float(fields.max()),'warning_points':warning_points,
            'legacy_forward_pass':bool(forward['pass']),
            'interpretation':'finite numerical approximation with measured diagnostics; not a physical-admissibility or uncertainty certificate'}

def covariance_error_budget(budgets,covariance):
    b=np.asarray(budgets,float);C=np.asarray(covariance,float)
    if C.shape!=(b.size,b.size) or not np.isfinite(C).all() or not np.isfinite(b).all() or np.any(b<0) or not np.array_equal(C,C.T):raise ValueError('Invalid covariance/budgets')
    L=np.linalg.cholesky(C);bound=np.abs(np.linalg.solve(L,np.eye(b.size)))@b
    return {'max_component_sigma_units':float(np.max(b/np.sqrt(np.diag(C)))),
            'whitened_norm_allowance':float(np.linalg.norm(bound)),
            'note':'empirical numerical allowance, not a rigorous error bound or added stochastic noise'}
