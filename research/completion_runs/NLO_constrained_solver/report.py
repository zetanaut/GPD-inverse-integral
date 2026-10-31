"""Summarize saved outcomes; no neural evaluation or optimization."""
from pathlib import Path
import json,collections
import numpy as np
from first_order import normal_residual
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())

def main():
    aggregate=read(HERE/'RESULT.json');rows=[]
    for label in (31,43):
        out=HERE/f'start{label}';r=read(out/'RESULT.json');opt=read(out/'OPTIMIZER.json')
        k=read(out/'KKT.json');m=read(out/'MOMENTS.json');full=read(out/'FULL_FORWARD.json')
        dat=np.load(out/'FIRST_ORDER.npz',allow_pickle=False)
        lam,res,ok=normal_residual(dat['original_gradient'],dat['working_jacobian'])
        near={'scope':'Diagnostic only: best nonnegative normal cancellation using all saved near rows, not a relaxed KKT gate',
              'indices':dat['working_indices'].tolist(),'multipliers':lam.tolist(),
              'stationarity_residual_L2':float(np.linalg.norm(res)),'linear_solve_success':bool(ok)}
        (out/'NEAR_NORMAL_DIAGNOSTIC.json').write_text(json.dumps(near,indent=2)+'\n')
        row={'label':label,'accepted':r['accepted'],'J':r['J'],'J_components':r['J_components'],
             'original_gradient_L2':r['original_gradient_L2'],'KKT_L2':r['KKT_L2'],
             'primal_max_excess':r['primal_max_excess'],'complementarity':r['complementarity'],
             'near_normal_residual_L2':near['stationarity_residual_L2'],
             'full_points_passed':sum(v['pass'] for v in full['records']),
             'CFF_max_ratio':max(max(a/b for a,b in zip(v['CFF_estimated_total'],v['CFF_limits'])) for v in full['records']),
             'field_max_budget':max(max(v['field_totals'].values()) for v in full['records']),
             'moment80_RMS384':m['grids']['enforced80']['quadrature_RMS']['384'],
             'moment160_RMS384':m['grids']['independent160']['quadrature_RMS']['384'],
             'moment_refinement_max':max(v['residual_refinement_RMS_192_384'] for v in m['grids'].values()),
             'enforced80_pass':m['grids']['enforced80']['pass'],
             'disjoint160_pass':m['grids']['independent160']['pass'],
             'refinement_pass':all(v['residual_refinement_RMS_192_384']<=1e-7 for v in m['grids'].values()),
             'moment_pass':m['pass'],'old_moment96_pass':m['grids']['enforced80']['quadrature_RMS']['96']<=1e-3,
             'derivatives_pass':read(out/'DERIVATIVES.json')['pass'],
             'optimizer_status':opt['status'],'counts':opt.get('counts',{}),'iterations':len(opt.get('history',[])),
             'optimizer_oracle_counts':opt.get('oracle_counts',{}),
             'optimization_seconds':opt.get('seconds'),'total_seconds':r['total_seconds'],
             'process_peak_RSS_bytes':r['process_peak_RSS_bytes'],'gates':r['gates'],
             'active_indices':k['evaluated_jacobian_indices'],'rank':k['constraint_qualification']}
        rows.append(row)
    (HERE/'COMPARISON.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Constrained solver continuation results','',
       f"Accepted candidates: **{aggregate['accepted_count']}/2**. These are warm-start development results only.",'',
       '| Quantity | Continuation 31 | Continuation 43 |','|---|---:|---:|']
    fields=[('Original J','J'),('Raw original gradient L2','original_gradient_L2'),('KKT L2 (limit 1e-6)','KKT_L2'),
        ('Primal excess (must be 0)','primal_max_excess'),('Complementarity (limit 1e-8)','complementarity'),
        ('Near-normal diagnostic residual','near_normal_residual_L2'),('Full forward checks /15','full_points_passed'),
        ('CFF budget / original limit','CFF_max_ratio'),('Largest field budget (limit 1e-3)','field_max_budget'),
        ('Enforced80 refined moment RMS','moment80_RMS384'),('Disjoint160 refined moment RMS','moment160_RMS384'),
        ('Largest moment192/384 refinement residual','moment_refinement_max'),('Final derivatives pass','derivatives_pass'),
        ('Outer iterations','iterations'),('Optimization seconds','optimization_seconds'),('Total seconds','total_seconds'),
        ('Process peak RSS bytes','process_peak_RSS_bytes')]
    fmt=lambda v:f'{v:.10g}' if isinstance(v,float) else str(v)
    for title,key in fields:lines.append(f'| {title} | {fmt(rows[0][key])} | {fmt(rows[1][key])} |')
    lines+=['','## Bounded solver and original-coordinate diagnostics','',
       'One deterministic feasible trust-region Gauss–Newton/SQP configuration was frozen before either continuation. No augmented-Lagrangian loop ran. Objective gradients and all inequalities retain their original definitions; the working window only selects a step subproblem. Final acceptance uses the unchanged active tolerance, strict primal rule, signed multipliers, complementarity, original-coordinate KKT L2 and active-row rank check. Full-forward inequalities must remain inactive when their gradients are not represented in the multiplier fit.','',
       'All accepted steps are nonlinearly feasible for all 166 differentiable inequalities and reduce the original J. Every accepted state and trial/failure diagnostic is retained. Trust-radius trials are part of one solve, not new starts. Termination is bounded by 600 iterations, 12,000 value evaluations, 6,600 seconds of optimization and 7,200 seconds including final validation, with a three-unchanged-termination guard. The last returned iterate is preserved without post-selection.','']
    for row in rows:
        lines += [f"### Continuation {row['label']}",'',f"Termination: `{row['optimizer_status']}`. Gates: `{row['gates']}`.",'',
          f"Solver counts: `{row['counts']}`. Value/Jacobian counts during optimization: `{row['optimizer_oracle_counts']}`.",'',
          f"Final active indices: `{row['active_indices']}`. Active-row rank record: `{row['rank']}`.",'']
    lines += ['## Evidence and limits','',
       'The 12 supplied first-order tests, 16 inherited KKT tests and 10 new solver tests passed before neural execution. Analytic active-boundary, curved-constraint, multiplier-sign, infeasible-point and unchanged-state controls are retained. Both supplied tangent witnesses reproduce. Residual identities, original objective gradients, constraint rows and directional derivatives were checked before execution. PREEXEC_FIXES.md preserves the pre-execution test failures and their corrections.','',
       'The temporary derivative span is rebuilt from all 3,867 neural coordinates; it is not a GPD basis substitution or a parameter freeze. All 42 allowed moment coefficients remain free. The scientific objective, normalizations, data/covariance, 0.95 numerical margin, reference-scale architecture, NLO operators and contractions are unchanged.','',
       'Every returned candidate has all 15 authoritative full-forward checks and refined 80/160-node moment checks. The 160 nodes do not enter optimization. No historical full audit, other dataset, kernel, profile, calibration, symbolic selection or experiment ran.','',
       'Necessary KKT and finite numerical/moment checks cannot certify a global optimum, universal continuum accuracy, all-order polynomiality or statistical coverage. A passing warm-start repair would still require a separately reviewed end-to-end estimator contract. Process peak RSS is cumulative within the two-continuation process.','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    details=[]
    for r in rows:
        details.append(f"Continuation {r['label']}: J={r['J']:.9g}; KKT L2={r['KKT_L2']:.6g}; primal excess={r['primal_max_excess']:.3g}; complementarity={r['complementarity']:.3g}; runtime={r['total_seconds']:.1f}s. Full-forward checks {r['full_points_passed']}/15; enforced80 pass={r['enforced80_pass']}; quadrature refinement pass={r['refinement_pass']}; disjoint160 RMS={r['moment160_RMS384']:.6g} (limit 0.001), pass={r['disjoint160_pass']}. Termination: {r['optimizer_status']}.")
    summary=f"""# NLO constrained-solver result

{aggregate['accepted_count']}/2 continuations accepted. Only the two saved feasible checkpoints and unchanged d000 data/covariance/contractions were used.

{details[0]}

{details[1]}

One trust-region constrained Gauss–Newton implementation was frozen before execution. All 3,867 FiLM parameters, 42 free moment coefficients, original J, 18/492 normalization, lambda=1, 166 inequalities, numerical margin and NLO physics remain unchanged. All inequalities affect trial acceptance; the 160 validation nodes were not used in optimization.

The required original-coordinate KKT L2 remains 1e-6, complementarity 1e-8, active tolerance 1e-6, and primal feasibility strict. Termination reasons, accepted states, rejected trials, original gradients, relevant Jacobian rows, mapped multipliers, full constraint vectors and resource counts are preserved.

All 38 supplied/inherited/new mechanical tests passed. Both saved descent witnesses and directional derivatives were verified before continuation. Final authoritative checks and a read-only replay accompany the packet. See REPORT.md and any BLOCKER.md for limitations.

No historical training restart, AL penalty escalation, kernels, new datasets, profiles, calibration or experimental fits ran. These warm-start development results do not certify an end-to-end estimator or statistical coverage.
"""
    assert len(summary.split())<=250
    (HERE/'SUMMARY.md').write_text(summary)
    if aggregate['accepted_count']<2:
        text=['# Remaining bounded solver/validation blocker','',
          'Failed continuations remain failed. This is failure of the frozen bounded numerical procedure, not proof of scientific infeasibility or failure of the accepted NLO operators.','']
        text+=details
        for r in rows:
            if not r['accepted']:text+=['',f"Start {r['label']} termination: `{r['optimizer_status']}`. Failed gates: "+', '.join(k for k,v in r['gates'].items() if not v)+'.']
        text+=['','Reproduce the saved original-coordinate first-order evidence, without optimization:','','```sh',
          '/Users/dustin/miniforge3/bin/python completion_runs/NLO_constrained_solver/replay_saved.py','```','',
          'The immutable previous packet and exact dependency hashes are specified in README.md and DEPENDENCIES.json. No further continuation or tolerance change is authorized.','']
        (HERE/'BLOCKER.md').write_text('\n'.join(text))
    print(json.dumps({'accepted_count':aggregate['accepted_count'],'summary_words':len(summary.split())}))

if __name__=='__main__':main()
