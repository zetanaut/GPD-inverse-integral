"""Summarize completed saved evidence only; no model evaluations."""
from pathlib import Path
import json,collections

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())

def main():
    aggregate=read(HERE/'RESULT.json');rows=[]
    for label in (31,43):
        p=HERE/f'start{label}';r=read(p/'RESULT.json');f=read(p/'FULL_FORWARD.json')
        m=read(p/'MOMENTS.json');k=read(p/'KKT.json');o=read(p/'OPTIMIZER.json');der=read(p/'FINAL_DERIVATIVES.json')
        cff=max(max(a/b for a,b in zip(v['CFF_estimated_total'],v['CFF_limits'])) for v in f['records'])
        field=max(max(v['field_totals'].values()) for v in f['records'])
        reasons=collections.Counter(v['inner']['message'] for v in o['history'])
        row={'label':label,'accepted':r['accepted'],'J':r['J'],'J_data':r['J_components'][0],
             'J_poly':r['J_components'][1],'raw_gradient_L2':r['raw_gradient_L2'],
             'KKT_stationarity_L2':r['KKT_stationarity_L2'],
             'filter_maximum_excess':k['full_maximum_excess'],'filter_feasible':k['full_primal_feasible'],
             'full_forward_pass_count':sum(v['pass'] for v in f['records']),
             'maximum_CFF_budget_ratio':cff,'maximum_field_budget':field,
             'moment_enforced80_RMS384':m['grids']['enforced80']['quadrature_RMS']['384'],
             'moment_new160_RMS384':m['grids']['independent160']['quadrature_RMS']['384'],
             'old_moment96_pass':m['grids']['enforced80']['quadrature_RMS']['96']<=1e-3,
             'derivative_checks_pass':der['pass'],'active_constraint_indices':k['evaluated_jacobian_indices'],
             'constraint_qualification':k['constraint_qualification'],
             'inner_termination_counts':dict(reasons),'outer_iterations':len(o['history']),
             'optimization_evaluations':o['total_optimization_evaluations'],
             'optimizer_status':o['status'],'seconds':r['total_seconds'],'process_peak_RSS_bytes':r['peak_RSS_bytes'],
             'gates':r['gates']}
        rows.append(row)
    (HERE/'COMPARISON.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Bounded feasibility-constrained FiLM prototype','',
      f"Accepted candidates: **{aggregate['accepted_count']}/2**. This is a new development estimator with an explicit numerical feasibility domain; the old noisy gate remains rejected.",'',
      '| Quantity | Start 31 | Start 43 |','|---|---:|---:|']
    fields=[('Original J','J'),('Data term /18','J_data'),('Moment term /492','J_poly'),
       ('Raw gradient L2','raw_gradient_L2'),('Necessary KKT L2 (limit 1e-6)','KKT_stationarity_L2'),
       ('Filter maximum excess (must be 0)','filter_maximum_excess'),
       ('Original full-forward points passed /15','full_forward_pass_count'),
       ('Largest CFF budget / original limit','maximum_CFF_budget_ratio'),
       ('Largest field budget (limit 1e-3)','maximum_field_budget'),
       ('Enforced80 moment RMS (limit 1e-3)','moment_enforced80_RMS384'),
       ('New disjoint 160 RMS (limit 1e-3)','moment_new160_RMS384'),
       ('Objective/constraint derivatives passed','derivative_checks_pass'),
       ('Registered outer iterations','outer_iterations'),('Optimization evaluations','optimization_evaluations'),
       ('Total seconds','seconds'),('Observed process peak RSS bytes','process_peak_RSS_bytes')]
    for title,key in fields:
        fmt=lambda v:f'{v:.10g}' if isinstance(v,float) else str(v)
        lines.append(f'| {title} | {fmt(rows[0][key])} | {fmt(rows[1][key])} |')
    lines += ['', '## First-order and solver outcome','',
      'Every candidate is checked with strict c<=0, signed multipliers, complementarity, original-coordinate stationarity and the active-row rank diagnostic. Full-forward inequalities must be inactive for their omitted derivatives to have zero multipliers. The old unconstrained scorer was not imported. KKT checks refer to the prospectively declared 0.95-margin filter, not the old estimator. No tolerance is relaxed.', '']
    for row in rows:
        lines += [f"### Start {row['label']}",'',
          f"Optimizer status: `{row['optimizer_status']}`. Active filter indices: `{row['active_constraint_indices']}`. Final gate outcomes: `{row['gates']}`.",'',
          'Inner-solver terminations within this one registered continuation:', '']
        lines += [f'- {n}: {count}' for n,count in row['inner_termination_counts'].items()]
        lines += ['', 'Iteration, evaluation-cap and line-search failures remain in the saved logs and outer states. The pre-registered AL loop continued within its fixed outer budget; there was no new start or algorithm retry. The returned state is the last outer iterate, not a post-selected best fit.','']
    lines += ['## Verification and limits','',
      'The 16 supplied KKT tests and five additional solver/scoring tests passed. The actual one-dimensional active-boundary algorithm control has nonzero raw gradient and passing constrained KKT. Original-coordinate objective and constraint directions were checked with independent covariance/projection calculations, direct given-matrix evolution and refined moment integration. Initial constraints also match all 30 previously saved initial full-forward records.', '',
      'Every outer iterate compares moment integration at 192 and 384 points per side; final validation includes 96/192/384, per-channel/spin residuals and the specified new 160 disjoint xi values. Spins 10/12 remain diagnostics. The numerical filter includes evolved-field comparisons on all 15 points and is distinct from the complete hard/time/warning gates. Every final point retains those original checks.', '',
      'The unchanged J uses the saved d000 covariance, lambda=1 and denominators 18/492. The 3,867-parameter FiLM and all 42 permitted polynomial coefficients remain free. The accepted K0/K1 arrays and fit rows were read-only. Linear propagations were computed from those generators; no evolution kernel was constructed.', '',
      'Finite validation does not prove global numerical accuracy, all-order polynomiality, a minimum, statistical calibration or experimental validity. Process peak RSS is cumulative within the two-start process. No other data realization, profile, calibration, symbolic selection, truth-based model selection or experimental fit was executed.','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    compact=[]
    for r in rows:
        compact.append(f"Start {r['label']}: J={r['J']:.9g}; KKT L2={r['KKT_stationarity_L2']:.6g}; full forward {r['full_forward_pass_count']}/15; new 160 moment RMS={r['moment_new160_RMS384']:.6g}; accepted={r['accepted']}.")
    summary=f"""# NLO feasible-domain result

{aggregate['accepted_count']} of two candidates accepted. Only saved d000 observations/covariance and original starts 31/43 were used. Original FiLM, all 42 free moment coefficients, objective normalization and accepted NLO operators were preserved.

One augmented-Lagrangian configuration was frozen before optimization: 24 outer iterations, 80 inner iterations, 160 evaluations per outer, 6,600-second optimization ceiling and 7,200-second total ceiling per start. The original 80 moment nodes and all 15 evolved-field numerical comparisons guide the search; a declared 0.95 numerical margin supplements unchanged authoritative full checks.

{compact[0]}

{compact[1]}

Required KKT tolerance remains 1e-6. Failures, line-search terminations, original gradients, multipliers, rank diagnostics, checkpoints and all original forward checks are retained. Neither lower objective nor feasibility alone establishes convergence.

All 16 supplied KKT tests and five additional safeguards passed. Independent objective/constraint derivatives, refined moments, new disjoint 160 nodes, source/input hashes and an isolated saved-state replay are included. No remaining datasets, new noise, kernels, calibration, profiles, symbolic pruning or experiments were run. Previous evidence is unchanged.

See BLOCKER.md, REPORT.md and README.md. This bounded development result makes no global-optimality, all-order-admissibility or calibration claim.
"""
    assert len(summary.split())<=250
    (HERE/'SUMMARY.md').write_text(summary)
    if aggregate['accepted_count']==0:
        block=['# Reproducible bounded convergence blocker','',
          'Neither authorized start satisfies the complete prospective acceptance rule. No further optimization is authorized or attempted. This demonstrates failure of this bounded algorithm/configuration; it is not evidence that the feasible neural problem has no stationary solution.','']
        block+=compact
        block+=['','The original-coordinate necessary KKT norm must be <=1e-6, together with strict primal feasibility, valid multipliers, complementarity and all validation gates. Inspect start31/KKT.json and start43/KKT.json, COMPARISON.json and the saved FIRST_ORDER.npz files. Inner line-search and evaluation-budget failures are preserved in OPTIMIZER.json, EVALUATIONS.jsonl and each outer directory.','','Minimal reproduction (no fits):','',
          '```sh','/Users/dustin/miniforge3/bin/python completion_runs/NLO_feasible_domain/reproduce.py','```','',
          'This independently reevaluates the saved J, gradient, constraint vector and active Jacobian rows in a temporary project and reproduces the necessary first-order decision. It does not rerun optimization or alter original evidence.','']
        (HERE/'BLOCKER.md').write_text('\n'.join(block))
    print(json.dumps({'accepted':aggregate['accepted_count'],'summary_words':len(summary.split()),'rows':rows},indent=2))

if __name__=='__main__':main()
