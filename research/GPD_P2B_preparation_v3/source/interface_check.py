"""Original Oracle/constraint/checkpoint value paths, without a Jacobian or step."""
import argparse
from pathlib import Path
import time
from p2b_common import ROOT, setup, information, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capsule', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError('Use a new empty interface output')
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    core = setup(args.capsule)
    import numpy as np
    import torch
    from pdf_information import PDFProblem
    from closure.solver_binding import oracle_at, isolated_solver
    import common as legacy_common
    if Path(legacy_common.__file__).name != 'common.py' or 'P2B' in str(legacy_common.__file__):
        raise RuntimeError('Legacy module was shadowed')
    base, domain = core.problem(core.ENDPOINTS[0])
    joint = PDFProblem(base, information(base))
    disabled = PDFProblem(base, enabled=False)
    checks = {}
    def compare(name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        checks[name] = bool(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=1e-10,atol=1e-11))
    with torch.no_grad():
        original_c = domain.constraints(base, base.x0).numpy()
        compare('enabled_constraints_unchanged', domain.constraints(joint,base.x0), original_c)
        compare('disabled_constraints_unchanged', domain.constraints(disabled,base.x0), original_c)
        reference = joint.components(base.x0).numpy()
    oracle = oracle_at(joint, domain, base.x0)
    values = oracle.values(base.x0.numpy())
    compare('Oracle_components', values['components'], reference)
    compare('Oracle_J', values['J'], reference.sum())
    compare('Oracle_residual_norm', values['J'], values['residual']@values['residual'])
    # Binding construction only; no derivatives/model/step are invoked.
    solver, bound_oracle, stats, settings = isolated_solver(oracle)
    checks['legacy_binding_constructed'] = bound_oracle is oracle and settings['max_iterations']==60
    core.checkpoint(joint, base.x0, 'DIAGNOSTIC_VALUES_ONLY', args.out/'CHECKPOINT.pt')
    ck = torch.load(args.out/'CHECKPOINT.pt', weights_only=True)
    compare('checkpoint_joint_J', ck['J'], values['J'])
    compare('checkpoint_flat_identity', ck['flat'], base.x0)
    ok = all(checks.values())
    write(args.out/'INTERFACE_CHECK.json', dict(status='PASS' if ok else 'FAIL', checks=checks,
        legacy_common_path=str(legacy_common.__file__), constraint_count=len(original_c),
        residual_count=len(values['residual']), seconds=time.monotonic()-start,
        peak_RSS_bytes=core.rss_bytes(), full_Jacobian_or_step=False, new_joint_fits=0))
    print(dict(status='PASS' if ok else 'FAIL', checks=len(checks), seconds=time.monotonic()-start, peak_RSS_bytes=core.rss_bytes()))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
