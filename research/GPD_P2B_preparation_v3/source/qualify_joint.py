"""Allocated-node integration probe, at most one outer diagnostic step; no fits."""
import argparse
from pathlib import Path
import json
import os
import sys
import time
from p2b_common import ROOT, setup, information, write


def allocation_guard():
    if sys.platform != 'linux' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Requires an allocated Linux compute node')
    if os.environ.get('SLURM_JOB_NUM_NODES') != '1' or os.environ.get('SLURM_CPUS_PER_TASK') != '1':
        raise RuntimeError('Exactly one node and one CPU required')
    if int(os.environ.get('SLURM_NTASKS', '1')) != 1:
        raise RuntimeError('Exactly one task required')
    if int(os.environ.get('SLURM_MEM_PER_NODE', '0')) < 32768:
        raise RuntimeError('A 32 GiB per-node allocation must be verified')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capsule', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    allocation_guard()
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError('Use a new empty qualification output')
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from verify_package import verify
    verify(ROOT)
    core = setup(args.capsule)
    core.verify_manifest(full=True)
    import numpy as np
    import torch
    from torch.func import jvp
    from pdf_information import PDFProblem
    from closure.solver_binding import oracle_at, isolated_solver, accepted
    start = time.monotonic()
    checks, events = {}, []

    def compare(name, a, b, rtol=2e-7, atol=1e-8):
        a, b = np.asarray(a), np.asarray(b)
        ok = a.shape==b.shape and bool(np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=rtol,atol=atol))
        checks[name] = dict(pass_=ok, max_abs=float(abs(a-b).max()) if a.shape==b.shape else None)
        write(args.out/'CHECKS.json', checks)
        if not ok:
            raise RuntimeError('Integration check failed: '+name)

    def event(name, **kwargs):
        events.append(dict(name=name, seconds=time.monotonic()-start, **kwargs))
        write(args.out/'PATHS.json', events)

    base, domain = core.problem(core.TASKS[0])
    with np.load(ROOT/'evidence/DIAGNOSTIC_STATES.npz', allow_pickle=False) as z:
        flat = torch.tensor(z['half_baseline'].copy())
    p = PDFProblem(base, information(base))
    off = PDFProblem(base, enabled=False)
    with torch.no_grad():
        compare('disabled_residual', off.residual(flat), base.residual(flat), 0, 0)
        compare('original_prefix', p.residual(flat)[:6010], base.residual(flat), 0, 0)
        compare('disabled_constraints', domain.constraints(off,flat), domain.constraints(base,flat), 0, 0)
        compare('enabled_constraints', domain.constraints(p,flat), domain.constraints(base,flat), 0, 0)
        named = {k:float(v) for k,v in p.named_components(flat).items()}
    local = json.loads((ROOT/'local/checks/LOCAL_CHECKS.json').read_text())
    compare('Mac_PDF_value', named['J_PDF'], local['details']['states']['half_baseline']['J_PDF'])
    oracle = oracle_at(p, domain, flat)
    before = oracle.values(flat.numpy())
    compare('Oracle_components', before['components'], p.components(flat).detach().numpy())
    compare('Oracle_objective', before['J'], sum(before['components']))
    solver, oracle, stats, settings = isolated_solver(oracle)
    original_derivatives = oracle.derivatives
    def derivatives(a, indices):
        event('derivatives_enter', active_constraint_count=len(indices))
        result = original_derivatives(a, indices)
        theta = torch.tensor(a, dtype=torch.float64)
        with torch.no_grad():
            r = p.residual(theta).numpy()
        gradient, B, C = result
        if B.shape != (6046,3867):
            raise RuntimeError('Wrong joint residual Jacobian shape')
        compare('gradient_2BTr', gradient, 2*B.T@r)
        direction = torch.sin(torch.arange(1,3868,dtype=torch.float64))
        direction /= torch.linalg.norm(direction)
        directional = jvp(p.residual, (theta,), (direction,))[1].detach().numpy()
        compare('full_residual_directional_derivative', B@direction.numpy(), directional)
        compare('PDF_Jacobian_suffix', B[6010:], p.information.jacobian(theta).numpy())
        event('derivatives_complete', shape=list(B.shape), constraint_rows=len(C))
        return result
    oracle.derivatives = derivatives
    original_model = solver.model_coordinates
    def coordinates(*args, **kwargs):
        result = original_model(*args, **kwargs)
        event('local_model_complete', rank=int(result['rank']))
        return result
    solver.model_coordinates = coordinates
    original_values = oracle.values
    def values(a):
        result = original_values(a)
        event('initial_value' if np.array_equal(a,flat.numpy()) else 'trial_value', J=result['J'], constraint_max=float(np.max(result['c'])))
        return result
    oracle.values = values
    settings = dict(settings, max_iterations=1, max_evaluations=64)
    final, diagnostics = solver.solve(oracle, accepted.old.kkt, lambda *a: None, settings)
    after = oracle.values(np.asarray(final))
    seen = {v['name'] for v in events}
    missing = sorted({'derivatives_complete','local_model_complete','trial_value'}-seen)
    # Exercise original checkpoint value path on a diagnostic state only.
    core.checkpoint(p, torch.tensor(final), 'P2B_DIAGNOSTIC_ONLY', args.out/'DIAGNOSTIC_STEP.pt')
    checkpoint = torch.load(args.out/'DIAGNOSTIC_STEP.pt', weights_only=True)
    compare('checkpoint_objective', checkpoint['J'], after['J'])
    write(args.out/'QUALIFICATION.json', dict(
        status='INCOMPLETE_PATHS' if missing else 'INTEGRATION_CHECKS_PASS_REVIEW_REQUIRED',
        checks=checks, events=events, missing_paths=missing,
        before=named, after_J=after['J'], diagnostics=diagnostics, binder_stats=stats,
        seconds=time.monotonic()-start, peak_RSS_bytes=core.rss_bytes(), environment=core.environment(),
        new_joint_fits=0, new_noise_draws=0, diagnostic_step_ceiling=1,
        finalizer_scoring_and_continuation_driver_qualified=False,
        nonempty_active_path_observed=any(v.get('active_constraint_count',0)>0 for v in events),
        next_action='Review paths and resource evidence; construct/qualify the two-case execution driver before a separate scientific fit release'))
    return 2 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
