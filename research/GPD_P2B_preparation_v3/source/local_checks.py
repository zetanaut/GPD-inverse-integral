"""Bounded boundary/mapping/derivative checks; no joint fit or base Jacobian."""
import argparse
from pathlib import Path
import json
import time
from p2b_common import ROOT, setup, field_only, information, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capsule', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError('Use a new empty check output')
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    core = setup(args.capsule)
    import numpy as np
    import torch
    from torch.func import jacrev, jvp, vjp
    from pdf_information import map_xf_to_stored, ForwardInformation, PDFProblem
    from closure.truths import Truth
    checks = {}
    details = {}

    def compare(name, a, b, rtol=2e-7, atol=1e-9):
        a = a.detach().numpy() if torch.is_tensor(a) else np.asarray(a)
        b = b.detach().numpy() if torch.is_tensor(b) else np.asarray(b)
        ok = a.shape == b.shape and bool(np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, rtol=rtol, atol=atol))
        checks[name] = dict(pass_=ok, max_abs=float(abs(a-b).max()) if a.shape == b.shape else None, rtol=rtol, atol=atol)

    def expect_reject(name, fn):
        try:
            fn()
        except ValueError:
            checks[name] = dict(pass_=True)
        else:
            checks[name] = dict(pass_=False)

    base = field_only(core)
    info = information(base)
    with np.load(ROOT/'inputs/CENTRAL_FORWARD.npz', allow_pickle=False) as z:
        x, y, cov = [z[k].copy() for k in ('x', 'observed', 'covariance')]
    # Algebraic flavor/xf test, independent of the Z1 channel generator.
    u, ub, d, db, s, sb, g = 2+x, .4+x*x, .7+3*x, .2+x, .1+x/3, .1+x/4, 1.5+2*x
    mapped = map_xf_to_stored(x, x*u, x*ub, x*d, x*db, x*s, x*sb, x*g)
    U, D, S = u+ub, d+db, s+sb
    compare('xf_quark_division_and_xg_retention', mapped, np.stack((U+D+S, U-(D+S)/2, x*g), axis=-1), 1e-13, 1e-14)
    compare('charge_identity', (2/9)*(mapped[:, 0]+mapped[:, 1]), (4*U+D+S)/9, 1e-13, 1e-14)
    compare('generator_forward_input_identity', y, Truth('Z1').values(x, 0., 0.), 0, 0)
    compare('cholesky_covariance', info.cholesky@info.cholesky.T, cov, 1e-12, 1e-14)
    expect_reject('rank_deficient_covariance_rejected', lambda: ForwardInformation(base, x, y, np.zeros_like(cov)))
    expect_reject('wrong_common_normalizer_rejected', lambda: ForwardInformation(base, x, y, cov, 36))
    expect_reject('unsupported_x_rejected', lambda: ForwardInformation(base, np.linspace(0., .8, 12), y, cov))
    expect_reject('nonfinite_observation_rejected', lambda: ForwardInformation(base, x, y*np.nan, cov))
    states = {'fresh_seed31': base.x0}
    with np.load(ROOT/'evidence/DIAGNOSTIC_STATES.npz', allow_pickle=False) as z:
        states.update({k: torch.tensor(z[k].copy()) for k in z.files})
    k = torch.arange(1, 3868, dtype=torch.float64)
    d1 = torch.sin(k); d1 /= torch.linalg.norm(d1)
    mask = torch.zeros_like(k); pos = 0
    for name, shape, n in base.spec:
        if name.startswith('conditioners.'):
            mask[pos:pos+n] = 1
        pos += n
    d2 = torch.cos(k)*mask; d2 /= torch.linalg.norm(d2)
    details['states'] = {}
    for name, flat in states.items():
        pred = info.prediction(flat).detach().numpy().reshape(12, 3)
        delta = (pred-y).reshape(-1)
        r = info.residual(flat).detach()
        independently_whitened = np.linalg.solve(np.linalg.cholesky(cov), delta)
        compare(name+'_mahalanobis', float(r@r)*160, float(delta@np.linalg.solve(cov, delta)), 1e-10, 1e-10)
        details['states'][name] = dict(
            boundary_channel_relative_L2=np.sqrt(np.sum((pred-y)**2, axis=0)/np.sum(y**2, axis=0)).tolist(),
            chi2_PDF=float(independently_whitened@independently_whitened),
            whitened_RMS=float(np.sqrt(np.mean(independently_whitened**2))),
            J_PDF=float(r@r), truth_use='post hoc diagnostic, not checkpoint selection or initialization')
        if name not in ('fresh_seed31', 'half_baseline'):
            continue
        B = info.jacobian(flat)
        theta = flat.clone().requires_grad_(True)
        g = torch.autograd.grad(info.residual(theta).square().sum(), theta)[0]
        compare(name+'_gradient_2BTr', g, 2*B.T@r)
        v = torch.cos(torch.arange(1, 37, dtype=torch.float64)); v /= torch.linalg.norm(v)
        _, back = vjp(info.residual, flat)
        compare(name+'_adjoint', back(v)[0], B.T@v)
        for dname, direction in (('all', d1), ('conditioners', d2)):
            forward = jvp(info.residual, (flat,), (direction,))[1]
            compare(name+'_'+dname+'_JVP', forward, B@direction)
            errors = []
            for eps in (1e-5, 3e-6, 1e-6):
                h = eps*max(1., float(torch.linalg.norm(flat)))
                fd = (info.residual(flat+h*direction)-info.residual(flat-h*direction))/(2*h)
                error = float(torch.linalg.norm(fd-forward))
                bound = 1e-8+1e-5*max(float(torch.linalg.norm(fd)), float(torch.linalg.norm(forward)))
                errors.append(dict(epsilon=eps, error=error, bound=bound, pass_=error <= bound))
            checks[name+'_'+dname+'_finite_difference'] = dict(pass_=any(errors[i]['pass_'] and errors[i+1]['pass_'] for i in (0, 1)), steps=errors)
    # A full residual VALUE evaluation uses cached original NLO rows, no map build.
    p, _ = core.problem(core.ENDPOINTS[0], numerical=False)
    pdf = information(p)
    joint = PDFProblem(p, pdf)
    off = PDFProblem(p, enabled=False)
    with torch.no_grad():
        original = p.residual(p.x0)
        whole = joint.residual(p.x0)
        compare('unchanged_original_residual_prefix', whole[:6010], original, 0, 0)
        compare('disabled_exact_residual_identity', off.residual(p.x0), original, 0, 0)
        compare('disabled_exact_objective_identity', off.objective(p.x0), p.objective(p.x0), 0, 0)
        named = joint.named_components(p.x0)
        compare('joint_components_sum', joint.components(p.x0).sum(), whole.square().sum(), 1e-12, 1e-12)
        compare('common_loss_normalization', named['J_joint']-named['J_P1'], named['chi2_PDF']/160, 1e-12, 1e-12)
        checks['dimensions'] = dict(pass_=len(original)==6010 and len(whole)==6046 and p.x0.numel()==3867)
        checks['no_grad_values'] = dict(pass_=all(v.grad_fn is None and not v.requires_grad for v in named.values()))
        details['fresh_joint_components'] = {n: float(v) for n, v in named.items()}
    # One deterministic output-head linear solve. It never enters a joint fit.
    cfg = json.loads((ROOT/'DESIGN.json').read_text())
    rc = cfg['representation_check']
    train_x = np.linspace(x[0], x[-1], rc['training_points'])
    edges = np.linspace(x[0], x[-1], rc['checking_points']+1)
    check_x = (edges[:-1]+edges[1:])/2
    indices = []; pos = 0
    for name, shape, n in base.spec:
        if name.startswith('output.'):
            indices.extend(range(pos, pos+n))
        pos += n
    ids = torch.tensor(indices, dtype=torch.long)
    zero = torch.zeros(1, dtype=torch.float64)
    train_t = torch.tensor(train_x[None])
    def head_field(head):
        theta = base.x0.index_copy(0, ids, head)
        return base.field(theta, train_t, zero, zero)[0].T.reshape(-1)
    hzero = torch.zeros(len(ids), dtype=torch.float64)
    A = jacrev(head_field, chunk_size=8)(hzero).detach().numpy()
    target = Truth('Z1').values(train_x, 0., 0.).reshape(-1)
    head, _, rank, singular = np.linalg.lstsq(A, target, rcond=rc['svd_rcond'])
    witness = base.x0.index_copy(0, ids, torch.tensor(head))
    check_truth = Truth('Z1').values(check_x, 0., 0.)
    with torch.no_grad():
        fitted = base.field(witness, torch.tensor(check_x[None]), zero, zero)[0].T.numpy()
    relative = np.sqrt(np.sum((fitted-check_truth)**2, axis=0)/np.sum(check_truth**2, axis=0))
    sigma = cfg['covariance']['relative_sigma']*np.maximum(abs(check_truth), cfg['covariance']['floor_fraction_of_channel_grid_maximum']*abs(y).max(axis=0))
    maximum = float(np.max(abs(fitted-check_truth)/sigma))
    checks['forward_representation_channel_errors'] = dict(pass_=bool(np.all(relative <= rc['channel_relative_L2_max'])), measured=relative.tolist(), limit=rc['channel_relative_L2_max'])
    checks['forward_representation_max_standardized_error'] = dict(pass_=maximum <= rc['maximum_absolute_standardized_error'], measured=maximum, limit=rc['maximum_absolute_standardized_error'])
    details['representation'] = dict(rank=int(rank), head_columns=len(ids), singular_values=singular.tolist(),
        fitted_head_L2=float(np.linalg.norm(head)), original_head_L2=float(torch.linalg.norm(base.x0[ids])),
        training_points=len(train_x), disjoint_checking_points=len(check_x),
        simultaneous_CFF_or_moment_feasibility_tested=False, witness_used_for_scientific_initialization=False,
        caveat='Sampled forward witness only; potentially large canceling head coefficients are reported and are not evidence of stable joint optimization')
    np.savez_compressed(args.out/'REPRESENTATION_CHECK.npz', x=check_x, target=check_truth, prediction=fitted)
    ok = all(v['pass_'] for v in checks.values())
    write(args.out/'LOCAL_CHECKS.json', dict(status='PASS' if ok else 'FAIL', checks=checks, details=details,
        seconds=time.monotonic()-start, peak_RSS_bytes=core.rss_bytes(), environment=core.environment(),
        new_joint_fits=0, new_noise_draws=0, forward_head_diagnostic_solves=1,
        full_joint_jacobian_or_solver_step_executed=False))
    print(json.dumps(dict(status='PASS' if ok else 'FAIL', checks=len(checks), representation_relative=relative.tolist(), representation_max_sigma=maximum, seconds=time.monotonic()-start, peak_RSS_bytes=core.rss_bytes())))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
