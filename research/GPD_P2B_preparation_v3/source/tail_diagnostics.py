"""Truth-only interpretation of omitted forward tails, never fit residuals."""
import argparse
from pathlib import Path
from p2b_common import ROOT, setup, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capsule', type=Path, required=True)
    a = p.parse_args()
    setup(a.capsule)
    import numpy as np
    from closure.truths import Truth
    truth = Truth('Z1')
    def integrals(order):
        u, w = np.polynomial.legendre.leggauss(order)
        pieces = []
        for lo, hi in [(0., .02), (.02, .8), (.8, 1.)]:
            x = lo+(hi-lo)*(u+1)/2
            f = truth.values(x, 0., 0.)
            moments = np.stack((x*f[:, 0], x*f[:, 1], f[:, 2], x*(2/9)*(f[:, 0]+f[:, 1])), axis=1)
            pieces.append((hi-lo)/2 * (w@moments))
        return np.array(pieces)
    fine, coarse = integrals(160), integrals(80)
    total = fine.sum(axis=0)
    analytic = np.array([truth.moment(c, 2, 0., 0.)/2 for c in range(3)])
    error = float(abs(fine-coarse).max())
    match = bool(np.allclose(total[:3], analytic, rtol=1e-12, atol=1e-13))
    write(ROOT/'local/TAIL_DIAGNOSTICS.json', dict(status='PASS' if error<1e-12 and match else 'FAIL',
        channels=['positive_x_Sigma_first_moment','positive_x_T_first_moment','positive_x_Hg_integral','positive_x_Hplus_first_moment'],
        intervals=[[0.,.02],[.02,.8],[.8,1.]], integrals=fine.tolist(),
        fractions=fine.__truediv__(total).tolist(), analytic_first_three=analytic.tolist(),
        refinement_max_abs=error, analytic_match=match,
        scope='Generator-only diagnostic, not observed tail data, a constraint, or an uncertainty estimate; x-grid also leaves unresolved behavior between points'))
    print(dict(status='PASS' if error<1e-12 and match else 'FAIL', Hplus_unmeasured_tail_fraction=float((fine[0,3]+fine[2,3])/total[3])))


if __name__ == '__main__':
    main()
