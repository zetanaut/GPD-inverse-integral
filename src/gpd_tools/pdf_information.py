"""Finite supported boundary observations; no truth import or inferred PDF prior."""
import math
import os
import sys
import numpy as np
import torch
from torch.func import jacrev


def map_xf_to_stored(x, xu, xubar, xd, xdbar, xs, xsbar, xg):
    """LHAPDF-convention input arrays x*f, at one already matched Q2."""
    arrays = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in
                                  (x, xu, xubar, xd, xdbar, xs, xsbar, xg)])
    x, xu, xubar, xd, xdbar, xs, xsbar, xg = arrays
    if not all(np.isfinite(v).all() for v in arrays) or np.any((x <= 0) | (x >= 1)):
        raise ValueError('Finite inputs and 0<x<1 required; no endpoint clipping')
    U, D, S = (xu+xubar)/x, (xd+xdbar)/x, (xs+xsbar)/x
    return np.stack((U+D+S, U-(D+S)/2, xg), axis=-1)


class ForwardInformation:
    """Observation vector and covariance only; output order is x-major."""
    def __init__(self, base, x, observed, covariance, normalizer=160):
        x = np.asarray(x, dtype=float)
        observed = np.asarray(observed, dtype=float)
        cov = np.asarray(covariance, dtype=float)
        if normalizer != 160:
            raise ValueError('P2B v1 uses common CFF/PDF normalizer 160')
        if x.shape != (12,) or observed.shape != (12, 3) or cov.shape != (36, 36):
            raise ValueError('P2B v1 expects 12 x coordinates and three channels')
        if not all(np.isfinite(a).all() for a in (x, observed, cov)):
            raise ValueError('Nonfinite forward information')
        if not np.allclose(x, np.geomspace(.02, .8, 12), rtol=0, atol=1e-15):
            raise ValueError('Supported forward grid changed')
        if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-15):
            raise ValueError('Nonsymmetric covariance')
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[-1] <= 0 or eigenvalues[0]/eigenvalues[-1] <= 1e-12:
            raise ValueError('Covariance is not sufficiently positive definite; no silent pseudoinverse/jitter')
        self.base = base
        self.x = torch.tensor(x[None], dtype=torch.float64)
        self.zero = torch.zeros(1, dtype=torch.float64)
        self.observed = torch.tensor(observed.reshape(-1), dtype=torch.float64)
        self.cholesky = torch.tensor(np.linalg.cholesky(cov), dtype=torch.float64)
        self.normalizer = normalizer
        self.eigenvalues = eigenvalues

    def prediction(self, flat):
        return self.base.field(flat, self.x, self.zero, self.zero)[0].T.reshape(-1)

    def residual(self, flat):
        delta = self.prediction(flat)-self.observed
        return torch.linalg.solve_triangular(self.cholesky, delta[:, None], upper=False)[:, 0]/math.sqrt(self.normalizer)

    def jacobian(self, flat):
        return jacrev(self.residual, chunk_size=8)(flat.detach()).detach()


class PDFProblem:
    """Append 36 real whitened residuals to the unchanged 6010-row P1 block."""
    def __init__(self, base, information=None, enabled=True):
        if enabled and (information is None or information.base is not base):
            raise ValueError('Enabled information must use the identical base field')
        self.base = base
        self.information = information
        self.enabled = bool(enabled)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def residual(self, flat):
        r = self.base.residual(flat)
        return torch.cat((r, self.information.residual(flat))) if self.enabled else r

    def objective(self, flat):
        return self.residual(flat).square().sum()

    def original_objective(self, flat):
        return self.base.objective(flat)

    def components(self, flat):
        old = self.base.components(flat)
        return torch.cat((old, self.information.residual(flat).square().sum().reshape(1))) if self.enabled else old

    def named_components(self, flat):
        old = self.base.components(flat)
        pdf = self.information.residual(flat).square().sum() if self.enabled else old.new_zeros(())
        return dict(J_CFF=old[0], J_poly=old[1], J_PDF=pdf,
                    chi2_PDF=160*pdf, J_P1=old.sum(), J_joint=old.sum()+pdf)

    def residual_jacobian(self, flat):
        if sys.platform != 'linux' or not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Full joint Jacobian requires allocated-node qualification')
        old = self.base.residual_jacobian(flat)
        return torch.cat((old, self.information.jacobian(flat))) if self.enabled else old

    def jacobian(self, flat, chunk_size=8):
        return self.residual_jacobian(flat)
