"""Task-02 spacelike vector hard coefficients on the frozen LO GPD grid.

Only the hard function is one-loop.  Evolution and running coupling remain LO;
the combined order is explicitly ``NLO_COEFF_LO_EVOL``.  The coefficient is
linear in nodal GPD values, so the returned complex rows are differentiable.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

CF = 4.0 / 3.0
TF = 0.5
CHARGE_SQUARES = np.array([4, 1, 1, 4, 1], dtype=float) / 9.0
ORDER_LABEL = "NLO_COEFF_LO_EVOL"


def _log_lower(z, xi):
    """R3 Log((z+xi)/(2xi)-i0), including the lower negative-axis bank."""
    a = np.asarray((z + xi) / (2 * xi), dtype=complex)
    # The infinitesimal is used only to select a branch, not to regularize a
    # denominator or define a finite-i0 numerical approximation.
    a = np.where(np.imag(a) == 0, a - 1e-300j, a)
    return np.log(a)


def _unskewed_q1(z, xi, alpha, log_q2_muf2=0.0):
    z = np.asarray(z, dtype=complex)
    ell = _log_lower(z, xi)
    rho = (z + xi) / (z - xi)
    hard = 9 - 3 * rho * ell - ell * ell
    coll = -3 - 2 * ell
    return alpha * CF / (4 * np.pi) * (hard + log_q2_muf2 * coll) / (z + xi)


def _unskewed_g1(z, xi, alpha, log_q2_muf2=0.0):
    z = np.asarray(z, dtype=complex)
    ell = _log_lower(z, xi)
    rho = (z + xi) / (z - xi)
    hard = 2 * (z + 3 * xi) / (z - xi) * ell - rho * ell * ell
    coll = -2 * rho * ell
    return alpha * TF / (4 * np.pi) * (hard + log_q2_muf2 * coll) / ((z + xi) * (z - xi))


def correction_kernel(z, xi, alpha, channel, log_q2_muf2=0.0, nf=3):
    """Positive-x manuscript-sign one-loop kernel, excluding input GPD.

    Quark input is H_charge^+=wbar*Sigma+T.  Gluon input is H_g=xg on
    positive x; its charge sum and full-to-positive-x factor two are here.
    """
    if not 0 < xi < 1 or not 3 <= nf <= 5 or alpha < 0:
        raise ValueError("Invalid xi, nf, or alpha")
    if channel == "quark":
        return -(_unskewed_q1(z, xi, alpha, log_q2_muf2) -
                 _unskewed_q1(-np.asarray(z), xi, alpha, log_q2_muf2))
    if channel == "gluon":
        charge_sum = float(np.sum(CHARGE_SQUARES[:nf]))
        return -2 * charge_sum * (
            _unskewed_g1(z, xi, alpha, log_q2_muf2) +
            _unskewed_g1(-np.asarray(z), xi, alpha, log_q2_muf2))
    raise ValueError("channel must be quark or gluon")


def _constant_integral(xi, alpha, channel, log_q2_muf2, nf, height):
    """Integrate the constant crossover mode above the physical cut.

    This is the analytic i0 boundary value.  The deformation is allowed for
    the *constant* mode; a piecewise-polynomial field is separately subtracted
    before real-axis integration, so no false analytic continuation of its
    element boundaries is made.
    """
    def f(t):
        z = t + 1j * height * np.sin(np.pi * t)
        dz = 1 + 1j * height * np.pi * np.cos(np.pi * t)
        return correction_kernel(z, xi, alpha, channel, log_q2_muf2, nf) * dz
    real = quad(lambda t: float(np.real(f(t))), 0, 1, points=[xi],
                epsabs=2e-12, epsrel=2e-12, limit=300)
    imag = quad(lambda t: float(np.imag(f(t))), 0, 1, points=[xi],
                epsabs=2e-12, epsrel=2e-12, limit=300)
    return real[0] + 1j * imag[0], max(real[1], imag[1])


def _subtracted_nodes(grid, order):
    xi = float(grid.xi)
    edges = np.unique(np.r_[0.0, grid.edges[(grid.edges > 0) & (grid.edges < 1)], xi, 1.0])
    z, w = leggauss(order)
    u = (z + 1) / 2
    ys, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if abs(a - xi) < 1e-13:
            y = a + (b - a) * u**2
            wt = (b - a) * u * w
        elif abs(b - xi) < 1e-13:
            y = b - (b - a) * u**2
            wt = (b - a) * u * w
        else:
            y = a + (b - a) * u
            wt = (b - a) * w / 2
        ys.append(y)
        ws.append(wt)
    return np.concatenate(ys), np.concatenate(ws)


def correction_rows(grid, alpha, order=96, log_q2_muf2=0.0, nf=3,
                    contour_height=None):
    """Complex quark/gluon rows with crossover subtraction, no pole mask.

    Returns (quark_row, gluon_row, diagnostics).  Each row acts on the real
    nodal values of the positive-x quark-plus or even gluon GPD.  The constant
    mode is restored by an independently converged contour integral.  The
    residual vanishes at x=xi and is integrated on each spectral element.
    """
    if order < 16:
        raise ValueError("Order too small for a logarithmic crossover")
    xi = float(grid.xi)
    height = min(0.25 * xi, 0.08) if contour_height is None else contour_height
    if not 0 < height < 1:
        raise ValueError("Contour height must be positive and less than one")
    y, wt = _subtracted_nodes(grid, order)
    interpolant = grid.interpolation(y)
    endpoint = grid.interpolation([xi])[0]
    residual = interpolant - endpoint
    rows = []
    constants = {}
    for channel in ("quark", "gluon"):
        kernel = correction_kernel(y, xi, alpha, channel, log_q2_muf2, nf)
        constant, estimated_error = _constant_integral(
            xi, alpha, channel, log_q2_muf2, nf, height)
        row = (wt * kernel) @ residual + constant * endpoint
        rows.append(row)
        constants[channel] = dict(value=[constant.real, constant.imag],
                                  quad_error=estimated_error)
    return rows[0], rows[1], dict(order=order, contour_height=height,
                                  constant_integrals=constants,
                                  subtracted_nodes=len(y))


def reduced_amplitude(grid, sigma, t, gluon, alpha, order=96,
                      log_q2_muf2=0.0, nf=3,
                      requested_order=ORDER_LABEL):
    """LO baseline plus NLO hard correction, never full NLO evolution."""
    if requested_order != ORDER_LABEL:
        raise NotImplementedError("Only NLO_COEFF_LO_EVOL is available; full NLO evolution is task 03")
    # Import lazily so this task-scoped module does not alter the LO package.
    from gpdpert.matching import coefficient_rows
    lo = coefficient_rows(grid, order=max(64, order), convention="manuscript")
    cq, cg, diagnostics = correction_rows(grid, alpha, order,
                                          log_q2_muf2, nf)
    wbar = float(np.mean(CHARGE_SQUARES[:nf]))
    hcharge = wbar * np.asarray(sigma) + np.asarray(t)
    out = complex(*(lo @ hcharge)) + cq @ hcharge + cg @ np.asarray(gluon)
    return out, dict(order_label=ORDER_LABEL, hard_order="NLO",
                     evolution_order="LO", coupling_order="LO",
                     diagnostics=diagnostics)
