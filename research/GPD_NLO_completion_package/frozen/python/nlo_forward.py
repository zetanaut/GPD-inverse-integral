"""Fixed-nf=3, MSbar two-loop running and ordered grid evolution.

This module accepts *dimensionless, separately validated* K0 and K1 matrices.
It does not manufacture or certify the two-loop nonforward kernels.  The
evolution equation is dH/dln(mu_F^2) = [a K0 + a^2 K1] H, a=alpha_s/(2 pi).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Prescription:
    q02: float = 2.0
    alpha0: float = 0.35691594851514724
    nf: int = 3
    scheme: str = "MSbar"

    def __post_init__(self):
        if self.nf != 3 or self.scheme != "MSbar":
            raise ValueError("Only frozen fixed-nf=3 MSbar prescription is supported")
        if self.q02 <= 0 or not 0 < self.alpha0 < 1:
            raise ValueError("Invalid coupling boundary")

    @property
    def beta0(self):
        return 11 - 2 * self.nf / 3

    @property
    def beta1(self):
        return 102 - 38 * self.nf / 3


def alpha_two_loop(q2: float, p: Prescription = Prescription()) -> float:
    """Solve da_s/dln Q² = -beta0*a_s²-beta1*a_s³; a_s=alpha/(4pi)."""
    if q2 <= 0:
        raise ValueError("Q² must be positive")
    t = log(q2 / p.q02)
    if t == 0:
        return p.alpha0
    a0 = p.alpha0 / (4 * pi)

    def beta(_, y):
        a = y[0]
        return [-p.beta0 * a * a - p.beta1 * a * a * a]

    sol = solve_ivp(beta, (0, t), [a0], rtol=2e-13, atol=1e-15)
    if not sol.success or sol.y[0, -1] <= 0:
        raise ValueError("Two-loop coupling failed or reached its Landau region")
    return float(sol.y[0, -1] * 4 * pi)


def _matrices(k0, k1):
    k0 = np.asarray(k0, dtype=float)
    k1 = np.asarray(k1, dtype=float)
    if k0.ndim != 2 or k0.shape[0] != k0.shape[1] or k1.shape != k0.shape:
        raise ValueError("K0 and K1 must be equal-size square matrices")
    if not np.isfinite(k0).all() or not np.isfinite(k1).all():
        raise ValueError("K0/K1 contain nonfinite values")
    return k0, k1


def _generator(t, k0, k1, p):
    a = alpha_two_loop(p.q02 * np.exp(t), p) / (2 * pi)
    return a * k0 + a * a * k1


def evolve_field(field, k0, k1, q2: float, p: Prescription = Prescription(),
                 steps: int = 128):
    """RK4 ordered forward action, with no commuting-K approximation."""
    k0, k1 = _matrices(k0, k1)
    v = np.asarray(field, dtype=float).copy()
    if v.shape != (k0.shape[0],):
        raise ValueError("Field dimension does not match kernel")
    if steps < 1 or q2 <= 0:
        raise ValueError("Invalid step count or Q²")
    t1 = log(q2 / p.q02)
    h = t1 / steps
    for j in range(steps):
        t = j * h
        m0 = _generator(t, k0, k1, p)
        mm = _generator(t + h / 2, k0, k1, p)
        m1 = _generator(t + h, k0, k1, p)
        d1 = m0 @ v
        d2 = mm @ (v + h * d1 / 2)
        d3 = mm @ (v + h * d2 / 2)
        d4 = m1 @ (v + h * d3)
        v += h * (d1 + 2 * d2 + 2 * d3 + d4) / 6
    return v


def precompute_row(row, k0, k1, q2: float,
                   p: Prescription = Prescription(), steps: int = 128):
    """Backward adjoint action C U(q2,q02), also the initial-field gradient."""
    k0, k1 = _matrices(k0, k1)
    c = np.asarray(row, dtype=complex).copy()
    if c.shape != (k0.shape[0],):
        raise ValueError("Coefficient row dimension does not match kernel")
    if steps < 1 or q2 <= 0:
        raise ValueError("Invalid step count or Q²")
    t1 = log(q2 / p.q02)
    h = -t1 / steps
    for j in range(steps):
        t = t1 + j * h
        m0 = _generator(t, k0, k1, p)
        mm = _generator(t + h / 2, k0, k1, p)
        m1 = _generator(t + h, k0, k1, p)
        d1 = -(c @ m0)
        d2 = -((c + h * d1 / 2) @ mm)
        d3 = -((c + h * d2 / 2) @ mm)
        d4 = -((c + h * d3) @ m1)
        c += h * (d1 + 2 * d2 + 2 * d3 + d4) / 6
    return c
