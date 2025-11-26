#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a 1D H^{(+)} band from the PyTorch GPD ensemble.

Reads:
  - gpd_inverse_from_ReH/gpd_inverse_summary.json
  - gpd_inverse_from_ReH/gpd_models/gpd_rep_*.pt

Evaluates each GPD replica at fixed (xi0, t0, Q20) on an x-grid,
then plots the ensemble mean ± 1σ band:

  H^{(+)}(x, xi0, t0, Q0^2)

with vertical dashed lines at x = ±xi0.
"""

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =====================
#  User settings
# =====================
OUT_DIR    = "gpd_inverse_from_ReH"
MODELS_DIR = os.path.join(OUT_DIR, "gpd_models")
SUMMARY_JSON = os.path.join(OUT_DIR, "gpd_inverse_summary.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If you want to override the slice from the summary, set these to floats.
# If they are None, we use the xi0, t0, Q20 values stored in gpd_inverse_summary.json.
XI0_OVERRIDE  = None   # e.g. 0.22
T0_OVERRIDE   = None   # e.g. -0.3
Q20_OVERRIDE  = None   # e.g. 4.0


# =====================
#  Model classes
# =====================
class GPDBaseNet(nn.Module):
    """Same architecture as used in GPD-H_pt.py."""
    def __init__(self, in_dim: int, hidden: int, layers: int):
        super().__init__()
        modules = []
        d = in_dim
        for _ in range(layers):
            modules.append(nn.Linear(d, hidden))
            modules.append(nn.Tanh())
            d = hidden
        modules.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def eval_H_single_replica(
    base: nn.Module,
    ckpt: dict,
    xgrid: torch.Tensor,
    xi0: float,
    t0: float,
    Q20: float,
    device: torch.device
) -> np.ndarray:
    """
    Evaluate H^{(+)}(x, xi0, t0, Q20) for a single GPD replica.

    Uses the same oddness projection + envelope used in training:
        H^{(+)}(x,...) = (1 - |x|)^beta * [f(x,...) - f(-x,...)] / 2
    """
    xi_mean = float(ckpt["xi_mean"])
    xi_std  = float(ckpt["xi_std"])
    t_mean  = float(ckpt["t_mean"])
    t_std   = float(ckpt["t_std"])
    Q2_mean = float(ckpt["Q2_mean"])
    beta    = float(ckpt["beta"])

    x = xgrid.to(device)  # [nx]

    xi = torch.full_like(x, float(xi0))
    t  = torch.full_like(x, float(t0))
    Q2 = torch.full_like(x, float(Q20))

    xi_n = (xi - xi_mean) / max(xi_std, 1e-6)
    t_n  = (t  - t_mean)  / max(t_std, 1e-6)
    qn   = (torch.log(Q2) - torch.log(torch.tensor(Q2_mean + 1e-12, device=device))) / 0.7

    # Features at x and -x
    feats_p = torch.stack([ x,  xi_n, t_n, qn], dim=1)
    feats_m = torch.stack([-x,  xi_n, t_n, qn], dim=1)

    with torch.no_grad():
        out_p = base(feats_p)
        out_m = base(feats_m)

    Hplus = 0.5 * (out_p - out_m)
    env   = torch.clamp(1.0 - x.abs(), min=0.0) ** beta
    H     = env * Hplus
    return H.detach().cpu().numpy()


# =====================
#  Main plotting
# =====================
def main():
    device = torch.device(DEVICE)
    print("[plot] using device:", device)

    if not os.path.isfile(SUMMARY_JSON):
        raise FileNotFoundError(f"Summary JSON not found: {SUMMARY_JSON}")
    with open(SUMMARY_JSON, "r") as f:
        summary = json.load(f)

    # Slice kinematics (use summary unless overridden)
    slice_info = summary.get("slice", {})
    xi0 = float(XI0_OVERRIDE if XI0_OVERRIDE is not None else slice_info.get("xi0"))
    t0  = float(T0_OVERRIDE  if T0_OVERRIDE  is not None else slice_info.get("t0"))
    Q20 = float(Q20_OVERRIDE if Q20_OVERRIDE is not None else slice_info.get("Q20"))

    nx_summary = int(summary.get("nx", 181))
    hidden = int(summary.get("gpd_hidden", 64))
    layers = int(summary.get("gpd_layers", 2))

    # Load all GPD replicas
    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "gpd_rep_*.pt")))
    if not model_files:
        raise FileNotFoundError(f"No gpd_rep_*.pt found in {MODELS_DIR}")

    nrep = len(model_files)
    print(f"[plot] found {nrep} GPD replicas")

    xgrid = torch.linspace(-1.0, 1.0, nx_summary, device=device)
    H_stack = []

    for path in model_files:
        ckpt = torch.load(path, map_location=device)
        base = GPDBaseNet(in_dim=4, hidden=hidden, layers=layers).to(device)
        base.load_state_dict(ckpt["state_dict"])
        base.eval()
        Hx = eval_H_single_replica(base, ckpt, xgrid, xi0, t0, Q20, device)
        H_stack.append(Hx)

    H_stack = np.stack(H_stack, axis=0)  # [nrep, nx]
    H_mean  = H_stack.mean(axis=0)
    H_std   = H_stack.std(axis=0)
    x_np    = xgrid.detach().cpu().numpy()

    # ================== Plot ==================
    plt.figure(figsize=(6.8, 4.4))
    plt.fill_between(x_np, H_mean - H_std, H_mean + H_std,
                     alpha=0.3, label="ensemble ±1σ")
    plt.plot(x_np, H_mean, linewidth=1.8, label="mean reconstruction")

    # Dashed lines at x = ±xi0
    plt.axvline(+abs(xi0), linestyle="--", linewidth=1)
    plt.axvline(-abs(xi0), linestyle="--", linewidth=1)

    plt.xlabel("x")
    plt.ylabel(r"$H^{(+)}(x,\xi_0,t_0,Q_0^2)$")

    title = (r"Ensemble H$^{+}$ band at "
             + rf"$\xi={xi0:.3f}$, $t={t0:.3f}$ GeV$^2$, "
             + rf"$Q_0^2={Q20:.2f}$ GeV$^2$ "
             + rf"(Nrep={nrep})")
    plt.title(title)

    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"H_band_xi{xi0:.3f}_t{t0:.3f}_Q2{Q20:.2f}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
