#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D GPD surface with ±1σ band from the PyTorch GPD ensemble.

Plots H^{(+)}(x0, xi, t, Q0^2) for:
  - the ensemble mean (colored surface),
  - mean + 1σ (translucent surface above),
  - mean - 1σ (translucent surface below).

Inputs (produced by GPD-H_pt.py):
  - gpd_inverse_from_ReH/gpd_inverse_summary.json
  - gpd_inverse_from_ReH/gpd_models/gpd_rep_*.pt

Output:
  - gpd_inverse_from_ReH/H_surface_band_x{X0}_Q2{Q20}.png
  and an interactive window you can rotate with the mouse.
"""

import os
import glob
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ==========================
#  User config
# ==========================

OUT_DIR       = "gpd_inverse_from_ReH"
MODELS_DIR    = os.path.join(OUT_DIR, "gpd_models")
SUMMARY_JSON  = os.path.join(OUT_DIR, "gpd_inverse_summary.json")

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed x0 and Q0^2 for the slice H^{(+)}(x0, xi, t, Q0^2)
# Set USE_SUMMARY_Q2 = True to use the Q^2 from the summary slice instead.
X0           = 0.20      # fixed x
USE_SUMMARY_Q2 = True
Q20_OVERRIDE = 2.12      # only used if USE_SUMMARY_Q2 = False

# Grid resolution in (xi, t)
N_XI = 45
N_T  = 45


# ==========================
#  Model definition
# ==========================

class GPDBaseNet(nn.Module):
    """Same architecture as in GPD-H_pt.py."""
    def __init__(self, in_dim: int = 4, hidden: int = 64, layers: int = 2):
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


# ==========================
#  Helpers
# ==========================

def load_summary(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing summary JSON: {path}")
    with open(path, "r") as f:
        summary = json.load(f)

    nx     = int(summary["nx"])
    beta   = float(summary["beta"])
    hidden = int(summary["gpd_hidden"])
    layers = int(summary["gpd_layers"])
    grid   = summary["grid"]

    slice_info = summary.get("slice", {})
    xi0_s = slice_info.get("xi0", slice_info.get("xi"))
    t0_s  = slice_info.get("t0",  slice_info.get("t"))
    Q20_s = slice_info.get("Q20", slice_info.get("Q2"))

    return summary, nx, beta, hidden, layers, grid, xi0_s, t0_s, Q20_s


def eval_H_surface_for_model(
    ckpt_path: str,
    XI: np.ndarray,
    TT: np.ndarray,
    x0: float,
    Q20: float,
    hidden: int,
    layers: int,
    beta_default: float,
    device: torch.device
) -> np.ndarray:
    """
    Evaluate H^{(+)}(x0, xi, t, Q20) on the (XI,TT) grid for one GPD replica.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    net = GPDBaseNet(in_dim=4, hidden=hidden, layers=layers).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    xi_mean = torch.tensor(ckpt["xi_mean"], device=device, dtype=torch.float32)
    xi_std  = torch.tensor(ckpt["xi_std"],  device=device, dtype=torch.float32)
    t_mean  = torch.tensor(ckpt["t_mean"],  device=device, dtype=torch.float32)
    t_std   = torch.tensor(ckpt["t_std"],   device=device, dtype=torch.float32)
    Q2_mean = torch.tensor(ckpt["Q2_mean"], device=device, dtype=torch.float32)
    beta    = float(ckpt.get("beta", beta_default))

    # Flatten grid
    xi_flat = XI.ravel().astype(np.float32)
    t_flat  = TT.ravel().astype(np.float32)
    N = xi_flat.size

    x_flat  = np.full(N, float(x0), dtype=np.float32)
    q2_flat = np.full(N, float(Q20), dtype=np.float32)

    x  = torch.from_numpy(x_flat).to(device)
    xi = torch.from_numpy(xi_flat).to(device)
    t  = torch.from_numpy(t_flat).to(device)
    Q2 = torch.from_numpy(q2_flat).to(device)

    # Normalize kinematics
    xi_n = (xi - xi_mean) / xi_std.clamp_min(1e-12)
    t_n  = (t  - t_mean)  / t_std.clamp_min(1e-12)
    qn   = (torch.log(Q2) - torch.log(Q2_mean + 1e-12)) / 0.7

    feats_p = torch.stack([ x,  xi_n, t_n, qn], dim=1)
    feats_m = torch.stack([-x,  xi_n, t_n, qn], dim=1)

    with torch.no_grad():
        out_p = net(feats_p)   # [N]
        out_m = net(feats_m)

    Hplus = 0.5 * (out_p - out_m)
    env   = torch.clamp(1.0 - x.abs(), min=0.0) ** beta  # scalar per point
    H     = (env * Hplus).detach().cpu().numpy().reshape(XI.shape)
    return H


# ==========================
#  Main
# ==========================

def main():
    device = torch.device(DEVICE)
    print("[plot] using device:", device)

    summary, nx, beta, hidden, layers, grid_meta, xi0_s, t0_s, Q20_s = load_summary(SUMMARY_JSON)

    # Choose Q0^2
    if USE_SUMMARY_Q2 and Q20_s is not None:
        Q20 = float(Q20_s)
    else:
        Q20 = float(Q20_OVERRIDE)

    print(f"[plot] x0 = {X0:.3f}, Q0^2 = {Q20:.2f} GeV^2")

    # Build (xi,t) grid from inversion grid meta
    xi_min, xi_max = float(grid_meta["xi_min"]), float(grid_meta["xi_max"])
    t_min,  t_max  = float(grid_meta["t_min"]),  float(grid_meta["t_max"])

    xi_vals = np.linspace(xi_min, xi_max, N_XI, dtype=np.float32)
    t_vals  = np.linspace(t_min,  t_max,  N_T,  dtype=np.float32)
    XI, TT  = np.meshgrid(xi_vals, t_vals)

    # Load all GPD replicas
    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "gpd_rep_*.pt")))
    if not model_files:
        raise FileNotFoundError(f"No gpd_rep_*.pt found in {MODELS_DIR}")

    nrep = len(model_files)
    print(f"[plot] found {nrep} GPD replicas")

    # Evaluate ensemble on the grid
    H_stack = []
    for path in model_files:
        Hk = eval_H_surface_for_model(
            ckpt_path=path,
            XI=XI,
            TT=TT,
            x0=X0,
            Q20=Q20,
            hidden=hidden,
            layers=layers,
            beta_default=beta,
            device=device
        )
        H_stack.append(Hk)

    H_stack = np.stack(H_stack, axis=0)  # [nrep, N_T, N_XI]
    H_mean  = H_stack.mean(axis=0)
    H_std   = H_stack.std(axis=0)

    # ---------- 3D plot ----------
    fig = plt.figure(figsize=(8.0, 5.8))
    ax = fig.add_subplot(111, projection='3d')

    # Mean surface
    surf_mean = ax.plot_surface(XI, TT, H_mean,
                                linewidth=0, antialiased=True)

    # ±1σ surfaces (translucent)
    ax.plot_surface(XI, TT, H_mean + H_std,
                    linewidth=0, antialiased=True, alpha=0.25)
    ax.plot_surface(XI, TT, H_mean - H_std,
                    linewidth=0, antialiased=True, alpha=0.25)

    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$t$ [GeV$^2$]")
    ax.set_zlabel(r"$H^{(+)}(x_0,\xi,t,Q_0^2)$")

    title = (r"H$^{+}$ surface at "
             + rf"$x_0={X0:.3f}$, "
             + rf"$Q_0^2={Q20:.2f}$ GeV$^2$ "
             + rf"(Nrep={nrep})")
    ax.set_title(title)

    # Colorbar for mean H
    fig.colorbar(surf_mean, ax=ax, shrink=0.7, pad=0.08, label=r"$H^{(+)}$ (mean)")

    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, f"H_surface_band_x{X0:.3f}_Q2{Q20:.2f}.png")
    fig.savefig(out_png, dpi=150)
    print("[plot] saved:", out_png)

    # Interactive view so you can rotate
    plt.show()


if __name__ == "__main__":
    main()
