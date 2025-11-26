#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D GPD surface H^{(+)}(x0, xi, t, Q0^2) with translucent ±1σ sheets.

Mean: colored by H^{(+)} with a colormap.
Band: two single-shade translucent grey sheets at mean ± 1σ.

Inputs (from GPD-H_pt.py):
  - gpd_inverse_from_ReH/gpd_inverse_summary.json
  - gpd_inverse_from_ReH/gpd_models/gpd_rep_*.pt
  - gpd_inverse_from_ReH/ReH_smooth_for_inverse.csv
"""

import os
import glob
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap


# ----------------- User settings -----------------
OUT_DIR      = "gpd_inverse_from_ReH"
MODELS_DIR   = os.path.join(OUT_DIR, "gpd_models")
SMOOTH_CSV   = os.path.join(OUT_DIR, "ReH_smooth_for_inverse.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "gpd_inverse_summary.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Override slice here (optional); if None, use slice from summary JSON
X0_OVERRIDE  = None      # e.g. 0.20
Q20_OVERRIDE = None      # e.g. 2.12

# Resolution of xi–t grid for plotting
N_XI = 60
N_T  = 60

ALPHA_BAND = 0.25      # transparency for ±1σ sheets
ELEV_INIT  = 25.0
AZIM_INIT  = -60.0

# Must match your GPD inversion architecture
GPD_HIDDEN = 64
GPD_LAYERS = 2


# ----------------- Model definition -----------------
class GPDBaseNet(nn.Module):
    """Same architecture as used in GPD-H_pt.py."""
    def __init__(self, in_dim: int = 4, hidden: int = GPD_HIDDEN, layers: int = GPD_LAYERS):
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


def load_gpd_ensemble(models_dir: str, summary_json: str, device: torch.device):
    """Load all gpd_rep_*.pt and their normalizations."""
    if not os.path.isfile(summary_json):
        raise FileNotFoundError(f"Missing summary JSON: {summary_json}")
    with open(summary_json, "r") as f:
        summary = json.load(f)

    model_files = sorted(glob.glob(os.path.join(models_dir, "gpd_rep_*.pt")))
    if not model_files:
        raise FileNotFoundError(f"No gpd_rep_*.pt found in {models_dir}")

    models = []
    metas  = []
    for path in model_files:
        ckpt = torch.load(path, map_location=device)
        base = GPDBaseNet(in_dim=4, hidden=GPD_HIDDEN, layers=GPD_LAYERS).to(device)
        base.load_state_dict(ckpt["state_dict"])
        base.eval()
        models.append(base)

        metas.append({
            "xi_mean": float(ckpt["xi_mean"]),
            "xi_std":  float(ckpt["xi_std"]),
            "t_mean":  float(ckpt["t_mean"]),
            "t_std":   float(ckpt["t_std"]),
            "Q2_mean": float(ckpt["Q2_mean"]),
            "beta":    float(ckpt.get("beta", 2.5)),
        })

    print(f"[plot] loaded {len(models)} GPD replicas from {models_dir}")
    return models, metas, summary


def eval_H_surface_single_model(
    model: GPDBaseNet,
    meta: dict,
    x0: float,
    XI: np.ndarray,
    TT: np.ndarray,
    Q20: float,
    device: torch.device,
) -> np.ndarray:
    """Evaluate H^{(+)}(x0, xi, t, Q20) on the (XI,TT) grid for one replica."""
    xi_mean, xi_std = meta["xi_mean"], meta["xi_std"]
    t_mean,  t_std  = meta["t_mean"],  meta["t_std"]
    Q2_mean         = meta["Q2_mean"]
    beta            = meta["beta"]

    xi_flat = XI.reshape(-1).astype(np.float32)
    t_flat  = TT.reshape(-1).astype(np.float32)
    q2_flat = np.full_like(xi_flat, Q20, dtype=np.float32)
    x_flat  = np.full_like(xi_flat, x0,  dtype=np.float32)

    x_t  = torch.from_numpy(x_flat).to(device)
    xi_t = torch.from_numpy(xi_flat).to(device)
    t_t  = torch.from_numpy(t_flat).to(device)
    Q2_t = torch.from_numpy(q2_flat).to(device)

    # Normalize kinematics
    xi_n = (xi_t - xi_mean) / max(xi_std, 1e-6)
    t_n  = (t_t  - t_mean)  / max(t_std, 1e-6)
    qn   = (torch.log(Q2_t) - torch.log(torch.tensor(Q2_mean + 1e-12, device=device))) / 0.7

    feats_p = torch.stack([ x_t,  xi_n, t_n, qn], dim=1)
    feats_m = torch.stack([-x_t,  xi_n, t_n, qn], dim=1)

    with torch.no_grad():
        out_p = model(feats_p)
        out_m = model(feats_m)

    Hplus = 0.5 * (out_p - out_m)
    env   = torch.clamp(1.0 - x_t.abs(), min=0.0) ** beta
    H     = env * Hplus

    return H.detach().cpu().numpy().reshape(XI.shape)


# ----------------- Main -----------------
def main():
    device = torch.device(DEVICE)
    print("[plot] using device:", device)

    models, metas, summary = load_gpd_ensemble(MODELS_DIR, SUMMARY_JSON, device)
    nrep = len(models)

    # Choose slice (x0, Q0^2)
    slice_info = summary.get("slice", {})
    x0_val  = float(X0_OVERRIDE if X0_OVERRIDE is not None else slice_info.get("xi0", 0.2))
    Q20_val = float(Q20_OVERRIDE if Q20_OVERRIDE is not None else slice_info.get("Q20", slice_info.get("Q2", 2.0)))

    print(f"[plot] using slice x0 = {x0_val:.3f}, Q0^2 = {Q20_val:.2f} GeV^2")

    # xi,t coverage from smooth CFF grid
    if not os.path.isfile(SMOOTH_CSV):
        raise FileNotFoundError(f"Smooth CFF CSV not found: {SMOOTH_CSV}")
    df = pd.read_csv(SMOOTH_CSV)
    if not {"xi", "t"}.issubset(df.columns):
        raise ValueError(f"{SMOOTH_CSV} must contain 'xi' and 't' columns")

    xi_data = df["xi"].values
    t_data  = df["t"].values
    xi_min, xi_max = float(xi_data.min()), float(xi_data.max())
    t_min,  t_max  = float(t_data.min()),  float(t_data.max())

    XI, TT = np.meshgrid(
        np.linspace(xi_min, xi_max, N_XI),
        np.linspace(t_min,  t_max,  N_T),
        indexing="xy"
    )

    # Evaluate ensemble on the grid
    H_stack = []
    for model, meta in zip(models, metas):
        Hk = eval_H_surface_single_model(model, meta, x0_val, XI, TT, Q20_val, device)
        H_stack.append(Hk)
    H_stack = np.stack(H_stack, axis=0)  # [nrep, Nt, Nxi]
    H_mean  = H_stack.mean(axis=0)
    H_std   = H_stack.std(axis=0)

    print(f"[plot] H_mean range: [{H_mean.min():.4f}, {H_mean.max():.4f}]")

    # ---------- 3D plot ----------
    fig = plt.figure(figsize=(8.0, 6.0))
    ax  = fig.add_subplot(111, projection="3d")

    # Colormap & normalization based on physical H values
    cmap = get_cmap("viridis")
    norm = Normalize(vmin=float(H_mean.min()), vmax=float(H_mean.max()))
    facecolors = cmap(norm(H_mean))

    # Mean surface with explicit facecolors (shade=False avoids extra shading)
    surf_mean = ax.plot_surface(
        XI, TT, H_mean,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    # ±1σ sheets: fixed grey color, translucent
    ax.plot_surface(
        XI, TT, H_mean + H_std,
        color="0.7",
        alpha=ALPHA_BAND,
        linewidth=0,
        antialiased=False,
    )
    ax.plot_surface(
        XI, TT, H_mean - H_std,
        color="0.7",
        alpha=ALPHA_BAND,
        linewidth=0,
        antialiased=False,
    )

    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$t$ [GeV$^2$]")
    ax.set_zlabel(r"$H^{(+)}(x_0,\xi,t,Q_0^2)$")

    title = (
        r"H$^{+}$ surface with $\pm 1\sigma$ band at "
        + rf"$x_0={x0_val:.3f}$, $Q_0^2={Q20_val:.2f}$ GeV$^2$ "
        + rf"(Nrep={nrep})"
    )
    ax.set_title(title)

    # Good starting view; you can rotate with the mouse
    ax.view_init(elev=ELEV_INIT, azim=AZIM_INIT)

    # Colorbar based on the same norm & cmap (in physical H units)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])  # needed for colorbar
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label(r"$H^{(+)}$ (mean)")

    fig.tight_layout()

    out_png = os.path.join(
        OUT_DIR,
        f"H_surface_band_simple_x{float(x0_val):.3f}_Q2{float(Q20_val):.2f}.png"
    )
    fig.savefig(out_png, dpi=150)
    print("[plot] saved:", out_png)

    plt.show()


if __name__ == "__main__":
    main()
