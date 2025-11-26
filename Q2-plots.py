#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D GPD surfaces H^{(+)}(x0, xi, t, Q^2) at multiple fixed Q^2 values
(on the same axes, no error band).

Uses the GPD ensemble from:
  gpd_inverse_from_ReH/gpd_models/gpd_rep_*.pt
and the kinematic grid from:
  gpd_inverse_from_ReH/ReH_smooth_for_inverse.csv
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
from matplotlib.cm import get_cmap, ScalarMappable


# ----------------- User settings -----------------
OUT_DIR      = "gpd_inverse_from_ReH"
MODELS_DIR   = os.path.join(OUT_DIR, "gpd_models")
SMOOTH_CSV   = os.path.join(OUT_DIR, "ReH_smooth_for_inverse.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "gpd_inverse_summary.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed x0 for all surfaces (if None, use xi0 from summary JSON)
X0_OVERRIDE = None   # e.g. 0.20

# Q^2 values (GeV^2) to plot
Q2_LIST = [1.0, 4.0, 8.0]

# Resolution of xi–t grid
N_XI = 60
N_T  = 60

SAVE_FIG = True
SHOW_INTERACTIVE = True

# Must match GPD-H_pt.py architecture
GPD_HIDDEN = 64
GPD_LAYERS = 2


# ----------------- Model definition -----------------
class GPDBaseNet(nn.Module):
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
    """Load all gpd_rep_*.pt + meta info + summary."""
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
        net  = GPDBaseNet(in_dim=4, hidden=GPD_HIDDEN, layers=GPD_LAYERS).to(device)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        models.append(net)
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


def main():
    device = torch.device(DEVICE)
    print("[plot] using device:", device)

    models, metas, summary = load_gpd_ensemble(MODELS_DIR, SUMMARY_JSON, device)
    nrep = len(models)

    # Slice: x0 from override or summary
    slice_info = summary.get("slice", {})
    x0_val = float(X0_OVERRIDE if X0_OVERRIDE is not None else slice_info.get("xi0", 0.2))

    # xi,t coverage from CFF grid
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

    # Evaluate ensemble-mean H for each Q^2 in Q2_LIST
    H_means = []
    for Q20 in Q2_LIST:
        print(f"[plot] evaluating ensemble mean at Q^2 = {Q20:.2f} GeV^2")
        H_stack = []
        for model, meta in zip(models, metas):
            Hk = eval_H_surface_single_model(model, meta, x0_val, XI, TT, Q20, device)
            H_stack.append(Hk)
        H_stack = np.stack(H_stack, axis=0)
        H_mean_Q = H_stack.mean(axis=0)
        H_means.append(H_mean_Q)

    H_means = np.stack(H_means, axis=0)  # [nQ, Nt, Nxi]
    H_min = float(H_means.min())
    H_max = float(H_means.max())
    print(f"[plot] H_mean global range over Q^2-list: [{H_min:.4g}, {H_max:.4g}]")

    # Colormap & normalization shared across all surfaces
    cmap = get_cmap("viridis")
    norm = Normalize(vmin=H_min, vmax=H_max)

    # ---------- Plot ----------
    fig = plt.figure(figsize=(8.0, 6.0))
    ax  = fig.add_subplot(111, projection="3d")

    # Draw each Q^2 surface
    for i, Q20 in enumerate(Q2_LIST):
        H_mean_Q = H_means[i]
        colors = cmap(norm(H_mean_Q))

        surf = ax.plot_surface(
            XI, TT, H_mean_Q,
            facecolors=colors,
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=0.9
        )

        # Label near one corner of the surface
        xi_label = xi_max
        t_label  = t_min
        z_label  = float(H_mean_Q[0, -1])
        ax.text(
            xi_label, t_label, z_label,
            rf"$Q^2 = {Q20:.1f}\,\mathrm{{GeV}}^2$",
            fontsize=9, ha="right", va="bottom"
        )

    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$t$ [GeV$^2$]")
    ax.set_zlabel(r"$H^{(+)}(x_0,\xi,t,Q^2)$")

    title = (
        r"H$^{+}$ surfaces at "
        + rf"$x_0={x0_val:.3f}$, "
        + r"$Q^2 = 1, 4, 8\ \mathrm{GeV}^2$ "
        + rf"(ensemble mean, Nrep={nrep})"
    )
    ax.set_title(title)

    # Colorbar for H_mean (shared)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label(r"$H^{(+)}$ (mean)")

    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()

    if SAVE_FIG:
        out_png = os.path.join(
            OUT_DIR,
            f"H_surface_multiQ_x{x0_val:.3f}_Q2s_{'_'.join(str(int(q)) for q in Q2_LIST)}.png"
        )
        fig.savefig(out_png, dpi=150)
        print("Saved:", out_png)

    if SHOW_INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
