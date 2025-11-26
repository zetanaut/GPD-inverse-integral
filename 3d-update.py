#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot 3D GPD surface H^{(+)}(x0, xi, t, Q0^2) from the PyTorch GPD ensemble
trained by GPD-H_pt.py.

Outputs (under OUT_DIR):
  - H_surface3d_x{X0}_Q2{Q20}.png   (3D surface of mean H)
  - H_std_heatmap_x{X0}_Q2{Q20}.png (2D heatmap of std(H))
  - coverage_xi_t_x{X0}.png         (xi–t coverage + ERBL line)
  - surface_summary.json
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


# =====================
#  User settings
# =====================
OUT_DIR    = "gpd_inverse_from_reH"              # must match GPD-H_pt.py
MODELS_DIR = os.path.join(OUT_DIR, "gpd_models")
SMOOTH_CSV = os.path.join(OUT_DIR, "ReH_smooth_for_inverse.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "gpd_inverse_summary.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X0        = 0.20      # fixed x
Q20_MODE  = "median"  # "median" or float (e.g. 4.0)
N_XI      = 45        # grid points in xi
N_T       = 45        # grid points in t
CLIP_Z    = None      # e.g. (-0.5, 0.5) to clip mean H; None = no clip


# =====================
#  GPD base net (must match training)
# =====================
class GPDBaseNet(nn.Module):
    """Same architecture as in GPD-H_pt.py."""
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


# =====================
#  Loading models & meta
# =====================
def load_gpd_ensemble(models_dir: str, summary_json: str, device: torch.device):
    if not os.path.isfile(summary_json):
        raise FileNotFoundError(f"Missing summary JSON: {summary_json}")
    with open(summary_json, "r") as f:
        summary = json.load(f)

    hidden = int(summary["gpd_hidden"])
    layers = int(summary["gpd_layers"])
    beta   = float(summary["beta"])

    model_files = sorted(glob.glob(os.path.join(models_dir, "gpd_rep_*.pt")))
    if not model_files:
        raise FileNotFoundError(f"No gpd_rep_*.pt found in {models_dir}")

    models = []
    metas  = []

    for path in model_files:
        ckpt = torch.load(path, map_location=device)
        # state dict + normalization params
        base = GPDBaseNet(in_dim=4, hidden=hidden, layers=layers).to(device)
        base.load_state_dict(ckpt["state_dict"])
        base.eval()
        models.append(base)

        metas.append({
            "xi_mean": float(ckpt["xi_mean"]),
            "xi_std":  float(ckpt["xi_std"]),
            "t_mean":  float(ckpt["t_mean"]),
            "t_std":   float(ckpt["t_std"]),
            "Q2_mean": float(ckpt["Q2_mean"]),
        })

    print(f"[plot] loaded {len(models)} GPD replicas from {models_dir}")
    return models, metas, beta, summary


# =====================
#  H evaluator
# =====================
def eval_H_single_model(
    model: GPDBaseNet,
    meta: dict,
    beta: float,
    x0: float,
    XI: np.ndarray,
    TT: np.ndarray,
    Q20: float,
    device: torch.device
) -> np.ndarray:
    """
    Evaluate H^{(+)}(x0, xi, t, Q20) for all points in the (XI,TT) grid
    for a single replica model.
    """
    xi_mean, xi_std = meta["xi_mean"], meta["xi_std"]
    t_mean, t_std   = meta["t_mean"], meta["t_std"]
    Q2_mean         = meta["Q2_mean"]

    xi_flat = XI.reshape(-1).astype(np.float32)
    t_flat  = TT.reshape(-1).astype(np.float32)
    q2_flat = np.full_like(xi_flat, Q20, dtype=np.float32)
    x_flat  = np.full_like(xi_flat, x0, dtype=np.float32)

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


# =====================
#  Main plotting
# =====================
def main():
    device = torch.device(DEVICE)
    print("[plot] using device:", device)

    os.makedirs(OUT_DIR, exist_ok=True)

    models, metas, beta, summary = load_gpd_ensemble(MODELS_DIR, SUMMARY_JSON, device)
    nrep = len(models)

    # Load coverage (xi,t,Q2)
    if not os.path.isfile(SMOOTH_CSV):
        raise FileNotFoundError(f"Smooth CFF CSV not found: {SMOOTH_CSV}")
    df = pd.read_csv(SMOOTH_CSV)
    if not {"xi", "t", "Q2"}.issubset(df.columns):
        raise ValueError(f"{SMOOTH_CSV} must have columns 'xi','t','Q2'")

    xi_data = df["xi"].values
    t_data  = df["t"].values
    Q2_data = df["Q2"].values

    # Choose Q0^2
    if isinstance(Q20_MODE, (int, float)):
        Q20 = float(Q20_MODE)
    else:
        Q20 = float(np.median(Q2_data))

    # Build (xi,t) grid from coverage range
    xi_min, xi_max = float(np.min(xi_data)), float(np.max(xi_data))
    t_min,  t_max  = float(np.min(t_data)),  float(np.max(t_data))
    XI, TT = np.meshgrid(
        np.linspace(xi_min, xi_max, N_XI),
        np.linspace(t_min,  t_max,  N_T)
    )

    # Evaluate ensemble
    H_stack = []
    for m, meta in zip(models, metas):
        Hm = eval_H_single_model(m, meta, beta, X0, XI, TT, Q20, device)
        H_stack.append(Hm)
    H_stack = np.stack(H_stack, axis=0)   # [rep, Nt, Nxi]
    H_mean  = H_stack.mean(axis=0)
    H_std   = H_stack.std(axis=0)

    if CLIP_Z is not None:
        lo, hi = CLIP_Z
        H_mean = np.clip(H_mean, lo, hi)

    # ---------- 3D surface of mean H ----------
    fig = plt.figure(figsize=(8.0, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XI, TT, H_mean, cmap=plt.get_cmap(), linewidth=0, antialiased=True)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$t$ [GeV$^2$]")
    ax.set_zlabel(r"$H^{(+)}(x_0,\xi,t,Q_0^2)$")
    ax.set_title(
        r"H$^{+}$ surface at "
        + rf"$x_0={X0:.3f}$, "
        + rf"$Q_0^2={Q20:.2f}$ GeV$^2$ "
        + rf"(Nrep={nrep})"
    )
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label=r"$H^{(+)}$ (mean)")
    fig.tight_layout()
    surf_path = os.path.join(OUT_DIR, f"H_surface3d_x{X0:.3f}_Q2{Q20:.2f}.png")
    fig.savefig(surf_path, dpi=150)
    plt.close(fig)

    # ---------- 2D heatmap of std(H) ----------
    plt.figure(figsize=(6.8, 4.8))
    im = plt.imshow(
        H_std,
        origin="lower",
        extent=[XI.min(), XI.max(), TT.min(), TT.max()],
        aspect="auto"
    )
    plt.colorbar(im, label=r"std[$H^{(+)}$]")
    plt.axvline(abs(X0), linestyle="--", linewidth=1)
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$t$ [GeV$^2$]")
    plt.title(r"Uncertainty: std($H^{(+)}$) at fixed $x_0$, $Q_0^2$")
    plt.tight_layout()
    std_path = os.path.join(OUT_DIR, f"H_std_heatmap_x{X0:.3f}_Q2{Q20:.2f}.png")
    plt.savefig(std_path, dpi=150)
    plt.close()

    # ---------- Coverage plot ----------
    plt.figure(figsize=(6.4, 4.6))
    plt.scatter(xi_data, t_data, s=10, alpha=0.65)
    plt.axvline(abs(X0), linestyle="--", linewidth=1)
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$t$ [GeV$^2$]")
    plt.title(r"Kinematic coverage; dashed: $\xi=|x_0|$")
    plt.tight_layout()
    cov_path = os.path.join(OUT_DIR, f"coverage_xi_t_x{X0:.3f}.png")
    plt.savefig(cov_path, dpi=150)
    plt.close()

    # ---------- Summary JSON ----------
    summary_out = {
        "out_dir": OUT_DIR,
        "models_dir": MODELS_DIR,
        "smooth_cff_csv": SMOOTH_CSV,
        "x0": float(X0),
        "Q20": float(Q20),
        "n_replicas": int(nrep),
        "xi_range": [float(xi_min), float(xi_max)],
        "t_range":  [float(t_min),  float(t_max)],
        "grid": {"N_XI": int(N_XI), "N_T": int(N_T)},
        "figures": {
            "surface_mean": surf_path,
            "std_heatmap":  std_path,
            "coverage":     cov_path,
        }
    }
    with open(os.path.join(OUT_DIR, "surface_summary.json"), "w") as f:
        json.dump(summary_out, f, indent=2)

    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
