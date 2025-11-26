#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPD-H_pt.py  (heavily commented / teaching version)

Goal
----
Given:
  • A *global* neural-network fit to the CFF ReH(ξ, t, Q²), trained on
    experimental data with realistic errors (the "ReH ensemble" in reh_model_v3).

We want to:
  • Solve the inverse problem:
        ReH(ξ, t, Q²)  =  PV ∫ dx  K(ξ, x) H^{(+)}(x, ξ, t, Q²)
    where K is the usual DVCS kernel,
    and H^{(+)} is the charge-weighted C-even GPD combination.

Strategy
--------
Two-stage approach:

  1) **CFF stage (forward)**:
     - Load the ReH ensemble from reh_model_v3/.
       Each member was trained on a Monte Carlo replica of the experimental
       data, so together they represent the propagated *experimental* uncertainty.
     - Build a *fine* regular grid in (ξ, t, Q²) inside the region where
       the ReH model is trained (using ranges from training_summary.json).
     - Evaluate on that grid:
         • the ensemble mean and predictive std of ReH (for diagnostics
           and weights in χ²),
         • the mean prediction of each individual ReH model (each model
           is then treated as one correlated CFF replica).

  2) **GPD stage (inverse)**:
     - For each ReH ensemble member k, we train one GPD replica
       H^{(+)}_k(x, ξ, t, Q²) so that its PV convolution reproduces
       that model's ReH field on the grid.
     - Importantly: we do *not* add extra Gaussian noise in this stage.
       The only source of randomness is the ReH ensemble, which already
       encodes the experimental errors from the original CFF CSV.
     - The collection of GPD replicas {H^{(+)}_k} then represents the
       propagated uncertainty on the GPD.

Outputs (in OUT_DIR = "gpd_inverse_from_ReH")
--------------------------------------------
  • ReH_smooth_for_inverse.csv
      - The (ξ, t, Q²) grid with ensemble mean & std for ReH.
      - Useful for plotting and diagnostics.

  • gpd_models/gpd_rep_*.pt
      - PyTorch checkpoints of the GPD replicas (one per ReH model used).

  • H_band_from_ReHfit.npz
      - x-grid and H-band (mean ±1σ) at a representative slice
        (ξ₀, t₀, Q₀²), chosen from the middle of the grid.

  • H_band_from_ReHfit.png
      - A quick visualization of that 1D H^{(+)} band.

  • gpd_inverse_summary.json
      - Metadata: hyperparameters, grid ranges, slice used, etc.
"""

import os
import glob
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# 1. Global configuration
# ============================================================

# Path to the *CFF* (ReH) global fit directory.
# This should match the OUT_DIR of the separate CFF training script.
REH_MODEL_DIR = "reh_model_v3"

# Where we store all outputs of the GPD inversion stage.
OUT_DIR       = "gpd_inverse_from_ReH"

# Use GPU if available for speed; else fall back to CPU.
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 123  # global random seed for reproducibility

# How dense the ReH grid is in (ξ, t, Q²).
# Higher numbers → smoother approximation of the integral, but more compute.
N_XI_GRID  = 50
N_T_GRID   = 50
N_Q2_GRID  = 5

# How many GPD replicas to train.
# MAX_GPD_REPS = None  → use all ReH models in reh_model_v3/models.
MAX_GPD_REPS = None

# Hyperparameters for the GPD inversion model (stage 2).
NX               = 181      # number of x points in [-1, 1]
BETA             = 2.5      # exponent in endpoint envelope (1 - |x|)^β
PV_EPS_MULT      = 2.0      # how wide to cut out around singularities in PV integral (in units of dx)
GPD_HIDDEN       = 64       # MLP width
GPD_LAYERS       = 2        # number of hidden layers
GPD_LR           = 2e-2     # learning rate for Adam
GPD_WEIGHT_DECAY = 0.0      # L2 penalty on weights (set to 0; we use curvature reg instead)
GPD_EPOCHS       = 120      # epochs per GPD replica
GPD_BATCH_SIZE   = 64
GPD_LAM_X        = 1e-3     # strength of curvature regularization in x
PRINT_EVERY      = 10       # how often to print loss for monitoring


# ============================================================
# 2. Small utilities
# ============================================================

def set_seed(seed: int = 123):
    """Set random seeds for NumPy and PyTorch for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # These make cuDNN deterministic (slower but reproducible).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Standardizer:
    """
    Keep track of feature mean and std to normalize inputs.

    The ReH model was trained on standardized kinematic variables.
    We must apply the *same* normalization here to get consistent predictions.
    """
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize: (x - mean) / std."""
        return (x - self.mean) / self.std.clamp_min(1e-12)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform: x * std + mean."""
        return x * self.std.clamp_min(1e-12) + self.mean


# ============================================================
# 3. ReH ensemble model (CFF stage)
# ============================================================

class MLPHet(nn.Module):
    """
    Heteroscedastic MLP for the CFF fit: predicts both mean and log σ.

    - Input: standardized kinematics (ξ, Q², t, [optional extra variable k]).
    - Output:
        mu     : mean ReH in normalized label space
        logsig : log of model standard deviation (aleatoric uncertainty)

    Aleatoric here means: uncertainty intrinsic to the data (e.g. experimental
    errors), as opposed to epistemic (spread between ensemble members).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 num_layers: int = 5, dropout: float = 0.05):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head_mu = nn.Linear(hidden_dim, 1)
        self.head_logsig = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        mu   : shape [batch], mean ReH (normalized units)
        logs : shape [batch], log σ_model (normalized units)
        """
        h = self.backbone(x)
        mu = self.head_mu(h).squeeze(-1)
        logs = self.head_logsig(h).squeeze(-1)
        return mu, logs


def load_reH_ensemble(model_dir: str, device: torch.device):
    """
    Load the ReH ensemble and its normalization info.

    Parameters
    ----------
    model_dir : str
        Path to reh_model_v3 (or equivalent).
    device    : torch.device

    Returns
    -------
    models  : list of MLPHet
        Each model corresponds to one Monte Carlo replica of the experimental
        data; together they approximate the posterior over ReH.
    scaler  : Standardizer
        Input normalization used during CFF training.
    y_scale : float
        Overall label scaling used to normalize ReH during CFF training.
    support : dict
        Ranges of ξ, t, Q² used in the global fit.
    summary : dict
        Full training summary (for documentation / calibration factor).
    """
    # Input normalization
    scaler_path = os.path.join(model_dir, "scaler.pt")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Could not find scaler.pt in {model_dir}")
    scaler_dict = torch.load(scaler_path, map_location=device)
    mean = scaler_dict["mean"].to(device)
    std  = scaler_dict["std"].to(device)
    y_scale = float(scaler_dict["y_scale"])  # used later to un-normalize ReH
    scaler = Standardizer(mean=mean, std=std)

    # Training summary: contains architecture + support ranges + calibration
    summary_path = os.path.join(model_dir, "training_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Could not find training_summary.json in {model_dir}")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    mconf   = summary["model"]
    support = summary["support"]

    in_dim  = int(mconf["in_dim"])
    hidden  = int(mconf["hidden"])
    layers  = int(mconf["layers"])
    dropout = float(mconf["dropout"])
    include_k = bool(support.get("include_k", False))

    # Ensemble members: model_000.pt, model_001.pt, ...
    model_files = sorted(glob.glob(os.path.join(model_dir, "models", "model_*.pt")))
    # If no explicit ensemble was saved, fall back to the base model.
    if not model_files:
        base_path = os.path.join(model_dir, "models", "model_base.pt")
        if not os.path.isfile(base_path):
            raise FileNotFoundError(f"No ensemble model_*.pt or model_base.pt in {model_dir}/models")
        model_files = [base_path]

    models: List[MLPHet] = []
    for path in model_files:
        m = MLPHet(in_dim=in_dim, hidden_dim=hidden, num_layers=layers, dropout=dropout).to(device)
        state = torch.load(path, map_location=device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    print(f"[ReH ensemble] Loaded {len(models)} models from {model_dir} (include_k={include_k})")
    return models, scaler, y_scale, support, summary


def eval_reH_ensemble(models: List[MLPHet],
                      X_std: torch.Tensor,
                      y_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the ReH ensemble on a standardized feature grid.

    We combine:
      - Epistemic variance: spread of ensemble means μ_k across models.
      - Aleatoric variance: each model's predicted σ_model² (from logs).

    Total predictive variance at each point:
        Var_pred = Var_epistemic + E[ Var_aleatoric ].

    Parameters
    ----------
    models   : list of MLPHet
    X_std    : tensor [N, in_dim], standardized features
    y_scale  : float, label scale factor (to go back to physical units)

    Returns
    -------
    mean : np.ndarray [N]
        Ensemble mean ReH in physical units.
    std  : np.ndarray [N]
        Predictive standard deviation in physical units.
    """
    mu_list, var_list = [], []
    with torch.no_grad():
        for m in models:
            mu, logs = m(X_std)
            mu_list.append(mu.detach().cpu().numpy())
            var_list.append(np.exp(2.0 * logs.detach().cpu().numpy()))
    MU = np.stack(mu_list, axis=0)  # shape (E, N)
    VA = np.stack(var_list, axis=0)

    mean_n     = MU.mean(axis=0)
    var_epist  = MU.var(axis=0)     # spread between models
    var_alea   = VA.mean(axis=0)    # mean σ_model²
    var_pred_n = var_epist + var_alea

    mean = mean_n * y_scale
    std  = np.sqrt(np.clip(var_pred_n, 1e-12, None)) * y_scale
    return mean, std


def eval_reH_single_model(model: MLPHet,
                          X_std: torch.Tensor,
                          y_scale: float) -> np.ndarray:
    """
    Evaluate a *single* ReH model on the standardized grid.

    We only use the mean μ here, ignoring σ_model, because each model is
    already a sample from the CFF posterior (data replica). The ensemble
    spread across models then encodes the uncertainty.
    """
    with torch.no_grad():
        mu, _ = model(X_std)
        mu_np = mu.detach().cpu().numpy() * y_scale
    return mu_np


# ============================================================
# 4. Build ReH grid & evaluate ensemble (Stage 1)
# ============================================================

def build_reH_grid_and_evaluate(models: List[MLPHet],
                                scaler: Standardizer,
                                y_scale: float,
                                support: dict,
                                summary: dict,
                                device: torch.device):
    """
    Construct a regular grid in (ξ, t, Q²) within the support of the CFF fits
    and evaluate the ReH ensemble on that grid.

    Why do we need a grid?
    ----------------------
    The inversion formula relates ReH at *all* ξ for a given GPD H(x, ξ, t, Q²).
    Rather than only using the original experimental kinematics (which may be
    sparse or irregular), we exploit the smooth global CFF model to populate
    a dense grid in (ξ, t, Q²). This makes the inverse problem numerically
    better behaved and allows us to evaluate the PV integral consistently.

    Returns
    -------
    xi_flat, t_flat, Q2_flat : 1D arrays with the grid points
    X_std_grid               : standardized feature tensor
    ReH_mean                 : ensemble mean ReH on the grid
    sigma_eff                : predictive std (possibly calibrated)
    grid_meta                : dict with ranges and grid sizes
    """
    xi_min, xi_max = float(support["xi_min"]), float(support["xi_max"])
    q2_min, q2_max = float(support["q2_min"]), float(support["q2_max"])
    t_min,  t_max  = float(support["t_min"]),  float(support["t_max"])
    include_k      = bool(support.get("include_k", False))

    print("[ReH grid] xi in [%.4f, %.4f]" % (xi_min, xi_max))
    print("[ReH grid] Q2 in [%.4f, %.4f]" % (q2_min, q2_max))
    print("[ReH grid] t  in [%.4f, %.4f]" % (t_min,  t_max))

    xi_grid = np.linspace(xi_min, xi_max, N_XI_GRID, dtype=np.float32)
    q2_grid = np.linspace(q2_min, q2_max, N_Q2_GRID, dtype=np.float32)
    t_grid  = np.linspace(t_min,  t_max,  N_T_GRID,  dtype=np.float32)

    # Make a full 3D Cartesian product grid and then flatten.
    # XI, Q2, TT each has shape (N_xi, N_q2, N_t).
    XI, Q2, TT = np.meshgrid(xi_grid, q2_grid, t_grid, indexing="ij")
    xi_flat = XI.ravel()
    Q2_flat = Q2.ravel()
    t_flat  = TT.ravel()
    N = xi_flat.size

    # Assemble feature vectors. If the CFF model used an additional kinematic
    # variable "k" (e.g. beam energy), we fix it to a reference value here.
    if include_k:
        # Here we simply set k = 0 for all points; since the model was trained
        # on a range of k, this corresponds to probing a particular beam energy.
        k_flat = np.zeros_like(xi_flat, dtype=np.float32)
        feats  = np.stack([xi_flat, Q2_flat, t_flat, k_flat], axis=1)
    else:
        feats  = np.stack([xi_flat, Q2_flat, t_flat], axis=1)

    X_grid     = torch.from_numpy(feats.astype(np.float32)).to(device)
    X_std_grid = scaler.transform(X_grid)

    # Evaluate ensemble statistics on the grid
    ReH_mean, ReH_std = eval_reH_ensemble(models, X_std_grid, y_scale)

    # Optionally apply a calibration factor τ that was extracted in the
    # ReH training (so that 1σ intervals have ≈68% coverage on validation).
    tau = 1.0
    if "calibration" in summary and "tau" in summary["calibration"]:
        tau = float(summary["calibration"]["tau"])
        print(f"[ReH grid] applying calibration factor tau = {tau:.3f} to ReH std")
    sigma_eff = np.maximum(ReH_std * tau, 1e-3)

    # Save grid + ensemble mean/std for diagnostics and plotting.
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame({
        "xi":   xi_flat.astype(np.float64),
        "Q2":   Q2_flat.astype(np.float64),
        "t":    t_flat.astype(np.float64),
        "ReH_smooth":   ReH_mean.astype(np.float64),
        "sigma_smooth": sigma_eff.astype(np.float64),
    })
    df.to_csv(os.path.join(OUT_DIR, "ReH_smooth_for_inverse.csv"), index=False)

    grid_meta = {
        "xi_min": xi_min, "xi_max": xi_max,
        "q2_min": q2_min, "q2_max": q2_max,
        "t_min":  t_min,  "t_max":  t_max,
        "N_xi_grid": N_XI_GRID,
        "N_t_grid":  N_T_GRID,
        "N_q2_grid": N_Q2_GRID,
        "N_points":  int(N),
        "include_k": include_k,
    }

    return xi_flat, t_flat, Q2_flat, X_std_grid, ReH_mean, sigma_eff, grid_meta


# ============================================================
# 5. GPD model + principal-value (PV) operator
# ============================================================

class GPDBaseNet(nn.Module):
    """
    Neural parametrization of the GPD *shape* before symmetry and envelope.

    Input  : (x, ξ_normalized, t_normalized, logQ²_normalized)
    Output : scalar "raw" function f(x, ξ, t, Q²).

    The final GPD used in physics is:
        H^{(+)}(x, ξ, t, Q²) = (1 - |x|)^β * [f(x) - f(-x)] / 2
    i.e. we impose:
      - exact oddness under x ↦ −x via anti-symmetrization
      - a known endpoint behavior via the (1 - |x|)^β envelope.
    """
    def __init__(self, in_dim: int = 4, hidden: int = GPD_HIDDEN, layers: int = GPD_LAYERS):
        super().__init__()
        modules: List[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            modules.append(nn.Linear(d, hidden))
            modules.append(nn.Tanh())
            d = hidden
        modules.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PVForwardOperator(nn.Module):
    """
    Wrapper that takes the base GPD network and:

      • builds an x-grid in [-1, 1],
      • enforces C-even oddness & envelope,
      • performs the principal-value (PV) integral to obtain ReH.

    The PV integral for the C-even combination roughly looks like:
        ReH(ξ) ∼ PV ∫_{-1}^{1} dx H^{(+)}(x, ξ, t, Q²)
                                   * [ 1 / (ξ - x) - 1 / (ξ + x) ].

    Numerically we:
      - discretize x ∈ [-1,1] on NX points with trapezoidal weights,
      - remove small windows around ±ξ where the integrand diverges
        (PV_EPS_MULT × Δx),
      - multiply by the kernel and sum.
    """
    def __init__(self,
                 base: GPDBaseNet,
                 xi_mean: float,
                 xi_std: float,
                 t_mean: float,
                 t_std: float,
                 Q2_mean: float,
                 nx: int = NX,
                 beta: float = BETA,
                 pv_eps_mult: float = PV_EPS_MULT,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.base = base
        self.nx = nx

        # Discretize x ∈ [-1,1] with trapezoidal weights for integration.
        xgrid = torch.linspace(-1.0, 1.0, nx, device=device)
        dx = xgrid[1] - xgrid[0]
        w = torch.ones_like(xgrid) * dx
        w[0] *= 0.5
        w[-1] *= 0.5

        # register_buffer = store tensors that are needed but not trained
        self.register_buffer("xgrid", xgrid)
        self.register_buffer("w", w)
        self.dx = dx

        # Envelope exponent and PV exclusion radius (in multiples of dx)
        self.beta = float(beta)
        self.pv_eps_mult = float(pv_eps_mult)

        # Normalization constants for ξ, t, Q² (estimated from the grid)
        self.xi_mean = torch.tensor(xi_mean, device=device, dtype=torch.float32)
        self.xi_std  = torch.tensor(xi_std,  device=device, dtype=torch.float32)
        self.t_mean  = torch.tensor(t_mean,  device=device, dtype=torch.float32)
        self.t_std   = torch.tensor(t_std,   device=device, dtype=torch.float32)
        self.Q2_mean = torch.tensor(Q2_mean, device=device, dtype=torch.float32)

    def Hx_batch(self, xi: torch.Tensor, t: torch.Tensor, Q2: torch.Tensor) -> torch.Tensor:
        """
        Compute H^{(+)}(x, ξ, t, Q²) on the whole x-grid for a batch of
        kinematic points (ξ_i, t_i, Q²_i).

        Returns a tensor of shape [batch_size, nx].
        """
        B = xi.shape[0]
        x = self.xgrid.unsqueeze(0).expand(B, -1)  # shape [B, nx]

        # Normalize ξ, t, Q² using the means/std from the grid.
        xi_n = (xi - self.xi_mean) / self.xi_std.clamp_min(1e-12)
        t_n  = (t  - self.t_mean) / self.t_std.clamp_min(1e-12)
        qn   = (torch.log(Q2) - torch.log(self.Q2_mean + 1e-12)) / 0.7

        # Broadcast ξ_n,t_n,qn across the x dimension.
        xi_n_exp = xi_n.unsqueeze(1).expand_as(x)
        t_n_exp  = t_n.unsqueeze(1).expand_as(x)
        qn_exp   = qn.unsqueeze(1).expand_as(x)

        # Build features for x and -x to enforce oddness.
        feats_p = torch.stack([ x,  xi_n_exp, t_n_exp, qn_exp], dim=-1)
        feats_m = torch.stack([-x,  xi_n_exp, t_n_exp, qn_exp], dim=-1)

        flat_p = feats_p.view(-1, 4)
        flat_m = feats_m.view(-1, 4)

        out_p = self.base(flat_p).view(B, self.nx)
        out_m = self.base(flat_m).view(B, self.nx)

        # C-even combination: odd in x → (f(x) - f(-x))/2
        Hplus_raw = 0.5 * (out_p - out_m)

        # Envelope to enforce reasonable endpoint behavior.
        env   = torch.clamp(1.0 - x.abs(), min=0.0) ** self.beta
        return env * Hplus_raw

    def ReH_pred(self, xi: torch.Tensor, t: torch.Tensor, Q2: torch.Tensor):
        """
        Given ξ, t, Q² arrays (1D), compute:

          • Hx : H^{(+)}(x, ξ_i, t_i, Q²_i) on the x-grid
          • ReH_pred : PV integral yielding ReH(ξ_i, t_i, Q²_i)

        Returns
        -------
        ReH : tensor [batch]
        Hx  : tensor [batch, nx]
        """
        B = xi.shape[0]
        x = self.xgrid.unsqueeze(0).expand(B, -1)
        Hx = self.Hx_batch(xi, t, Q2)

        # Kernel for the C-even combination.
        kernel = 1.0 / (xi.unsqueeze(1) - x + 1e-12) - 1.0 / (xi.unsqueeze(1) + x + 1e-12)

        # Implement principal value by excising a small window around x=±ξ.
        pv_eps = self.pv_eps_mult * self.dx
        mask = ((x - xi.unsqueeze(1)).abs() > pv_eps) & ((x + xi.unsqueeze(1)).abs() > pv_eps)
        mask = mask.to(Hx.dtype)

        integrand = Hx * kernel * mask * self.w.unsqueeze(0)
        ReH = integrand.sum(dim=1)
        return ReH, Hx

    def curvature_reg_x(self, Hx: torch.Tensor) -> torch.Tensor:
        """
        Simple smoothness penalty in x:

          ∫ (∂²H/∂x²)² dx  ≈  mean over grid of (discrete H'')² * Δx

        This discourages rapid oscillations in x that are not supported by
        the CFF data and helps regularize the ill-posed inversion.
        """
        # Second finite difference along x: H_{j+1} - 2H_j + H_{j-1}
        D2 = Hx[:, 2:] - 2.0 * Hx[:, 1:-1] + Hx[:, :-2]  # [B, nx-2]
        Hpp = D2 / (self.dx * self.dx)
        return (Hpp ** 2).mean() * self.dx


# ============================================================
# 6. GPD training loop: one replica per ReH model
# ============================================================

def train_gpd_ensemble_from_reH(
    models: List[MLPHet],
    X_std_grid: torch.Tensor,
    xi_flat: np.ndarray,
    t_flat: np.ndarray,
    Q2_flat: np.ndarray,
    sigma_eff: np.ndarray,
    y_scale: float,
    grid_meta: dict,
    device: torch.device
):
    """
    Main inversion routine.

    For each ReH ensemble member m:
      1) Evaluate its mean ReH on the grid: ReH_vals(ξ_i, t_i, Q²_i).
         (No extra noise: each model is already a CFF replica.)
      2) Train a GPD model H^{(+)}_m(x,ξ,t,Q²) so that:
             ReH_pred(ξ_i,t_i,Q²_i) ≈ ReH_vals_i
         under a χ² loss with weights ~ 1 / σ_eff².
      3) Save the trained GPD as one replica.
      4) Also store H^{(+)}_m(x, ξ₀, t₀, Q₀²) on x-grid for later bands.

    The spread over these replicas encodes the propagated experimental
    uncertainty from the original CFF data.
    """
    # Convert the grid coordinates & σ into tensors.
    xi_t = torch.from_numpy(xi_flat.astype(np.float32)).to(device)
    t_t  = torch.from_numpy(t_flat.astype(np.float32)).to(device)
    Q2_t = torch.from_numpy(Q2_flat.astype(np.float32)).to(device)
    sig_t = torch.from_numpy(sigma_eff.astype(np.float32)).to(device)

    # Means and stds of ξ, t, Q² used for normalization inside the GPD model.
    xi_mean = float(xi_t.mean().item())
    xi_std  = float(xi_t.std().clamp_min(1e-6).item())
    t_mean  = float(t_t.mean().item())
    t_std   = float(t_t.std().clamp_min(1e-6).item())
    Q2_mean = float(Q2_t.mean().item())

    print("[GPD] xi_mean/std =", xi_mean, xi_std)
    print("[GPD] t_mean/std  =", t_mean, t_std)
    print("[GPD] Q2_mean     =", Q2_mean)

    # Choose a "typical" slice (ξ₀, t₀, Q₀²) from the middle of the grid
    # for later visualization of the H-band.
    xi0 = float(0.5 * (grid_meta["xi_min"] + grid_meta["xi_max"]))
    t0  = float(0.5 * (grid_meta["t_min"]  + grid_meta["t_max"]))
    Q20 = float(0.5 * (grid_meta["q2_min"] + grid_meta["q2_max"]))

    xgrid = torch.linspace(-1.0, 1.0, NX, device=device)
    H_slices = []  # will collect H^{(+)}_m(x, ξ₀, t₀, Q₀²) for each replica

    total_models = len(models)
    n_reps = total_models if MAX_GPD_REPS is None else min(total_models, MAX_GPD_REPS)
    print(f"[GPD] training {n_reps} GPD replicas (one per ReH model)")

    for rep in range(n_reps):
        m = models[rep]
        print(f"\n[GPD] ==== Replica {rep:03d}/{n_reps-1:03d} (from ReH model {rep}) ====")

        # 1) CFF replica: evaluate this ReH model on the grid (no extra noise!)
        ReH_vals = eval_reH_single_model(m, X_std_grid, y_scale)
        y_cen = torch.from_numpy(ReH_vals.astype(np.float32)).to(device)

        # Dataset for this replica's inversion:
        ds = TensorDataset(xi_t, t_t, Q2_t, y_cen, sig_t)
        loader = DataLoader(ds, batch_size=GPD_BATCH_SIZE, shuffle=True)

        # Build a fresh GPD MLP and PV operator.
        base = GPDBaseNet(in_dim=4, hidden=GPD_HIDDEN, layers=GPD_LAYERS).to(device)
        op   = PVForwardOperator(
            base,
            xi_mean=xi_mean,
            xi_std=xi_std,
            t_mean=t_mean,
            t_std=t_std,
            Q2_mean=Q2_mean,
            nx=NX,
            beta=BETA,
            pv_eps_mult=PV_EPS_MULT,
            device=device
        )
        opt = torch.optim.Adam(base.parameters(), lr=GPD_LR, weight_decay=GPD_WEIGHT_DECAY)

        # 2) Training loop for this replica.
        for ep in range(1, GPD_EPOCHS + 1):
            base.train()
            total_loss = 0.0
            npts = 0
            for xi_b, t_b, Q2_b, y_b, s_b in loader:
                opt.zero_grad()

                # Forward PV transform: from H^{(+)} → ReH_pred
                y_pred, Hx = op.ReH_pred(xi_b, t_b, Q2_b)

                # χ² loss with pointwise weights w = 1/σ²
                w = 1.0 / (s_b**2 + 1e-12)
                data_loss = torch.mean(w * (y_pred - y_b)**2)

                # Curvature regularization in x to suppress oscillations.
                reg_x = GPD_LAM_X * op.curvature_reg_x(Hx)

                loss = data_loss + reg_x
                loss.backward()
                opt.step()

                total_loss += loss.item() * xi_b.shape[0]
                npts += xi_b.shape[0]

            if ep % PRINT_EVERY == 0 or ep == GPD_EPOCHS:
                print(f"[GPD rep {rep:03d}] epoch {ep:4d}/{GPD_EPOCHS}  "
                      f"loss={total_loss / max(1, npts):.4f}")

        # 3) Evaluate H^{(+)} for this replica at the reference slice (ξ₀, t₀, Q₀²).
        base.eval()
        with torch.no_grad():
            xi0_t = torch.full((1,), float(xi0), dtype=torch.float32, device=device)
            t0_t  = torch.full((1,), float(t0),  dtype=torch.float32, device=device)
            Q20_t = torch.full((1,), float(Q20), dtype=torch.float32, device=device)
            Hx0 = op.Hx_batch(xi0_t, t0_t, Q20_t)[0].detach().cpu().numpy()
            H_slices.append(Hx0)

        # 4) Save this GPD replica for later evaluation / plotting.
        rep_dir = os.path.join(OUT_DIR, "gpd_models")
        os.makedirs(rep_dir, exist_ok=True)
        torch.save(
            {
                "state_dict": base.state_dict(),
                "xi_mean": xi_mean,
                "xi_std": xi_std,
                "t_mean": t_mean,
                "t_std": t_std,
                "Q2_mean": Q2_mean,
                "nx": NX,
                "beta": BETA,
                "pv_eps_mult": PV_EPS_MULT,
            },
            os.path.join(rep_dir, f"gpd_rep_{rep:03d}.pt")
        )

    # --------------------------------------------------------
    # After all replicas: build an H-band at (ξ₀, t₀, Q₀²)
    # --------------------------------------------------------
    H_slices = np.stack(H_slices, axis=0)   # shape [n_reps, NX]
    H_mean = H_slices.mean(axis=0)
    H_std  = H_slices.std(axis=0)
    xgrid_np = xgrid.detach().cpu().numpy()

    np.savez(
        os.path.join(OUT_DIR, "H_band_from_ReHfit.npz"),
        xgrid=xgrid_np,
        H_mean=H_mean,
        H_std=H_std,
        xi0=float(xi0),
        t0=float(t0),
        Q20=float(Q20)
    )

    # Quick 1D visualization of the H-band.
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.6, 4.2))
        plt.fill_between(xgrid_np, H_mean - H_std, H_mean + H_std,
                         alpha=0.3, label="ensemble ±1σ")
        plt.plot(xgrid_np, H_mean, label="mean reconstruction", linewidth=1.8)
        plt.axvline(+abs(xi0), linestyle="--", linewidth=1)
        plt.axvline(-abs(xi0), linestyle="--", linewidth=1)
        plt.xlabel("x")
        plt.ylabel(r"$H^{(+)}(x,\xi_0,t_0,Q_0^2)$")
        plt.title(
            f"GPD H$^+$ band from ReH ensemble at "
            + rf"$\xi_0={xi0:.3f},\, t_0={t0:.3f}$ GeV$^2$, "
            + rf"$Q_0^2={Q20:.2f}$ GeV$^2$ (Nrep={n_reps})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "H_band_from_ReHfit.png"), dpi=150)
        plt.close()
    except ImportError:
        pass

    # Summary JSON for reproducibility and plotting scripts.
    summary = {
        "reh_model_dir": REH_MODEL_DIR,
        "out_dir": OUT_DIR,
        "device": str(device),
        "nx": NX,
        "beta": BETA,
        "pv_eps_mult": PV_EPS_MULT,
        "gpd_hidden": GPD_HIDDEN,
        "gpd_layers": GPD_LAYERS,
        "gpd_epochs": GPD_EPOCHS,
        "gpd_batch": GPD_BATCH_SIZE,
        "gpd_lr": GPD_LR,
        "gpd_weight_decay": GPD_WEIGHT_DECAY,
        "gpd_lam_x": GPD_LAM_X,
        "n_gpd_replicas": int(n_reps),
        "grid": grid_meta,
        "slice": {"xi0": float(xi0), "t0": float(t0), "Q20": float(Q20)},
        "n_points": int(grid_meta["N_points"]),
    }
    with open(os.path.join(OUT_DIR, "gpd_inverse_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[GPD] inversion summary saved to", os.path.join(OUT_DIR, "gpd_inverse_summary.json"))


# ============================================================
# 7. Main entry point
# ============================================================

def main():
    # Set seeds and make sure output directory exists.
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(DEVICE)
    print("[main] using device:", device)

    # ---- Stage 1: load CFF (ReH) global fit and build grid ----
    models, scaler, y_scale, support, summary = load_reH_ensemble(REH_MODEL_DIR, device)

    xi_flat, t_flat, Q2_flat, X_std_grid, ReH_mean, sigma_eff, grid_meta = \
        build_reH_grid_and_evaluate(models, scaler, y_scale, support, summary, device)
    print(f"[ReH grid] total grid points: {grid_meta['N_points']}")

    # ---- Stage 2: invert ReH → H^{(+)} using the ensemble ----
    train_gpd_ensemble_from_reH(
        models=models,
        X_std_grid=X_std_grid,
        xi_flat=xi_flat,
        t_flat=t_flat,
        Q2_flat=Q2_flat,
        sigma_eff=sigma_eff,
        y_scale=y_scale,
        grid_meta=grid_meta,
        device=device
    )


if __name__ == "__main__":
    main()
