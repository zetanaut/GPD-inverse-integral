
# DNN Inversion of CFF ReH to GPD H

This repository implements a **two–stage neural–network pipeline** to extract the
charge–weighted, C–even GPD \(H^{(+)}\) from experimental DVCS data via the
Compton Form Factor (CFF) \(\Re\mathcal{H}\).

The key script is **`GPD-H_pt.py`**, which takes as input a **global PyTorch
ensemble fit of the CFF** and performs a **model–based inverse transform** to
obtain an ensemble of GPDs \(H^{(+)}(x,\xi,t,Q^2)\).

The code is written to be **educational**: it keeps the physics structure
explicit (oddness, endpoint behavior, principal–value integral) while staying
numerically simple enough for students to study and modify.


## 1. Conceptual Overview

The CFF and GPD are related schematically by a principal–value integral:

\[
\Re\mathcal{H}(\xi,t,Q^2)
\;=\;
\text{PV}\int_{-1}^{1}\!dx\;
H^{(+)}(x,\xi,t,Q^2)\,
\Big[ \frac{1}{\xi-x} - \frac{1}{\xi+x} \Big]
\]

This repository treats **both sides** with neural networks:

1. **CFF stage (forward):**  
   A separate script trains a **global heteroscedastic NN** on experimental
   data to model \(\Re\mathcal{H}(\xi,t,Q^2)\). It produces an **ensemble** of
   networks \( \{\mathcal{M}_k\} \) in a directory called `reh_model_v3/`.
   Each model corresponds to a Monte‑Carlo replica of the experimental data.

2. **GPD stage (inverse):**  
   `GPD-H_pt.py` reads that CFF ensemble and:

   - builds a **dense regular grid** in \((\xi,t,Q^2)\),
   - evaluates each ReH model on that grid, giving a smooth **CFF replica field**
     \( \Re\mathcal{H}_k(\xi,t,Q^2)\),
   - trains one **GPD replica** \(H^{(+)}_k\) per CFF replica, such that
     its **PV convolution reproduces** that \(\Re\mathcal{H}_k\),
   - saves the collection \(\{H^{(+)}_k\}\) as a **GPD ensemble**.

The spread of the GPD ensemble reflects **experimental uncertainties** propagated
through the learned CFF model into the GPDs.


## 2. Files & Directory Layout

A typical layout after running both stages looks like:

```text
.
├── CFF_combined_minstd.csv         # experimental CFF input used in stage 1
├── ReH_global_fit.py               # (example) script that trained the CFF ensemble
├── reh_model_v3/                   # OUTPUT of stage 1 (CFF fit)
│   ├── models/
│   │   ├── model_000.pt
│   │   ├── model_001.pt
│   │   └── ...
│   ├── scaler.pt
│   └── training_summary.json
├── GPD-H_pt.py                     # **this script** (stage 2: inversion)
├── gpd_inverse_from_ReH/           # OUTPUT of stage 2
│   ├── ReH_smooth_for_inverse.csv
│   ├── gpd_models/
│   │   ├── gpd_rep_000.pt
│   │   ├── gpd_rep_001.pt
│   │   └── ...
│   ├── H_band_from_ReHfit.npz
│   ├── H_band_from_ReHfit.png
│   └── gpd_inverse_summary.json
├── plot_gpd_surface_band_simple.py # optional: 3D surface with ±1σ
└── plot_gpd_surface_multiQ.py      # optional: 3D surfaces at several Q²
````

You only **need** `reh_model_v3/` and `GPD-H_pt.py` to run the inversion,
but the plotting scripts are useful for visualization.

## 3. Dependencies & Installation

* Python ≥ 3.8
* PyTorch (CPU or GPU)
* NumPy
* pandas
* matplotlib (optional, but used for quick plots)

Install via:

```bash
pip install torch numpy pandas matplotlib
```

Make sure `reh_model_v3/` has already been produced by the CFF training script
(e.g. `ReH_global_fit.py` in the same repo).

## 4. Running the Inversion

1. **Check the CFF ensemble exists**

   You should have:

   ```text
   reh_model_v3/
     ├── models/model_000.pt  (and friends)
     ├── scaler.pt
     └── training_summary.json
   ```

2. **Run the GPD inversion**

   ```bash
   python3 GPD-H_pt.py
   ```

   The script will:

   * print the device it is using (`cpu` or `cuda`),
   * build the ReH grid,
   * loop over CFF replicas, training one GPD replica per CFF model.

   Progress lines like

   ```text
   [GPD rep 003] epoch  40/120  loss=1.2345
   ```

   show the loss for each replica during training.

3. **Inspect outputs**

   * `gpd_inverse_from_ReH/ReH_smooth_for_inverse.csv`
     → a table of the regular grid and ensemble mean/std of ReH.

   * `gpd_inverse_from_ReH/gpd_models/gpd_rep_XXX.pt`
     → one trained **GPD replica** per CFF replica.

   * `gpd_inverse_from_ReH/H_band_from_ReHfit.png`
     → 1D plot of (H^{(+)}(x,\xi_0,t_0,Q_0^2)) with an uncertainty band.

   * `gpd_inverse_from_ReH/gpd_inverse_summary.json`
     → metadata about the inversion (grid ranges, hyperparameters, etc.).

4. **Optional visualization**

   Once the GPD ensemble exists, you can run, e.g.,

   ```bash
   python3 plot_gpd_surface_band_simple.py
   python3 plot_gpd_surface_multiQ.py
   ```

   to make 3D GPD surfaces and uncertainty bands in ((\xi,t)) for fixed
   (x_0, Q^2_0).

## 5. How `GPD-H_pt.py` Works (High‑Level)

### 5.1 Stage 1 — CFF grid from the ReH ensemble

* **Load CFF ensemble**
  `load_reH_ensemble()` reads:

  * `scaler.pt` → input normalization (mean & std) and `y_scale` for ReH,
  * `training_summary.json` → model architecture and kinematic support,
  * `models/model_*.pt` → heteroscedastic MLPs ( \mathcal{M}_k ).

* **Build regular ((\xi,t,Q^2)) grid**
  `build_reH_grid_and_evaluate()` uses the `support` ranges from the
  training summary to create a dense, rectangular grid:

  ```python
  xi_grid = np.linspace(xi_min, xi_max, N_XI_GRID)
  t_grid  = np.linspace(t_min,  t_max,  N_T_GRID)
  q2_grid = np.linspace(q2_min, q2_max, N_Q2_GRID)
  ```

* **Evaluate ensemble on grid**

  * `eval_reH_ensemble()` computes the **ensemble mean** and **predictive std**
    at each grid point (epistemic + aleatoric variance).

  * Optionally multiplies the std by a **calibration factor** τ (computed in
    the CFF training step) so that ~68% of validation points lie within ±1σ.

* **Save smooth CFF field**

  The grid and ensemble statistics are saved to
  `ReH_smooth_for_inverse.csv`:

  ```text
  xi, Q2, t, ReH_smooth, sigma_smooth
  ```

  This file is useful by itself for CFF plots and cross‑checks.

### 5.2 Stage 2 — GPD replicas via PV inversion

* **GPD parameterization**

  The core network `GPDBaseNet` approximates a function (f(x,\xi,t,Q^2)).
  The physical C‑even GPD is built as:

  [
  H^{(+)}(x,\xi,t,Q^2)
  ====================

  (1 - |x|)^{\beta} \frac{f(x,\xi,t,Q^2) - f(-x,\xi,t,Q^2)}{2}
  ]

  This enforces:

  * **oddness in (x)** (C-even combination),
  * a known **endpoint falloff** via the ((1-|x|)^\beta) envelope.

* **Principal–value operator**

  `PVForwardOperator` discretizes the integral defining (\Re\mathcal{H}):

  ```python
  xgrid  = torch.linspace(-1, 1, NX)
  kernel = 1/(xi - x) - 1/(xi + x)

  # exclude a small band around x=±xi for PV
  mask = (|x - xi| > eps) & (|x + xi| > eps)

  ReH_pred = Σ_x H(x) * kernel * mask * w_x
  ```

  Here `NX` is the number of points on the x‑grid, `w_x` are trapezoidal
  integration weights, and `eps = PV_EPS_MULT * dx` is the PV cut‑out width.

* **Loss function**

  For each replica, we minimize:

  [
  \mathcal{L} =
  \frac{1}{N}\sum_i
  \frac{\left[\Re\mathcal{H}^{\text{pred}}_i -
  \Re\mathcal{H}^{\text{grid}}*i\right]^2}
  {\sigma*{\text{eff},i}^2}

  * \lambda_x \int dx; \big(\partial_x^2 H^{(+)}\big)^2
    ]

  - The first term is a **weighted χ²** using the effective ReH uncertainty
    `sigma_eff`.
  - The second term is a **curvature penalty in x** (`GPD_LAM_X`) which
    suppresses oscillatory solutions and stabilizes the inversion.

* **Replica logic**

  For each CFF model `m_k`:

  1. evaluate its smooth CFF field `ReH_vals` on the grid,
  2. build a dataset of `(xi, t, Q2, ReH_vals, sigma_eff)`,
  3. train a fresh GPD network + PV operator to minimize the loss,
  4. save the resulting network as `gpd_rep_XXX.pt`,
  5. record the slice (H_k^{(+)}(x,\xi_0,t_0,Q_0^2)) for band plots.

  The ensemble `{gpd_rep_*}` is then the **GPD uncertainty band**.

## 6. Key Hyperparameters to Experiment With

All are defined at the top of `GPD-H_pt.py`:

* **ReH grid density**

  * `N_XI_GRID`, `N_T_GRID`, `N_Q2_GRID`
    → more points = smoother, but slower.

* **GPD x‑grid and structure**

  * `NX`           – number of x points in ([-1,1]),
  * `BETA`         – endpoint envelope exponent,
  * `GPD_HIDDEN`   – MLP width,
  * `GPD_LAYERS`   – number of hidden layers.

* **PV & regularization**

  * `PV_EPS_MULT`  – controls how aggressively you excise the kernel
    singularity; too small → numerical noise, too large → bias.
  * `GPD_LAM_X`    – strength of curvature regularization in x.

* **Replica control**

  * `MAX_GPD_REPS` – cap the number of CFF replicas you invert (for speed).

Students can change these and see how the GPD band and smoothness respond.

## 7. Typical Workflow for Students

1. **Inspect the CFF stage**

   * Look at `reh_model_v3/training_summary.json` to see the ranges of
     (\xi, t, Q^2) and the network architecture.
   * Plot `ReH_smooth_for_inverse.csv` with a quick 3D surface or contour
     plot to see the CFF landscape.

2. **Study the inversion script**

   * Read `GPD-H_pt.py` from top to bottom:

     * How are ReH ensemble predictions combined?
     * How is the PV integral implemented on the x‑grid?
     * Where are physics constraints (oddness, envelope) imposed?

3. **Run and visualize**

   * Run `python3 GPD-H_pt.py`.
   * Examine `H_band_from_ReHfit.png`.
   * Use the plotting scripts to view 3D surfaces in ((\xi,t)) at various
     (Q^2).

4. **Modify and re‑run**

   * Decrease or increase `NX`, `GPD_LAM_X`, or `N_XI_GRID` and observe
     effects on smoothness and uncertainty.
   * Try restricting to fewer replicas via `MAX_GPD_REPS` and see how the
     band changes.



You can find the ReH CFF model used at Rivanna under /project/UVA-Spin:
https://ood.hpc.virginia.edu/pun/sys/dashboard/files/fs//project/UVA-Spin
