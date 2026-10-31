# Reproduce the analysis and figures

This directory contains the numerical inputs and scripts for the six figures and comparison tables in the current NLO GPD manuscript. Its default workflow reads saved results, checks their integrity, recomputes diagnostics, and renders plots. An additional workflow evaluates the saved neural fields and contracts them with the experimental NLO row bank.

## 1. Tables and figures from saved arrays

From the repository root, in a Python 3.9+ virtual environment:

```bash
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py
```

This route needs NumPy, pandas, and Matplotlib. It does not need PyTorch, TeX, a GPU, a cluster allocation, or native QCD compilation. Matplotlib uses its noninteractive Agg backend.

To choose a destination:

```bash
python analysis/reproduce.py --out /tmp/gpd-analysis
```

The default is `outputs/analysis/`, which is ignored by Git. The frozen input tree is protected from use as the reproduction output destination. Running the command again replaces its generated outputs.

The two stages can also be run separately:

```bash
python analysis/analyze_results.py --out outputs/analysis/tables
python analysis/make_figures.py --out outputs/analysis/figures
```

`make_figures.py` has an optional `--data` argument for an alternative data directory with the same schema. For the bound study results, use `reproduce.py`, which verifies the input manifest first.

### Generated products

| Output under `outputs/analysis/` | Meaning |
|---|---|
| `figures/fig01_exact_closure.{pdf,png}` | Exact-input slices and disjoint CFF predictions |
| `figures/fig02_exact_surface.{pdf,png}` | Exact-input truth, reconstructed surface, and signed difference |
| `figures/fig03_exact_noisy.{pdf,png}` | Exact/noisy real-CFF comparison for two initializations |
| `figures/fig04_experimental_family.{pdf,png}` | Local-CFF, mean-CFF, and individual-replica GPD slices |
| `figures/fig05_experimental_surface.{pdf,png}` | Mean-CFF GPD surface and replica-001 response |
| `figures/fig06_budget_response.{pdf,png}` | Dependence of CFF residual and GPD response on iteration count |
| `tables/closure_errors.csv` | Wider-chart field errors from bound score records; CFF errors recomputed from arrays |
| `tables/experimental_residuals.csv` | Own-input, mean-input, and local-data RMS residuals for nine saved comparisons |
| `tables/experimental_comparison.csv` | Four-fit comparison used in the manuscript |
| `tables/field_response.csv` | Six matched replica-minus-mean GPD response norms |
| `tables/cff_response.csv` | Matched input/output CFF norms, directional gain, and cosine |
| `tables/mean_distance_decomposition.csv` | Mean mismatch and 32-replica dispersion at three iteration counts |
| `tables/forward_information_comparison.csv` | Exact/half-noise CFF-only and joint CFF+PDF errors from review records |
| `tables/RESULTS.md` | Readable tables generated from the same calculations |
| `tables/ANALYSIS_CHECK.json` | Numerical agreement checks and their scope |
| `REPRODUCTION_CHECK.json` | Completion record for the requested workflow |

The [gallery](../docs/figures.md) explains each plot and links its inputs. Committed previews in `analysis/figures/` are generated from the same script; fresh outputs go to the chosen output directory.

## 2. Replay the saved neural fields and experimental CFF map

Install the repository package and plotting dependencies, then enable checkpoint replay:

```bash
python -m pip install -e '.[dev,plot]'
python analysis/reproduce.py --replay-checkpoints
```

Or run only the replay:

```bash
python analysis/replay_checkpoints.py --out outputs/analysis/replay
```

The replay uses 54 compact NPZ model states: five closure fits, the local-CFF fit, and three experimental-input fits at each additional iteration from 5 through 20. These are exported state dictionaries, containing 3,867 numerical parameters each, loaded with `allow_pickle=False`. No optimizer state or executable pickle is needed. Their [index](checkpoints/index.json) records the source checkpoint hashes, case labels, and iteration counts. Coordinate normalization comes from [FIT_CONFIG.json](config/FIT_CONFIG.json).

The script:

1. Recomputes the five closure field errors on the original 33-by-9 physical chart, using order-160 quadrature on each side of the crossover.
2. Evaluates the plotted closure slices and all experimental slice comparisons at 5, 10, and 20 iterations, including the separate local-CFF fit.
3. Regenerates the `(x,t)` surface arrays at the reference scale.
4. Contracts the 49 experimental model states with the bundled nominal NLO row bank, giving 7,840 predictions at the 160 retained coordinates.
5. Compares recalculated values with the bound scores, CFF arrays, curves, and residual tables, failing on a mismatch.

Additional outputs are `closure_field_scores.json`, `experimental_predictions.csv`, `experimental_residuals.csv`, `sampled_surfaces.npz`, and `CHECKPOINT_REPLAY.json` in the replay directory. The model, target, and saved result remain unchanged throughout.

This is evaluation of saved estimators. It does not repeat the optimization trajectories, regenerate the ideal-grid/holdout CFF operators, independently validate the QCD kernels, or refit the P2B comparison. The [workflow map](../docs/workflows.md) identifies those original source entry points and their additional requirements.

## 3. Metric definitions

All experimental CFF residuals use the original quoted standard deviations once:

$$\mathrm{RMS}_{\sigma}(g,d)=\sqrt{\frac{1}{160}\sum_i\left(\frac{g_i-d_i}{\sigma_i}\right)^2}.$$

Changing from a run's own target to the unperturbed local values changes `d`, not `sigma`. The original table and saved CSV predictions use the source sign convention; the internal map's real prediction is converted by `ReH_source = -ReH_internal` before comparison.

The full-field closure metric is a relative L2 norm of $H^{(+)}$ on the declared physical chart. It weights each `(xi,u)` chart node equally and uses Gaussian weights in x. It is not an unweighted average of the display pixels, and it is not a uniform physical-t integral. Exact settings are in [GEOMETRY.json](config/GEOMETRY.json); the unchanged scoring implementation is [closure/metrics.py](../research/GPD_three_stage_closure/closure/metrics.py).

The experimental field response is

$$D_r^{(b)}=\widehat H_r^{(b)}-\widehat H_{\mathrm{mean}}^{(b)}.$$

Its reported RMS averages the matched full-x grids at all six displayed anchors. `b` is the number of additional outer iterations from the common starting network. The mean subscript denotes the **GPD fitted to the mean CFF function**, not an average of GPD fits. CSV fields named `budget` and `central13` are retained archive identifiers; the public descriptions use iteration counts and the local-CFF fit.

For CFF responses, both the input and predicted changes are measured relative to the mean-CFF run and divided by the same `sigma`. The ratio of their norms measures response magnitude. `gain` and `cosine` also test direction: a norm ratio of one does not imply that the supplied perturbation has been recovered.

For the 32 supplied CFF replica vectors $R_r$, the identity checked is

$$\frac{1}{mn}\sum_r\|W_\sigma(g-R_r)\|^2=\frac{1}{n}\|W_\sigma(g-\bar R)\|^2+\frac{1}{mn}\sum_r\|W_\sigma(R_r-\bar R)\|^2.$$

This separates finite-sample mean mismatch from CFF dispersion. It is not a decomposition of GPD uncertainty into physical sources.

## 4. Numerical inputs and provenance

| Input group | Format and role |
|---|---|
| `data/*Z1_s*.npz` | Display coordinates, truth/reconstructed slices, and fitted/disjoint CFF arrays for five closure cases |
| `data/*_scores.json` | Recorded wider-chart scores and CFF checks for those cases |
| `data/sampled_surfaces.npz` | x[121], t[41], xi=0.275, and charge-weighted fields on the common grid |
| `data/experimental_family.csv` | case, iteration count, anchor, x, xi, t, and reconstructed H+ |
| `data/experimental_nominal.csv` | Saved local-CFF and diagnostic comparison curves; this workflow selects `central13` |
| `data/experimental_cff.csv` | 160 coordinates per run/stage, input CFFs, source-sign predictions, original errors, and pulls |
| `data/budget_stages.csv` | Residuals at every saved iteration from 5 through 20 |
| `data/budget_response.csv`, `data/cff_response.csv` | Bound reference values for the recomputed responses |
| `data/coherent_cff_replicas.npz` | 32-by-160 CFF predictions, row identifiers, mean, dispersion, and covariance |
| `data/p2b_exact.json`, `data/p2b_noisy.json` | Reviewed forward-information comparison records |
| `config/` | Original fit settings, geometry, study design, and forward-observation covariance design |
| `checkpoints/` | Portable state arrays and source identities for all 54 evaluated models |

[INPUT_MANIFEST.json](INPUT_MANIFEST.json) binds the 81 input/configuration/state files. [ORIGINAL_DATA_SOURCES.json](ORIGINAL_DATA_SOURCES.json) preserves the manuscript collection's original sources. The repository-wide [SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json) records imported files and adaptations. Paths naming the author workspace are provenance locators, not runtime prerequisites for the public commands.

To inspect or regenerate the state export from an available original workspace, use [export_analysis_states.py](../tools/export_analysis_states.py). Public reproduction uses the already exported arrays.

## 5. Validation and scientific scope

`analyze_results.py` checks that residuals and response norms agree with the saved results and verifies row/coordinate alignment before subtraction. `replay_checkpoints.py` independently reevaluates the saved fields using the published model and row interfaces. The [validation record](../docs/validation.md) reports the tested environment and numerical agreement.

Thirty-two CFF replicas and two propagated GPD replicas are different sample counts. The figures show the actual completed GPD comparisons, not percentile or coverage-calibrated experimental bands. The [results guide](../docs/results.md) states what each comparison establishes and what remains undetermined.
