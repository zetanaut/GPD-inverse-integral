# Software and reproduction guide

## Supported local entry points

The installable source uses two namespaces: `gpd_reh` for the reference field and CFF interface, and `gpd_tools` for reusable inference components. Most modules are byte-for-byte copies of the working implementation. `SOURCE_MANIFEST.json` identifies their origins and the small import/path changes in copied tests.

| API | Purpose | Required inputs |
|---|---|---|
| `gpd_reh.model.ExperimentalFiLM` | Reference-scale three-channel neural field | t normalization and float64 coordinates |
| `gpd_reh.moments.FixedTMoments` | Fixed-t polynomial-space residuals | t nodes, quadrature and spin choices |
| `gpd_reh.bank.ExperimentalForward` | Contract a field with exact NLO rows | Prepared observation CSV and validated bank index |
| `gpd_reh.problem.ExperimentalProblem` | ReH residual, objective, integrated Jacobian | Prepared data, exact-coordinate banks, random seed |
| `gpd_tools.import_rows.import_rows` | Import paired nominal/reference adjoint rows | Accepted export with source metadata |
| `gpd_tools.pdf_information.ForwardInformation` | Frozen P2B correlated boundary residual | Base field, 12 x nodes, 36 observations and covariance |
| `gpd_tools.direct_error` | Contract/fit/pair records, calibration and intervals | Frozen estimator/data identities and independent records |
| `gpd_tools.truths.Truth` | Z1/Z2 synthetic field and analytic moments | Truth label and coordinates |
| `gpd_tools.solver`, `residual_subproblem` | Constrained-solver building blocks | Declared objective/constraint interfaces |
| `gpd_tools.model_cache.ExactModelCache` | Reuse identical coordinate-model construction | Original builder, arrays and settings |

The synthetic generators are model-based controls. Their code documents the properties they satisfy; positivity and complete phenomenology are not certified.

## Data and reproduction

`data/experimental/final_summary_by_DNN.csv` is the author-identified source snapshot associated with [Le and Keller, arXiv:2504.15458](https://arxiv.org/abs/2504.15458). The preparation code binds its SHA-256 and retains 160 of 189 rows after the declared kinematic cuts. `prepared/` retains the selected coordinates, input summary, map requests, and observation arrays. `banks/` contains 102 exact `(xi,Q2)` row pairs, each with source and validation metadata.

The bank artifacts enable evaluating the current forward interface without native kernel compilation. They inherit their validation record; this repository packaging does not independently regenerate or certify all evolution kernels. The separate `tests/data/accepted_development_rows.npz` is a small development fixture and is not the full experimental geometry.

```python
from gpd_reh.problem import ExperimentalProblem
p = ExperimentalProblem('data/experimental/prepared',
                        'data/experimental/banks', seed=31)
value = p.objective(p.x0)  # fresh random state, not an experimental fit
```

The [quickstart](../examples/quickstart.py) evaluates the row map and a small moment diagnostic. `pytest` exercises source conventions, field structure, moment spaces, actual row contraction, derivatives, and calibration rules. These checks do not retrain scientific models.

## Research workflow sources

`research/` retains original project-relative directory names so the numerical and inference development remains inspectable:

- `GPD_NLO_completion_package`: Python/C++/Fortran evolution and hard-map sources, precision safeguards, source mappings, and captured conventions.
- `completion_runs/NLO_final_geometry/source`, `GPD_final_geometry_GG_local_repair`, `GPD_GG_coordinate_repair`: geometry construction, native-source repair, row validation, and export machinery.
- `GPD_ReH_experimental_run`: experimental preparation, bank import/export, domain adapter, constrained fit binding, and diagnostics.
- `completion_runs/NLO_constrained_solver`, `NLO_feasible_domain`, and `GPD_NLO_constrained_solver_review`: the solver and its numerical-domain checks.
- `GPD_three_stage_closure`, `GPD_accuracy_cost_study`: truth generation, exact/noisy closures, scoring, and cost studies.
- `completion_runs/NLO_symbolic_refit`: support proposals, refit branches, and held-out checks.
- `GPD_direct_error_calibration`: statistical CLI, fitting adapters, and replay tools.
- `GPD_P2B_preparation_v3`, `GPD_P2B_half_development_UVA_v1`: correlated PDF integration, profiling/cache tools, policy, checkpoint continuation, and Slurm transfer tooling.
- `GPD_upstream_bias_repair/input/study/src`: upstream CFF study sources.

These are research workflow snapshots, not a second turnkey package. Several entry points require the original source-hashed runtime capsule, native libraries, operator/contraction caches, configurations, or saved checkpoints. Their guards and historic paths are retained. The bundled row banks replace neither those runtime dependencies nor their numerical-domain contracts. Old job submission tools are historical source; executing the quickstart or test suite does not submit cluster work.

The multi-gigabyte native caches, full fit ensembles, scheduler logs, and duplicate return archives are outside this Git source distribution. Existing paper results are represented by compact reviewed evidence. The full selected-field replay therefore requires the associated run artifacts; the repository does not claim to reproduce the complete campaigns from the quickstart.

## Legacy compatibility

The six original root scripts are preserved without changing their scientific behavior. They use the earlier single-field checkpoint schema and CFF ensemble layout. Current FiLM fields have three channels, a different conditioning contract, and fixed-scale network outputs; applying a legacy plotter directly would be incorrect. Use the current diagnostic/plotting workflow sources with their corresponding checkpoints. No automatic checkpoint conversion is claimed.

## Provenance and rights

`SOURCE_MANIFEST.json` records copies and adaptations. Source hashes can be checked with `python tools/verify_sources.py`. Tiktaalik-derived sources carry their original license, copied at `research/GPD_NLO_completion_package/frozen/fortran/LICENSE.tiktaalik`. The surrounding research code retains its existing authorship and rights.
