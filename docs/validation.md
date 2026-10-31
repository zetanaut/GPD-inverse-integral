# Repository validation

Checked 31 October 2026 for the analysis and plotting update.

## Tested local workflow

The source package installed successfully as `gpd-inverse-integral 0.3.0.dev0` in an isolated Python 3.9.23 environment using the available scientific runtime. The tested package versions are recorded in [requirements-tested.txt](../requirements-tested.txt): NumPy 1.26.3, SciPy 1.13.0, PyTorch 2.8.0, pandas 2.1.4, Matplotlib 3.8.2, and pytest 8.4.2.

- **82 tests passed:** 78 existing model, convention, NLO-interface, derivative, calibration, PDF, and cache tests, plus four analysis tests covering coordinate matching, mean-distance decomposition, directional response, and relative-error definitions.
- **712 imported-file hashes matched** the source manifest. All 598 files from the preceding source distribution remain unchanged, including all six original root utilities.
- **81 analysis inputs matched** their SHA-256 manifest, including the exported states, configuration, and saved numerical products.
- **Six PDF and six PNG figures regenerated** through the public command. All six PNGs match the visually reviewed manuscript figures pixel for pixel in this environment.
- **Seven comparison tables regenerated**, with saved-array CFF errors, residuals, response norms, and the finite-replica mean-distance identity checked against the bound records.
- **54 model states replayed**, including five closure fits, the local-CFF fit, and three mean/replica fits at each saved iteration from 5 through 20.
- **7,840 experimental CFF predictions recomputed** using the bundled nominal NLO rows. All five closure field-error records and the plotted slice/surface values matched their saved counterparts.
- **54 portable state exports checked** against their source-hashed original checkpoints, including flat-vector/state-dictionary agreement.
- Maintained documentation links resolved, and the added public Python entry points parsed successfully.

## Numerical agreement

| Replayed quantity | Largest absolute difference from saved record |
|---|---:|
| Closure field scores | 0 in the tested environment |
| Plotted field curves | 4.45 × 10⁻¹⁶ |
| Surface grids | 0 in the tested environment |
| Experimental RMS residuals | 7.11 × 10⁻¹⁵ |
| Saved experimental CFF predictions | 4.27 × 10⁻¹⁴ |

These figures describe floating-point replay agreement in the tested environment. Cross-platform results are checked with numerical tolerances rather than requiring byte-identical plot files; PDF timestamps, font libraries, and rendering versions may differ.

Machine-readable records: [array analysis](../evidence/analysis_20261031/ANALYSIS_CHECK.json), [checkpoint replay](../evidence/analysis_20261031/CHECKPOINT_REPLAY.json), and [figure-generation completion](../evidence/analysis_20261031/REPRODUCTION_CHECK.json).

The existing derivative test emits one PyTorch warning about converting a differentiable scalar to a float for assertion. The numerical test passes; the inherited scientific implementation was not modified to suppress it.

## Continuous integration and scope

The [GitHub Actions workflow](../.github/workflows/analysis.yml) installs the constrained CPU runtime on Python 3.11, checks imported sources, runs the tests, and regenerates the analysis with checkpoint replay. Its live status is visible on the repository's Actions page; the local record above does not assert a remote CI outcome.

Validation here covers distribution integrity, analysis arithmetic, saved-model evaluation, and use of the bundled experimental NLO rows. It does not repeat an inverse optimization campaign, build the native evolution operators, independently validate the source kernels, or establish statistical coverage for the experimental GPD family. [Reproduction scope by operation](../analysis/README.md)
