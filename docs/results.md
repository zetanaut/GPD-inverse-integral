# Results and interpretation

The comparisons below use saved calculations included in the 31 October 2026 analysis snapshot. The [reproduction guide](../analysis/README.md) distinguishes quantities recomputed directly from arrays, quantities reevaluated from saved weights, and diagnostics retained from reviewed score records.

## Exact-input closure

The generating field, Z1, has three reference-scale channels, nonzero ERBL and DGLAP support, t dependence, and analytic polynomial moments. It is different from the neural inverse parameterization. Its [definition](truths.md), [implementation](../src/gpd_tools/truths.py), and [fit configuration](../analysis/config/FIT_CONFIG.json) are supplied. The inverse objective receives CFFs and free-coefficient moment constraints; it does not receive the true pointwise GPD or its moment coefficients.

The exact complex-CFF fit uses 45 input positions spanning five skewness values, three scales, and three physical-t chart positions. A disjoint check uses 512 positions excluded from fitting and fit selection.

| Quantity | Relative error (%) |
|---|---:|
| Charge-weighted GPD, wider chart | 4.89668 |
| Charge-weighted GPD, ERBL | 5.41309 |
| Charge-weighted GPD, DGLAP | 4.78678 |
| Real CFF, fitted / disjoint positions | 0.92740 / 0.74706 |
| Imaginary CFF, fitted / disjoint positions | 1.60140 / 2.74769 |
| Singlet, wider chart | 78.45925 |
| Nonsinglet, wider chart | 235.00874 |
| Gluon, wider chart | 5.89642 |

The reported physics result is recovery of the charge-weighted combination. Cancellation between poorly identified singlet and nonsinglet channels prevents interpreting that recovery as flavor separation. The small gluon error in this one closure does not establish general gluon identifiability. [Figures 1–2](figures.md#1-exact-input-multiscale-closure) show both norm-level agreement and the remaining shape differences.

The wider chart uses 33 skewness nodes, nine physical-t chart nodes, and order-160 x quadrature in each region. It gives equal weight to each chart node and Gaussian weight in x. The analysis's optional checkpoint replay recomputes this field metric from the saved weights.

## Real CFFs at experimental coordinates

The same generating field is fitted using only real CFFs at the 160 retained experimental coordinates. Two network initializations use the same exact targets or the same fixed Gaussian perturbation at the quoted experimental error scale.

| Initialization | Exact-input wider-chart error (%) | Noisy-input wider-chart error (%) |
|---|---:|---:|
| 31 | 0.7502 | 32.6917 |
| 43 | 3.0512 | 49.2089 |

The noisy change affects the field shape, not only normalization. These results show sensitivity to the specified input noise and reconstruction procedure. They do not estimate an irreducible experimental resolution limit or statistical coverage across all possible noise draws. [Figure 3](figures.md#3-exact-and-noisy-real-cff-inputs)

The ideal complex-CFF and experimental-coordinate real-CFF tests use different sampling, observation counts, and objective weights. Their numerical errors cannot rank the value of real versus complex measurements in isolation.

## Reconstruction from experimental CFFs

The source is the table of local real-CFF extractions associated with [Le and Keller](https://arxiv.org/abs/2504.15458). The cuts Q²>1.5 GeV² and |t|/Q²≤0.25 retain 160 of 189 rows, with 67 skewness values, 102 distinct `(xi,Q2)` pairs, and 79 momentum transfers. The fit uses the combined quoted statistical/systematic errors. It does not refit the original DVCS cross sections or supply measured imaginary CFFs.

Each of 32 perturbed local-CFF vectors is fitted with the same smooth 3–32–32–1 tanh CFF network. Each fitted replica is a coherent function of xi, t, and Q². The median replica standard deviation divided by the original error is 0.478; this is the response of the chosen smoothing procedure, not evidence of a more precise experiment. The empirical 32-member covariance is not inverted to replace the original weights.

Three GPD fits use the pointwise mean of those CFF functions or individual replicas 000 and 001. All start from the same GPD network fitted to a smooth representation of the unperturbed local data. Their output is compared after 5, 10, and 20 additional outer iterations. The separate local-CFF fit uses the 160 local CFF values directly.

| GPD target | Own-target RMS | Same-local-data RMS |
|---|---:|---:|
| Local CFF values | 2.43350 | 2.43350 |
| Replica-mean CFF | 1.32583 | 2.17274 |
| CFF replica 000 | 1.52782 | 2.19179 |
| CFF replica 001 | 1.58418 | 2.12619 |

The last three rows use 20 additional iterations. Residuals use the same original quoted errors. The comparison combines differences in targets and optimization procedures; it does not isolate a causal benefit from smoothing. [Figures 4–5](figures.md#4-experimental-gpd-slices)

The GPD fitted to the mean CFF function is not the mean of reconstructed GPDs. Thirty-two upstream CFF functions have been constructed, while only two individual CFF replicas have been propagated to GPDs in this comparison. Their responses form a conditional demonstration, not a calibrated experimental uncertainty band.

## Dependence on optimization iterations

| CFF replica | GPD response RMS after 5 iterations | After 10 | After 20 |
|---|---:|---:|---:|
| 000 | 0.00910 | 0.00893 | 0.01313 |
| 001 | 0.00702 | 0.01947 | 0.03370 |

These are GPD differences from the mean-CFF fit on the matched six full-x slice grids, not errors against a known experimental truth. Over the same interval, the mean-CFF fit's own-target residual drops from 1.38890 to 1.32583 in original error units. No response plateau is established. [Figure 6](figures.md#6-dependence-on-optimization-iterations)

After 20 iterations, the CFF output-to-input perturbation norm ratios are 0.339 and 0.434. Directional differences matter as well, so these cannot be read as fractions of experimental uncertainty recovered. The finite-replica mean-distance identity gives approximately 1.75782 + 0.30395 = 2.06176 for the mean-CFF fit. Mean mismatch remains even after accounting for the supplied CFF replica dispersion.

The [analysis script](../analysis/analyze_results.py) recomputes these responses with explicit coordinate matching and verifies them against the saved records. The [checkpoint replay](../analysis/replay_checkpoints.py) reevaluates the experimental predictions at all 48 saved mean/replica stages and the local-CFF state.

## Forward-information comparison

The additional constraint is a synthetic forward observation with 36 components: three stored channels at 12 x nodes from 0.02 to 0.8. The design includes an explicit correlated covariance; it is not an empirical PDF uncertainty set. [Design](../analysis/config/PDF_DESIGN.json) · [PDF interface](../src/gpd_tools/pdf_information.py)

At xi=0.275, t=−0.363 GeV², and Q₀²=2 GeV², define

$$A=5\int_{0.2}^{0.4}H^{(+)}dx,\qquad B=\int_0^1xH^{(+)}dx.$$

| Input | Fit | Observed-field error (%) | A error (%) | B error (%) | Wider-chart error (%) |
|---|---|---:|---:|---:|---:|
| Exact | CFF | 0.920 | 0.485 | 0.869 | 0.750 |
| Exact | CFF+PDF | 0.469 | 0.677 | 0.252 | 0.535 |
| Half noise | CFF | 11.463 | 4.990 | 1.281 | 28.613 |
| Half noise | CFF+PDF | 10.134 | 3.540 | 3.236 | 38.134 |

A and B columns are absolute relative errors. Half noise uses half the quoted CFF perturbation scale and is a separate comparison from the full-noise example above. Exact-input addition improves the wider-chart field while keeping both target errors below 1%. Under half noise it improves the observed field and A, but worsens B and the wider chart. This tests one generating field and one fixed half-noise realization; it does not establish general stabilization by forward information.

The analysis extracts this table from the [exact](../analysis/data/p2b_exact.json) and [half-noise](../analysis/data/p2b_noisy.json) review records. It does not rerun those fits. The half-noise CPU attempt completed 400 Adam updates and 60 accepted outer updates in 7:12:10 on its recorded allocation. That is execution evidence, not a GPU or comparative speedup measurement.

## Supporting studies

The source distribution also preserves related studies with distinct questions and estimator definitions:

- **Selected-functional calibration:** 442 cases comprised 160 calibration and 282 test cases. Joint selected-target coverage was 135/141 and 131/141 on two synthetic strata. This does not calibrate the experimental fields shown above. [Protocol](calibration_protocol.md)
- **Information-limit diagnostic:** a constructed pair separated by 25% in an off-forward quantity had whitened ReH distance 0.559 and 61.0% optimal binary identification under the specified observation model.
- **Regularization comparison:** the selected P2A penalty failed its three exact-input preservation checks. This rejects that tested penalty choice, not all regularization.
- **Upstream CFF comparison:** improved CFF RMSE in a synthetic study did not translate automatically into reliable downstream GPD recovery in its small bridge test.
- **Symbolic refits:** allowed moment-support hypotheses can be proposed and refitted, while leaving retained coefficients free. Small coefficients do not establish exact QCD zeros.

Their [program summary](../evidence/program_summary.json) and [review](../evidence/program_review.txt) preserve the source metrics. These separate studies should not be pooled into the experimental replica family or relabeled as its uncertainty calibration.
