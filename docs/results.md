# Completed evidence and scope

Results through 30 October 2026. This page summarizes saved reviewed records; the repository import runs no scientific fit or new noise draw.

## NLO closure and noise sensitivity

The noiseless complex-CFF multiscale example achieved 4.8967% wider-chart charge-weighted field error. Its individual channel errors were much larger in some channels, so this is not a successful complete flavor/gluon separation. At actual experimental coordinates, the exact-ReH Z1 closures gave wider-chart errors of 0.7502% and 3.0512% for seeds 31 and 43; the corresponding noisy cases gave 32.6917% and 49.2089%. These are individual controlled cases with specified settings.

P1 varied the amplitude of one fixed noise direction. At half amplitude, observed/wider-chart errors were 11.46%/28.61%; at full amplitude, 16.53%/32.69%. This is a sensitivity trajectory, not an ensemble estimate. [Source metrics](../evidence/program_summary.json)

## Forward information: P2B

The [README comparison table](../README.md#completed-evidence) uses one truth and the fixed exact/half-noise inputs. The [exact review](../evidence/GPD_P2B_exact_center_review_20261028/REVIEW.txt) accepts all four preservation checks. The [half-noise review](../evidence/GPD_P2B_half_development_review_20261030/REVIEW.txt) independently confirms the failed B safeguard, despite improvements in observed-slice H+ and A. Independent assessment remains unreleased for that design.

A is a mid-x average; B is an x-weighted global first moment at the stated off-forward point. Wider-chart accuracy probes different support. A lower PDF residual cannot substitute for success on these GPD quantities. The failure is a limitation of this tested addition, not a conclusion that PDF information can never help.

The half-noise run completed 400 Adam updates and 60 accepted outer updates in one CPU attempt, taking 7:12:10 elapsed on an 8-CPU/64-GiB allocation. It used 199 solver evaluations, with 81.43% of solver time in norm subproblems. This supports finite-budget feasibility, not terminal convergence or GPU acceleration. Durations of runs with different input trajectories do not establish a speedup.

## Other completed findings

- **P2A regularization:** the selected penalty failed all three exact-input 2% preservation gates (17.10%, 5.90%, 14.75%). This rejects the tested design, not all regularization.
- **Selected-functional calibration:** 442 completed cases comprised 160 calibration and 282 tests at two fixed synthetic truths. Joint coverage was 135/141 and 131/141, with common absolute half-width 0.01858363. This is evidence for that estimator, target pair, and observation law. The A/B truth-stratum labels here are distinct from P2B's functional labels.
- **Information limit:** a constructed 25%-separated off-forward pair had whitened ReH distance 0.559 and 61.0% optimal binary identification under the specified law. The ambiguity is conditional on that construction and observation model.
- **Upstream CFF improvement:** an upstream candidate reduced ReH RMSE by about 29%/39% on two truths and eight paired draws each. Its small downstream bridge still had 184.75% input-slice H+ error, demonstrating that improving CFFs does not alone ensure reliable inversion.
- **Symbolic refits:** the implemented proposal/refit comparisons concern allowed moment supports. They do not establish exact physical zeros or unrestricted analytic identification.

[Program review](../evidence/program_review.txt) · [Machine-readable summary](../evidence/program_summary.json) · [Calibration protocol](calibration_protocol.md)

## Experimental interpretation and publication

Experimental central reconstructions and their propagated uncertainties remain conditional on the numerical and statistical choices examined. The synthetic calibration cannot be relabeled an experimental error band. The current reduced basis does not determine E or separate all quark flavors.

The manuscript is a comprehensive revision on the existing arXiv record. Its contribution rests on the more complete NLO inference formulation and evidence for its capabilities and limits. Comparisons, ablations, additional truths, and repeated noise draws should support the particular claims made. A separate demonstration of superiority over the old draft is not a universal publication requirement.
