# Completion reset — NLO first, then one frozen final analysis

Date: 2026-09-26. This is a scope and sequencing decision, not new numerical validation.

## Objective
Complete the quantitative, explicitly qualified NLO analysis needed for the paper rewrite. Do not silently substitute an LO paper or present effective H-dominance CFFs as unrestricted physical H. Do not turn completion into a requirement to prove universal GPD uniqueness or every possible positivity relation.

## Status and evidence
- The bounded independent LO gate was reported passing; retain its audited sources, fields and caches. Do not repeat historical runs.
- NLO hard-coefficient source/convolution checks were completed after the recorded repair. This is not certification of the complete NLO forward calculation.
- Task 03 is blocked on full-support coupled NLO evolution/CFF numerical accuracy. The saved pilot is xi=0.135, Q0^2=2, Q^2=2.3 GeV^2, fixed nf=3. Finest saved grid-pair absolute differences are Re=0.0038631013123936464 and Im=0.0004771775856968752, both above 1e-4. These are inter-resolution differences, not errors against a converged independent reference and not proof of a kernel-formula error.
- Structural, regularization, symbolic-selection, and nuisance-profile studies are development evidence. Preserve them; do not broaden those surveys now.
- The legacy experimental merge/surface is rejected. The separate joint-CFF ensemble has source/replay support but is conditional on H dominance, an incomplete effective-amplitude closure, and the documented experimental-error assumptions. Its empirical covariance rank must not be interpreted as exact physical zero-uncertainty directions.

## Critical path

### A. Close task 03: the NLO forward map
Owner: scientific reviewer for diagnosis; Codex for specified implementation/validation batches.
1. Reuse the saved pilot and source-keyed products. Follow evidence/NLO_RESUMPTION.md.
2. Isolate continuous K0/K1 action, endpoint/crossover representation, kernel quadrature, evolution integration and CFF quadrature. Use an independent analytic-input action/reference, not the same interpolant twice.
3. Establish a bounded pilot with a matrix-free/small-basis reference or a genuinely independent matched backend. Do not claim an untested backend works, turn a finite conformal subspace into unrestricted certification, or mask nonfinite values.
4. Retain scheme, normalization, running-coupling/flavor conventions and numerical budgets. Do not fit conservation corrections or relax tolerances.
5. Only after the pilot passes, run the bounded full-support acceptance suite and complete hard-plus-evolution consistency/derivative checks. Record absolute errors near CFF zeros and refinement of the learned functions when fits begin.
Exit: a reproducible complete NLO forward model satisfying the frozen numerical requirements. A failed pilot returns one minimal scientific failure, not another increasing-grid sweep.

### B. Freeze the final analysis contract (no new broad surveys)
Before final fitting, declare the primary charge-weighted target/functionals, domains and kinematic cuts, all-term polynomiality rule, explicit regularization/physical assumptions, input uncertainty model and omitted-CFF/effective-amplitude branches.
Keep symbolic compression descriptive unless the selection procedure is itself calibrated. Retain unbounded or strongly assumption-dependent target ranges as scientific outcomes.
Choose the experimental interpretation explicitly: the audited candidate can support a named H-dominance analysis plus documented sensitivity, not an unrestricted H extraction. Additional CFF priors need independent justification. Existing per-set compatibility spans are not a coherent globally GPD-compatible uncertainty ensemble.

### C. Run one registered closure and uncertainty campaign
Dependencies: A and B. Compare LO-generated/LO-fitted, NLO-generated/NLO-fitted, and NLO-generated/LO-fitted pseudodata under the same declared truth/domain/estimator contract. Reuse validated components; generate new final truths and holdouts only after freezing the method. Separate numerical, perturbative, statistical, representation and omitted-amplitude effects.
Validate the chosen target-interval procedure over independent datasets, recording coverage precision, widths, empty sets, optimizer failures and assumptions. If a finite-class convex profile replaces neural inference, identify that change explicitly; its coverage is not automatically coverage for the neural estimator or out-of-class truths. Start with the bounded pilot, not a fresh architecture sweep.
Exit: a justified uncertainty statement for specified targets, or correspondingly narrowed claims. Do not hide a method failure by post-hoc truth selection.

### D. Experimental application
Dependencies: A–C. Use original supported kinematic rows, coherent realizations and/or an explicitly justified observable-level likelihood. Do not diagonalize missing covariance, count interpolated points as independent measurements, silently repair the legacy table, or convert half-difference/angular/finite-t conventions implicitly. Report effective-CFF, twist and external-input assumptions alongside results. No arbitrary +/-0.1 probes become physical nuisance priors.
Exit: auditable quantitative NLO results within the stated assumptions; no unrestricted full-GPD or uncertainty claim not supported by the analysis.

### E. Rewrite in parallel, finalize after D
Write introduction, inverse-problem formulation, numerical method, polynomiality/symbolic method and validated limitations now from existing evidence. Maintain a claim-to-artifact ledger; leave incomplete NLO and final-result sections explicitly pending. Finalize abstract/conclusions and the experimental section only after A–D. Reframe the paper around information-resolved conditional reconstruction, not unique inversion caused by differentiability.

## Stop doing
- Further LO kernel reruns, compiler portability work already passed, blind NLO grid sweeps, new FiLM searches, and additional local nuisance-profile surveys without a critical-path justification.
- New helper bundles solely to relay status or improve metadata; mechanical validations run locally.
- Repeated approval rounds for routine debugging. A worker owns its bounded stage; scientific review occurs at milestone completion or a reproducible scientific blocker.

## Immediate next work
Task 03, the saved full-support NLO pilot. Reuse the prior resumption specification. No additional external-input inventory is required to investigate that numerical blocker, and no statistical-profile campaign precedes it.

## Provenance
Status read from GPD_task03_resolution_decision.zip, including DECISION.md, NLO_RESUMPTION.md and results/RESULT.json. The exact extracted files are retained in evidence/. Other status items are the documented decisions from the completed reviews, not new reproductions in this reset.
Source archive SHA-256: fd3deea6a0852bc655d846b8cfa7e38953b0a0e1df3287aa7278907fe0a6f25e
