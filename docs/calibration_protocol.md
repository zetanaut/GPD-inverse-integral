# Direct target-error calibration: adopted development route

This explicitly replaces the proposed target-profile uncertainty calculation.
It does not replace FiLM, free moment coefficients, the NLO physics, or the
previously documented failed fits. It is not an experimental posterior.

## 1. A reproducible point estimator, not a global profile optimizer
The adapter learns the original 3,867-parameter reference-scale FiLM, with all
42 permitted moment coefficients left free, against the unchanged J and NLO
operators. Initialization seeds are now fixed to 31 and 43 for **every** dataset:
no truth, calibration/test role, or previous dataset state changes initialization.

Each start uses 400 Adam steps on J. Of its random/Adam fields that satisfy all
nonmoment fitting constraints, scale the existing affine output head only as
needed to put the predictive moment residual at 0.8 of its existing bound. The
lowest-J such field initializes the accepted residual-norm trust-region solver
for at most 60 iterations. All parameters remain available during those steps.
This starting restoration is an algorithmic choice, not a new physical prior or
a global norm restriction. Full residual, feasibility and source contracts are
unchanged. It may affect point bias; calibration must include this exact rule.

Candidates are recorded at steps 0/20/40/60 and deterministic solver termination.
Rank by J, then seed, then stage. Independently validate at most four candidates
in that order against the original disjoint finite-moment, numerical and target
quadrature gates. First passing candidate is the output; none means abstention.
The disjoint nodes are output-eligibility checks (not held-out experimental data).
No KKT pass/global optimum is claimed. KKT/gradient diagnostics remain recorded;
finite-budget stopping is a declared CHANGE to the old estimator contract.
A wall/process/resource termination aborts the dataset instead of selecting a
machine-time-dependent partial fit. No final calibration cutoff may be reused
across estimator, geometry, source, noise-model or software changes.

The current executable adapter is restricted to the accepted 18-scalar
DEVELOPMENT geometry. It rejects other sizes. Final 30-scalar geometry, new
truths and final forward verification remain gated work: do not replace them
with the development setup and call it the final campaign.

## 2. Joint score and reported rectangle
For two fixed targets, use positive scales s=(0.05,0.05) in the inherited
dimensionless conventions, declared before calibration. These are a relative
width allocation, not an estimate of errors or a nuisance prior:

    S(y,H) = max_j |fhat_j(y) - f_j(H)| / s_j.

Only the scoring controller sees true targets. Deployment never receives the
truth identity or stratum. At alpha=0.10 and m=80 per registered stratum the
calibration rank is exactly ceil(81*0.90)=73 (no interpolated quantile).
Use the maximum of the two within-stratum cutoffs, common to ALL deployments
and BOTH targets. Retain the inherited conservative override: any explicit
calibration abstention makes the affected stratum cutoff infinite, hence the
common cutoff infinite. No invalid or missing row is silently discarded.

    interval_j(y) = [fhat_j(y) - q*s_j, fhat_j(y) + q*s_j].

No target-equality fits, endpoint scanning or disconnected-set search is needed.
These are two-target rectangles, not certified feasible GPD sets. They may
include unattainable pairs of target values and they do not assert exclusion of
all possible physical functions outside the simulation assumptions.

## 3. Scope of coverage
For each registered stratum separately, if the fixed fitting algorithm yields
exchangeable calibration/test scores, its rank cutoff covers the joint target
error with probability at least 90% over calibration and test draws. The common
maximum is at least that stratum's cutoff and hence remains conservative.
For abstaining test fits we return the full real line; this only enlarges the
reported set. Failures still invalidate operational usefulness if too frequent.

The guarantee is NOT conditional on a particular realized calibration sample,
not a Bayesian posterior, not finite-output-conditional coverage, and not
uniform over arbitrary GPDs. It applies to the declared simulation distribution
or the specified fixed truths with repeatable observation/noise sampling.
Changing truth family, covariance, physical kinematics, missing CFF assumptions,
or using an experimental rank-deficient replica covariance requires a new
applicability analysis. Do not equate this with the experimental inference.

## 4. Counts and gate
The prospective final counts remain 80 calibration and 240 test datasets per
stratum (640 total). The score is now joint rather than separate marginal
profile scores. Old cutoffs are unusable. Freeze the entire method and all
streams BEFORE calibration. All already inspected datasets are DEVELOPMENT.

Report joint and each marginal coverage by stratum, Wilson intervals, finite
widths, abstentions, numerical failures and calibration-infinite outputs.
One-sided 95% Wilson lower bound >=0.85 is the registered joint coverage check;
failure fraction <=0.05 and finite-output fraction >=0.95 are separate gates.
Infinite intervals cannot masquerade as useful success. No optional sample-size
extension or post-test retuning. Auxiliary out-of-class tests are diagnostic,
not grounds to choose a more favorable nominal cutoff afterward.

## 5. What is authorized now
Only the bounded four-dataset DEVELOPMENT smoke on the original d000--d003
registered streams, without retuning. No final truths, final calibration,
statistical experimental fit, symbolic pruning or kernel construction. The two
current local cases and their outcomes are recorded in REPORT.md.

Reference: Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511v6,
Secs.1.1,2.3,4.1 and Appendix D.
https://arxiv.org/html/2107.07511v6
This supports rank-calibration mathematics, not new GPD results or prior-free
experimental validity. The original extraction remains neural.
