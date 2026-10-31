# Physics and inference conventions

The [technical note](GPD_AI_Methods_Proposal_Note.pdf) gives the full mathematical account. This page connects its definitions to the software.

## Reference fields and forward physics

`gpd_reh.model.ExperimentalFiLM(t_mean, t_std)` represents three fields at $Q_0^2=2$ GeV². It receives `x[B,N]`, `xi[B]`, and `t[B]` as float64 tensors and returns `[B,3,N]`, ordered as `Sigma`, `T_rescaled`, `Hg=xg`. `T_rescaled` is the historical code label for $T=U-(D+S)/2$.

The model uses a width-24 x trunk and two residual FiLM blocks. Conditioning depends on normalized $\xi^2$ and t. Odd projection in x applies to both quark channels, even projection to the gluon. The envelope $(1-x^2)^2$ sets endpoints and imposes a representation assumption. The network has no Q² input.

The bank contract is fixed three-flavor MSbar NLO, two-loop running, $\alpha_s(Q_0^2)=0.35691594851514724$, and factorization/renormalization scales equal to Q. Ordered evolution and hard coefficients are combined into exact-coordinate rows. `ExperimentalForward` produces `[B,2]` predictions in internal real/imaginary conventions. Bank hashes, source identities, and exact `(xi,Q2)` coordinates are checked. Missing maps are errors; nearest-coordinate substitution is not used.

For the supplied Le–Keller CFF source, the preparation code maps `ReH_internal = -ReH_source`. This is an explicit source-to-internal convention. Rows already include the charge factors. Do not add a second factor or apply the sign flip to a different source without establishing its convention.

## Unknown-coefficient polynomiality

For even spins N=2,4,6,8, quark moments integrate $x^{N-1}F_q$, and gluon moments integrate $x^{N-2}H_g$. At each fixed t they are constrained to the full allowed span of $1,\xi^2,\ldots,\xi^N$. QR complements measure the component outside that span. `FixedTMoments` does not insert known-truth moment values, fix coefficients, or fit a polynomial through different t values.

These are finite-spin, finite-quadrature constraints. Positivity, all-order polynomiality, individual flavor separation, and unconstrained small-x behavior are not guaranteed by this representation.

## Objective and derivative interfaces

`ExperimentalProblem` combines the mean squared standardized ReH residual with the normalized moment penalty. The supplied experimental table does not establish a complete joint covariance. A known covariance would require its declared whitening.

The P2B study adds a 36-component correlated forward observation at 12 fixed x values. Its `ForwardInformation` uses a Cholesky factor and the common 160-row normalization. Its PDF errors are a controlled synthetic sensitivity design, not an empirical PDF error set. `map_xf_to_stored` converts LHAPDF-style x*f inputs into the stored basis at an already matched scale.

The exported PDF adapter retains the frozen study's grid, dimensions, and allocated-node guard on the full joint Jacobian. It is not a general arbitrary-grid PDF fitter. The public model, moment, row, and calibration examples work without Slurm.

The integrated Jacobian differentiates each contracted observable or moment before stacking the residual derivatives. The retained solver uses temporary parameter-space coordinates and finite budgets. Numerical eligibility, optimizer stationarity, field recovery, and statistical calibration are separate claims.

## Symbolic hypotheses and uncertainty

Moment-support compression proposes restrictions after an unrestricted fit; coefficients remain free during refitting. Such branches must be compared against the full allowed support at matched computational budgets. A small fitted coefficient is not evidence of an exact physical zero.

The direct-error calibration code binds an estimator and observation law, scores chosen functionals jointly, uses exact order-statistic cutoffs, and returns full-line intervals when finite calibration is unsupported. It does not produce a posterior or a full-field band. The 442-case reported calibration has its own truths and observation law; it cannot be transferred to the current experimental fit or the P2B targets by relabeling A/B.
