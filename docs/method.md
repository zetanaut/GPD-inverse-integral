# Physics and inference conventions

This guide connects the current mathematical formulation to the software and the saved-run comparisons. The [analysis guide](../analysis/README.md) provides executable commands; the [workflow map](workflows.md) identifies the underlying fitting and numerical sources.

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

## The inverse map and reusable physics rows

For a fixed perturbative prescription, the complex amplitude is

$$\mathcal H(\xi,t,Q^2)=\mathcal C_{\mathrm{NLO}}(\xi,Q^2)\,\mathcal U_{\mathrm{NLO}}(Q^2,Q_0^2;\xi)\,\boldsymbol F_0(\cdot,\xi,t).$$

The evolution operator is ordered in log Q². The singlet and gluon mix, while the nonsinglet follows its own channel. Hard coefficients and evolution use the same scheme, coupling, flavor count, and factorization scales. Numerical kernels and subtraction terms remain explicit in the [source map](../research/GPD_NLO_completion_package/candidate/SOURCE_MAP.md).

At fixed coordinates, this map is linear in the reference field. Its discretized action can be stored as

$$y_{ia}(\theta)=\sum_{c,k}W_{ia,ck}\,F_{0,c}(x_{ik},\xi_i,t_i;\theta),\qquad a\in\{\mathrm{Re},\mathrm{Im}\}.$$

The rows include evolution, hard matching, quadrature, and charge normalization. They are shared between observations with the same `(xi,Q2)` because the perturbative kernels do not depend on t. The field still receives each observation's own t. Differentiation acts through the neural field and the contracted observable; a learned surrogate does not replace the QCD map.

Paired subtractions and higher-precision evaluation near delicate kernel limits address numerical cancellation. A nominal/reference row comparison measures discretization effects within this implementation. Shared source kernels mean such a refinement check is not an independent QCD calculation.

## Polynomial-space residuals

At each fixed t, evaluate a moment on the collocation skewness nodes and form its vector m. If V contains the allowed columns `1, xi², ..., xi^N`, an orthonormal basis Q-perp for the complement defines a residual

$$p_{cN,j}=\frac{(Q_N^\perp)^T m_{cN,j}}{s_{cN}}.$$

The penalty sums squared residuals with the declared normalization. This is equivalent to eliminating the unknown polynomial coefficients by least squares. It constrains membership in the allowed space without supplying the true generalized form factors. The nominal configuration uses 41 skewness nodes and 13 fixed t values for four even spins. Generalized form factors at different t values are not forced to share coefficients.

## Optimization and compared runs

For real-only input, the nominal objective is

$$J(\theta)=\frac{1}{n}\sum_i\left[\frac{\operatorname{Re}\mathcal H_i(\theta)-d_i}{\sigma_i}\right]^2+\lambda_{\mathrm{poly}}J_{\mathrm{poly}}(\theta),\qquad\lambda_{\mathrm{poly}}=1.$$

The ideal complex-CFF closure includes both components and its declared normalized product-quadrature weights. Its residual scale of 0.04 is an objective scale, not an injected noise amplitude.

Fresh closure fits use 400 Adam steps at learning rate 0.003 followed by up to 60 outer constrained-solver iterations. Candidate outputs are ranked using the prescribed objective and deterministic tie-breaking, without using known GPD truth or unfitted imaginary data. Trial steps minimize a local Gauss–Newton residual model inside a trust region and the current derivative span. The solver retains the norm of affine moment-feasibility residuals, and checks the original nonlinear conditions before accepting a step.

The mean-CFF and individual-replica GPD comparisons instead start from one common fitted network and use 5, 10, or 20 additional outer iterations. The separate local-CFF fit has its own optimization history. The saved comparisons therefore describe specified reconstruction rules; they are not controlled experiments that vary only CFF smoothing.

The procedure stops at declared iteration limits. A small CFF residual, numerical feasibility, or a smooth plotted field does not certify a unique GPD or a global optimizer solution.

## Propagating CFF inputs

An input replica perturbs the complete vector once: `d_i^(r) = d_i + sigma_i z_i^(r)`, with independent standard-normal z under the declared marginal-error model. A 3–32–32–1 tanh network then smooths each vector as a function of `(xi,t,log Q2)`. The training/validation split, model seed, target standardization, epoch selection, and all-row refit procedure are repeated consistently. This creates coherent CFF functions, with correlations induced by the smoothing map.

The GPD stage fits either the pointwise mean CFF function or an individual CFF replica. Nonlinearity of the estimator means fitting the mean CFF does not generally give the mean reconstructed GPD. A finite number of optimization iterations is part of that estimator; changing it changes the response to the same supplied perturbation.

The [analysis guide](../analysis/README.md#3-metric-definitions) gives the residual, field-response, directional-response, and finite-replica mean-distance formulas used in the public scripts. The [results guide](results.md) applies those definitions to the actual comparisons.
