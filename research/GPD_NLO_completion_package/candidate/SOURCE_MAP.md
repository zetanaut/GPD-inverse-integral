# Source and numerical mapping

Physics inputs are `PILOT.json`, `ACCEPTANCE.json`, and the captured contracts.
The source equations are BFM, hep-ph/9912379v1, Eqs. (15), (73)-(79),
(174)-(185), and the full-support continuation in section 4:
https://arxiv.org/pdf/hep-ph/9912379 . Eq. (185) was visually checked against the
locally retained PDF from `completion_runs/03/source/R2_hep-ph-9912379v1.pdf`.
The added mpmath GG transcription is separately written from these equations.

For a plus kernel P, Eq. (79) gives the action

    integral P(x,y) [f(y)-f(x)] dy
      + f(x) PV integral [P(x,y)-P(y,x)] dy + d f(x).

The local counterterm d is the explicit delta coefficient. For GG it is
`-nf*(35*CA+74*CF)/108` from Eqs. (175)-(176). NS+ has the explicit term in
Eq. (177), and NS- has zero explicit delta. Regular parts act on f(y).
Our quadratures pair the two sides of y=+x and y=-x in the same physical
exclusion distance, and split at +/-xi. The independent local check computes
the transposed-kernel difference directly. The frozen callback and candidate
quadrature retain their failures; divergent sides are never separately
accepted as finite integrals.

## Diagnosed GG arithmetic

The frozen GG point functions exhibit cancellation near input y=+/-xi.
At x=.09, xi=.135, y=.135000001, the frozen total K1 GG is about 94% away
from the 50/80-digit Eq. (185) value. The exact local quadrature call at
x=.13481531067484 returned nonfinite. This is observable without any
matrix construction, representation change, or time integration.

`gg_wide.f90` promotes the captured GG rational/log/dilog expressions to the
existing four-component IEEE-binary64 arithmetic. `kernel_guard.diff` adds
only a guarded call when min(|y-xi|,|y+xi|)<1e-4*xi. It changes numerical
arithmetic, not formulas, support, plus prescription, delta terms, or physics
acceptance tolerances. Decimal rational coefficients in this branch are
formed at the wider precision. Outside it the frozen expressions are used.
The separate `gg_reference.py` implements the real log and dilogarithm
branches and checks 50 versus 80 digits. This is independent arithmetic and
an equation transcription check for GG; it does not certify every sector.

## Candidate representation

`Elements` uses continuous Lagrange polynomials on physical-x intervals;
shared interface values leave the two derivatives independent. Parity extends
positive-x fields: Sigma/T odd, Hg even. Quarks vanish at x=0 by oddness;
all fields vanish at x=1, as the frozen input does and support-preserving
finite-time evolution requires. The gluon value at zero is free. No raw kernel
is evaluated at endpoints, zero, or crossover.

At interior Gauss points we first calculate the action on every basis function,
then solve M K = B^T W A with M=B^T W B. No rows are projected onto conservation
constraints. The Gram conditioning, polynomial input reproduction, continuous
action, conservation, quadrature and representation errors must be checked
separately. A second resolution sharing this source is only numerical evidence,
not an independent physical oracle.

## Evolution and hard map

The candidate API retains the supplied ordered DOP853 evolution and two-loop
coupling, with K0*alpha/(2*pi)+K1*(alpha/(2*pi))^2. Its hard row is
[2/9*(LO+Cq1), 2/9*(LO+Cq1), Cg1], acting on [Sigma,T_rescaled,Hg].
The direct gluon coefficient already includes its charge sum and positive-x
factor. R3 hep-ph/1301.3819v1 Eqs. (12)-(14) and the accepted captured hard
source are unchanged. No legacy `reduced_amplitude` T convention is used.
The API remains unapproved until the full acceptance suite passes.

## Independent retained-order factorization identity

`analytic_scale_control.py` integrates R2 Eqs. (73)-(79) directly for the
frozen polynomial initial fields, using rational antiderivatives at 40 digits.
It does not call the Fortran bridge or use a represented evolved field. For
x>=0, the first transformed support is y>x. The reflected support is y<x
with positive sign for x<xi, and y>x with negative sign for x>xi.
After writing f(y)-f(x)=(y-x)G(y,x), the QQ pole cancels algebraically.
The even-gluon QG endpoint layer has the nonzero limit

    (K0_QG g)(xi) = nf integral_xi^1 g(y)/(y+xi)^2 dy
                    + nf*g(xi)/(2*xi).

The second term follows from delta integral_delta dy/y^2 -> 1; it is a
limit of the regular kernel, not an added counterterm. The independent
one-sided limit and direct CFF integral agree with the accepted hard
collinear-log coefficient: Re residual 1.571e-11, Im 7.78e-16. This controls
the retained one-loop scale cancellation and the full-support initial
quark/gluon normalization. It does not certify the two-loop evolution.

## Refined crossover arithmetic and local tolerance

The nearest localized output, x=.13500072961293158, also exposed nonfinite
NS+ and NS- local callbacks. Direct fixed quadrature in double arithmetic
varied by about 5e-6, exceeding their inherited local criterion. That version
was not accepted. `mp_kernel.cpp` now implements the same Eq. (177) and (185)
expressions using locally installed MPFR 4.1.0 at 192 bits. The guard covers
min(|y-xi|,|y+xi|,|y-x|,|y+x|)<1e-4*xi, preserving accurate reduced-coordinate
pole differences. Point references are separate mpmath transcriptions; 128/192
bit results agree and match those references within 2.1e-16 scaled error.
Outside the guard the captured functions remain in use. The accepted QG
software guard is unchanged. No source equation, delta, support, coupling or
acceptance threshold changes. The frozen local callback still enforces its
1e-8*max(1,abs(total)) convergence criterion. Direct paired quadrature verifies
its finite result on both sides of the nearest crossover point.

`continuous_action.geometry` uses a linear coordinate map on any tiny panel
where squaring would move nodes inside the source ABI's 1e-14 exclusion.
This covers the entire same interval; it neither clips nor removes nodes or
integration support. Failures and pre-repair source versions are retained.

## Independent spin-2 source moments

Altenbuchinger et al., arXiv:1012.4409v2, Eqs. (7),(8) and Table 1
(https://arxiv.org/pdf/1012.4409) give the MSbar singlet/nonsinglet spin-2
anomalous dimensions. Dividing the alpha/(4pi) coefficients by 2 and 4
converts them to this task's alpha/(2pi) convention. For nf=3 the [Mq,Mg]
matrices are

    K0 = [[-16/9, 1], [16/9, -1]],
    K1 = [[-2000/243, 611/108], [2000/243, -611/108]],

and the T coefficients are -16/9 and -2360/243. Here Mq=int_0^1 x Sigma,
Mg=int_0^1 Hg and MT=int_0^1 x T. These are local-operator identities at
any xi, not experimental inputs. `check_raw_moments.py` tests continuous
analytic-input actions before projection; `check_physics.py` measures the
additional represented-field error. The endpoint-zero approximation space
excludes constant/x test functions, so weak projection need not preserve
these moments exactly. No conservation constraint is imposed on the matrices.

## QG arithmetic and checkpoint reuse

`make_fast_qg.py` translates the captured QG arithmetic-expression trees to
MPFR, including constants formed at the wider precision. `FastQG` uses this
evaluator only inside the already accepted 0.02*xi guard region. Its 360 point
values (three xi values, ERBL/DGLAP/crossover) are bitwise identical to the
captured four-component software evaluator. Two complete source blocks are
also bitwise identical. Local callbacks still use the original MP bridge.
The slow pilot was deliberately checkpointed after 125 rows (1131.65 seconds).
`assemble_fast.py` copies those rows only after checking the complete geometry,
prescription, common-method hashes and arithmetic comparison; every copied row
hash and its parent key is recorded. Future rows use the faster guard. This is
an arithmetic implementation change, not an independent physical reference.

## Crossover representation and QG endpoint subtraction

`enriched.py` adds s(1-s)log(s)^k, k=1..3, on only the two cells adjoining xi.
These are free approximation coefficients; parity, endpoint values and the
polynomial span remain unchanged. Local QR conditioning and diagonal mass
equilibration change coordinates only. Strong QR collocation remains a
separate discretization control sharing the point source.

Near y=x the captured vector QG half-kernel (`KV1_qG_half`, BFM Eq. (183)) contains
TF(CF-CA)(F+Fbar)log²(|Y-X|/|X|). At Y=X,
F+Fbar=-1/[X(1-X)], while F+Fbar+1/[Y(1-Y)]=0, so
the explicitly linear-log terms have zero leading coefficient. Here
X=(1+x/xi)/2 and Y=(1+y/xi)/2. Applying the reflected half, its rho
support, nf=3 and the physical Jacobian gives c=2.5/(xi²-x²):
for t=y-x>0, c log²(t/(xi+x)); additionally for x>xi,
c log²(t/(x-xi)); for x<xi and t<0, -c log²((-t)/(xi-x)).
The endpoint functions are subtracted times f(x) on the existing paired
radius r and added back exactly using
int_0^r log²(t/L)dt = r[log²(r/L)-2log(r/L)+2].
There is no new delta or change of physics. `qg_subtraction.py` implements
this identity; `correct_qg_rows.py` applies the resulting rank-one numerical
quadrature correction to saved rows, without new kernel evaluations.

The diagnostic at x=xi+5.7841e-9 changes the n48-to96 analytic-input
QG action difference from 4.87e-2 to 8.59e-6. At x=.09 and .19 the
subtracted n48-to96 differences are below 2.2e-11. The original quadrature
failures and all native rows are retained. These checks establish a numerical
integration repair; full forward/reference acceptance is reported separately.

## Independent coefficient convolution of evolved fields

`hard_reference.py` evaluates the source coefficient formulas from the accepted
independent contour implementation. It subtracts a full-support smooth polynomial
with the evolved crossover value, integrates that polynomial along a deformed
contour, and integrates the remaining represented-field residual on each real
interval with adaptive SciPy quadrature. It uses neither nominal hard rows,
their Gauss nodes, nor their constant-mode integrals. Its analytic subtraction
is distinct from the nominal constant subtraction. The pilot agrees within
2.5e-10 per CFF component with no quadrature warnings. The field representation
is shared; this is an independent coefficient-convolution control, not another
evolution reference. Two adaptive tolerances and all warnings are recorded.

`check_structure.py` checks full-source reflection in every K0/K1 sector,
physical field parity, and ordered evolution composition with the appropriate
two-loop intermediate boundary coupling. The pilot source parity is exact in
the sampled comparisons and composition changes the CFF by1.55e-14.

The numerical error budget adds candidate/reference discretization changes,
observed discrepancy, independent source/output/hard/time quadrature changes,
and an arithmetic allowance. It is a conservative estimate, not a rigorous
error bound. Its scope is the frozen six development fixtures.

## Independent evolved moments and D-term support

`check_evolved_moments.py` evolves the published MSbar spin-2 quark/gluon
matrix and the NS-plus spin-2 anomalous dimension together with the two-loop
coupling in a four-variable ODE. It compares these local-operator moments
with integrals of each saved evolved field, for both inputs and each weak/strong
degree. No field generator enters the moment reference. The same check records
DGLAP leakage and the imaginary CFF of the isolated added D-term against their
zero reference, without division by zero. No moment or support is projected
back onto the candidate fields.
