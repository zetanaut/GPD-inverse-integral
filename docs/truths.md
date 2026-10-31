# Independent known truths, not an inverse ansatz

For positive or negative x, the bulk quark basis functions used to define a truth are

    x (1-x^2)^beta P_i^(beta,1/2)(2x^2-1) xi^(2k) exp(B t),  k<=i+1.

The gluon uses `(1-x^2)^beta P_i^(beta,-1/2)(2x^2-1)` instead. Outside |x|<=1 the
functions vanish. Z1 uses beta=2; Z2 uses beta=1.6 and additional D terms. The explicit
numbers and t slopes are in closure/truths.py and the frozen STUDY.json. This generator
expansion is **never imposed on the neural inverse**. It has nonfactorized shape/t
correlations because modes have different t slopes. The two generators have no fitted
PDF/form-factor normalization, positivity certification, or physical-flavor extraction.

For even spin N=2j+2, quark moments use x^(N-1) and gluon moments x^(N-2). In u=x^2,
Jacobi orthogonality makes radial order i>j contribute zero. Orders i<=j carry powers
at most 2i+2<=N, so all Mellin moments have the required upper degree. The D-term is
q: z(1-z^2)^2(d0+d2 z^2), g: |xi|(1-z^2)^2(d0+d2 z^2), z=x/|xi|, on |x|<|xi|.
It contributes the permitted xi^N highest power with the correct Hg=xg convention.
Even extension in xi and the corresponding odd/even x parities are explicit.

Analytical beta-function integration is independent of the numerical moment quadrature.
The completed checks cover 1,296 channel/spin/xi/t cases through spin 18. They test these
formulas, not all possible physical constraints or exact representability by a finite MLP.

The NLO pseudodata are generated with the existing strong reference discretization;
fitting uses the weak discretization. They share physical kernels, so their comparison
is an independent numerical representation check, not two independent theories/codes.
A zero-closure error below the demonstrated forward resolution must not be assigned
corresponding physical precision. No b-space transform or alternative factorization
scheme is introduced.

Theory context: H. Dutrieux et al., arXiv:2112.10528 (moment polynomiality and flexible
neural GPD modeling); E. Moffat et al., arXiv:2303.12006 (scale information and shadow
GPDs). The actual new generator coefficients and tests are defined by this package.
