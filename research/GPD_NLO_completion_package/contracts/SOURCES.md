# Task 01 source and capability ledger

Primary formula sources were inspected at the stated versions; these are formula maps, not numerical certifications.

| ID | Versioned primary source | Applicable location | Use and caveat |
|---|---|---|---|
| R1 | [Belitsky et al., hep-ph/9908337v1](https://arxiv.org/pdf/hep-ph/9908337) | Eqs. (3), (6), (9)–(10), (14)–(17) | Spacelike kinematics and an independent hard-coefficient expression. Its skewed-distribution/charge conventions must be transformed before comparison. |
| R2 | [Belitsky, Freund and Müller, hep-ph/9912379v1](https://arxiv.org/pdf/hep-ph/9912379) | Eq. (3); LO Eqs. (73)–(78); NLO matrix and local terms Eqs. (175)–(176); vector channels Eqs. (177)–(185); section 4 | MSbar two-loop nonforward evolution source. The full-support continuation, plus-prescription, delta terms, sector and physical-x normalization require validation. |
| R3 | [Moutarde et al., 1301.3819v1](https://arxiv.org/pdf/1301.3819) | Eqs. (11)–(14) | Spacelike vector quark and gluon DVCS coefficients, factorization logarithms and i0 prescriptions. Its Eq. (13) and (14) one-loop terms already include alpha_s/(4π), color factors and charges. |
| R4 | [Diehl, hep-ph/0307382v2](https://arxiv.org/pdf/hep-ph/0307382) | Eqs. (21)–(24), (30), (192), (200); section 3.12.3 | Forward GPD conventions, valence sum-rule scope, positivity prerequisites and order/scheme caveat. |

Public reference-code inventory (tiktaalik and Gepard shallow source checkouts were inspected read-only in `/private/tmp/gpd_task01_refs`; APFEL++ was checked through its public repository metadata/README):

- [tiktaalik](https://github.com/quantom-collab/tiktaalik), `62ad1f40acfb58f483fdd9d36e1e4923db439d91`: `f90src/kernels/kernels_nlo.f90` explicitly maps R2 vector channels; `tiktaalik/matrices.py` exposes `nlo=True` kernels and `tiktaalik/evolution.py` exposes NLO evolution. Its convenience evolver can cross flavor thresholds. A fixed-`nf=3` reference must use/configure explicit kernels and common coupling. Code inspection is not a compiled or numerical comparison. The file header's “(1855)” appears to be a typo; R2's last relevant equation is (185).
- [Gepard](https://github.com/kkumer/gepard), `770f0dc1a3ea4f5d844f26e8f67a2a8e8dda681b`: `src/gepard/c1dvcs.py` has Mellin-space NLO quark/gluon Wilson coefficients. It is a prospective matched-model amplitude cross-check, not an arbitrary full-support x-grid evolution oracle; not run here.
- [APFEL++](https://github.com/vbertone/apfelxx), public HEAD `27deaec493d95bad0686b3b1c91fbbc910c891ff`: no nonforward NLO capability established. It is not accepted as the required GPD reference.

Local immutable sources: `completion_runs/00/baseline/gpd_poly_symbolic_v5` (delivered archive SHA-256 `caf5616f4d676adc88c85039d5e1fc54b886d4acaa130d7c6c6c751029c90408`), nested `vendor/gpd_lo_baseline_v4/vendor/gpd_perturbative_v3/PRESCRIPTION.md`, v5 `PLAN.json`, `completion_runs/00/source_map.json`, `completion_runs/00/input_inventory.json`, and accepted `completion_runs/00/DECISION.md`. No project-specific HEAD is asserted. Raw observation rows, model weights and checkpoints are intentionally omitted.

Evidence limit: source availability supports a proposed contract. It does not yet certify R3-to-baseline sign, the R2 full-support normalization, fixed-flavor reference interoperability, numerical convergence, uncertainty calibration or physical proton admissibility. Those remain later gated tests.
