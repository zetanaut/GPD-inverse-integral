# Related methods

Literature context used for the 31 October 2026 analysis snapshot. The reconstruction target, perturbative physics, inputs, and evidence distinguish the methods.

| Line of work | Relevant evidence and relation |
|---|---|
| [Dutrieux et al., EPJC 82, 252 (2022)](https://doi.org/10.1140/epjc/s10052-022-10211-5) | Neural GPD modeling in x/xi and double-distribution coordinates with theoretical constraints; pseudo-data demonstrations. |
| [Dall'Olio et al., PRD 109, 096013 (2024)](https://doi.org/10.1103/PhysRevD.109.096013) | ANN and finite-element reconstruction from partial DGLAP information using Lorentz symmetry. |
| [Riberdy et al., EPJC 84, 201 (2024)](https://doi.org/10.1140/epjc/s10052-024-12513-2) | ANN GPD prior, GK phenomenology and mock lattice data. |
| [NNGPD proceedings](https://arxiv.org/abs/2605.13000) and [expanded report](https://arxiv.org/abs/2605.06994) | The brief proceedings introduce the framework. The expanded Xu et al. report demonstrates neural learning from CFF and Mellin projections using spectator-model closure; its evidence is more substantial than the proceedings alone. |
| [Chu et al., JHEP 05 (2026) 210](https://doi.org/10.1007/JHEP05(2026)210) | Neural lattice framework extended to zero-skewness GPDs and tomography. |
| [Puhan et al., arXiv:2608.07085](https://arxiv.org/abs/2608.07085) | Pion quark/gluon neural GPD extraction using PDF replicas, form factors and lattice inputs. |
| [Chu et al., arXiv:2610.29726](https://arxiv.org/abs/2610.29726) | Neural parameterization of proton GPD moments at nonzero skewness from lattice data. |
| [GUMP1.0, PRL 135, 261903 (2025)](https://doi.org/10.1103/qct5-y7rp) | Global NLO conformal-moment extraction combining experiment and lattice; non-neural physics benchmark. |
| [Dutrieux et al., PRX accepted paper (2026)](https://doi.org/10.1103/77pv-msts) | Full-kinematic lattice GPD reconstruction with Gaussian-process regularization; non-neural field-reconstruction benchmark. |

Neural CFF fits address the upstream amplitude stage: [Kumerički et al. (2011)](https://doi.org/10.1007/JHEP07(2011)073), [Moutarde et al. (2019)](https://arxiv.org/abs/1905.02089), and [Huang and Cao (2026)](https://arxiv.org/abs/2606.09152). A [2026 neural-field conference presentation](https://indico.physi.uni-heidelberg.de/event/1277/contributions/4070/) also studies double-distribution reconstruction and prior dependence; its conference status should remain explicit.

The contribution here is the coupled reference-scale neural field with explicit NLO evolution and hard matching, free-coefficient finite-spin moment constraints, and the controlled progression from exact-input closure to noisy and experimental CFF inputs. Generating moment values are withheld in the nominal closure. The current experimental evidence is a two-member conditional propagation, rather than a precision ranking against methods with different inputs and validation questions. No claim of first neural GPD modeling, first learning from integral observables, or first NLO extraction is made.

GUMP is a non-neural global fit and a physics benchmark. Neural CFF fits operate upstream at the amplitude level; partial-DGLAP, lattice/GFF, and pion studies solve different reconstruction problems. These distinctions are preserved in the current [method](method.md) and [results](results.md).
