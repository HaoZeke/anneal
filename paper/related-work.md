# Related-work watch

Checked notes for the JCTC manuscript on one basin-hopping preset for Lennard-Jones clusters. The manuscript replaces arXiv:2302.02811. Keys are in `related-work.bib`. A key appears only when the DOI or arXiv id was read from the publisher or arXiv record.

## 2026-11-11 — GOAT

de Souza, Angewandte Chemie International Edition 64, e202500393 (2025), doi:10.1002/anie.202500393, describes GOAT, a global optimisation algorithm for molecules and atomic clusters that does not run molecular dynamics. The abstract reports organic molecules, water clusters, metal complexes, and metal nanoparticles, and a comparison with CREST. It says GOAT is in general more efficient and accurate than previous algorithms, and that the potential can be chosen freely, including hybrid DFT.

The manuscript's sealed hexamer result is the 192-seed hop at 121 of 192 against CREST at 97 of 192, plus six single draws from CREST 3.0.2. GOAT is a 2025 method that already claims a CREST comparison on water clusters, and the manuscript does not cite it.

Cite `deSouzaGoat2025`. Add a head-to-head on the water hexamer at a shared energy and gradient budget. The abstract gives no hexamer size and no seed count, so it does not contradict 121 of 192 or 97 of 192.

## 2026-11-11 — Water hexamer and octamer landscapes

Hernández-Rojas, Calvo, and Wales, Journal of Chemical Physics 164, 064301 (2026), doi:10.1063/5.0317247, survey the hexamer and the octamer by basin hopping and discrete path sampling on MB-pol and on TIP4P/2005. The two potentials give qualitatively similar landscapes. The energetic order of corresponding minima changes, including the global minimum. Hexamer isomerisation paths differ between the models. Paths between low-lying cubic octamers agree, aside from that reordering.

The manuscript uses the 1998 Wales--Hodges minima for TIP4P, prints a Cambridge TIP4P hexamer energy in the Supporting Information, and runs a separate GFN2-xTB hexamer comparison with CREST. This paper does not contradict that Jorgensen TIP4P energy. It does contradict any reading that treats the TIP4P hexamer, or a GFN2 prism, cage, book, or chair hit, as the MB-pol or TIP4P/2005 global minimum.

Cite `hernandezRojasWater2026` beside the water section. Keep the three potentials apart. Do not retarget the CREST table.

## 2026-11-11 — Adaptive parallel basin hopping for Lennard-Jones clusters

Carmona, Rodríguez-Kessler, Salazar-Colores, and Muñoz-Castro, arXiv:2510.25938 (29 October 2025), give a Python basin-hopping code for Lennard-Jones clusters. Every ten steps the step size follows the acceptance rate. Several trials are minimised together and the lowest goes to the Metropolis test. They report near-linear wall-clock speedup for up to eight concurrent minimisations. On LJ38 they recover the truncated octahedron and a C5v icosahedral isomer, and they say both appear within a few hundred iterations. They also say that no public, standardised Python basin-hopping implementation exists.

The sealed LJ38 map has the recommended hop on the fcc minimum and one basin-hopping arm 2.429 above it. A few hundred iterations is not a charged force-call budget, so that sentence does not contradict the map. The claim that no public Python basin-hopping code exists is not right: ASE and pele already ship basin hopping, and the manuscript already names ASE.

Cite `carmonaBasinHopping2025`. A head-to-head on LJ38 needs the same call ceiling. Do not treat their iteration count as a hit rate.

## 2026-11-11 — Deterministic bounds for small Lennard-Jones clusters

Kuznetsov and Sahinidis, Journal of Global Optimization (2025), doi:10.1007/s10898-025-01476-7, tighten distance bounds and prove global optimality for the putative 5-atom and 6-atom Lennard-Jones clusters with a general-purpose solver. They say a proved optimum is still out of reach beyond the smallest sizes. The record has no volume or page numbers yet.

The manuscript scores LJ38, LJ55, LJ75, LJ98, and LJ104 against the published Cambridge energies, within 0.001. This paper does not give a lower energy at those sizes, so it does not contradict the hit targets. It does contradict any wording that calls the 38-, 75-, 98-, or 104-atom structures proved optima. The manuscript calls them the published minima and does not say they are proved.

Cite `kuznetsovLennardJones2025`. Keep the hit rule as agreement with the published energy.

## 2026-11-11 — A necessary condition on published Lennard-Jones lists

Kiessling, Journal of Chemical Physics 159, 014301 (2023), doi:10.1063/5.0158931, states a necessary condition on a list of putative lowest pair energies. On a pooled list from N = 2 to N = 1610, the N = 447 entry fails. Passing the test is not a proof.

The manuscript sizes 38, 55, 75, 98, and 104 are not among the failures he names. This does not contradict the Cambridge energies used as references.

Cite `kiesslingLennardJones2023` if the reference list is extended. No change to the current energies.

## 2026-11-11 — AGOX through v3.11.1

Christiansen, Rønne, and Hammer, Journal of Chemical Physics 157, 054701 (2022), doi:10.1063/5.0094165, is the AGOX paper. The newest repository tag is v3.11.1, committed 4 February 2026. The changelog through 3.11.0 (15 August 2025) adds Ray parallel tempering in 3.8.0 on 26 August 2024, with the older implementation renamed concurrent tempering, then replica-exchange modules and universal-potential models in 3.10.0, then grand-canonical modules in 3.11.0.

The manuscript names AGOX and says the ASE, AGOX, and GMIN counts are not in a sealed kit through v1.30. These releases do not fill that kit. The parallel-tempering module is a cluster-search code beyond the spin-glass papers cited for replica exchange.

Cite `christiansenAgox2022`. Pin a head-to-head to v3.11.1, and say whether the arm is the local surrogate search or the replica-exchange search. The open count is not contradicted.

## 2026-11-11 — ASE 3.29.0

Larsen and co-workers, Journal of Physics: Condensed Matter 29, 273002 (2017), doi:10.1088/1361-648X/aa680e, is the ASE paper. Repository tag 3.29.0 is dated 21 June 2026. The manuscript still prints no ASE count.

Cite `larsenAse2017` for the code. Run the missing arm on tag 3.29.0, name basin hopping as the ASE optimiser, and charge force calls as the hop does. No printed ASE number is contradicted, because none is printed.

## 2026-11-11 — pele 0.1.2

pele is a Python energy-landscape code that continues a rewrite of GMIN, OPTIM, and PATHSAMPLE. Tag v0.1.2 is dated 12 October 2026, and v0.1.1 is dated 8 October 2026. The 0.1.2 notes add a combined Python and C++ documentation site. The README lists basin hopping, a Lennard-Jones potential, and an LJ38 global-minimum figure with a disconnectivity graph. It publishes no hit rate.

The external basin-hopping code in the manuscript is GMIN. The Supporting Information records a GMIN arm at 64 of 64 on an earlier LJ98 campaign. pele does not replace that row. It is the Python basin-hopping code to name if a new comparison cannot build the GMIN binary.

No bibliographic entry: this release has no verified DOI or arXiv id. Do not invent one. When the open GMIN head-to-head is run, record whether the arm is GMIN or pele. The 64 of 64 row is not contradicted.

## 2026-11-11 — Active delta-learning

Pitfield, Christiansen, and Hammer, Physical Chemistry Chemical Physics 28, 912--926 (2026), doi:10.1039/d5cp04302f, also arXiv:2507.18485, compare random structure search, basin hopping, GOFEE, and a replica-exchange formulation while a universal potential is corrected by a SOAP Gaussian-process delta model. The systems are silver--sulfur clusters and sulfur reconstructions of Ag(111) and Ag(100). Judged by recovery of global minima, the delta model is robust. Judged by CPU time, replica exchange is the most efficient. The foundation models are CHGNet, MACE-MP0, and MACE-MPA.

This is the current search from the AGOX authors at a DFT target. It does not report LJ38 through LJ104 or the water hexamer, so it does not contradict those counts.

Cite `pitfieldActiveDelta2026`. Do not describe the fixed Lennard-Jones preset as a substitute for that delta-learned search.

## 2026-11-11 — MARS

Suárez-Dou, Gallegos, Brünig, and Tkatchenko, Journal of Chemical Theory and Computation 22, 11125--11141 (2026), doi:10.1021/acs.jctc.6c01588, release MARS for conformer ensembles, infrared spectra, and microsolvation. Sampling combines metadynamics, multitemperature molecular dynamics, and a genetic algorithm on the machine-learned force fields SO3LR and MACE. They say semiempirical quantum chemistry has been the practical state of the art for automated conformer sampling, and that these force fields improve on it for high-level relative energies.

The hexamer comparison is against CREST at GFN2-xTB, which is that semiempirical practice. MARS does not report the hexamer hit rates, so it does not contradict 121 of 192 or 97 of 192.

Cite `suarezDouMars2026` beside CREST. Keep the result as a budget-matched hit rate against CREST 3.0.2, not as a ranking of every conformer method.

## 2026-11-11 — AIMNet2 molecular clusters

Giri and Anoop, Journal of Computational Chemistry 46, e70287 (2025), doi:10.1002/jcc.70287, search water, ammonia, hydrogen peroxide, methanol, and acetic acid clusters of one to ten molecules. The driver is a tabu interface, PyAR, on the AIMNet2 neural-network potential, used outside the organic molecules it was mostly trained on.

The manuscript's water clusters are TIP4P, with the Wales--Hodges minima as the reference, and the hexamer campaign is GFN2-xTB. AIMNet2 is a third surface.

Cite `giriAimnetClusters2025`. A head-to-head counts only if both sides relax on the same potential. No TIP4P or GFN2 energy is contradicted.

