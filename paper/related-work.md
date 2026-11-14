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

## 2026-11-11 — Complementary energy landscapes

Slavensky, Christiansen, and Hammer, Journal of Chemical Physics 159, 024123 (2023), doi:10.1063/5.0156218, propose candidates by relaxing them on a deliberately incomplete machine-learned landscape, then returning to the true surface. They report a new global minimum of the olivine cluster (Mg2SiO4)4, and a test on reduced rutile SnO2(110).

The manuscript's moves are geometric: a surface penalty and Thompson sampling on quench depth, not a surrogate landscape. This paper does not report a Lennard-Jones or water-hexamer energy, so it does not contradict those claims.

Cite `slavenskyComplementary2023` as the AGOX candidate generator. No change to the preset.

## 2026-11-11 — Universal potentials as Bayesian priors

Lyngby, Larsen, and Jacobsen, Physical Review Materials 8, 123802 (2024), doi:10.1103/PhysRevMaterials.8.123802, also arXiv:2408.15590, take a universal machine-learning potential as the prior of a Gaussian process, so the process learns only the difference from the target surface. They report faster location of the global optimum when that potential behaves well, on bulk crystals, surfaces, and a cluster.

The Lennard-Jones preset uses no prior, and this paper does not re-rank LJ38 through LJ104.

Cite `lyngbyUniversalPrior2024`. No contradiction. The 2026 Digital Discovery paper that uses the name BEACON is a different method.

## 2026-11-11 — Fractional existence

Larsen, Kaappa, Vishart, Bligaard, and Jacobsen, Physical Review B 107, 214101 (2023), doi:10.1103/PhysRevB.107.214101, add atoms of fractional existence so a search can move atoms past barriers. Positions may stay on a lattice while the existence variables move, or both may move. On copper clusters the existence variables speed up the harder sizes.

The hard sizes in the manuscript are Lennard-Jones 38, 75, and 98, not copper. This does not contradict the fixed preset.

Cite `larsenFractional2023`. A head-to-head would need one shared potential.

## 2026-11-11 — Population basin hopping on LJ20, LJ30, and LJ40

Baioletti, Santucci, and Tomassini, Journal of Global Optimization 89, 803--832 (2024), doi:10.1007/s10898-024-01373-5, also arXiv:2403.05877, compare basin hopping and two population variants with differential evolution, particle swarm optimisation, and CMA-ES. On Lennard-Jones and Morse clusters of 20, 30, and 40 atoms, their population basin hopping does well on Lennard-Jones and is slower on Morse. They say basin hopping beats CMA-ES on these cluster problems.

The manuscript's population is a fixed-size resampled set of replicas with a Good--Turing census, not replacement of the worst member. Their sizes stop at 40, so LJ38 is nearby and LJ75, LJ98, and LJ104 are absent. This does not contradict the withheld LJ75 hit count, or the LJ38 and LJ98 counts removed from the article.

Cite `baiolettiBasinHopping2024` as a population basin-hopping baseline. Do not import the 20, 30, and 40 results into the article.

## 2026-11-11 — Grand-canonical search with an on-the-fly potential

Quinlivan Domínguez, Christiansen, Neyman, Hammer, and Bruix, Journal of Chemical Physics 165, 034115 (2026), doi:10.1063/5.0302876, train a SOAP Gaussian process during a grand-canonical search and rank candidates by an ab initio Gibbs energy. The abstract says the method reproduces selected literature examples across stoichiometries.

AGOX 3.11.0 added grand-canonical modules in the same period. The manuscript's preset is fixed-N Lennard-Jones and fixed-size water. No fixed-N energy is contradicted.

Cite `quinlivanGrandCanonical2026`, so a fixed-N hit is not described as a search over composition.

## 2026-11-11 — CREST 3.0.2 still has no user seed

The newest numbered CREST release is still v3.0.2, published 25 August 2024, the build named in the manuscript. The development line was updated on 16 June 2026. It adds g-xTB refinement, dated July 2025, and input-parser maintenance. The input parser on v3.0.2 and on that later line has no seed keyword. The dynamics initialiser either draws an unseeded generator or, when a same-random flag is set, writes a seed of all ones.

This does not contradict the sentence that CREST 3.0.2 exposes no seed. Keep the hexamer table pinned to 3.0.2. Do not rerun it on the unreleased line unless the question is the g-xTB refiner itself.

The CREST paper is already in `refs.bib` as `prachtCrestProgramExploration2024`. No new entry.

## 2026-11-11 — A different BEACON

Tang, Chakrabarty, and Paulson, Digital Discovery 5, 3116--3127 (2026), doi:10.1039/d6dd00123h, use the name BEACON for a Bayesian novelty search aimed at diverse outcomes rather than one optimum. The abstract is not about atomic clusters, basin hopping, or water.

Do not cite this paper as an atomic-structure code. The structure-search papers from this watch are `lyngbyUniversalPrior2024` and `larsenFractional2023`, and neither uses this title. No anneal claim is contradicted.

## 2026-11-11 — Thomson clusters of thousands of particles

Amore and Wales, arXiv:2610.08122 (6 October 2026), search low-energy minima for thousands of particles on a sphere. Symmetric or known structures seed a run, and a new minimum is reused at neighbouring sizes. On the Thomson problem they improve many earlier solutions, especially at large size. They say the defect patterns may carry over to other spherical packings, from atomistic to mesoscopic scales.

The manuscript keeps one move library at every Lennard-Jones size and does not install a lattice template that already names the minimum. This paper propagates known structures from one size to the next. The potential is the Thomson problem, not Lennard-Jones, so the new energies do not contradict the Cambridge hit targets or the sealed maps.

Cite `amoreThomson2026`. Do not put the Thomson energies in the Lennard-Jones table.

## 2026-11-11 — A GFN2-xTB reference set is not the low-energy set

Vahediahmar, Zweig, Zhang, and Azizi, arXiv:2610.12523 (8 October 2026), ask whether the GEOM-DRUGS reference ensemble is complete on the GFN2-xTB surface used to build it. After screening, 2.3 to 7.0 percent of conformers from five generators remain outside the 0.75 Å reference radius. On 15 percent of molecules, with an interval from 8 to 22 percent, some generator finds an intact structure below the entire released ensemble. A complementary CREST search matches the released minimum for 65 of 69 molecules, and for 11 of those 69 it lies more than 0.5 kcal/mol below the whole released ensemble.

The manuscript scores the hexamer by an IRA match to a prism, cage, book, or chair reoptimised at GFN2-xTB, and it compares polished energies with CREST 3.0.2. This preprint does not treat the water hexamer, so it does not contradict 121 of 192, 97 of 192, or the six polished gaps. It does contradict any reading in which a CREST ensemble, or a GFN2-xTB reference set, is the complete low-energy answer.

Cite `vahediahmarGeomDrugs2026` beside the CREST section. Keep the hexamer result as a motif hit rate and a polished gap.

## 2026-11-11 — Seven atoms are proved, and the diameter bound is tighter

Shen, arXiv:2611.09925 (23 September 2026), reports a proof that the pentagonal bipyramid is the unique global minimiser of the seven-particle Lennard-Jones energy, up to rigid motion and relabelling. The same letter treats the seven-charge Thomson problem.

The manuscript said Kuznetsov and Sahinidis prove optima only for five and six atoms. That "only" does not survive this letter. The proof stops at seven atoms. It gives no energy at 38, 55, 75, 98, or 104, and it does not prove those sizes.

Lecomte, arXiv:2607.09555 (10 July 2026), certifies a one-layer improvement of the Kuznetsov--Sahinidis diameter bound at 92 sizes from 5 to 200 atoms. The note says the improvement resolves no open global-optimisation case. It also says the diameter box is not what limits a solver at six atoms or fewer, and that no solver is run at 38 atoms or larger, the sizes the refinement would affect.

Cite `shenSevenParticle2026` and `lecomteDiameter2026` beside Kuznetsov. Drop "only" from the sentence. Keep the hit targets as the published Cambridge energies.

## 2026-11-11 — Searches that recover the published Lennard-Jones minima

Durand, Journal of Optimization Theory and Applications 210, 53 (2026), doi:10.1007/s10957-026-03085-0, chooses one atom with an interval-arithmetic heuristic and moves it inside a population search. The abstract says the method recovers every best-known energy from 6 to 150 atoms, finds the current best-known energies at 200, 250, and 300 atoms, and finds no new optimum.

The manuscript sizes 38, 75, 98, and 104 all sit inside that 6-to-150 range. This paper does not contradict the Cambridge energies. It says a 2026 search found nothing lower. A head-to-head has to use the same force-call budget. Do not copy those sizes into the hit table.

Chakraborty and Anoop, Journal of Computational Chemistry 47, e70484 (2026), doi:10.1002/jcc.70484, put a PyAR-style tabu initialiser in front of a bonobo optimiser. Without that initialiser the search is unreliable past 13 atoms, and local relaxation with similarity filtering reaches the LJ38 minimum in 6 of 25 runs. With the initialiser, LJ38 is found in all 20 runs. From LJ2 through LJ100 they report 943 successes in 1030 runs, and the failures that remain are LJ75, LJ76, and LJ77.

The article no longer prints an LJ38 or LJ98 hit count, and the LJ75 count is withheld, so these rates contradict no printed number. They do not report an energy below the Cambridge values. They do ask for a head-to-head on LJ38 and on LJ75 through LJ77, at one relaxation budget, against the lean preset rather than against a template that already names the minimum.

Sozykin and co-workers, arXiv:2601.18592 (26 January 2026), encode a cluster as a tensor train and search Lennard-Jones clusters from 5 to 45 atoms. They single out LJ38 and report success rates of 16 percent and 33 percent, depending on the encoding. They describe the hit as the known LJ38 minimum. The search stops at 45 atoms, so LJ75, LJ98, and LJ104 are absent. A head-to-head on LJ38 has to charge the evaluations the hop charges. Do not read a tensor-train success rate as a force-call hit rate.

Gochitashvili, Meyers, Wang, and Kolmogorov, Physical Chemistry Chemical Physics 27, 25636--25647 (2025), doi:10.1039/d5cp02412a, also arXiv:2507.13791, test hyperspatial relaxation and a TETRIS initialiser on LJ37, LJ38, LJ39, LJ47, and LJ55. They take the Wales--Doye structures as the ground states. An extra dimension raises the chance of hitting those states, and the TETRIS initialiser helps more than the extra dimension once the cost is counted. They report no lower Lennard-Jones energy. LJ75, LJ98, and LJ104 are not in the set. A head-to-head is LJ38 and LJ55.

Schärf and Kühne, Theoretical Chemistry Accounts 145, 70 (2026), doi:10.1007/s00214-026-03321-w, benchmark minima-hopping escape directions on Lennard-Jones clusters of 60 to 74 particles. Softened random directions with L-BFGS remain the lowest-cost baseline. A dense BFGS history can cut repeated local exploration, and one deterministic mode oversamples a funnel. The set stops at 74 particles, so it does not reach LJ75, and the move is a short molecular-dynamics escape, not the preset. No sealed basin-hopping count is contradicted. A comparison, if it is run, is minima hopping on 60 to 74 atoms.

Kiessling and Wales, Molecular Physics 124, e2590148 (2025), doi:10.1080/00268976.2025.2590148, also arXiv:2511.15008, prove that a stationary Lennard-Jones configuration has a nearest-neighbour distance no larger than the dimer equilibrium distance, with equality for a global minimum only at 2, 3, and 4 atoms. They conjecture that every global minimum has a nearest neighbour longer than 1, in units where the dimer distance is the sixth root of two. On the Cambridge list through 150 atoms, and on an extension through 1000, the shortest distance they record is at 923 atoms. The sizes 38, 55, 75, 98, and 104 are not named as failures. This does not replace the Cambridge energies.

Cite `durandLennardJones2026`, `chakrabortyBonobo2026`, `sozykinTensorTrain2026`, `gochitashviliGosh2025`, `schaerfMinimaHopping2026`, and `kiesslingPairDistance2025`. Do not import their rates into the article.

## 2026-11-11 — Basin hopping away from the Lennard-Jones preset

Carmona, Arroyo-Valdez, Muñoz-Castro, and Rodríguez-Kessler, Journal of Nanoparticle Research 28, 197 (2026), doi:10.1007/s11051-026-06714-x, run adaptive basin hopping on ORCA density-functional surfaces, with species exchange and a hierarchical refinement. This is not the Lennard-Jones preprint already cited as `carmonaBasinHopping2025`. It does not report LJ38 through LJ104 or the water hexamer, so it does not contradict those counts, and it does not fill the open ASE, AGOX, and GMIN kit. A head-to-head would be on one shared density-functional surface.

Shukla, ChemRxiv (30 October 2026), doi:10.26434/chemrxiv.15011323/v1, builds Stillinger--Weber silicon clusters by trilateration at a fixed bond length. At an equal number of relaxations that initialiser reaches lower minima than a basin-hopping code with single-atom, collective, and surface moves, at every reported size from 32 to 60 atoms. The manuscript preset is Lennard-Jones and TIP4P, not silicon, so those gaps do not contradict the sealed maps. A head-to-head is the silicon cluster at a shared relaxation budget.

Cite `carmonaAbhDft2026` and `shuklaSilicon2026`.

## 2026-11-11 — GOFEE on LJ55 and LJ75

Slavensky and Hammer, Journal of Chemical Physics 161, 014713 (2024), doi:10.1063/5.0207801, also arXiv:2407.13471, put a spectral-graph and Voronoi classifier in front of the sampling step in GOFEE. The classifier drops structures from basins the search has already exploited. They test it on two-dimensional pyroxene, on olivine (Mg2SiO4)4, on Au12, and on the Lennard-Jones clusters LJ55 and LJ75. The Lennard-Jones targets are the known global minima. On LJ55, with the filter, they say the search finds the minimum almost every time within 2000 single-point evaluations. On LJ75 they turn the surrogate off and relax every candidate in the Lennard-Jones potential, because the surrogate was taking the time. Without the filter they call that modified search really difficult. With the filter the chance of finding the minimum rises. The LJ75 curves count local optimisations, not single-point evaluations.

The article prints no LJ55 hit count, and the LJ75 count is withheld, so these curves contradict no printed number. They give no energy below the Cambridge values. A head-to-head on LJ55 and on LJ75 has to use one relaxation budget. Their single-point count is not the force-call count the hop charges.

Cite `slavenskyClassifier2024`. Keep the GOFEE curves out of the basin-hopping hit table.

## 2026-11-11 — Population annealing on the LJ38 funnels

Kowaguchi and Hukushima, Journal of Chemical Physics 164, 214110 (2026), doi:10.1063/5.0333811, also arXiv:2603.04157, estimate the free energy of LJ38 by population annealing. They treat the global minimum as the face-centred cubic truncated octahedron, in competition with an icosahedral funnel. The low-temperature energy extrapolates to the known minimum, which they write as -173.92. Quenches of configurations drawn below the crossover relax to that truncated octahedron.

The Doye Lennard-Jones table, read on 11 November 2026, still lists -173.928427 for the 38-atom Oh minimum. The manuscript's LJ38 map puts the recommended hop on that same truncated octahedron. This paper does not report a lower potential energy, and it does not report a force-call hit rate. The population is a free-energy sample, not a seed count.

Cite `kowaguchiLj382026`. Leave the sealed LJ38 map as a search record.

## 2026-11-11 — Nested sampling of the 7-atom and 36-atom clusters

Maillard, Finocchi, Godinho, and Trassinelli, Physical Sciences Forum 12, 8 (2025), doi:10.3390/psf2025012008, also arXiv:2501.11370, use nested sampling in nested_fit to estimate the partition function of the 7-atom and 36-atom Lennard-Jones clusters. They recover the known phase transitions. For the 36-atom cluster that includes the low-temperature solid-solid transition, provided the live-point count is high enough. They do not report a new global minimum.

Seven atoms are the pentagonal bipyramid already proved by Shen. The sizes 38, 55, 75, 98, and 104 are absent. This does not contradict that proof or the Cambridge hit targets.

Cite `maillardNested2025`. A nested-sampling likelihood count is not a hop hit rate.

## 2026-11-11 — Extra printed digits through twenty atoms

Willnecker and Moodley, arXiv:2411.18087 (27 November 2024), run an adaptive genetic algorithm on Lennard-Jones clusters from 2 to 20 atoms. The abstract says the energies are more accurate than the literature. The table matches a basin-hopping reference through six decimals. The longer print is the same energies at machine precision. At ten atoms the extra digits sit above the six-decimal reference, so they are not a lower minimum. The search stops at 20 atoms.

This does not contradict the Cambridge energies at 38 atoms and above. It does not call for a head-to-head at the manuscript sizes.

Cite `willneckerGenetic2024`. Read "more accurate" as a longer print of the known energies.

## 2026-11-11 — Journal version of the Thomson search

Amore and Wales, Journal of Chemical Physics 165, 164104 (22 October 2026), doi:10.1063/5.0355867, is the journal version of the preprint already filed as `amoreThomson2026`. The abstract is the same search on the Thomson problem. The Lennard-Jones hit targets are unchanged.

The key `amoreThomson2026` now carries the journal DOI beside the arXiv id.

## 2026-11-11 — Lower energies on two Cambridge tables that are not Lennard-Jones

Olivier, doi:10.5281/zenodo.23775350 (29 October 2026), deposits six clusters and reports energies below the minima listed in the Cambridge Cluster Database for the modified Dzugutov potential at 74, 134, 177, and 186 atoms, and for the Z1 potential at 54 and 68 atoms. The deposit says nothing is claimed about global optimality. The Doye copies of those two tables, read on 11 November 2026, still show the listed energies the deposit quotes. Modified Dzugutov 74 is -180.982763, 134 is -343.386067, 177 is -462.579581, and 186 is -487.786110. Z1 54 is -146.333923 and 68 is -185.275450. Both pages were last modified on 4 March 2025. This note does not recompute his geometries.

The Lennard-Jones table on the same site was last modified the same day. It still lists the published minima the manuscript uses: -173.928427 at 38 atoms, -279.248470 at 55, -397.492331 at 75, -543.665361 at 98, and -582.086642 at 104. Those six deposited energies are modified Dzugutov and Z1 energies. They do not contradict the Lennard-Jones hit targets.

Cite `olivierPairTables2026`. Keep them out of the Lennard-Jones table.

## 2026-11-11 — An AIMNet2 genetic search through ten molecules

Zhang, Cheng, and Zhang, Journal of Computational Chemistry 47, e70389 (2026), doi:10.1002/jcc.70389, release GMMLP. An improved genetic algorithm searches on AIMNet2, trained at ωB97M-D3/def2-TZVPP. The benchmarks are nine molecular clusters with one to ten molecules: formaldehyde, methylamine, methanol, methane, water, sulfuric acid, nitric acid, ammonia, and urea. Across those benchmarks the abstract counts 9869 isomers.

Giri and Anoop already search water clusters of that size on AIMNet2, with a tabu driver rather than this genetic algorithm. The manuscript's water surfaces are TIP4P and GFN2-xTB. GMMLP does not report those energies, so it does not contradict them. A comparison with the sealed hexamer counts only if both sides relax on the same potential.

Cite `zhangGmmlp2026` beside `giriAimnetClusters2025`.

## 2026-11-11 — One argon atom on the water hexamer isomers

Rock, Clothier, and Tschumper, ACS Physical Chemistry Au (2026), doi:10.1021/acsphyschemau.6c00129, optimise Ar(H2O)6. The water topologies are prism, cage, two books, bag, glove, ring, and two boats. Geometry and frequencies are MP2, and the relative energies are compared at CCSD(T). At every level they report, one argon atom changes the spacing of those topologies, and the harmonic zero-point correction is nearly the same with and without the argon, so the reordering is electronic. The record has no volume or pages yet.

The manuscript scores bare GFN2-xTB motif hits, and it keeps a separate TIP4P reference. This paper does not report 121 of 192 or 97 of 192. An argon-tagged MP2 order is a different surface from GFN2-xTB.

Cite `rockArgonHexamer2026` beside the water section. Leave the CREST table on the bare hexamer.

## 2026-11-11 — Hydrophobic guests around six water molecules

Baruah, ACS Omega (2026), doi:10.1021/acsomega.6c10192, optimises assemblies of six water molecules at ωB97X with the def2-TZV basis, with hydrophobic guests present. The guests stabilise an open book, a closed bag, a prism, and a double envelope. Weak contacts, including C-H···O, O-H···π, and O-H···F-C, move the hydrogen-bond pattern. The record has no volume or pages yet. This is a guest-stabilised density-functional assembly, not a bare-cluster global optimisation.

It does not contradict 121 of 192, 97 of 192, or the Wales-Hodges TIP4P geometries.

Cite `baruahWaterSix2026`. Keep those guest-stabilised geometries apart from the GFN2 motifs.

## 2026-11-11 — A local tightening is not a new basin

Doust Mohammadi, ChemRxiv (26 October 2026), doi:10.26434/chemrxiv.15011046/v1, separates two ways a geometry optimisation can fail. Inside one basin, on the stiff closed-shell cases in the benchmarks he reviews, bond lengths settle to a few tenths of a picometre and the coordinate system mostly changes the cost. A jump into another basin, including a wrong cluster isomer, can move the energy by more than a change of functional or basis. He does not report a water-hexamer hit rate or a Lennard-Jones energy.

The manuscript's polish is a local relaxation of a structure already scored as a motif. This note does not contradict the polished gaps. A lower energy in another basin would be another isomer.

Cite `doustMohammadiBasins2026` beside the polish. A tighter threshold inside one basin is not a new motif.

## 2026-11-11 — Surface-supported basin hopping

Zhang, Cheng, Zhang, and Dai, Journal of Chemical Theory and Computation 22, 7436--7455 (2026), doi:10.1021/acs.jctc.6c00328, release SGMSC. A refined basin hop on embedded-atom potentials searches surface-supported clusters: Cu7 on Zr, W, Ti, Ta, Pt, Pd, Pb, Ni, Mo, Mg, Fe, Cu, Co, Au, Al, and Ag, Pt10 on Au(100), and Au20 on Pt(110). On the cases they checked, the embedded-atom minimum stayed the lowest isomer after a density-functional relaxation. The supported Pt10 and Au20 minima differ from the gas-phase structures.

The manuscript preset is a free Lennard-Jones or water cluster. SGMSC does not report those energies, and it does not fill the open ASE, AGOX, and GMIN kit. A head-to-head would be a supported cluster on one shared potential.

Cite `zhangSgmsc2026`. A free-cluster hit is not a supported-cluster hit.

## 2026-11-11 — Minima hopping on top of ASE

Krummenacher, Gubler, Finkler, Huber, Sommer-Jörgensen, and Goedecker, SoftwareX 25, 101632 (2024), doi:10.1016/j.softx.2024.101632, implement minima hopping on the Atomic Simulation Environment, for clusters and for bulk. A corrigendum is SoftwareX 26, 101645 (2024), doi:10.1016/j.softx.2024.101645. The current code deposit is doi:10.5281/zenodo.24035625, published 7 November 2026. The deposit says any ASE calculator can be attached, and that periodic runs can move the cell. It publishes no hit rate.

The missing ASE arm in the manuscript is basin hopping on tag 3.29.0. This package is minima hopping. It does not fill that kit, and it does not contradict the open count, because no ASE number is printed.

Cite `krummenacherMinimaHopping2024` for the paper and `krummenacherAseMh2026` for the November deposit. A comparison that uses this code has to be named as minima hopping, and it has to charge force calls.

## 2026-11-11 — A MACE basin hop on a doped boron cluster

Rodríguez-Kessler, arXiv:2609.31533 (25 September 2026), searches B18Y2 by basin hopping on a pretrained MACE potential and then relaxes the candidate with density-functional theory. He reports a double-ring global minimum, with one yttrium above the boron framework and one below. The note gives no Lennard-Jones energy and no water energy.

This is not the adaptive basin hop on ORCA surfaces already filed as `carmonaAbhDft2026`. It does not contradict the sealed maps, and it does not fill the open kit. A head-to-head would be this boron cluster on one shared potential.

Cite `rodriguezKesslerYttrium2026`.

## 2026-11-11 — Release tags checked again

ASE tag 3.29.0, dated 21 June 2026, is still the newest tag. Master moved on 11 November 2026, with a fix for a fixed cell that has no positional degrees of freedom and a fix for copying atoms.info. Those commits are not a release. AGOX tag v3.11.1, dated 4 February 2026, is still the newest tag. The dev branch moved on 5 May 2026 so the constraint import accepts ASE 3.28, and that commit is not a tag. CREST's newest numbered release is still v3.0.2. The master tip is still the 16 June 2026 CMake fix. pele is still v0.1.2, dated 12 October 2026, and that tag is still the tip. The GMIN page was last modified on 17 March 2022 and still points at the group git image.

Pin the open arms to ASE 3.29.0, AGOX v3.11.1, CREST 3.0.2, and pele 0.1.2. The May AGOX commit and the November ASE commits are not those arms. No bibliographic entry for this check: a tag date is not a DOI.

## 2026-11-11 — Two names that are not the cluster search

Nozari, Abdi, and Szmelter-Jarosz, arXiv:2503.02331 (4 March 2025), name a bio-inspired metaheuristic the Goat Optimization Algorithm. The tests are analytic benchmark functions. It is not de Souza's GOAT, and it does not search a molecule or an atomic cluster.

Lauga, Molinari, and Vaiter, arXiv:2605.18364 (18 May 2026), propose proximal basin hopping. The abstract states a high-probability guarantee for a finite sample of their proximal scheme. The tests they report are synthetic functions and fits of deep-learning scaling laws. The note does not treat a Lennard-Jones cluster. It does not prove a minimum at 38 atoms or beyond, so the sentence that leaves those sizes unproved still stands.

Cite `nozariGoatAlgorithm2025` and `laugaProximal2026` only to keep the names apart. The molecular GOAT citation stays `deSouzaGoat2025`.

## 2026-11-11 — A geometric argument for symmetric Lennard-Jones minima

Schneider, arXiv:2603.00011 (27 January 2026), argues that a landscape invariant under a finite group pushes the global minimum onto a symmetric stratum, because the fully asymmetric part of the real quotient is metrically rare. For LJ13 he reads the Cambridge database as about 78 percent of the documented local minima having a trivial stabiliser, while the global minimum is the regular icosahedron. The figure of about 83 percent of global minima from one to 150 atoms having a non-trivial point group is his citation of Wales and Doye, not a new census of the database.

The manuscript already says a global minimum is often more symmetric than the minima around it. This preprint does not replace the Cambridge energies, and it does not change the symmetrisation preset.

Cite `schneiderSymmetry2026`. No contradiction, and no new head-to-head.

## 2026-11-11 — Known minima through thirteen atoms

Barrón Romero, Pesquisa e Docência em Ciências Exatas e Naturais, pages 32--47 (27 June 2026), doi:10.37572/edart_2706261093, enumerates clusters of 4 to 13 particles on Lennard-Jones and Morse wells. The abstract says a proved optimum is known only for 2, 3, and 4 particles. The same abstract then says a 2022 argument fixes the 13-particle icosahedron, and that the known candidate geometries from 5 to 13 particles are the true global minima of those wells. The chapter points at the Cambridge Cluster Database as the source of those models. The 13-atom Lennard-Jones ranking in the chapter is a count of near-neighbour contacts. It is not an energy from the Cambridge table.

The manuscript cites proofs for five and six atoms, and Shen's proof for seven. The abstract sentence that stops at four particles does not survive those proofs. The enumeration stops at 13 particles. It gives no energy at 38, 55, 75, 98, or 104. The Cambridge hit targets stand, and Lecomte's statement that 38 atoms and larger stay unproved stands.

Cite `barronRomeroLennardJones2026` beside the small-size proofs. Keep the five-, six-, and seven-atom sentences. Leave the chapter's contact sum out of the Lennard-Jones table. No head-to-head at the manuscript sizes.

## 2026-11-11 — A TIP4P generator ranked on another surface

Bhatt, Chacko, Bijewar, and Nagare, arXiv:2411.00754 (1 November 2024), generate water clusters of 1 to 20 molecules with an artificial bee colony on TIP4P, then relax the lowest candidates at HCTH/407 with the 6-311++G(d,p) basis. They compare those geometries with coordinates from the Cambridge Cluster Database. At their density functional they call stacked pentagonal and cage-like structures lower than the stacked cubes in the database for 11 molecules and for 13 through 20 molecules. The hexamer is outside that list. At the same level they say a prism relaxes into a book.

The manuscript's water reference is the 1998 Wales--Hodges TIP4P minima, and the sealed hexamer table is GFN2-xTB. This preprint reports neither a TIP4P energy below those Wales--Hodges minima nor the sealed hit counts. An HCTH book is a third surface.

Cite `bhattWaterClusters2024` beside the water section. Keep TIP4P, GFN2-xTB, and HCTH apart. Leave the CREST table on GFN2-xTB. A comparison with this search counts only if both sides relax at HCTH.

## 2026-11-11 — RPA relaxation reorders the larger water clusters

Tahir, Shang, Li, and Ren, arXiv:2404.10492 (16 April 2024), relax water clusters with random-phase-approximation forces. For the five low-energy isomers of the 25-molecule cluster, the energy order on RPA geometries differs from the order on geometries relaxed by lower-level theories. They say the water hexamer does not show that reordering. The record has no DOI.

The manuscript's hexamer is TIP4P and GFN2-xTB. This preprint leaves those two surfaces alone. The 25-molecule reordering stays out of the hexamer table.

Cite `tahirRpaWater2024`. The hexamer comparison remains the one already sealed.

## 2026-11-11 — Lennard-Jones benchmarks that score something else

Yallup and Handley, arXiv:2609.29413 (24 September 2026), benchmark Neural Transport Nested Sampling on Lennard-Jones clusters of 13 and 55 particles. The score is a Wasserstein error against a Markov-chain reference, and a partition function from one run. They recover the known phase structure. They report no new global minimum and no force-call hit rate.

Chatbipho, Yang, Wexler, and Pártay, Journal of Chemical Physics 163, 174701 (3 November 2025), doi:10.1063/5.0283538, also arXiv:2506.01295, run nested sampling of adsorbates on the truncated-octahedral 38-atom Lennard-Jones cluster. The cluster is the substrate. The search is the adsorbed layer. They report no lower cluster energy.

Garg, Deep, and Kulkarni, Engineering Optimization 58, 1000--1031 (21 May 2025), doi:10.1080/0305215x.2025.2496941, put the Lennard-Jones potential problem among the tests of a hybrid metaheuristic. The Crossref record gives the title, the pages, and no energy and no cluster size.

The Cambridge hit targets stand. A Wasserstein error, an adsorption partition function, and a Lennard-Jones test with no named size are outside the hop's hit rate. Open a head-to-head only when a paper names a size and charges force calls.

Cite `yallupNested2026`, `chatbiphoAdsorbate2025`, and `gargLennardJones2025`.

## 2026-11-11 — Further basin-hopping searches of doped boron clusters

Rodríguez-Kessler, arXiv:2601.01261 (3 January 2026), arXiv:2603.02428 (2 March 2026), and arXiv:2604.25908 (28 April 2026), report basin-hopping global minima for B18Ca2, B7Ca2, and B18Ag2. Each minimum is a boron ring or a double ring with the dopant atoms on opposite sides. None of the three reports a Lennard-Jones energy or a water energy.

These are the same kind of search as `rodriguezKesslerYttrium2026`. The sealed maps stand, and the open ASE, AGOX, and GMIN kit stays open. A head-to-head would be one of these clusters on one shared potential. The Lennard-Jones and water counts do not require it.

Cite `rodriguezKesslerB18ca2026`, `rodriguezKesslerB7ca2026`, and `rodriguezKesslerB18ag2026`.

## 2026-11-11 — xtb 6.8.0 moves GFN2 total energies

The grimme-lab xtb release v6.8.0 is dated 5 November 2026. The release page says absolute total energies can differ from earlier releases by as much as a millihartree, especially for large systems, and that relative energies usually move very little. Against xtb 6.7.1, the GMTKN55 WTMAD2 for GFN0-xTB through GFN2-xTB changes by less than 0.0002 kcal/mol. The same note gives a 1988-atom nylon complex whose GFN2-xTB total energy changes by about 0.0013 hartree and whose binding energy changes by 0.041 kcal/mol. The notes also mention constraint-algorithm work that includes minimum hopping. They publish no cluster hit rate.

The hexamer single-point logs print xtb 6.7.1, commit edcfbbe, and those single points reproduce the polished energies. The article said the polish was xtb at GFN2 and did not name the release. It now names 6.7.1. CREST's newest numbered release is still v3.0.2, so that build does not absorb 6.8.0. ASE tag 3.29.0, AGOX tag v3.11.1, pele v0.1.2, and the GMIN page of 17 March 2022 are unchanged.

0.0002 kcal/mol is about \(3\times 10^{-7}\) Eh, and 0.041 kcal/mol is about \(6.5\times 10^{-5}\) Eh. The printed gaps sit at \(1\times 10^{-7}\) Eh. The release does not show that those six gaps survive on 6.8.0. It also does not recompute them, so it does not contradict the 6.7.1 gaps, 121 of 192, or 97 of 192. It does contradict any reading in which the gaps are properties of GFN2 with no release attached. Keep the table on 6.7.1. A later check can re-polish the same finals on 6.8.0. Do not start a new CREST search for this release.

No bibliographic entry: the release has no DOI.

## 2026-11-11 — g-xTB is a different tight-binding method

Froitzheim, Müller, Hansen, and Grimme, ChemRxiv (23 June 2025), doi:10.26434/chemrxiv-2025-bjxvt, present g-xTB. The abstract says it aims at the ωB97M-V functional with a large basis, keeps tight-binding speed, and is parameterized through the actinides. The record has no journal volume.

The CREST development line already noted here adds a g-xTB refiner. This preprint is that method. It is not GFN2-xTB, and it does not report the hexamer hit rates, so it does not contradict 121 of 192, 97 of 192, or the 6.7.1 gaps. Cite `froitzheimGxtb2025` beside the CREST section. Leave the hexamer table on GFN2-xTB with xtb 6.7.1.

## 2026-11-11 — COMPLAX keeps a rigid solute

Lauria and Maranzana, Journal of Chemical Information and Modeling 66, 13050--13055 (2026), doi:10.1021/acs.jcim.6c02105, release COMPLAX for targeted microsolvation. The abstract says it is unlike an unconstrained global optimiser: a rigid core holds the solute, and a Fibonacci lattice places the solvent. The comparison is CREST, the ORCA Solvator, OGOLEM, and ABCluster, followed by a density-functional relaxation at ωB97X-D3/def2-SVP. Relative to the COMPLAX minimum, CREST lies 4.6 kcal/mol higher for an organolithium in tetrahydrofuran and matches it for pyridine with two waters. ABCluster lies 4.7 kcal/mol lower for acetate in water. They say CREST put that water on the pyridine π system rather than the nitrogen lone pair. The same section names GOAT and QCG among unconstrained searches.

The manuscript's hexamer is a bare GFN2-xTB cluster, scored by motif. COMPLAX does not report 121 of 192, 97 of 192, or the polished gaps. A head-to-head would be one of their microsolvated solutes, and they say the score is fidelity to the input solute. Do not retarget the hexamer table. The GOAT comparison already asked for stays the bare hexamer at one energy and gradient budget.

Cite `lauriaComplax2026`.

## 2026-11-11 — A symmetry bias in the local optimisation of LJ38 and LJ75

Tao, Shao, and Zhu, arXiv:2401.13953 (25 January 2024), add a fingerprint energy so a local optimisation prefers high-symmetry structures. They test silicon, silicon carbide, and Lennard-Jones clusters. For LJ38 and LJ75 they name the double funnels. Two hundred random starts come from CALYPSO. Energies and forces use the ASE calculator, and the trajectories use FIRE. The fingerprint search shifts the density of local minima toward lower energy. The note gives no Cambridge energy and no force-call hit rate.

The missing ASE arm is basin hopping on tag 3.29.0. This preprint is a biased local relaxation, and it does not fill that arm. It does not contradict a printed ASE count, because none is printed, and it gives no energy below the Cambridge values. A comparison, if it is run, is LJ38 and LJ75 at one relaxation budget, and it has to be named as this fingerprint relaxation.

Cite `taoFingerprint2024`. Keep it out of the basin-hopping hit table.

## 2026-11-11 — A firefly search that reports a gain on basin hopping

Lee, Stampfl, and Soon, AI Agent 2, no. 1 (31 March 2026), doi:10.20517/aiagent.2025.13, release NARA. A firefly search keeps several structures and trains on the fly. For the "8" surface oxide on Cu(111) they say it is more efficient than basin hopping. For gold clusters one run recovers planar and non-planar structures. The record has no page numbers. They do not report LJ38 through LJ104 or the water hexamer.

This does not contradict those counts, and it does not fill the open ASE, AGOX, and GMIN kit. A head-to-head would be the surface oxide, or the gold cluster, on one shared potential.

Cite `leeNara2026`.

## 2026-11-11 — Minima hopping on ionized gold

Safa, Rahmatizad Khajehpasha, and Goedecker, arXiv:2602.15646 (17 February 2026), search gold clusters of 22 to 100 atoms. Neutral clusters in the upper part of that range prefer compact structures. They say a large enough positive charge makes planar structures the ground states, and that finite temperature stabilizes the planar forms further. The search is minima hopping on a machine-learned potential, with a charge correction because the potential was not trained on ions.

The note gives no Lennard-Jones energy and no water energy. It does not contradict the sealed maps. A head-to-head would be one of these ions on the same potential. The Lennard-Jones and water counts do not require it.

Cite `safaGoldIons2026`.

## 2026-11-11 — Journal version of the tensor-train search

Sozykin and co-workers, Physical Review B 113, 224111 (22 June 2026), doi:10.1103/sv5r-mryz, is the journal version of the preprint already filed as `sozykinTensorTrain2026`. The arXiv record still has no journal-ref line. The DOI, the volume, and the article number were read from the Crossref record, and that record names the same eight authors. The Lennard-Jones success rates in the earlier note are unchanged.

The key `sozykinTensorTrain2026` now carries the journal DOI beside the arXiv id.

## 2026-11-12 — Two packages for giant cluster global minima

Cheng, Dai, Zhang, and Zhang, Journal of Molecular Modeling 32, 414 (13 October 2026), doi:10.1007/s00894-026-07000-2, release GMUPC for fullerene-like uranyl peroxide clusters from U20 to U10260. The workflow builds symmetric structures from Wyckoff positions, perturbs isomers, drops duplicates, and searches with particle swarm optimisation, a genetic algorithm, standard basin hopping, and an improved basin hop. The generator prefers high-symmetry cages. A machine-learning potential carries the broad search.

The same authors, Structural Chemistry (23 October 2026), doi:10.1007/s11224-026-02938-8, release GMSGC for fullerene-like gadolinium--oxygen clusters from 50 to 25650 gadolinium atoms. The record has no volume or pages yet. The package again combines symmetric generation, a random perturbation, duplicate removal, standard and improved basin hopping, a genetic algorithm, and particle swarm optimisation. Relativistic density-functional calculations follow. They say the giant icosahedral shells they report obey the isolated-pentagon rule.

Zhang, Cheng, and Zhang already search molecular clusters through ten molecules on AIMNet2, filed as `zhangGmmlp2026`. GMUPC and GMSGC are later packages, on uranyl peroxide and on gadolinium oxide. Neither reports a Lennard-Jones energy or a water energy, so neither contradicts the Cambridge hit targets, 121 of 192, or 97 of 192. A head-to-head would be one of their clusters, on the potential that package uses. The symmetry-seeded cage is not the manuscript's move library.

Cite `chengGmupc2026` and `chengGmscg2026`.

## 2026-11-12 — A reinforcement-learning search against the Birmingham genetic algorithm

Mubeen, Farooq, and Chen, AIP Advances 16, 095003 (1 September 2026), doi:10.1063/5.0344056, search transition-metal nanoclusters with proximal policy optimisation on an embedded-atom potential. The clusters are Ag1Ni12, Ag6Ni7, Ag7Ni6, Ag12Ni1, and Pt6Co7. The abstract says the search finds lower-energy minima than a Birmingham cluster genetic algorithm, and that the changes are largest for Ag6Ni7 and for Pt6Co7. A later atomic-cluster-expansion relaxation is used to compare those embedded-atom minima with effective-medium ones. The abstract gives no Lennard-Jones energy.

The manuscript names cut-and-splice as the move in the Birmingham pool methods. It prints no genetic-algorithm hit rate. These bimetallic energies do not contradict the Cambridge Lennard-Jones targets or the sealed water counts. A head-to-head would be one of these clusters on the embedded-atom surface.

Cite `mubeenDrLClusters2026`.

## 2026-11-12 — Basin hopping away from the Lennard-Jones sizes

Park, Pillai, Pracht, and Wales, Journal of Chemical Physics 165, 054310 (6 August 2026), doi:10.1063/5.0343219, map n-alkanes from n-C14H30 to n-C20H42 with the Cambridge energy-landscape software. Basin hopping finds the low minima, and discrete path sampling connects the families. On their many-body surface the lowest family changes with chain length: all-trans at 14 carbons, a hairpin from 15 to 17, and a compact fold from 18. DLPNO-CCSD(T1)-F12 calculations put the hairpin lower again for the longer chains. The compact folds remain relevant. The chains are not Lennard-Jones clusters, and the note gives no water energy.

Du, Kim, and Warren, arXiv:2608.27680 (27 August 2026), test four nonlocal perturbations, each followed by a local relaxation, as basin-hopping moves on amorphous alumina. Moving one oxygen so that two aluminium coordination numbers change reaches low-energy configurations in fewer trials than a conventional Monte Carlo trajectory. The moves are not the manuscript's six geometric moves. The note gives no Lennard-Jones energy.

Xu and co-workers, arXiv:2608.17149 (17 August 2026), globally optimise supported Zn3 clusters on Cu(111) with a machine-learning potential. Compact triangles are favoured among the zero-temperature minima. Free-energy calculations shift the weight toward extended linear forms as the temperature rises, and the shift is mostly entropic. The cluster is supported zinc, not Lennard-Jones or water. A zero-temperature hit is not that finite-temperature ensemble.

None of the three contradicts the Cambridge hit targets, 121 of 192, or 97 of 192. The GMIN rows stay the LJ98 and LJ104 comparison. Cite `parkAlkaneLandscape2026`, `duBasinHopping2026`, and `xuZincClusters2026`.

## 2026-11-12 — CREST, a GOAT geometry search, and a Bayesian conformer search

Moshchenkov, Ryzhako, Rozov, Otlyotov, and Minenkov, Journal of Computational Chemistry 47, e70520 (14 October 2026), doi:10.1002/jcc.70520, compare an exhaustive conformer search with CREST metadynamics for 13 gas-phase forms of clinical drugs. GFN1-xTB and GFN2-xTB complement each other on which conformer is the lowest. The GFN force field does worse, because its surface and the density-functional surface differ. Conformational entropies from the systematic search, and from the standard CREST entropy mode with the single-point Hessian off, agree reasonably with a molecular-dynamics estimate. Turning the single-point Hessian on, after a density-functional refinement, makes the CREST entropy too large for some species. The set is not the water hexamer, so the note does not contradict 121 of 192, 97 of 192, or the polished gaps.

Masys, Jonauskas, and Rinkevičius, Lithuanian Journal of Physics 66, no. 3 (11 September 2026), doi:10.3952/physics.2026.66.3.3, apply an algorithm they call GOAT to dangling bonds on hydroxylated and aminated nanodiamonds. The abstract reports more than five million geometry optimisations. The landing page does not name ORCA and does not name de Souza. The search is a geometry optimisation of a nanodiamond, so it is not the analytic Goat Optimization Algorithm already filed as `nozariGoatAlgorithm2025`. It does not report a water cluster. The missing hexamer comparison with de Souza's GOAT stays open.

Lewis-Atwell, Farrar, and Grayson, Chem Catalysis, 101905 (November 2026), doi:10.1016/j.checat.2026.101905, use Bayesian optimisation to locate the minimum-energy conformer. The record has no volume. The abstract says the method needs fewer high-level geometry optimisations than a screen based on a force field, on GFN2-xTB, or on single-point energies, and that a few tens of high-level optimisations are typical. Transition states are treated in the same way. The abstract does not report the water hexamer, so it does not contradict 121 of 192 or 97 of 192. A GFN2-xTB screen is not that high-level minimum. A comparison with this search is one of their catalytic transition states.

Cite `moshchenkovCrestEntropy2026`, `masysGoatNanodiamond2026`, and `lewisAtwellConformers2026`.

## 2026-11-12 — Structure searches that are not the cluster preset

Song, Shi, She, Sun, Cao, and Xuan, Crystal Growth and Design (5 November 2026), doi:10.1021/acs.cgd.6c00574, also arXiv:2604.08972, release selective random structure search. The record has no volume or pages yet. Symmetry-constrained random crystals are filtered for diversity and relaxed with universal machine-learning potentials. The tests are SiC, BaPtAs, layered NbSe2, and GaN nanotubes. They recover known ground states and report further polymorphs. The search is a crystal, not a Lennard-Jones cluster and not a water cluster.

Agada, Wang, and Biswas, arXiv:2609.20592 (17 September 2026), use the name CrystalMO-TuRBO for multi-objective trust-region Bayesian optimisation of a joint X-ray and neutron crystal refinement. The test is single-crystal Ho2Ti2O7. The objective is a diffraction residual. The manuscript's trust-region arm is the local Bayesian optimiser measured on the hop and not kept. This preprint does not run that arm, and it does not report a Lennard-Jones hit rate.

Cite `songSelectiveSearch2026` and `agadaCrystalTurbo2026`. Keep CrystalMO-TuRBO apart from the hop's trust-region arm.

## 2026-11-12 — Cross-field optimisers checked against the printed arms

Nedanovski, Nenov, and Pilev, arXiv:2606.15830 (14 June 2026), seed CMA-ES restarts from a nearest-better clustering of a Sobol sample. On the composition functions of the CEC suites they report, the fixed-budget target coverage is higher than BIPOP-CMA-ES. The tests are analytic benchmark functions. They do not run LJ98. The cross-field section prints a BIPOP arm at 6 of 6 on LJ98, with median 1324848 calls, against retry rows at 6 of 6 with median 813591. Those counts are this hop's arm. The preprint does not contradict them. Do not replace the Hansen citation with this restart.

Wolfson, arXiv:2606.26164 (24 June 2026), and Swarm and Evolutionary Computation 110, 102574 (November 2026), doi:10.1016/j.swevo.2026.102574, release ChiSao, a GPU-resident multimodal optimiser. The preprint names basin hopping among sequential baselines and reports a wall-clock speedup against it on analytic functions where the methods it compares all succeed. The suite is analytic. It is not a Lennard-Jones force-call hit rate, and it does not contradict a printed basin-hopping count.

Miyata, Arai, and Takabe, Physical Review E (6 November 2026), doi:10.1103/664f-wjpj, also arXiv:2601.13542, choose replica-exchange temperatures by a gradient step on the variance of the acceptance rates. The record has no volume or pages yet. The tests are the Ising model, the XY model, and the Edwards--Anderson model. The manuscript says parallel tempering exchanges states on a temperature ladder. This paper does not report a cluster hit rate, so that sentence stands.

Cite `nedanovskiCmaEs2026`, `wolfsonChisao2026`, and `miyataReplicaExchange2026`.

## 2026-11-12 — A planar bound, and two Cambridge tables that are not Lennard-Jones

Colbrook, doi:10.5281/zenodo.23835694 (31 October 2026), proves a lower bound on the classical planar Lennard-Jones energy. For every finite set of distinct points in the plane, the energy is at least the particle count times the energy per particle of the optimally scaled triangular lattice, and finite triangular patches meet that bound in the thermodynamic limit. The potential is two-dimensional. The note gives no energy at 38, 55, 75, 98, or 104 atoms in three dimensions. Lecomte's statement that every case of 38 atoms and larger stays unproved is a statement about those three-dimensional clusters. This deposit does not change it.

A deposit under the name mizopl, doi:10.5281/zenodo.23466237 (19 October 2026), claims eight putative global minima for 55-atom copper--silver Gupta nanoalloys below entries in the Cambridge Cluster Database. The largest gap it names is Cu39Ag16, with a difference of -1.18984440 eV. This note does not recompute the geometries. The potential is the Gupta potential, not Lennard-Jones. The Cambridge Lennard-Jones energies used as hit targets are a different table.

Meadows, doi:10.5281/zenodo.24111878 (10 November 2026), deposits candidate improvements to packing and energy tables. One file is a Thomson configuration at 436 particles. The deposit says it lies 0.0348 below the minimum over the live tables it compares, including the Cambridge Cluster Database and Wayne Deeter's table. This note does not recompute that configuration. The potential is the Thomson problem. Amore and Wales search that problem at large size. Their citation is not a claim that the Cambridge Thomson table is frozen, and this deposit is not a Lennard-Jones energy.

Cite `colbrookPlanarLennardJones2026`, `mizoplGuptaRecords2026`, and `meadowsThomson2026`. Keep all three out of the Lennard-Jones hit table.

## 2026-11-12 — What a layer relaxation can certify

Lecomte, arXiv:2607.12134 (13 July 2026), asks what a Kuznetsov--Sahinidis layer relaxation can certify. Above a singleton-layer cost of \(-1/2\), and for the geometric Lennard-Jones system down through that endpoint, the certified deficit is at most of order \(\sqrt{N}\). Farther below the threshold a valid relaxation of the Lennard-Jones system has a linear deficit. Once the singleton cost passes the stability constant, the relaxation certifies nothing. The six-atom minimiser enters only as a lower bound on that constant. The letter does not name 38, 55, 75, 98, or 104 atoms, and it proves no minimum at those sizes.

The same author, doi:10.5281/zenodo.21280557 (9 July 2026), shows that a one-dimensional layer bound with a summable kernel cannot certify a diameter of order \(N^{1/3}\). Inside that class the best correction is of order \(\sqrt{N}\). The 13 July letter cites this deposit beside the diameter-bound preprint already filed as `lecomteDiameter2026`.

The manuscript says that diameter preprint leaves every case of 38 atoms and larger unproved. These two notes do not prove those cases, so that sentence stands. They give no energy below the Cambridge values. They do not warrant a head-to-head.

Cite `lecomteCertification2026` and `lecomteScalingCeiling2026` beside Kuznetsov. Keep the hit targets as the Cambridge energies.

The Wales copy of the table from 3 to 150 atoms, read on 12 November 2026, was last modified on 22 February 2026. It still lists \(-173.928427\) at 38 atoms, \(-279.248470\) at 55, \(-397.492331\) at 75, \(-543.665361\) at 98, and \(-582.086642\) at 104. The Doye copy of that table is still last modified on 4 March 2025. The February timestamp is not a new minimum.

## 2026-11-12 — Lennard-Jones coordinates through sixteen atoms, in several dimensions

Varas, Acevedo, Carrasco, Rogan, and Valdivia, doi:10.5281/zenodo.19826810 (27 April 2026), deposit minimum-energy coordinates for the Lennard-Jones potential in more than one dimension. The directories run from 9 to 16 atoms. The files are labelled by dimension, and the highest label in the archive is dimension 15, on the 16-atom set. This note does not recompute the energies.

The sizes 38, 55, 75, 98, and 104 are absent. The deposit does not contradict the Cambridge hit targets, and it does not warrant a head-to-head at those sizes.

Cite `varasLennardJones2026`. Keep the coordinates out of the Lennard-Jones table.

## 2026-11-12 — Fullerene, platinum, and alloy searches

Chen, Liang, Chen, and Wang, Journal of Chemical Theory and Computation 22, 6218--6226 (2026), doi:10.1021/acs.jctc.6c00329, search \(({\rm C}_{60})_N\) with a fuzzy global optimisation that keeps three-body terms. On the PPR surface they extend the search from 105 to 320 molecules. Against the literature from 11 to 105 molecules they report 14 new lower energies on the pair potential and 3 on the potential that includes three-body terms. With the three-body terms, the structures at 85, 87, 94, 98, 100, 153, 224, 231, 253, 269, and 285 molecules change from decahedral to close-packed.

The 98 in that list counts \({\rm C}_{60}\) molecules. It is not the 98-atom Lennard-Jones cluster. The paper gives no Cambridge energy. It does not contradict the hit targets, and it does not warrant a head-to-head on LJ38 through LJ104. A comparison, if it is run, is \(({\rm C}_{60})_N\) on the PPR potential.

Li, Yang, and Guo, Physical Chemistry Chemical Physics 28, 16106--16120 (2026), doi:10.1039/d6cp00865h, search platinum clusters from 6 to 50 atoms with a deep potential and a hybrid differential-evolution algorithm. The abstract says the structures are more stable than structures taken from the S-C empirical potential, and it names magic sizes 10, 17, 30, 36, 42, and 43. The potential is platinum. No sealed Lennard-Jones count is contradicted. A head-to-head would be one of these platinum clusters on their deep potential, not LJ38 through LJ104.

Wu, Tang, He, Shi, and Lin, Physica Scripta 101, 345911 (2026), doi:10.1088/1402-4896/ae980f, optimise PtCo clusters at 13 and 38 atoms. The 13-atom clusters stay icosahedral. The 38-atom clusters change motif with composition. That 38-atom cluster is PtCo, not the Lennard-Jones truncated octahedron. The abstract gives no Cambridge energy. It does not contradict the LJ38 map, and it does not warrant a head-to-head on LJ38.

Tang, Zhang, Shi, Chen, and Wu, Journal of Nanoparticle Research 28, 188 (2026), doi:10.1007/s11051-026-06711-0, optimise Au--Cu clusters on the Gupta potential at 13, 19, 23, and 38 atoms. The landing page says the search reaches the 38-atom global minimum in 70 percent of runs, and that twenty runs at 55 atoms have a best-energy standard deviation of 0.002 eV. It says the search outperforms basin hopping, a genetic algorithm, and plain differential evolution. It also says the structures are a consistency check: a copper core, a gold surface, and the truncated octahedron at 38 atoms. Those sizes are Au--Cu on the Gupta potential. They do not contradict the Cambridge Lennard-Jones energies or a printed Lennard-Jones hit count. They do warrant a head-to-head on that Gupta surface, against the basin-hopping baseline the paper names, and not on LJ38 or LJ55.

Cite `chenFullerene2026`, `liPlatinum2026`, `wuPtco2026`, and `tangAucur2026`.

## 2026-11-12 — Glycine, a zeolite basin hop, and a CHGNet correction

Roth First, Anabi, Licht, Major, and Toker, Journal of Chemical Physics 165, 174314 (4 November 2026), doi:10.1063/5.0350852, search protonated glycine clusters through seven molecules. The mass spectrum has magic numbers at three and at six. The hexamer is the first size at which a fully zwitterionic structure is preferred. The trimer's abundance is kinetic. This hexamer is glycine. The note gives no GFN2-xTB energy and no TIP4P energy. It does not contradict 121 of 192 or 97 of 192, and it does not warrant a head-to-head on the CREST table.

Bengtsson, Hewitt, Saha, Sarwar, Collis, Erlebach, Grajciar, and Heard, Digital Discovery (2026), doi:10.1039/d6dd00356g, also ChemRxiv (5 February 2026), doi:10.26434/chemrxiv.10001878/v2, place aluminium in chabazite, mordenite, and zeolite RHO by basin hopping on a neural-network potential. The journal record has no volume or pages yet. Low-energy aluminium arrangements are a few percent of the structures they count. Random placements sit high relative to the basin-hopping minima. The search is a zeolite framework. It does not fill the open ASE, AGOX, and GMIN kit, and it does not contradict the sealed maps. A head-to-head would be one of these zeolites on their potential. It does not warrant a head-to-head on the Lennard-Jones preset.

Christiansen and Hammer, Journal of Chemical Physics 162, 184701 (8 May 2025), doi:10.1063/5.0268264, correct a universal potential by a Gaussian process on the potential's own embedding. In a global structure optimisation, CHGNet describes the "8" surface oxide on Cu(111) and prefers a structure the abstract calls more favorable than one in the density-functional literature. Sulfur adlayers on Cu(111), Ag(111), and Au(111) need the correction. MACE-MP0, SevenNet-0, and ORB-v2-only-MPtrj show the same kind of error. This precedes the silver--sulfur delta search filed as `pitfieldActiveDelta2026`, and the oxide is the system in `leeNara2026`. The note gives no Lennard-Jones energy and no water energy. It does not fill the open kit, and it does not warrant a head-to-head on LJ38 through LJ104 or on the hexamer.

Cite `rothGlycine2026`, `bengtssonZeolite2026`, `bengtssonZeoliteChemrxiv2026`, and `christiansenDelta2025`. The ChemRxiv key is the 5 February 2026 preprint of the Digital Discovery record.

## 2026-11-12 — BHSMS

Zhang, Cheng, Zhang, and Dai, Physical Chemistry Chemical Physics 28, 26341--26362 (4 November 2026), doi:10.1039/d6cp00638h, release BHSMS. The abstract says the package searches the global minimum of atomic and molecular clusters. The pieces it names are an improved basin hop, a structure relaxation, a similarity filter, a planarity check, and a constraint on molecular rotation. The interfaces it names include xTB, ORCA, and VASP. The tests it names are Au6, Au20, B8, B40, C20, C60, Be2O2, Si2O2, (CO2)3, (CO2)6, (NH3)4, (NH3)5, (H2O)5, (H2O)8, (HNO3)3(NH3)3, and (HNO3)3(NH3)3(H2O)2. The abstract gives no energy and no hit rate.

These authors already have the surface-supported package filed as `zhangSgmsc2026`, the AIMNet2 genetic search filed as `zhangGmmlp2026`, and the giant-cluster packages filed as `chengGmupc2026` and `chengGmscg2026`. BHSMS is the free-cluster basin hop. It does not name LJ38 through LJ104, and it does not name the water hexamer, so it does not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. It does not fill the open ASE, AGOX, and GMIN kit. A comparison on (H2O)5, (H2O)8, or Au20 would need one shared potential, and the abstract does not name that potential. It does not warrant a head-to-head on the sealed Lennard-Jones maps or on the CREST hexamer table.

Cite `zhangBhsms2026`.

## 2026-11-12 — Release tags checked on 12 November

ASE tag 3.29.0, dated 21 June 2026, is still the newest tag. The master tip is still the 11 November 2026 merge. AGOX tag v3.11.1, dated 4 February 2026, is still the newest tag. The dev tip is still the 5 May 2026 ASE 3.28 import fix. The GOFEE page on the documentation site is titled as AGOX 3.11.2. The changelog on dev has an unreleased note and no 3.11.2 section, and the tag list has no 3.11.2. That banner is not a release. CREST's newest numbered release is still v3.0.2, and the master tip is still the 16 June 2026 CMake fix. pele is still v0.1.2, and that tag is still the tip, dated 12 October 2026. xtb v6.8.0, dated 5 November 2026, is still the newest release.

Pin the open arms to ASE 3.29.0, AGOX v3.11.1, CREST 3.0.2, and pele 0.1.2. Do not move the AGOX arm to 3.11.2. No bibliographic entry for this check: a tag date is not a DOI.

## 2026-11-12 — A reinforcement-learning search of Pd6Mo7

Ahmad, Bin Qamar, Farooq, and Chen, Next Materials 13, 103127 (October 2026), doi:10.1016/j.nxmate.2026.103127, search Pd6Mo7 with proximal policy optimisation on an embedded-atom potential. The abstract says no earlier global-optimisation study had resolved that cluster. Candidates are re-ranked with a MACE potential and with density-functional calculations. Those calculations find the sixth-ranked embedded-atom configuration to be the lowest structure. A molecular-dynamics run at 500 K keeps that structure compact for 200 ps.

Farooq and Chen are also authors of the embedded-atom search already filed as `mubeenDrLClusters2026`. That paper is Ag--Ni and Pt--Co. This one is Pd6Mo7. It gives no Lennard-Jones energy and no water energy, so it does not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. The embedded-atom order is not the density-functional order. A head-to-head would be this cluster on one shared potential. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST hexamer table.

Cite `ahmadPdmo2026`.

## 2026-11-12 — An agent-rewritten local optimiser in ASE

Woo, arXiv:2611.13143 (9 November 2026), runs an agent loop on algorithms in the Atomic Simulation Environment. One target rewrites an implementation so that it runs faster and returns the same results. The other redesigns the method so that a relaxation uses fewer force evaluations. On problems held out from the agent, the note says the best optimisers need 3.1 to 3.4 times fewer force evaluations than L-BFGS for a structure relaxation, and 4.6 to 17 times fewer than a standard climbing-image nudged elastic band. The record has no DOI.

The missing ASE arm is basin hopping on tag 3.29.0. This preprint is a local relaxation and a band calculation. It does not run that arm, and it publishes no basin-hopping hit rate. ASE tag 3.29.0, checked again on 12 November 2026, is still the newest tag, and the master tip is still the 11 November 2026 merge. AGOX is still v3.11.1, CREST is still v3.0.2 with the 16 June 2026 master tip, pele is still v0.1.2, and xtb is still v6.8.0. The preprint does not contradict the open count, because no ASE number is printed. A comparison that uses these optimisers has to be named as this local relaxation. Its force-call ratio is not the hop's hit rate.

Cite `wooAseOptimizers2026`. Keep the open arm on stock basin hopping in ASE 3.29.0.

## 2026-11-12 — GrowPAL

López-Castro, Ortiz-Chi, and Merino, Journal of Chemical Theory and Computation 20, 4939--4948 (2024), doi:10.1021/acs.jctc.4c00365, introduce GrowPAL. An interstitial addition builds a seed of \(n+1\) atoms from a cluster of \(n\) atoms. The abstract says the Lennard-Jones tests run through 80 atoms, and that the search finds the LJ38 minimum and the partially icosahedral LJ69 minimum with fewer local optimisations than the methods they compare. A deconstruction of the growth path names the forebear of a minimum. On Sutton-Chen clusters from 5 to 80 atoms the abstract reports three new lowest-energy forms. On boron clusters from 8 to 15 atoms it reports the known minima.

The manuscript sizes 38 and 55 sit inside that range, and so does 75. The abstract does not name LJ75, LJ98, or LJ104, and it gives no energy below the Cambridge values. A count of local optimisations is not the force-call hit rate. The three new forms are Sutton-Chen energies. They do not contradict the Cambridge Lennard-Jones targets, 121 of 192, or 97 of 192.

The repository that continues the search says the 2024 tests showed the growth strategy through 74 atoms, and that a later diversity-preserving selection recovers the reported Lennard-Jones minima through 130 atoms, including LJ75 through LJ78, LJ98, and LJ102 through LJ104. That statement is the repository, not the 2024 abstract. It names the reported minima, so it does not contradict the hit targets. A head-to-head is LJ38, LJ75, LJ98, and LJ104, at one relaxation budget, against this growth search. The growth step is not the manuscript's move library. No bibliographic entry for the repository: it has no verified DOI or arXiv id.

Cite `lopezCastroGrowpal2024`. Keep the Sutton-Chen forms out of the Lennard-Jones table. The open arms stay on ASE 3.29.0, AGOX v3.11.1, CREST 3.0.2, and pele 0.1.2. Those tags were unchanged on this check.

## 2026-11-12 — Carbon dioxide clusters, platinum--nickel, and a silver--copper chart

Ramteke and Jose, Journal of Chemical Information and Modeling 66, 2597--2609 (24 February 2026), doi:10.1021/acs.jcim.5c02782, build molecular clusters with a method they call LOGOS. A network proposes a binding site on a parent, and the daughter is relaxed in a molecular-electrostatic-potential feature space. The benchmark is carbon dioxide, with fewer than 30 molecules. The abstract says the structures match minimum-energy structures already reported.

The manuscript's molecular cluster is the water hexamer, on TIP4P and on GFN2-xTB. LOGOS does not report those energies, and it does not name LJ38 through LJ104, so it does not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. GrowPAL already grows a cluster by adding atoms. LOGOS grows a molecular cluster by adding a molecule, on carbon dioxide. A comparison counts only if both sides relax on that carbon-dioxide surface. It does not warrant a head-to-head on the sealed maps or on the CREST table.

Usman and Chen, Journal of Nanoparticle Research 28, 182 (27 June 2026), doi:10.1007/s11051-026-06698-8, search platinum--nickel nanoclusters with proximal policy optimisation on an embedded-atom potential. MACE reoptimises the candidates. Density-functional energies are used to judge the global minimum. A molecular-dynamics run at 500 K follows. Chen is an author of `mubeenDrLClusters2026` and of `ahmadPdmo2026`. Those searches are silver--nickel, platinum--cobalt, and Pd6Mo7. This one is platinum--nickel. The abstract says a classical structure search converges slowly. It gives no basin-hopping hit rate and no Lennard-Jones energy. It does not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. A head-to-head would be one of these platinum--nickel clusters on the embedded-atom surface. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST hexamer table.

Settem, Telari, Tinti, Ferrando, and Giacomello, Nanoscale 18, 19082--19093 (2026), doi:10.1039/d6nr01194b, also arXiv:2603.23442 (24 March 2026), map 38-atom silver--copper nanoalloys. Parallel tempering molecular dynamics samples the equilibrium distribution, and a machine-learning map sorts the configurations. The chart is a finite-temperature survey across composition. The abstract gives no zero-temperature energy. The arXiv record has no DOI line and no journal-ref line. The DOI, the volume, and the pages were read from the Crossref record, and that record names the same five authors. The 38 in that chart counts silver and copper atoms. It is not the Lennard-Jones truncated octahedron. The mizopl deposit already on file claims lower Gupta energies at 55 atoms. This paper does not replace that note, and it does not contradict the Cambridge Lennard-Jones energies. A finite-temperature silver--copper chart is not the sealed LJ38 map.

Cite `ramtekeLogos2026`, `usmanPtni2026`, and `settemAgcu2026`.

## 2026-11-12 — Basin hopping on packings, and shells on a narrow well

Kravatskiy, Khrulkov, and Oseledets, arXiv:2602.10233 (10 February 2026), evolve an initialiser, a local improvement, and a perturbation, and iterate those operators as basin hopping. The tests are hexagon packing, an autocorrelation inequality, and spherical codes. On spherical codes the abstract says the program lowers the best-known maximum cosine for the majority of 90 codes, with a relative improvement up to 2.4 percent. The potential is not Lennard-Jones. A spherical code is not the Thomson problem filed as `amoreThomson2026`. The note does not prove a minimum at 38 atoms or beyond, so the sentence that leaves those sizes unproved still stands.

Barrón Romero, Pesquisa e Docência em Ciências Exatas e Naturais, pages 48--65 (27 June 2026), doi:10.37572/edart_2706261094, is the next chapter after `barronRomeroLennardJones2026`. The landing page calls the structures nano vehicles: shells of particles around a small core, on wells whose valley is narrower than a Lennard-Jones 12-6 well. The page says the shells remain minima when that well is varied. It gives no Cambridge energy. The sizes 38, 55, 75, 98, and 104 are not named. The Cambridge hit targets stand.

Cite `kravatskiyImprovEvolve2026` and `barronRomeroShells2026`. No head-to-head at the manuscript sizes. ASE tag 3.29.0, AGOX tag v3.11.1, CREST v3.0.2, pele v0.1.2, xtb v6.8.0, and the GMIN page of 17 March 2022 were unchanged on this check.

## 2026-11-13 — Quantum annealing on LJ13, LJ20, LJ25, and LJ38

Fiorentino and Marzari, arXiv:2606.03405 (2 June 2026), anneal a path-integral molecular-dynamics trajectory. On Lennard-Jones clusters the reference energies are the Cambridge Cluster Database, and the note says a lower structure could still be found. The scored sizes are 13, 20, and 25 atoms, and the 38-atom cluster. Unpinned simulated annealing and unpinned quantum annealing leave LJ38 unsolved in runs out to \(10^{6}\) steps. A replica-pinned variant reaches the reference minimum. They compare that variant with Wales and Doye: about 80 percent success on LJ38 in about 5000 local minimisations, against a comparable success with 32 replicas and 40 local optimisation rounds, one round every \(10^{4}\) steps. They call the comparison qualitative, because one side is Monte Carlo and the other is molecular dynamics. They also call the icosahedral 55-atom cluster easy. They do not name LJ75, LJ98, or LJ104, and they give no energy below the Cambridge values.

The manuscript's LJ38 map puts the recommended hop on the published truncated octahedron. This preprint does not report a lower potential energy, so it does not contradict the Cambridge hit targets. The 80 percent figure is their citation of Wales and Doye, not a printed anneal count. The article no longer prints an LJ38 hit count. A head-to-head on LJ38 has to charge the force calls the hop charges. Their count of local optimisation rounds is not that budget. Do not import the probabilities into the hit table.

Cite `fiorentinoQuantumAnnealing2026`.

## 2026-11-13 — Minima hopping on magic-size gold

Rahmatizad Khajehpasha, Safa, Eyvazi, Krummenacher, and Goedecker, arXiv:2601.10434 (15 January 2026), search paths among cuboctahedra, Ino's decahedra, and icosahedra for Au55, Au147, Au309, and Au561 on a machine-learned potential. Minima hopping finds low-symmetry minima. The abstract says some of the new global minima for Au309 and Au561 lie up to 2.8 eV below global minima reported earlier. The same note says an icosahedron is the ground state of a magic-size Lennard-Jones cluster and is not the ground state of these gold clusters.

Safa, Rahmatizad Khajehpasha, and Goedecker already search ionized gold from 22 to 100 atoms, filed as `safaGoldIons2026`. This preprint is the neutral magic sizes, and 309 and 561 sit outside that range. The 2.8 eV gaps are gold energies on their potential. They are not Cambridge Lennard-Jones energies, so they do not contradict the hit targets, 121 of 192, or 97 of 192. A head-to-head would be Au309 or Au561 on that potential. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST hexamer table.

Cite `rahmatizadGold2026`.

## 2026-11-13 — A charge-state search of NaPb12

Guevara-Vela, Rodríguez-Kessler, Muñoz-Castro, and Rocha-Rinza, The Journal of Physical Chemistry A (8 November 2026), doi:10.1021/acs.jpca.6c05536, search NaPb12. The abstract says a global structure search, followed by a density-functional refinement, puts a crown on the neutral cluster, an endohedral sodium on the monoanion, and an exohedral sodium on the dianion. The record has no volume or pages yet.

Rodríguez-Kessler and Muñoz-Castro are authors of the adaptive basin hop filed as `carmonaAbhDft2026` and of the boron-cluster preprints. This paper does not name that hop, and it does not name LJ38 through LJ104 or the water hexamer. It does not contradict those counts. A head-to-head would be NaPb12 on one shared surface. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST table.

Cite `guevaraVelaNapb2026`.

## 2026-11-13 — A refit of GFN2-xTB for liquid water

Nikolaeva, Kulik, and Stein, ChemRxiv (21 October 2026), doi:10.26434/chemrxiv.15010774/v1, reparameterize GFN2-xTB for aqueous electrolytes. The abstract says the default parameters overestimate the density of liquid water and of aqueous sodium chloride, and underestimate the sodium--water distance. They fit a reduced parameter set to cluster and periodic revPBE0-D3 data, first for hydrogen and oxygen, then for sodium. The abstract says the new parameters improve those densities and move the hydration distance into the experimental range. The fit is not a search for the hexamer, and the abstract gives no hexamer energy.

The hexamer table is unmodified GFN2-xTB in xtb 6.7.1. This preprint is a different parameter set, aimed at liquid densities. It does not recompute 121 of 192, 97 of 192, or the polished gaps, so it does not contradict those numbers. It does contradict any reading in which a refit of GFN2 for liquid water is the surface in that table. Keep the table on 6.7.1. Do not start a new CREST search on the refit.

Cite `nikolaevaGfn2Water2026`.

## 2026-11-13 — Journal version of the TIP4P water-cluster search

Bhatt, Chacko, Bijewar, and Nagare, The European Physical Journal B 98, 31 (2025), doi:10.1140/epjb/s10051-025-00881-x, is the journal version of the preprint already filed as `bhattWaterClusters2024`. The arXiv record still has no journal-ref line. The DOI, the volume, and the article number were read from the Crossref record, and that record names the same four authors. The landing page says the search is an artificial bee colony, optimised first with TIP4P, then at least ten of the lowest structures relaxed with density functional theory. It says they found new lowest-energy structures for every size from 2 to 20 molecules, against structures reported earlier.

The earlier note said the hexamer is outside the list of sizes where stacked pentagonal and cage-like structures lie below the stacked cubes. That list was the stacked-cube comparison. The journal sentence covers every size from 2 to 20, including the hexamer. It still does not quote a TIP4P energy below the Wales--Hodges minima, and it does not report 121 of 192 or 97 of 192. The new structures are the density-functional relaxations. Keep TIP4P, GFN2-xTB, and that density functional apart. Leave the CREST table on GFN2-xTB.

The key `bhattWaterClusters2024` now carries the journal DOI beside the arXiv id.

## 2026-11-13 — Coupled-cluster energies of small water clusters

Petty, Fowler, Ryu, Glick, Rock, Wang, Tschumper, and Shields, The Journal of Physical Chemistry A 129, 9291--9302 (2025), doi:10.1021/acs.jpca.5c04923, benchmark relative energies of water clusters from the trimer through the heptamer. The geometries are a CCSD(T):MP2 many-body optimisation. The reference single points are CCSD(T)-F12b. Those structures then test DLPNO-CCSD(T). Rock and Tschumper are authors of the argon-tagged hexamer already filed as `rockArgonHexamer2026`.

The abstract does not report 121 of 192 or 97 of 192, and it does not quote a TIP4P energy or a GFN2-xTB energy. A CCSD(T)-F12 ranking is another surface. Cite `pettyWaterClusters2025` beside the water section. Leave the CREST table on GFN2-xTB with xtb 6.7.1. A comparison counts only if both sides use these coupled-cluster geometries.

## 2026-11-13 — Quantum heat capacity on MB-pol

Hernández-Rojas, Calvo, and Wales, The Journal of Physical Chemistry Letters 17, 9231--9236 (2026), doi:10.1021/acs.jpclett.6c02032, study melting of water clusters on MB-pol. The abstract says the databases hold more than 800000 local minima for the 20-molecule cluster. Quantum effects change the main canonical melting features only weakly, and they suppress premelting. The octamer develops a convex intruder on the quantum microcanonical caloric curve that is absent classically.

These authors already surveyed the hexamer and the octamer on MB-pol and on TIP4P/2005, filed as `hernandezRojasWater2026`. This paper is the heat capacity. It does not report a new global minimum, a TIP4P energy, or the sealed hit counts. Cite `hernandezRojasMelting2026` beside that landscape paper. An MB-pol caloric curve is not the GFN2 motif table.

## 2026-11-13 — Gupta searches at 38, 55, and 98 atoms

Wu, Lin, He, Chen, and Shi, Nanoscale 18, 14001--14012 (2026), doi:10.1039/d6nr01123c, compare the Gupta, Sutton-Chen, and Finnis-Sinclair potentials for Pt--Cu clusters at 13, 23, and 38 atoms. The abstract calls the search CDE. At 13 atoms the three potentials meet on one icosahedron. In their density-functional comparison the Gupta and Sutton-Chen candidates are nearly degenerate, and the Finnis-Sinclair candidate lies 0.665 eV higher. At 38 atoms a Sutton-Chen Pt19Cu19 structure is the lowest of the single points they report, and Finnis-Sinclair reaches a different compact basin. They say the empirical ranking and the density-functional ranking do not collapse into one order.

The 38 in that comparison counts platinum and copper atoms. It is not the Lennard-Jones truncated octahedron. The 0.665 eV gap is a Pt--Cu density-functional difference. It does not contradict the Cambridge LJ38 energy. A head-to-head would be Pt--Cu at one of these sizes, on one of the three potentials, and not on LJ38.

Wu and Zhang, Journal of Nanoparticle Research 27, 205 (2025), doi:10.1007/s11051-025-06399-8, compare two Gupta parameter sets for Co--Pt. The landing page says the optimisations cover 38-atom clusters, 98-atom clusters, and part of the 147-atom series, with an adaptive immune optimisation algorithm. At 98 atoms one parameter set favours truncated octahedra and Leary tetrahedra. The other favours Leary tetrahedra and icosahedra. Those sizes are Co--Pt on the Gupta potential. They do not contradict the Cambridge Lennard-Jones energies. A head-to-head would be Co--Pt on one Gupta parameter set, against that immune optimisation, and not on LJ38 or LJ98.

Tsasse, Keyampi, Nana, and Zekeng, SPIN 15, 2550014 (2025), doi:10.1142/s2010324725500146, minimise molybdenum clusters on the Gupta potential and on the Sutton-Chen potential by quenching molecular dynamics. The abstract says most structures agree in point group, and that the dimer matches experiment. It gives no Lennard-Jones energy and no water energy. A head-to-head would be a molybdenum cluster on one of those potentials. It does not warrant a head-to-head on the Lennard-Jones preset.

Yusuf, Duan, Chen, Ouyang, and Wang, Physical Chemistry Chemical Physics (2026), doi:10.1039/d6cp02742c, run molecular dynamics of 55-atom pure and bimetallic copper, silver, and gold clusters on the Gupta potential. The abstract is a melting study. The record has no volume or pages yet. The 55-atom cluster is a noble-metal cluster, not the Lennard-Jones icosahedron. The abstract does not report a new global-minimum energy. It does not contradict the Cambridge energy at 55 atoms, and it does not warrant a head-to-head on the hop.

Cite `wuPtcu2026`, `wuCopt2026`, `tsasseMolybdenum2025`, and `yusufGuptaMelting2026`.

## 2026-11-13 — Ordering and gold searches away from the preset

Neumann, Margraf, Telari, Reuter, and Bruix, The Journal of Chemical Physics 164, 074104 (2026), doi:10.1063/5.0304275, add an optimal-exchange step to a global optimisation of PtAu nanoparticles. The abstract says the lowest chemical ordering changes the preferred shape, from high symmetry in the pure particles to distorted shapes at intermediate composition. The energy is an empirical surrogate. Telari is an author of the silver--copper chart filed as `settemAgcu2026`. This paper is platinum--gold. It gives no Lennard-Jones energy. A head-to-head would be a PtAu particle on that surrogate. It does not warrant a head-to-head on LJ38.

Elsborg, Hovmand, and Bhowmik, ACS Materials Au 6, 696--704 (2026), doi:10.1021/acsmaterialsau.5c00224, train a reinforcement-learning agent to swap atoms on a fixed icosahedron. The abstract says the agent recovers a previously established Ag--Au ground state, and that the same policy can be tried at an unseen size. The shape is held fixed. This is not the manuscript's geometric move library. It does not contradict the Cambridge hit targets, and it does not warrant a head-to-head on the Lennard-Jones preset.

Rodríguez-Kessler, Rodríguez-Domínguez, and Muñoz-Castro, Journal of Nanoparticle Research 28, 59 (2026), doi:10.1007/s11051-026-06580-7, search Pt\(_n\)V\(^-\) by basin hopping, for \(n\) from 1 to 13, and relax the candidates with density functional theory. The landing page names a \(C_{3v}\) pyramid as the global minimum of Pt\(_{10}\)V\(^-\). These authors already have the doped-cluster searches on file. This one is platinum--vanadium. It gives no Lennard-Jones energy and no water energy. A head-to-head would be this anion on one shared surface. It does not warrant a head-to-head on the sealed maps.

Aguado, Álvarez-Zapatero, Kostko, and von Issendorff, Nature Communications 17, 5099 (2026), doi:10.1038/s41467-026-71649-9, assign structures of gold cluster anions from 17 to 69 atoms. The abstract says extensive structural searches, then a comparison of measured and simulated photoelectron spectra. Many of the structures are highly symmetric. The searches are gold anions. They do not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. A head-to-head would be one of these anions. The minima-hopping gold papers already on file use other sizes and another potential.

Li, Wang, Tian, Yang, Lei, Xu, Xu, and Gu, RSC Advances 16, 17470--17479 (2026), doi:10.1039/d5ra09961g, report an octahedral Au32 cage beside the icosahedral Au32 fullerene. The abstract says the cage can hold up to four extra gold atoms, and that a scan over the elements finds 30 stable hollow cages of that symmetry, including gold, silver, copper, boron, and silicon. It does not describe a basin hop, and it gives no Cambridge energy. Keep the cage out of the Lennard-Jones table. No head-to-head at the manuscript sizes.

Cite `neumannPtAu2026`, `elsborgOrdering2026`, `rodriguezKesslerPtv2026`, `aguadoGoldAnions2026`, and `liAu32Cage2026`.

## 2026-11-13 — A survey, iridium clusters, and sulfur-dioxide water

Sanchez Alvarez and Calaminici, Journal of Computational Chemistry 46, e70243 (2025), doi:10.1002/jcc.70243, review global optimisation of molecular structures. The abstract covers conformations, crystal polymorphs, reaction pathways, and cluster structure prediction, and it separates stochastic methods from deterministic ones. It gives no Lennard-Jones hit rate and no water-hexamer count. Cite `sanchezAlvarezReview2025` as the survey. It does not contradict 121 of 192, 97 of 192, or the Cambridge energies, and it does not warrant a new head-to-head.

Gong, Liu, Shen, Dong, Liu, Yuan, and Lu, Crystals 16, 243 (2026), doi:10.3390/cryst16040243, search iridium clusters from 9 to 30 atoms with CALYPSO. The abstract says the Lennard-Jones potential is then used to compute the binding energy. Those energies belong to the iridium clusters. They are not the Cambridge Lennard-Jones minima. The search stops at 30 atoms, so 38, 55, 75, 98, and 104 are absent. Cite `gongIridium2026`. Keep the binding energies out of the Lennard-Jones table. No head-to-head at the manuscript sizes.

Modak and Chaudhury, Journal of the Indian Chemical Society 103, 102611 (2026), doi:10.1016/j.jics.2026.102611, title a gradient-directed parallel tempering strategy for global optimisation, applied to sulfur dioxide water clusters. The Crossref record gives the title, the volume, the issue, and the page, and no energy. The clusters contain sulfur dioxide. They are not the bare hexamer. The title does not contradict 121 of 192 or 97 of 192. A comparison, if it is run, is one of their sulfur dioxide water clusters. Do not retarget the CREST table.

Cite `modakSulfurDioxide2026`.

## 2026-11-13 — A diffusion search on a supported platinum cluster

Rønne, Vegge, and Bhowmik, arXiv:2510.13448 (15 October 2025), introduce GO-Diff. A diffusion model is trained from scratch on the energy function, with no prior structures and no relaxation inside the generator. The tests use MACE-MP0. One task places a platinum addatom on a stepped platinum surface. The other searches for a platinum heptamer on Pt(111). They compare that search with random structure search. GO-Diff finds the heptamer within 2560 energy evaluations. A model transferred from the addatom task finds it in 7 of 8 runs, at about 600 evaluations on average. The note cites basin hopping and does not run it. The record has no DOI.

The heptamer is a supported platinum cluster on MACE-MP0. It is not LJ38 through LJ104, and it is not the water hexamer, so the note does not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. An energy evaluation on MACE-MP0 is not the force-call count the hop charges. A comparison, if it is run, is the supported heptamer against random structure search. It does not warrant a head-to-head on the sealed Lennard-Jones maps or on the CREST table.

Cite `ronneGoDiff2025`.

## 2026-11-13 — RPA+rSE geometries leave the hexamer spacing almost unchanged

Tahir, Shang, and Ren, arXiv:2505.07357 (12 May 2025), give analytical gradients for the renormalized single-excitation correction on top of the random-phase approximation. On water hexamers the energy difference between the low-energy isomers is almost unchanged when the geometries change from RPA to RPA+rSE. On the WATER27 set the mean absolute error of RPA+rSE falls from 0.91 kcal/mol on RPA geometries to 0.70 kcal/mol on RPA+rSE geometries. The record has no DOI.

These authors already relaxed water clusters with RPA forces, filed as `tahirRpaWater2024`. That preprint said the hexamer does not show the reordering they report for the 25-molecule cluster. This letter is the gradient of the correction, and the hexamer spacing stays almost the same. It does not report 121 of 192 or 97 of 192, and it does not quote a TIP4P energy or a GFN2-xTB energy. Cite `tahirRpaRse2025` beside that preprint. Leave the CREST table on GFN2-xTB with xtb 6.7.1.

## 2026-11-13 — Confined Lennard-Jones melting at 55 and 147 atoms

Rocha, Dias, and Costa, Physical Review E 112, 014112 (2025), doi:10.1103/n6vd-ttx4, also arXiv:2502.00999, use parametric curves of the microcanonical entropy to classify phase transitions. One example is a Lennard-Jones cluster confined to a sphere. The sizes in that example are 55 and 147 particles. The quantities they report are a transition temperature and a latent heat. The low-temperature solid they describe is the Mackay icosahedron. The paper does not quote a Cambridge potential energy.

The 55-atom cluster is confined. It is not the free LJ55 minimum used as a hit target. The 147-atom cluster is outside the manuscript sizes. This does not contradict the Cambridge energies, and a melting temperature is not a hop hit rate. No head-to-head at the manuscript sizes.

Cite `rochaInflection2025`.

## 2026-11-13 — Girifalco fullerenes, an interface hop, and a crystal-search platform

Ren, Liang, Chen, and Wang, The Journal of Physical Chemistry Letters 16, 5855--5861 (2025), doi:10.1021/acs.jpclett.5c01355, search fullerene clusters with unbiased fuzzy global optimisation on the Girifalco potential. The cluster size runs from 11 to 400 molecules, and the fullerene size runs through C28, C36, C40, C60, C76, C84, and C96. Against the literature for (C60)N with N at most 150, they report new structures of lower energy at 129, 138, and 145 molecules.

Chen, Liang, Chen, and Wang already search (C60)N on the PPR surface, filed as `chenFullerene2026`. This letter is the Girifalco potential. The 129, 138, and 145 count fullerene molecules. They are not Lennard-Jones atoms, so the new energies do not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. Keep them out of the Lennard-Jones table. A comparison, if it is run, is (C60)N on the Girifalco potential.

Chou, Patel, Amsler, and Wolverton, Physical Review Materials 9, 113801 (2025), doi:10.1103/zg25-qf5p, adapt minima hopping to tilt grain boundaries in SrTiO3. An empirical search proposes candidates, and density-functional calculations refine the lower ones. The abstract says the search finds more stable interfacial structures than earlier genetic-algorithm studies of several of those boundaries. The limitation they name is a Buckingham potential. The system is an oxide interface. It does not report a Lennard-Jones energy or a water energy. A head-to-head would be one of those grain boundaries. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST table.

Song, She, Song, Shi, Xuan, and Cao, Computational Materials Science 275, 115049 (2026), doi:10.1016/j.commatsci.2026.115049, also arXiv:2604.21401, release GEWUM. The platform runs selective random structure search with universal machine-learning potentials. The case studies are Al--Sc--N, a P-62c phase of U3Si5, and ThH10 at 150 GPa. The selective-search paper is already filed as `songSelectiveSearch2026`. GEWUM is the workflow. The tests are crystals. They do not contradict the Cambridge hit targets or the sealed hexamer counts.

Cite `renGirifalco2025`, `chouInterface2025`, and `songGewum2026`.

## 2026-11-13 — The ASE tip moved, and the release tags did not

ASE tag 3.29.0, dated 21 June 2026, is still the newest tag. The master tip moved on 13 November 2026 to a merge of a GUI performance branch. That commit does not change basin hopping, and it is not a release. AGOX tag v3.11.1, dated 4 February 2026, is still the newest tag. The dev tip is still the 5 May 2026 import fix for ASE 3.28. CREST's newest numbered release is still v3.0.2, and the master tip is still the 16 June 2026 CMake fix. pele is still v0.1.2, dated 12 October 2026. xtb v6.8.0, dated 5 November 2026, is still the newest release.

Pin the open arms to ASE 3.29.0, AGOX v3.11.1, CREST 3.0.2, and pele 0.1.2. No bibliographic entry for this check: a tag date is not a DOI.

## 2026-11-13 — New lowest-energy silicon clusters at four sizes

Salazar-Barrientos, McCabe, Cummings, Rocha-Rinza, and Guevara-Vela, Theoretical Chemistry Accounts 145, 76 (17 August 2026), doi:10.1007/s00214-026-03329-2, search medium silicon clusters with a genetic algorithm on machine-learned potentials, then reoptimise candidates at PBE0/def2-TZVP and score them with DLPNO-CCSD(T). The potentials were trained on TPSSh/def2-TZVP and on r2SCAN-D3/def2-TZVP. The abstract says the workflow finds new lowest-energy structures for Si18, Si20, Si30, and Si32. The gap from the previously reported minimum shifts by 0.1 to 5.4 kcal/mol between cc-pVDZ and cc-pVTZ, and the Si20 gaps are 14.7 and 11.5 kcal/mol. That shift does not change the order. Density-functional candidates for Si17, Si22, Si23, and Si29 sit up to 20 kcal/mol above the literature minima once the correlated energies are used. The search files are doi:10.5281/zenodo.19494378.

Shukla already builds Stillinger--Weber silicon from 32 to 60 atoms, filed as `shuklaSilicon2026`. These four new structures are not Stillinger--Weber energies, and they are not Cambridge Lennard-Jones energies. They do not contradict the hit targets, 121 of 192, or 97 of 192. A head-to-head is Si18, Si20, Si30, and Si32 on the DLPNO-CCSD(T) ranking. The Stillinger--Weber comparison stays the one already asked for. Keep the silicon energies out of the Lennard-Jones table.

Cite `salazarSilicon2026` and `guevaraVelaSiliconData2026`.

## 2026-11-13 — Doped silicon isomers

Liu, Yang, Hao, Yang, and Wu, International Journal of Quantum Chemistry 126, e70193 (17 April 2026), doi:10.1002/qua.70193, search AgSi\(_n^-\) for \(n\) from 3 to 20 with a global search at mPW2PLYP. Through ten silicon atoms the silver sits on the surface of the pure silicon cluster. From eleven silicon atoms the silver replaces one silicon atom in the pure Si\(_{n+1}\) structure. The abstract compares the simulated photoelectron spectra with experiment. It gives no Lennard-Jones energy and no water energy.

Borshch, Condensed Matter and Interphases 28, 376--387 (22 September 2026), doi:10.17308/kcmf.2026.28/13966, proposes a modification of the Wade--Mingos rule for silicon clusters that enclose a fourth-period metal. The demonstration is MeSi\(_{15}^-\) with Me = Sc or V, at four density functionals, compared with an experimental photoelectron spectrum. The abstract does not name basin hopping. It gives no Lennard-Jones energy.

Neither paper contradicts the Cambridge hit targets, 121 of 192, or 97 of 192. A head-to-head would be AgSi\(_n^-\) on mPW2PLYP, or MeSi\(_{15}^-\) on one of the four functionals. Neither warrants a head-to-head on the Lennard-Jones preset or on the CREST hexamer table.

Cite `liuSilverSilicon2026` and `borshchSiliconMetal2026`.

## 2026-11-13 — Nickel clusters by basin hopping, and two small clusters by active learning

Saha, University of Illinois Chicago (15 July 2026), doi:10.25417/uic.32991926, searches nickel clusters from 2 to 150 atoms by basin hopping on a Gaussian approximation potential. The abstract says the search returns low-energy candidates at 149 sizes and compares them with structures reported earlier. It does not say those candidates lie below the earlier reports, and it gives no Lennard-Jones energy.

Wen, Jiang, Li, Zhang, and Wang, The Journal of Chemical Physics 165, 094301 (1 September 2026), doi:10.1063/5.0349333, test three uncertainty estimates for a graph network, then use Monte Carlo dropout to drive active learning. The cluster tests are Ta\(_2\)N\(_3^-\) and LaSi\(_{24}\). The abstract reports an error on the potential, not a new global-minimum energy.

Neither contradicts the Cambridge hit targets or the sealed hexamer counts. A head-to-head would be one nickel size on that Gaussian approximation potential, or Ta\(_2\)N\(_3^-\) or LaSi\(_{24}\) on the surface they train. Neither fills the open ASE, AGOX, and GMIN kit, and neither warrants a head-to-head on the Lennard-Jones preset or on the CREST table.

Cite `sahaNickel2026` and `wenClusterUncertainty2026`.

## 2026-11-13 — Allegro on the same grain boundaries, and Ag--Pd with Ag--Cu

Chou, Wang, Yang, van Aken, Perry, Kozinsky, and Wolverton, npj Computational Materials 12, 374 (15 July 2026), doi:10.1038/s41524-026-02214-7, run minima hopping with the Allegro potential on SrTiO\(_3\) tilt grain boundaries. Chou and Wolverton already search those boundaries with a Buckingham potential, filed as `chouInterface2025`. This paper names the \(\Sigma 3(112)[110]\) boundary and checks the \(\Sigma 5(310)[001]\) boundary against experiment. The potential is Allegro. It does not report a Lennard-Jones energy or a water energy. It does not fill the open ASE basin-hopping arm. A head-to-head would be one of those grain boundaries on Allegro. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST table.

Mubeen and Chen, Surfaces and Interfaces 98, 110424 (October 2026), doi:10.1016/j.surfin.2026.110424, title a dual proximal-policy search, on empirical potentials, for the global minimum of Ag--Pd and Ag--Cu nanoalloys. The Crossref record gives the title, the volume, and the page, and no energy. Mubeen and Chen already search Ag--Ni and Pt--Co, filed as `mubeenDrLClusters2026`. Settem and co-workers already map 38-atom Ag--Cu at finite temperature, filed as `settemAgcu2026`. This record is Ag--Pd and Ag--Cu, and it does not quote a Cambridge Lennard-Jones energy, so it does not contradict the hit targets, 121 of 192, or 97 of 192. A head-to-head would be Ag--Pd or Ag--Cu on the empirical potential the title names. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST hexamer table.

Cite `chouAllegro2026` and `mubeenAgpd2026`.

## 2026-11-13 — Two differential-evolution toolkits for metal clusters

Wu, He, and Lin, Journal of Chemical Information and Modeling 66, 1950--1954 (4 February 2026), doi:10.1021/acs.jcim.5c02790, release a C++ toolkit for collaborative differential evolution. Three subpopulations split exploration, exploitation, and balance. The case studies are Pt--Pd, Cu--Au, and monometallic platinum. The abstract says the search converges 50 to 100 percent faster than the conventional methods in that comparison, and that the runs are more stable.

The same three authors, Nanoscale 17, 26892--26906 (2025), doi:10.1039/d5nr03763h, describe adaptive multi-strategy differential evolution. Three populations again share exploration, exploitation, and diversity. The abstract says the search is faster and more stable than the traditional algorithms they compare, and that it reaches a lower energy on those tests. The clusters they then analyse are Pt--Ni, from 3 to 12 atoms and at the magic sizes 13, 19, and 23.

Wu, Lin, and He are also authors of the Pt--Cu potential comparison already filed as `wuPtcu2026`, whose abstract calls that search CDE. This note is the toolkit paper under the same name. It does not replace that ranking of Gupta, Sutton-Chen, and Finnis-Sinclair. The adaptive paper is a second algorithm, on Pt--Ni. Usman and Chen already search platinum--nickel with a policy gradient, filed as `usmanPtni2026`. Tang and co-workers already say another search beats plain differential evolution on Au--Cu, filed as `tangAucur2026`. Neither new abstract names LJ38 through LJ104 or the water hexamer, and neither quotes a Cambridge energy, so neither contradicts the hit targets, 121 of 192, or 97 of 192. A convergence speed is not the force-call hit rate. A comparison would be Pt--Pd, Cu--Au, platinum, or Pt--Ni on the potential that code uses. Neither warrants a head-to-head on the Lennard-Jones preset or on the CREST hexamer table, and neither fills the open ASE, AGOX, and GMIN kit.

Cite `wuCde2026` and `wuAmsde2025`.

## 2026-11-13 — Global minima of platinum-doped and ruthenium-doped silicon

Guevara-Vela, Rodríguez-Kessler, Muñoz-Castro, and Rocha-Rinza, Physical Chemistry Chemical Physics 28, 11107--11116 (2026), doi:10.1039/d6cp00681g, search neutral and anionic PtSi\(_n\) for \(n\) from 2 to 11. A genetic algorithm proposes the candidates, and PBE0/def2-TZVP refines them. The abstract reports the lowest structure at each size and the platinum coordination.

Lu, The Journal of Physical Chemistry A 130, 670--681 (8 January 2026), doi:10.1021/acs.jpca.5c08131, assigns RuSi\(_n\) for \(n\) from 13 to 17. Anion photoelectron spectra at 266 nm give vertical detachment energies from 2.87 to 3.28 eV, and the calculated values differ from experiment by less than 0.11 eV. The global minima come from CALYPSO and ABCluster, refined at B3LYP/def2-TZVPD, with DLPNO-CCSD(T) single points. The abstract names anionic RuSi\(_{17}^-\) and neutral RuSi\(_{14}\) as the compact magic sizes. The motif is a ruthenium atom inside a distorted bicapped pentagonal prism.

Guevara-Vela and Rocha-Rinza are authors of the pure-silicon search already filed as `salazarSilicon2026`, which reports new minima at Si18, Si20, Si30, and Si32. Shukla already builds Stillinger--Weber silicon from 32 to 60 atoms, filed as `shuklaSilicon2026`. Liu and co-workers already search AgSi\(_n^-\), filed as `liuSilverSilicon2026`. These two papers are platinum-doped and ruthenium-doped, and both stop at or below 17 silicon atoms. They give no Lennard-Jones energy and no Stillinger--Weber energy. The 0.11 eV figure is a detachment-energy residual, not a Cambridge gap. They do not contradict the Cambridge hit targets, the Salazar gaps, 121 of 192, or 97 of 192. A head-to-head would be PtSi\(_n\) on PBE0, or RuSi\(_n\) on the B3LYP ranking with the coupled-cluster single points. ABCluster is the microsolvation code already named beside CREST in `lauriaComplax2026`. This use is ruthenium-doped silicon. Neither paper warrants a head-to-head on the Lennard-Jones preset, on the Stillinger--Weber comparison, or on the CREST hexamer table.

Cite `guevaraVelaPtsi2026` and `luRutheniumSilicon2026`.

## 2026-11-13 — A modified basin hop on Y@Cu15

Rodríguez-Kessler and Muñoz-Castro, Physical Chemistry Chemical Physics 28, 13118--13124 (2026), doi:10.1039/d6cp00667a, search Y@Cu\(_{15}\) with a modified basin hop. The abstract reports a yttrium atom inside a \(C_{2v}\) cage of fifteen copper atoms. The interaction energy of that encapsulation is \(-220.4\) kcal/mol.

These authors already have the doped-cluster basin hops on file, including Pt\(_n\)V\(^-\) as `rodriguezKesslerPtv2026`. This cluster is yttrium in copper. The interaction energy is not a Cambridge Lennard-Jones energy, and the note gives no water energy, so it does not contradict the hit targets, 121 of 192, or 97 of 192. It does not fill the open ASE, AGOX, and GMIN kit. A head-to-head would be Y@Cu\(_{15}\) on the surface used for that hop. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST hexamer table.

Cite `rodriguezKesslerYcu2026`.

## 2026-11-13 — CREST on a cut-out binding pocket

Fellinger, Sappl, Szabadi, Merget, Schleifer, and Langer, Journal of Chemical Information and Modeling 66, 3432--3436 (23 March 2026), doi:10.1021/acs.jcim.6c00569, release AutoPocket2CREST. The workflow finds the ligand in a protein complex, cuts a pocket around it, can freeze the backbone, and hands that pocket to CREST. The abstract says the sampling is semiempirical and is aimed at the bound ligand.

The hexamer comparison is a bare cluster at GFN2-xTB in xtb 6.7.1, against CREST 3.0.2. This workflow does not report 121 of 192, 97 of 192, or the polished gaps. It does not mention a user seed, so the sentence that CREST 3.0.2 exposes no seed still stands. A pocket around a bound ligand is not the bare hexamer. Leave the CREST table on 3.0.2. It does not warrant a head-to-head on that table.

Cite `fellingerAutopocket2026`.

## 2026-11-14 — Gupta ordering of a 55-atom icosahedron, and copper--palladium at 19 atoms

Eryürek and Garip, Physica Scripta 100, 035406 (12 February 2025), doi:10.1088/1402-4896/adb10e, search the chemical ordering of 55-atom Au\(_N\)Al\(_{(N-42)}\)Ni\(_{13}\) nanoalloys. The geometry is a Mackay icosahedron. Monte Carlo basin hopping on the Gupta potential relaxes the ordering. In the lowest structures the nickel atoms occupy the icosahedral core, and gold and aluminium stay on the surface. The abstract identifies Au\(_{17}\)Al\(_{25}\)Ni\(_{13}\) as the lowest mixing energy at the Gupta level. The nickel core is under compression.

The 55 in that search counts gold, aluminium, and nickel. It is not the Lennard-Jones icosahedron. The mixing energy is a Gupta energy, so it does not contradict the Cambridge energy at 55 atoms, 121 of 192, or 97 of 192. The shape is held fixed, so the search is the ordering. It does not fill the open ASE, AGOX, and GMIN kit. A head-to-head would be this composition on the Gupta potential. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST hexamer table.

Yıldırım, Afyon Kocatepe University Journal of Sciences and Engineering 25, 1053--1061 (18 September 2025), doi:10.35414/akufemubid.1643569, treats every composition of the 19-atom Cu\(_n\)Pd\(_{19-n}\) series. The abstract combines the Gupta potential with density-functional calculations. It says the site preference, core or surface, follows the atomic radius, and that the excess energy at both levels favours mixing. The number of copper--palladium bonds is tied to that excess energy. The abstract does not name basin hopping, and it gives no Lennard-Jones energy.

Nineteen atoms sits outside 38, 55, 75, 98, and 104. This series does not contradict the Cambridge hit targets or the sealed hexamer counts. A head-to-head would be one of these 19-atom clusters on the Gupta potential. It does not warrant a head-to-head on the Lennard-Jones preset.

Cite `eryurekAuAlNi2025` and `yildirimCupd2025`.

## 2026-11-14 — Gupta basin hopping inside a palladium-covered truncated octahedron

Göcen, Nanomaterials 15, 1895 (17 December 2025), doi:10.3390/nano15241895, studies Pd\(_{60}\)Ir\(_n\)Rh\(_{19-n}\) clusters held in a truncated octahedron. Basin hopping on the Gupta potential searches the global optimisation of that family, and density-functional calculations check the configurations. The surface is only palladium and is held fixed. Iridium and rhodium occupy the 19-atom core. The abstract says Pd\(_{60}\)Ir\(_{6}\)Rh\(_{13}\) has the lowest mixing energy. The melting temperature is not monotonic in the iridium count, and the iridium-rich clusters are the most resistant to melting.

The formula fixes the size at 79 atoms. That is not LJ75 or LJ98. The mixing energies are Gupta energies, so they do not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. The outer shape and the palladium surface are held fixed, so the search is the core ordering. It does not fill the open ASE, AGOX, and GMIN kit. A head-to-head would be this composition on the Gupta potential. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST hexamer table.

Cite `gocenPdirrh2025`.

## 2026-11-14 — Identity, existence, and extra coordinates in one search

Larsen, Kaappa, Vishart, Bligaard, and Jacobsen, npj Computational Materials 11, 222 (10 July 2025), doi:10.1038/s41524-025-01656-9, add three kinds of coordinates to a global optimisation: the chemical identity of an atom, its degree of existence, and its position in four to six dimensions. A machine-learning model with a vectorial fingerprint is trained on density-functional energies and forces. The abstract says the extra coordinates let the search pass barriers on the ordinary landscape. The tests include clusters and periodic cells, with the cell vectors free, and a closing example of an iron--cobalt pair in nitrogen-doped graphene.

These authors already describe fractional existence on copper clusters, filed as `larsenFractional2023`. This paper adds the chemical identity and the higher-dimensional coordinates. The abstract does not name LJ38 through LJ104 or the water hexamer, so it does not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. The iron--cobalt pair is not the sealed map. A comparison that uses this search has to name the extra coordinates and the learned surface. It does not warrant a new head-to-head on the Lennard-Jones preset. The shared-potential comparison already asked for the 2023 paper stays that comparison.

Cite `larsenExtraDimensions2025`.

## 2026-11-14 — Shell rules for multi-component icosahedra, and a 2869-atom ordering

Canestrari, Nelli, and Ferrando, Nature Communications 16, 1655 (15 February 2025), doi:10.1038/s41467-025-56952-1, give a construction for multi-component icosahedra. Concentric shells of different chiral and achiral types, built from particles of different sizes, are assembled into one aggregate. Shell sequences become paths on the hexagonal lattice, and the rules name magic icosahedra and the size mismatch between shells. The abstract says molecular dynamics and density-functional calculations confirm the rules for several multi-component atomic clusters and nanoparticles.

The manuscript clusters are one component. The abstract does not quote a Cambridge Lennard-Jones energy, so it does not contradict the hit targets, 121 of 192, or 97 of 192. The construction installs a shell sequence. That is not the manuscript's move library. It does not warrant a head-to-head on the unary preset or on the CREST hexamer table.

Alkhatib, Salem, Hesselink, and Mpourmpakis, ACS Omega 10, 30996--31008 (14 July 2025), doi:10.1021/acsomega.5c04082, order the atoms of multimetallic nanoparticles with a bond-centric model and a genetic algorithm. Weighting factors turn a monometallic bond strength into a bimetallic one. The abstract reports the ordering of 2869-atom cuboctahedra across 15 bimetallic combinations, compared with experiment and with other calculations.

The size 2869 is outside 38, 55, 75, 98, and 104. The shape is held as a cuboctahedron, and the energy is the bond-centric cohesive energy. The abstract gives no Cambridge Lennard-Jones energy and no water energy, so it does not contradict the hit targets or the sealed hexamer counts. It does not fill the open ASE, AGOX, and GMIN kit. It does not warrant a head-to-head on the Lennard-Jones preset.

Cite `canestrariIcosahedra2025` and `alkhatibOrdering2025`.

## 2026-11-14 — A reinforcement-learning search of Ag15

Mubeen, Chen, and Ur Rehman, Journal of Chemistry and Environment, pages 1--17 (11 January 2025), doi:10.56946/jce.v4i1.589, search Ag\(_{15}\) with deep reinforcement learning. The record has no volume. The abstract says genetic algorithms and basin hopping usually perform poorly on complex surfaces, and that the reinforcement-learning search finds stable configurations more efficiently.

Mubeen and Chen already search Ag--Ni, Pt--Co, Ag--Pd, and Ag--Cu, filed as `mubeenDrLClusters2026` and `mubeenAgpd2026`. This cluster is fifteen silver atoms. The abstract gives no Lennard-Jones energy and no water energy, so it does not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. It does not fill the open ASE, AGOX, and GMIN kit. A head-to-head would be Ag\(_{15}\) on the surface used for that search. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST hexamer table.

Cite `mubeenAg152025`.

## 2026-11-14 — RAFFLE searches interfaces, and the named release tags did not move

Taylor, Pitfield, Davies, and Hepplestone, npj Computational Materials 11, 258 (8 August 2025), doi:10.1038/s41524-025-01749-5, release RAFFLE. The package searches low-energy interfaces between a pair of crystals. A genetic algorithm places atoms with two-body, three-body, and four-body distributions, and active learning updates those distributions. The abstract says the search recovers the known bulk phases of aluminium and of MoS\(_2\), and that it treats intercalation compounds, grain boundaries, and the silicon--germanium interface.

Pitfield is an author of the active delta-learning search already filed as `pitfieldActiveDelta2026`. RAFFLE is not AGOX and not GOFEE. On 14 November 2026 the release tags were unchanged: ASE 3.29.0 on the GitLab tag list and on PyPI, AGOX v3.11.1 on the GitLab tag list and on PyPI, CREST v3.0.2 on the GitHub release list, pele v0.1.2 on the GitHub release list and on PyPI, and xtb v6.8.0 on the GitHub release list. The interface search gives no Lennard-Jones energy and no hexamer count, so it does not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. It does not fill the open ASE, AGOX, and GMIN kit. It does not warrant a head-to-head on the Lennard-Jones preset or on the CREST hexamer table.

Cite `taylorRaffle2025`.

## 2026-11-14 — OCCat searches global minima of cluster catalysts

Wang, Xu, and Cheng, Journal of Computational Chemistry 47, e70583 (13 November 2026), doi:10.1002/jcc.70583, release OCCat. The record gives issue 30 and no page numbers. The abstract says the package searches the global minimum of nanoalloy and oxide clusters and then scores catalytic properties in the same workflow. The search is a revised particle swarm optimisation with a centered binary adjustment that keeps interatomic distances inside a constraint. The potentials it names are Lennard-Jones, Gupta, VASP density-functional calculations, and a LASP neural-network potential. The abstract says the benchmarks are Lennard-Jones clusters, transition-metal systems at the Gupta and density-functional levels, and oxidized silver at the neural-network potential, and that those benchmarks support the structures and the efficiency. It gives no size, no energy, and no hit rate. A later analyzer scores adsorption from a generalized coordination number and a turnover frequency from a microkinetic model. That score is not a basin-hopping hit rate. The demonstrations it names are CO oxidation, hydrogen evolution, and oxygen reduction on sub-nanometer clusters.

The manuscript sizes 38, 55, 75, 98, and 104 are not named. The release repository ships a Lennard-Jones example directory named LJ21. That name is not a reported hit rate, and 21 atoms is outside those sizes. The abstract does not quote a Cambridge energy, so it does not contradict the hit targets, 121 of 192, or 97 of 192. OCCat is not ASE, AGOX, or GMIN, so it does not fill that open kit. It is a cluster global-optimisation code whose abstract already benchmarks Lennard-Jones clusters and Gupta metal clusters. The manuscript has no OCCat arm. A head-to-head is those benchmarks, at one shared potential and one relaxation budget. Because the abstract does not name the sealed sizes, that comparison is not yet the sealed maps, and it is not the CREST hexamer table. No bibliographic entry for the repository: it has no verified DOI or arXiv id.

Cite `wangOccat2026`.

## 2026-11-14 — Journal versions of the two gold minima-hopping searches

Safa, Rahmatizad Khajehpasha, Finkler, and Goedecker, Nanoscale Advances 8, 3605--3612 (2026), doi:10.1039/d6na00102e, is the journal version of the preprint already filed as `safaGoldIons2026`. The arXiv record still has no journal-ref line. The DOI, the volume, and the pages were read from the Crossref record. That record adds Jonas A. Finkler to the three authors on the arXiv record, and the title is the same. The search is still minima hopping on gold clusters from 22 to 100 atoms, and a large enough positive charge still makes planar structures the ground states. Those energies are gold energies. They do not contradict the Cambridge hit targets, 121 of 192, or 97 of 192.

Rahmatizad Khajehpasha, Safa, Eyvazi, Krummenacher, and Goedecker, Nanoscale Advances 8, 5146--5157 (2026), doi:10.1039/d6na00012f, is the journal version of the preprint already filed as `rahmatizadGold2026`. The arXiv record still has no journal-ref line. The DOI, the volume, and the pages were read from the Crossref record, and that record names the same five authors. The new global minima for Au309 and Au561 still lie up to 2.8 eV below global minima reported earlier. Those gaps are gold energies on their potential. They do not contradict the Cambridge hit targets.

The keys `safaGoldIons2026` and `rahmatizadGold2026` now carry the journal DOI beside the arXiv id. Neither journal version warrants a new head-to-head. The comparisons already named stay Au ions from 22 to 100 atoms, and Au309 or Au561, each on the potential used in that search.

## 2026-11-14 — Adaptive diffusion on Lennard-Jones clusters through forty atoms

Adel, Szmitek, Hartl, Wanzenböck, and Madsen, Journal of Chemical Information and Modeling 66, 7414--7425 (25 June 2026), doi:10.1021/acs.jcim.6c00843, put a diffusion model inside an evolutionary search and relax the samples. The Lennard-Jones benchmark runs from 5 to 40 atoms, against CMA-ES. They say CMA-ES finds the global minimum only up to about 17 atoms, depending on the seed, and that one diffusion evolution finds the global minimum at every size they tried, with an energy difference below \(10^{-6}\) eV. Clusters smaller than 25 atoms are found in fewer than 10 generations. Larger clusters are found in fewer than 150 generations, except LJ34 and LJ38. They single out LJ31 and LJ38. For LJ38 they name the face-centred cubic truncated octahedron as the global minimum and an incomplete Mackay icosahedron as the second minimum. The same paper treats neutral gold clusters Au13, Au31, and Au38.

The manuscript's LJ38 map puts the recommended hop on that truncated octahedron. This paper does not report a lower potential energy, so it does not contradict the Cambridge energy at 38 atoms. The search stops at 40 atoms, so LJ55, LJ75, LJ98, and LJ104 are absent. The cross-field section prints a BIPOP arm at 6 of 6 on LJ98. This CMA-ES comparison stops at 40 atoms and does not run that arm, so the 6 of 6 row stands. A generation count is not the force-call budget the hop charges. The article no longer prints an LJ38 hit count, so these runs contradict no printed number. The gold clusters are not the Lennard-Jones map. It does warrant a head-to-head on LJ38, at one force-call budget, against this diffusion search. Do not import the generation counts into the hit table.

Cite `adelGuidedDiffusion2026`.

## 2026-11-14 — CREST on vitamins, and GOAT with one explicit water

Moshchenkov, Ryzhako, Rozov, Otlyotov, and Minenkov, Physical Chemistry Chemical Physics 28, 19708--19720 (2026), doi:10.1039/d6cp02127a, compare a systematic Uniconf search with CREST metadynamics on vitamins A, B1, B2, B3, B5, B6, B7, B9, and C. The semiempirical methods are GFN1-xTB, GFN2-xTB, and QM3. At the tightest settings both searches find the lowest conformer on a given semiempirical surface. When one semiempirical method misses the density-functional order, another can match it. Some CREST entropies are atypically large, and the abstract says every entropy they compute is an estimate.

These authors already compare CREST with a systematic search on clinical drugs, filed as `moshchenkovCrestEntropy2026`. This paper is the vitamins. It does not report 121 of 192, 97 of 192, or the polished gaps. A CREST entropy on a vitamin is not the hexamer motif table. Leave the CREST table on 3.0.2 and on GFN2-xTB with xtb 6.7.1. It does not warrant a head-to-head on that table.

Dooley and Vyas, Physical Chemistry Chemical Physics 28, 25271--25283 (2026), doi:10.1039/d6cp01697a, sample conformers with GOAT and with CENSO, with one explicit water. The abstract says that water breaks an intramolecular hydrogen bond and improves an openCOSMO-RS solvation energy. On hydrogen-bond donors in the SM8 set the root-mean-square error falls by 7.3 to 13.0 percent, depending on the workflow. The abstract calls GOAT the global optimisation algorithm and names it beside CENSO. That is the molecular GOAT, not the analytic Goat Optimization Algorithm already filed as `nozariGoatAlgorithm2025`.

The missing hexamer comparison with de Souza's GOAT stays open. This paper does not report a water-cluster hit rate, so it does not contradict 121 of 192 or 97 of 192. One explicit water on a solute is not the bare hexamer. It does not warrant a head-to-head on the CREST table.

Cite `moshchenkovVitamers2026` and `dooleyGoatMicrosolvation2026`.

## 2026-11-14 — The ASE master tip moved to an MPI hook

The release tags recorded earlier on 14 November 2026 are unchanged. ASE 3.29.0, dated 21 June 2026, is still the newest tag on the GitLab tag list and on PyPI. AGOX v3.11.1, dated 4 February 2026, is still the newest tag on the GitLab tag list and on PyPI, and the dev tip is still the 5 May 2026 import fix. CREST v3.0.2, published 25 August 2024, is still the newest numbered release on the crest-lab GitHub release list, and the master tip is still the 16 June 2026 CMake fix. pele v0.1.2, dated 12 October 2026, is still the newest release on the GitHub release list and on PyPI. xtb v6.8.0, dated 5 November 2026, is still the newest release on the GitHub release list. The GMIN directory page is still last modified on 17 March 2022.

The ASE master tip is not that tag. It moved on 14 November 2026 to a merge of an MPI mechanism branch. The files are the parallel hook, the command-line entry, and the parallel documentation. That commit does not change basin hopping, and it is not a release. Pin the open arms to ASE 3.29.0, AGOX v3.11.1, CREST 3.0.2, and pele 0.1.2. No bibliographic entry for this check: a tag date is not a DOI.

## 2026-11-14 — Multi-cage water clusters, and pentagonal ice bilayers

Kirov, The Journal of Chemical Physics 164, 184110 (12 May 2026), doi:10.1063/5.0324777, identifies polyhedral and multi-cage water clusters from the structural graph. The abstract gives explicit formulas for the n-membered cycles and a generalization of the Euler formula for adjacent polyhedra. The dataset is the Database of Water Cluster Minima. The abstract does not report a new global-minimum energy.

The manuscript's water clusters are the 1998 Wales--Hodges TIP4P minima and the GFN2-xTB hexamer. This paper does not quote a TIP4P energy or a GFN2 energy, so it does not contradict those references, 121 of 192, or 97 of 192. A topological label is not a hop hit. It does not warrant a head-to-head on the sealed maps or on the CREST table.

Gudkovskikh and Kirov, Acta Crystallographica Section A 82, 454--463 (6 October 2026), doi:10.1107/s2053273326009629, compare two-layer ices whose layers are mostly pentagons. The potential is AMOEBA. The abstract says the spread in stabilization energy from proton disorder, per water molecule, is comparable with the heat of fusion of ice, and it discusses which self-assembly routes give the most stable bilayers. The supporting coordinates are AMOEBA cells of those bilayers. The bilayers are not the gas-phase hexamer. They do not contradict 121 of 192, 97 of 192, or the Cambridge energies. An AMOEBA bilayer is not the TIP4P table and not the CREST table.

Cite `kirovMulticage2026` and `gudkovskikhBilayer2026`.

## 2026-11-14 — CREST samples a side chain apart from the core

Rindt and Pracht, Journal of Chemical Theory and Computation (3 November 2026), doi:10.1021/acs.jctc.6c01611, detach flexible side chains from the molecular core, sample the pieces, and rebuild an approximate ensemble. A limiting variant leaves the side chains in extended reference conformations. The abstract says folded alkyl chains become competitive only past well-defined lengths, and that the method is implemented within CREST. The record has no volume or pages yet.

The hexamer comparison is a bare cluster at GFN2-xTB in xtb 6.7.1, against CREST 3.0.2. A water hexamer has no alkyl substituent, so this sampler does not run on that table. The abstract does not report 121 of 192, 97 of 192, or a user seed. The sentence that CREST 3.0.2 exposes no seed still stands. Leave the CREST table on 3.0.2. It does not warrant a head-to-head on that table.

Cite `rindtCrestSubstructure2026`.

## 2026-11-14 — Finite-temperature ensembles of phosphine gold clusters

McCandler, Sanwal, and Rogal, arXiv:2608.19404 (19 August 2026), build Markov state models from molecular dynamics on a machine-learned potential for phosphine-stabilized gold nanoclusters. The abstract says crystal structures from single-crystal diffraction are often minor metastable states or transient configurations, rather than the dominant finite-temperature structures. Increasing the ligand coverage drives the structures from planar toward three-dimensional. The record has no DOI.

Those populations are ligand-protected gold at finite temperature. They are not Lennard-Jones energies and not the water hexamer, so they do not contradict the Cambridge hit targets, 121 of 192, or 97 of 192. A Markov state model is not the basin-hopping preset. It does not fill the open ASE, AGOX, and GMIN kit. It does not warrant a head-to-head on the sealed maps or on the CREST table.

Cite `mccandlerGold2026`.

