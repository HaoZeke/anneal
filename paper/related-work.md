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
