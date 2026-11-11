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

