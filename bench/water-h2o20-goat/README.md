# GFN2 (H2O)20, both arms in one session

Anneal and CREST both ran on this VM, serial, arm order shuffled per seed. The wrapper is `sudo unshare --uts` with hostname `bench`, working directory `/work`, one thread. Anneal is 28000 charged calls. CREST is `crest --gfn2 -T 1 --legacy -xnam /work/xtb-gfn2 -len 0.2`. Seeds 15001–15006. A hit is a vtight GFN2 energy within 0.01 eV of the reference, −2768.551087 eV. xtb is 6.7.1 (edcfbbe). GOAT is absent because ORCA 6 is login-gated.

The 0.90 Å gate spike fix is what moved the hits. Separation had been leaving every hop on that gate, where the force is a few hundred eV/Å and the line search accepts nothing. A non-bonded pair closer than 1.40 Å is now opened to 2.20 Å before the quench, and a compressed covalent bond is restored to the sum of its covalent radii.

The anneal mean wall is 1203.0 s. The CREST mean wall is 690.9 s. The ratio of those means is 1.74. Anneal is lower than CREST on five seeds. Anneal hits three seeds and CREST hits one. Time to first hit is the search wall scaled by the charged-call fraction of the first improvement inside the window. Calls to first hit are the charged calls of that same improvement, read from the existing `improvements.txt` of this session. CREST's stdout has a CREGEN energy after each MTD iteration and no evaluation count, so its column is that iteration. Only seed 15005 enters the window, at the tight optimization of MTD iteration 5. The other five CREST runs never do. `target_energy` was off; both arms ran their full course.

| Seed | Anneal eV | Anneal wall (s) | First hit (s) | Calls to first hit | CREST first in-window | CREST eV | CREST wall (s) | Hit | Converged | Basins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15001 | −2768.546660 | 1219.371 | 541.139 | 12426 | none | −2768.225151 | 569.091 | anneal | 185 | 69 |
| 15002 | −2768.380113 | 1153.458 | none | none | none | −2767.551086 | 997.525 | neither | 168 | 69 |
| 15003 | −2768.609629 | 1182.113 | 380.134 | 9004 | none | −2767.194128 | 611.398 | anneal | 165 | 75 |
| 15004 | −2768.714624 | 1283.792 | 358.224 | 7813 | none | −2768.378376 | 930.325 | anneal | 204 | 69 |
| 15005 | −2768.448144 | 1194.045 | none | none | MTD 5 tight | −2768.613713 | 603.027 | crest | 143 | 70 |
| 15006 | −2768.532340 | 1185.160 | none | none | none | −2764.338592 | 434.034 | neither | 158 | 72 |

## Where the (H2O)16 wall goes

Seed 16001, 28000 calls, same wrapper as the table above, scored −2214.506788 eV in 655.035 s. CREST on the same start scored −2214.590238 eV in 292.682 s. Of the anneal wall, 649.696 s is inside xtb and 5.338 s is outside it, 0.206 ms per call. That remainder does not account for the gap.

The 28000 charged calls are 191 converged quenches and 1 capped quench, 146.6 calls per converged quench. There were 177 hops. Every hop started inside the overlap gate (`proposal_overlap_rejected` 177) and was separated, then quenched in full (`screens` 0). Line-search overlap rejections (665) and geometry rejections (0) are not charged. SCC rejections were 0.

Of 177 proposals, 171 were accepted and 14 improved the incumbent:

| Move | Proposed | Accepted | Improved |
| --- | --- | --- | --- |
| hbrewire | 41 | 39 | 7 |
| gshake | 42 | 39 | 1 |
| gsurface | 29 | 29 | 1 |
| gcycle | 37 | 36 | 3 |
| gburst | 28 | 28 | 2 |

The wall is those full quenches. Shortening one is the quench-convergence work and is left untouched here.

## Move weights, same six seeds

The recommended water arm now pays the allocator only when a hop improves the incumbent, starts the hydrogen-bond rewire with a prior of three improvements, and after 32 hops without a new lowest minimum leaves from another stored basin for eight hops. Quench convergence is unchanged. A binary homotop polish was quenching every oxygen–hydrogen swap and spending the budget before a hop; that polish now stays on an atomic library. The search below is that binary, not the warm GFN2 calculator.

Anneal walls are the in-process clock around the 28000 calls. The shell clock ran longer on some seeds; those gaps are not search time. CREST walls are CREST's own total where it printed one. Seeds 15002 and 15004 aborted in CREST's structure crossing (`confcross_0.xyz` missing, exit 1). Their energies are the vtight score of the geometry CREST had already written, not a finished ensemble.

The anneal mean wall is 1185.3 s. The four CREST runs that finished average 782.7 s, and the anneal mean on those same four seeds is 1192.8 s, ratio 1.52. Anneal's scored energy is lower on all six seeds. Anneal hits three seeds and CREST hits none. Seed 15005, a miss in the table above, is now inside the window. Seed 15003 is 0.015 eV short of the reference. `target_energy` was off.

| Seed | Anneal eV | Anneal wall (s) | First hit (s) | Calls to first hit | CREST first in-window | CREST eV | CREST wall (s) | Hit | Converged | Basins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15001 | −2768.595186 | 1215.805 | 362.527 | 8349 | none | −2768.279689 | 439.020 | anneal | 182 | 95 |
| 15002 | −2768.493751 | 1155.901 | none | none | aborted | −2763.468598 | aborted | neither | 134 | 73 |
| 15003 | −2768.535856 | 1159.236 | none | none | none | −2767.414948 | 1430.880 | neither | 219 | 75 |
| 15004 | −2768.716925 | 1184.903 | 425.423 | 10053 | aborted | −2768.137684 | aborted | anneal | 172 | 79 |
| 15005 | −2768.555365 | 1194.294 | 759.912 | 17816 | none | −2768.459521 | 625.170 | anneal | 188 | 83 |
| 15006 | −2768.434243 | 1201.728 | none | none | none | −2764.305609 | 635.727 | neither | 172 | 77 |

Seed 16001 in this same session scored −2214.687531 eV in 785.859 s. CREST on that start scored −2214.521423 eV in 271.636 s. The 28000 calls are 220 converged quenches and 1 capped quench, 127.3 calls per converged quench, and 191 hops. Every hop started inside the overlap gate and was separated (`screens` 0, line-search overlap rejections 703, SCC rejections 0). Of 191 proposals, 186 were accepted and 29 improved the incumbent: hbrewire 44/44/6, gshake 48/46/13, gsurface 35/35/4, gcycle 40/38/6, gburst 24/23/0. The wall is still those quenches.
