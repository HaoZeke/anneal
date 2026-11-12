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
