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

## Warm SCC guess

The same six seeds ran again, both arms, same budget, same wrapper. The worker keeps one GFN2 calculator. The SCC wavefunction is the guess while no atom moves by more than 1 Å, and a larger move starts from xtb's own guess. Search single points use accuracy 1. That energy matches an accuracy 0.01 evaluation of the same geometry to 1e-8 Hartree. The geometry in the table is still the vtight accuracy 0.01 score.

A cold native `xtb --gfn 2 --acc 0.01` single point of the (H2O)20 start spends 44 ms in the SCC and 52 ms of wall. Of the SCC, 5 ms is integrals, 22 ms the iterations, and 16 ms the gradient. Reusing the wavefunction is what shortens the iterations. The mean single point here is 34.426 ms inside xtb and 0.206 ms outside it. The mean inside time is 963.980 s and the mean outside time is 5.770 s.

The anneal mean wall is 969.800 s. This session's CREST mean wall is 963.161 s. The ratio of those means is 1.007. Anneal is lower than CREST on all six seeds. Anneal hits three seeds and CREST hits none. The earlier full-course rows are unchanged.

| Seed | Anneal eV | Anneal wall (s) | ms inside | CREST eV | CREST wall (s) | Hit | Converged | Basins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15001 | −2768.679345 | 983.207 | 34.907 | −2768.208786 | 1028.142 | anneal | 155 | 65 |
| 15002 | −2768.475111 | 947.384 | 33.621 | −2767.472620 | 1307.079 | neither | 139 | 85 |
| 15003 | −2768.578540 | 902.479 | 32.026 | −2767.436057 | 1320.271 | anneal | 157 | 64 |
| 15004 | −2768.554796 | 988.487 | 35.098 | −2768.199175 | 819.176 | anneal | 174 | 72 |
| 15005 | −2768.435015 | 993.687 | 35.282 | −2768.323019 | 796.199 | neither | 163 | 72 |
| 15006 | −2768.526178 | 1003.558 | 35.623 | −2766.024110 | 508.100 | neither | 150 | 97 |

(H2O)16 seed 16001, same budget and the same session, is a check rather than a hit against the (H2O)20 reference. Anneal scored −2214.506788 eV in 655.074 s, 23.203 ms inside xtb. CREST scored −2214.590238 eV in 292.682 s.
