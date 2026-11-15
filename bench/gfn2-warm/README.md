# GFN2 warm start against a fresh guess

The wavefunction policy lives in rgpot. This tree only forwards
`RGPOT_XTB_WARM_MIN_GAP` when that variable is set. Unset, anneal does
not call the setter and the engine keeps its own policy. The setter is
not the default: on C10 a reused wavefunction missed a minimum the fresh
guess found.

Engine `4193327dd994b2a9a58f65df6579ded221eeffc7`. Consumer
`caf6f326`. Host `bench`, one core, serial.

## Frames

`gfn2_warm_gate` walks each `frames/<system>/*.xyz` in name order on one
session, then again with a restart before every frame. Electronic
temperature is 300 K. The gap guard is the engine default. A hit on the
energy comparison is a finite pair. Si13 frame 1 converges on neither
side. Every finite pair agrees to well inside 1e-6 eV.

| system | frames compared | max \|E_warm − E_fresh\| |
|---|---|---|
| Si13 | 6 of 7 | 0 eV |
| Si10 | 6 of 6 | 0 eV |
| C10 | 6 of 6 | 6.594e-12 eV |

`frames.csv` is that table. `frames/*/xtb-sp-eV.txt` is the standalone
xtb single point, `nan` where that reference does not converge.

## Search

Si10 and C10, seeds 2000–2011, budget 100000 calls, electronic
temperature 3000 K. Warm sets `RGPOT_XTB_WARM_MIN_GAP=1`. Fresh sets it
to 1e9, which never reuses a wavefunction. A hit is a recorded energy
within 1e-3 eV of the covalent reference: Si10 −460.68373809 eV, C10
−571.32273783 eV. `rows.csv` is one row per run. `pot_s` is time inside
the potential. `outside_s` is wall minus `pot_s`. Misses are entered at
the budget.

| system | mode | hits | median calls | median wall | median inside/call | median outside/call |
|---|---|---|---|---|---|---|
| Si10 | warm | 11/12 | 954 | 71.12 s | 0.06185 s | 1.62e-5 s |
| Si10 | fresh | 11/12 | 954 | 70.71 s | 0.06151 s | 1.62e-5 s |
| C10 | warm | 11/12 | 1174 | 5.97 s | 0.00500 s | 8.45e-6 s |
| C10 | fresh | 12/12 | 784 | 4.61 s | 0.00603 s | 1.11e-5 s |

Si10 call counts and best energies match on every seed. Seed 2011 misses
on both sides at −459.96894856 eV. Fisher exact p = 1. Mann–Whitney
U = 72, z = 0, two-sided p = 1.

C10 warm misses seed 2003 and stops at −568.81755877 eV, 2.505 eV above
the fresh hit. The other eleven seeds land on the same energy. Fisher
exact p = 1. Mann–Whitney on the call counts, misses at the budget,
normal approximation with the tie correction, U = 91, z = 1.10,
two-sided p = 0.273. The warm arm spends less time inside each call and
more calls overall, so the median wall is longer.

Warm start matches a fresh guess on the frames and does not win the
search. The consumer stays off unless the caller sets the gap.
