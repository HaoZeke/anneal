Copyright (c) anneal developers

(H2O)20, seeds 2000-2011.
All arms evaluate GFN2 plus the one-sided O-H wall, onset 1.15 angstrom,
200 eV per square angstrom. `PAPER_BUDGETS` is a 28000 charged-call cap.
The panel compares the default anneal quench, the opt-in capped Lindh
quench, and CREST. Each arm reuses the byte-identical random placement for
its seed; seeds 2000-2005 retain the original panel starts. The starts are
random rigid waters, not motifs, lattices, or group-indexed shells.

The host is bench, on core 0, in one UTS namespace. Runs are serial. The
shared rescore is
`xtb in.xyz --gfn 2 --opt vtight --acc 0.01 --iterations 500`.

Gradient diagnosis
------------------

The pinned engine's `rgpot_xtb_force` ABI returns physical force,
`-dE/dx`, in eV/angstrom. rgpot then negates that force once to expose the
objective gradient. Revision 746ec03f incorrectly negated the engine
buffer before rgpot, so both anneal and the CREST driver received
`-dE/dx`. That extra negation, not units, coordinate order, caching, or the
convergence test, caused the zero-quench and failed-preoptimization panel.
The callback now publishes the engine force unchanged.

The full rgpot objective was checked against central differences at two
random starts and one vtight minimum, nine components each. The maximum
absolute errors were 2.734134e-6, 2.849674e-6, and 9.162368e-7 eV/angstrom.
An rgmin FIRE quench of the known minimum converged in 1431 steps from
-2768.389962843636 to -2768.390859675824 eV, with final gradient norm
9.959564e-5 eV/angstrom.

The CREST driver uses the same callback, so all three arms were rerun at
131b411a31a2ef87669cba37b692e400819a595f. Every arm charged 28000 calls.
Default anneal produced 660-1045 converged quenches per seed, capped Lindh
336-686, and every final force audit passed. Every CREST run reached its
28000-call stop.

Corrected results
-----------------

The Cambridge / Wales--Hodges (H2O)20 minimum in this repository is the
TIP4P energy -872.9887544952 kJ/mol. This panel is GFN2-xTB, so that
energy is not a same-potential reference. The reference is therefore the
lowest corrected vtight rescore across all three arms:
capped Lindh seed 2008, -101.749249454294 Eh. A hit is at or below that
energy by 1e-6 Eh.

Against corrected CREST:

* Default anneal has 0 hits against 0, Fisher two-sided p 1, tie-corrected
  continuity-corrected Mann-Whitney p 3.658455354e-5, and paired Wilcoxon
  W+ 0, W- 78, exact two-sided p 0.00048828125, rank-biserial -1.
  The median paired gap, anneal minus CREST, is -0.01111905604 Eh.
* Capped Lindh has 1 hit against 0, Fisher two-sided p 1, tie-corrected
  continuity-corrected Mann-Whitney p 3.658455354e-5, and paired Wilcoxon
  W+ 0, W- 78, exact two-sided p 0.00048828125, rank-biserial -1.
  The median paired gap is -0.01022974055 Eh.

The default arm's best is seed 2001, -101.748641921094 Eh, which is
0.000607533200 Eh above the reference. CREST's best is seed 2006,
-101.738019426743 Eh, 0.011230027551 Eh above the reference.

Median search times, seconds:

* default anneal: 1251.828097 inside the potential, 489.955743 outside;
* capped Lindh: 1245.466106 inside, 67.2645165 outside;
* CREST: 1263.008673 inside the driver handle, 78.590811 outside.

Rows are append-only and include the code revision and host. The original
c2de652d and inverted 746ec03f rows remain unchanged.
