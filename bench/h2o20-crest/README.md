Copyright (c) anneal developers

(H2O)20, seeds 2000-2005.
All arms evaluate GFN2 plus the one-sided O-H wall, onset 1.15 angstrom,
200 eV per square angstrom. `PAPER_BUDGETS` is a 28000 charged-call cap.
The corrected panel compares the default anneal quench, the opt-in capped
Lindh quench, and CREST. Each arm reuses the byte-identical random placement
from the original panel for its seed. The starts are random rigid waters,
not motifs, lattices, or group-indexed shells.

The host is bench, on core 0, in one UTS namespace. Runs are serial. The
shared rescore is
`xtb in.xyz --gfn 2 --opt vtight --acc 0.01 --iterations 500`.

Gradient diagnosis
------------------

The original rows at c2de652dca0bc75e2b0cb4afffac91fd656404e4 used the
buggy path. `xtb_getGradient` supplied dE/dx, but `xtb_callback` published
that buffer as an rgpot force. rgpot negated it to construct its objective
gradient, so descent received -dE/dx. All six original anneal logs stamp
that revision, charge 28000 calls, and record 860-986 converged quenches,
which proves the callback was exercised. Their final force audits do not
test the historical search direction.

The CREST driver also calls `RgpotObjective::xtb(...).value_and_gradient`,
so the sign correction changed that path too. No original CREST row was
reused. All three arms were rerun at
746ec03fc290730a35d5478f14505222f3296e77.

Corrected results
-----------------

The Cambridge / Wales--Hodges (H2O)20 minimum in this repository is the
TIP4P energy -872.9887544952 kJ/mol. This panel is GFN2-xTB, so that
energy is not a same-potential reference. The reference is therefore the
lowest corrected vtight rescore across all three arms:
capped Lindh seed 2001, -101.736454652391 Eh. A hit is at or below that
energy by 1e-6 Eh.

Against corrected CREST:

* Default anneal has 0 hits against 0, Fisher two-sided p 1, tie-corrected
  continuity-corrected Mann-Whitney p 0.2297662705, and paired Wilcoxon
  W+ 7, W- 14, exact two-sided p 0.5625, rank-biserial -0.3333333333.
  The median paired gap, anneal minus CREST, is -0.001842610832 Eh.
* Capped Lindh has 1 hit against 0, Fisher two-sided p 1, tie-corrected
  continuity-corrected Mann-Whitney p 0.09269580256, and paired Wilcoxon
  W+ 3, W- 18, exact two-sided p 0.15625, rank-biserial -0.7142857143.
  The median paired gap is -0.007150391244 Eh.

The default arm's best is seed 2002, -101.735090408601 Eh, which is
0.001364243790 Eh above the reference. Corrected CREST's best is seed
2003, -101.729297715777 Eh, 0.007156936614 Eh above the reference.

Median search times, seconds:

* default anneal: 992.580433 inside the potential, 6.6255025 outside;
* capped Lindh: 1057.313669 inside, 25.1452255 outside;
* corrected CREST: 0.3088715 inside the driver handle, 0.267022 outside.

Failure accounting
------------------

The corrected direction exposed a campaign-level failure rather than a
successful search. Every anneal run charged 26400 calls under the 28000
cap, produced zero converged quenches, exhausted its quench steps, and
failed the final force bar. Thus its search objective best is `inf`; the
reported energies are the required vtight rescoring of each emitted
`best.xyz`.

Every corrected CREST pre-optimization terminated before the CREST search,
after 7-13 driver calls, with one failed call. Its timing is consequently
short and must not be interpreted as a successful 28000-call search.
Those failures are retained as measured outcomes.

Rows are append-only and include the code revision and host. The original
c2de652d rows remain unchanged.
