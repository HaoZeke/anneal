# PR 211 r-next8 same-session gate

This protocol was fixed before any gate row was generated.

## Arms

- A is revision `bd73f89d2879eeff2fb1323ce2eec8ac13d70ce3`,
  tree `fb5ae7cc254401140f67bfc9188078fe07f110ca`.
- B is that same revision with the r-next8 payload applied.
  `git apply --index` must produce tree
  `210f9611263fa07119ef78e3ccbc23cd3c2c0f4d`.
  A different tree stops the gate. The payload commit is not part of
  the delivered series.
- A2 is a second release build of A, from an empty target directory.
- B2 is a second release build of B, from an empty target directory.

Each row records the revision, the source tree, and the binary
SHA-256. Builds are release, locked, and feature `rgpot-ex`, producing
`h2h_covalent`, `gfn2_covalent_h2h`, and `slab_random_relax`. Runs are
serial and pinned to one core. OpenMP, BLAS, and Rayon thread counts
are 1. GMIN is not built and not run. No run uses a quick or reduced
budget except the slab accounting invocations defined below.

## Cells

Search seeds are 2000 through 2011. The slab accounting cell is seed
2000 only. Covalent cells use the general arm and the binary's own
paper budget and reference. GFN2 cells use the general arm, budget
100000, and the references below. `RGPOT_XTB_ETEMP` is unset, so the
engine's default electronic temperature of 300 K is used. A GFN2 run
also receives a 1500 s wall stop. If that stop fires before the budget
and before a hit, the row is censored.

| Cell | Potential | Budget | Reference |
| --- | --- | ---: | ---: |
| sw10 | Stillinger-Weber | 100000 | -13.79658026 |
| sw13 | Stillinger-Weber | 100000 | -18.50399759 |
| sw20 | Stillinger-Weber | 400000 | -30.18725174 |
| tc10 | Tersoff | 100000 | -55.08979464 |
| tc20 | Tersoff | 400000 | -120.69889105 |
| gfn2-si10 | GFN2-xTB | 100000 | -460.68373809 eV |
| gfn2-si13 | GFN2-xTB | 100000 | -599.34546694 eV |
| gfn2-c10 | GFN2-xTB | 100000 | -571.32273783 eV |
| slab | CuH2 | 1 and 2 | accounting only |

Execution order is slab, then gfn2-si10 (seed 2000 first), then sw10,
sw13, tc10, sw20, tc20, then gfn2-c10, then gfn2-si13.

A hit is a recorded energy no more than 0.001 above the reference.
A completed miss scores as budget plus one in the rank test. A
censored row is neither a hit nor a miss.

## Physical hit

`physical_hit` is true only when the row is a hit and the geometry is
acceptable.

Covalent geometry is one connected component. Stillinger-Weber
coordinates are scaled by the covalent library's sigma before the
test. Tersoff coordinates are already in angstrom.

GFN2 geometry must be one connected component, and a fresh xtb 6.7.1
single point at 300 K, accuracy 0.01 and 500 iterations, must converge
its SCC with a maximum absolute Cartesian force component below
1e-3 eV/A. The fresh point is a new process with restart files absent.
Connectivity uses the search rule: a bond exists when the separation
is strictly below 1.25 times the sum of the Cordero covalent radii
(C 0.76 A, Si 1.11 A). Each GFN2 row records the minimum
nearest-neighbour distance, the maximum per-atom nearest-neighbour
distance, the detached flag, the fresh energy, the SCC status, and
the maximum force component.

## Slab accounting

`slab_random_relax` runs on the repository fixture
`examples/fixtures/cuh2_fcc_slab.con`, seed 2000.

- Budget 1 and `RELAX_STEPS=0`: the initial combined evaluation only.
  A zero iteration limit does not enter the line search.
- Budget 2 and `RELAX_STEPS=0`: that evaluation plus the certifying
  gradient call. The ledger must show 1 then 2 charges. Any other
  ledger count makes the slab cell `PARTIAL`, while the raw engine
  totals are still reported.

A forwarder counts entries to `rgpot_cuh2_force`. The combined count
is the budget-1 total. The grad-only count is the budget-2 total
minus the budget-1 total. Both raw totals are kept.

## Order and rows

Even seeds run `A B B2 A2`. Odd seeds run `A2 B2 B A`. B2 is not
pooled into the B/A ratio. Rows are appended and flushed after every
run. Outside time is the binary wall time minus potential time. Peak
anonymous resident memory is sampled in-process from the process
status at 1 ms. The trajectory digest is SHA-256 of the result and
arm lines after clock fields are removed. The energy digest is
SHA-256 of the printed best energy and the coordinate text.

## Analysis

On completed seeds of each search cell:

- two-sided Fisher exact on hit against miss, A versus B;
- two-sided Mann-Whitney U on calls-to-hit, with a miss as budget
  plus one;
- the p-values are reported and do not by themselves set the verdict;
- energies within 0.001 in the cell unit count as equal, as do
  differences at or below 1e-6.

A physical hit present for A and absent for B is a hit loss. An A hit
that is not physical, and that B does not record, is listed per seed
and is not a hit loss. A B hit that is not physical is a defect.

Outside time and the anonymous-resident peak use the seed-paired
geometric mean of B/A. The interval is a 10000-draw bootstrap with
seed 20261123 and percentiles 2.5 and 97.5. The same interval is
computed for A2/A. Report the time ratio against 1.10 and against
`max(1.10, A2/A upper bound)`. A time regression is real only when
the B/A lower bound exceeds both 1 and the A2/A upper bound. Memory
fails only when its lower bound exceeds `max(1.10, A2/A upper bound)`.
Censored seeds are omitted from these ratios. A time or memory failure
requires at least six complete pairs.

Verdicts, and no other labels:

- `FIX` if a physical A hit is lost, or B records a non-physical hit.
  For the slab, `FIX` when A matches A2, B matches B2, and the
  grad-only engine count of B is strictly greater than A. Combined
  counts are reported and do not have to differ.
- `FAIL` if there is a real time regression or a memory failure, and
  the cell is not `FIX`.
- `PARTIAL` if the cell is not `FIX` or `FAIL` and any seed is
  censored, a search cell has fewer than 12 complete seeds, or the
  slab counts are not the PASS or FIX pattern.
- `PASS` only for a complete cell with none of the above.

`FIX` and `PARTIAL` are never relabelled `PASS`.
