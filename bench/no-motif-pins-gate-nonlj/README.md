# Non-LJ no-loss gate for the no-motif-pins kit

**Gate: FIX: Sutton-Chen (sc126-55 calls)**

Arm A is `ef1e1d07ad214d545cfa0bcb0c2e15b8be3bcbcb`. Arm B is A with the six
patches of the no-motif-pins kit applied, tree
`b1b89bd25f04bede362c41e65b0eed914d438489`. 25 systems in 6 panels, seeds 2000
to 2011, both arms: 600 runs, 0 failed.

- System tests lost by B at p < 0.05: sc126-55 (calls).
- Panel tests lost by B at p < 0.05 on the pooled tests: none.
- System tests won by B at p < 0.05: none.
- Over the 25 systems, B has more hits on 4 and fewer on 9. Its mean rank on
  calls to the minimum is lower on 5 and higher on 15. Total hits are 171 for A
  and 166 for B.
- The two arms return the same hit, calls and energy on 69 of the 300 (system,
  seed) pairs.
- sc126-55, lost on calls: hits 12/12 for A and 12/12 for B (Fisher p 1); median
  calls 18,379.5 for A and 46,601.5 for B (Mann-Whitney p 0.0351); on the same
  seed B needs more calls on 9 seeds, fewer on 3 and the same on 0.

## Panels

| panel | systems | hits A | hits B | Fisher p | p on calls | calls | wall A (s) | wall B (s) | inside A (s) | inside B (s) | outside A (s) | outside B (s) | same result | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BLJ | 7 | 30/84 | 25/84 | 0.328 | 0.55 | B more calls | 26,011 | 24,322 | 1,323 | 1,323 | 24,688 | 22,999 | 29/84 | PASS |
| Morse M75 rho14 | 1 | 9/12 | 8/12 | 1 | 0.231 | B more calls | 618 | 1,287 | 442 | 692 | 176 | 596 | 0/12 | PASS |
| Morse M80 rho14 | 1 | 6/12 | 5/12 | 1 | 0.975 | B more calls | 1,496 | 1,865 | 1,090 | 1,044 | 406 | 821 | 1/12 | PASS |
| Morse M80 rho6 | 1 | 12/12 | 11/12 | 1 | 0.795 | B fewer calls | 213 | 317 | 143 | 217 | 70.2 | 99.8 | 0/12 | PASS |
| Sutton-Chen | 9 | 75/108 | 78/108 | 0.696 | 0.22 | B more calls | 1,658 | 1,991 | 1,231 | 1,270 | 427 | 721 | 6/108 | FIX (sc126-55:calls) |
| water/TIP4P rigid | 6 | 39/72 | 39/72 | 1 | 0.734 | B more calls | 823 | 861 | 622 | 652 | 202 | 209 | 33/72 | PASS |

Fisher p is the two-sided exact test on hits, stratified by system for a panel
of several systems. The p on calls is the two-sided Mann-Whitney test with the
tie and continuity corrections for one system, and the van Elteren test (the
same rank test, stratified by system, tie-corrected, no continuity correction)
for a panel of several. Wall, inside and outside are the sums over the runs of a
panel, in seconds. Inside is the time inside the potential, outside is the rest
of the wall. "Same result" counts the pairs on which both arms return the same
hit, calls and energy.

| panel | wall B/A | inside B/A | outside B/A |
| --- | --- | --- | --- |
| BLJ | 0.9351 | 1.0007 | 0.9316 |
| Morse M75 rho14 | 2.0838 | 1.5651 | 3.3877 |
| Morse M80 rho14 | 1.2464 | 0.9574 | 2.0221 |
| Morse M80 rho6 | 1.4898 | 1.5230 | 1.4223 |
| Sutton-Chen | 1.2008 | 1.0317 | 1.6885 |
| water/TIP4P rigid | 1.0458 | 1.0498 | 1.0335 |
| all panels | 0.9943 | 1.0719 | 0.9798 |

## Systems

| system | budget | hits A | hits B | Fisher p | calls A | calls B | MW p | wall A (s) | wall B (s) | inside A (s) | inside B (s) | outside A (s) | outside B (s) | same result | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tip4p8 | 2,000,000 | 12/12 | 12/12 | 1 | 6,339 | 6,339 | 1 | 0.021 | 0.020 | 0.012 | 0.012 | 0.008 | 0.008 | 12/12 | PASS |
| tip4p12 | 2,000,000 | 12/12 | 12/12 | 1 | 41,418.5 | 48,553 | 0.795 | 0.225 | 0.270 | 0.153 | 0.182 | 0.073 | 0.088 | 10/12 | PASS |
| tip4p15 | 2,000,000 | 7/12 | 10/12 | 0.371 | 1,481,620 | 1,078,000 | 0.365 | 11.4 | 8.25 | 8.43 | 6.22 | 2.93 | 2.03 | 4/12 | PASS |
| tip4p16 | 2,000,000 | 2/12 | 3/12 | 1 | 2,000,000 | 2,000,000 | 0.745 | 17.8 | 17.7 | 13.2 | 13.2 | 4.63 | 4.55 | 2/12 | PASS |
| tip4p20 | 2,000,000 | 6/12 | 2/12 | 0.193 | 1,686,250 | 2,000,000 | 0.0632 | 20.0 | 24.5 | 15.6 | 18.6 | 4.37 | 5.32 | 2/12 | PASS |
| tip4p21 | 2,000,000 | 0/12 | 0/12 | 1 | 2,000,000 | 2,000,000 | 1 | 25.7 | 25.6 | 19.4 | 19.5 | 6.26 | 6.20 | 3/12 | PASS |
| sc126-38 | 400,000 | 12/12 | 12/12 | 1 | 113,504 | 116,912 | 0.751 | 1.04 | 1.12 | 0.789 | 0.828 | 0.247 | 0.294 | 0/12 | PASS |
| sc108-38 | 400,000 | 7/12 | 9/12 | 0.667 | 361,032 | 235,043 | 0.239 | 3.79 | 2.55 | 2.82 | 1.91 | 0.972 | 0.637 | 2/12 | PASS |
| sc96-38 | 400,000 | 12/12 | 12/12 | 1 | 50,230 | 125,848 | 0.0885 | 0.471 | 1.29 | 0.356 | 0.912 | 0.114 | 0.379 | 0/12 | PASS |
| sc126-55 | 1,000,000 | 12/12 | 12/12 | 1 | 18,379.5 | 46,601.5 | 0.0351 | 0.364 | 0.911 | 0.266 | 0.670 | 0.111 | 0.242 | 0/12 | FIX (calls) |
| sc108-55 | 1,000,000 | 5/12 | 5/12 | 1 | 1,000,000 | 1,000,000 | 0.949 | 20.7 | 20.7 | 16.1 | 16.2 | 4.63 | 4.50 | 0/12 | PASS |
| sc96-55 | 1,000,000 | 12/12 | 12/12 | 1 | 23,157 | 21,727.5 | 0.403 | 0.507 | 0.515 | 0.376 | 0.362 | 0.124 | 0.157 | 0/12 | PASS |
| sc126-80 | 1,000,000 | 5/12 | 5/12 | 1 | 1,000,000 | 1,000,000 | 0.797 | 44.2 | 43.5 | 31.8 | 31.2 | 12.2 | 12.7 | 1/12 | PASS |
| sc108-80 | 1,000,000 | 3/12 | 1/12 | 0.59 | 1,000,000 | 1,000,000 | 0.248 | 51.0 | 71.0 | 38.4 | 39.2 | 12.2 | 32.6 | 3/12 | PASS |
| sc96-80 | 1,000,000 | 7/12 | 10/12 | 0.371 | 989,848 | 830,532 | 0.121 | 42.9 | 36.1 | 31.7 | 27.3 | 11.1 | 8.44 | 0/12 | PASS |
| m75r14 | 4,000,000 | 9/12 | 8/12 | 1 | 963,277 | 2,039,210 | 0.231 | 33.2 | 90.0 | 24.4 | 48.9 | 8.81 | 41.1 | 0/12 | PASS |
| m80r14 | 4,000,000 | 6/12 | 5/12 | 1 | 3,901,430 | 4,000,000 | 0.975 | 152 | 199 | 110 | 110 | 38.2 | 83.6 | 1/12 | PASS |
| m80r6 | 4,000,000 | 12/12 | 11/12 | 1 | 62,677 | 45,108.5 | 0.795 | 3.12 | 1.96 | 1.98 | 1.39 | 1.15 | 0.625 | 0/12 | PASS |
| blj105n38 | 400,000 | 1/12 | 0/12 | 1 | 400,000 | 400,000 | 0.359 | 35.0 | 45.5 | 1.57 | 1.55 | 33.5 | 44.0 | 10/12 | PASS |
| blj115n38 | 400,000 | 9/12 | 7/12 | 0.667 | 138,864 | 79,757.5 | 1 | 2.56 | 3.37 | 0.603 | 0.351 | 1.93 | 3.03 | 2/12 | PASS |
| blj130n38 | 400,000 | 4/12 | 3/12 | 1 | 400,000 | 400,000 | 0.615 | 23.7 | 32.3 | 1.56 | 1.63 | 22.1 | 30.6 | 6/12 | PASS |
| blj105n55 | 1,000,000 | 12/12 | 12/12 | 1 | 204,047 | 210,096 | 1 | 15.9 | 12.2 | 1.77 | 1.89 | 13.7 | 10.3 | 0/12 | PASS |
| blj115n55 | 1,000,000 | 0/12 | 0/12 | 1 | 1,000,000 | 1,000,000 | 1 | 304 | 267 | 8.30 | 8.22 | 296 | 259 | 4/12 | PASS |
| blj130n55 | 1,000,000 | 4/12 | 3/12 | 1 | 1,000,000 | 1,000,000 | 0.746 | 119 | 125 | 7.35 | 7.33 | 112 | 118 | 4/12 | PASS |
| blj130n98 | 4,000,000 | 0/12 | 0/12 | 1 | 4,000,000 | 4,000,000 | 1 | 1,545 | 1,382 | 90.6 | 90.6 | 1,455 | 1,292 | 3/12 | PASS |

Calls are the median calls to the minimum with a miss entered at the budget.
Wall, inside and outside are medians over the runs of one arm, in seconds
(Hyndman-Fan type 7). The verdict of a system is FIX when B has fewer hits and
Fisher p < 0.05, when B ranks higher on calls and the Mann-Whitney p < 0.05, or
when a run of B fails where the same seed of A does not. Otherwise it is PASS.

## What ran

- Arm A: commit `ef1e1d07ad214d545cfa0bcb0c2e15b8be3bcbcb` as it is.
- Arm B: A plus `git am` of the kit. The tree is `b1b89bd25f04bede362c41e65b0eed914d438489`.
- Drivers: `examples/h2h_morse.rs`, `examples/h2h_sc.rs`, `examples/h2h_blj.rs`
  and `examples/h2h_seed.rs`, built once per arm from that arm's tree. The
  harness commits of this series add the M75 rho 14 system to the Morse driver,
  the BLJ driver, and the runner `scripts/no_motif_pins_gate/run_gate.py`. They
  touch no library source: `src/`, `Cargo.toml`, `Cargo.lock`, `build.rs` and
  `data/` are those of A in arm A and those of the kit tree in arm B. The
  Sutton-Chen driver is built with the `rgpot-ex` feature and loads the rgpot
  Sutton-Chen kernel at runtime.
- The `harness` column of `rows.csv` is the commit that carries the runner in
  the repository where the series was written. `git am` gives it another id.

| panel | systems | budget | reference |
| --- | --- | --- | --- |
| water/TIP4P rigid | (H2O)n, n = 8, 12, 15, 16, 20, 21 | 2,000,000 | stored TIP4P global minimum, `data/tip4p_ccd.csv` |
| Sutton-Chen | 12-6, 10-8 and 9-6 families at 38, 55 and 80 atoms | 400,000 at 38; 1,000,000 at 55 and 80 | the driver's table |
| Morse M75 rho 14 | 75 atoms, rho 14 | 4,000,000 | the driver's table |
| Morse M80 rho 14 | 80 atoms, rho 14 | 4,000,000 | the driver's table |
| Morse M80 rho 6 | 80 atoms, rho 6 | 4,000,000 | the driver's table |
| BLJ | Doye-Meyer binary Lennard-Jones, sigma_BB 1.05, 1.15, 1.30 at 38 and 55 atoms, 1.30 at 98 | 400,000 at 38; 1,000,000 at 55; 4,000,000 at 98 | Cambridge Cluster Database header energies |

`panels.csv` has the budget and the reference energy of every system. No budget
is set by the runner: each driver reports its own paper budget, and the runner
checks that the two arms report the same one.

The brief names water/TIP4P and a rigid/molecular case as two panels. The only
rigid-body molecular search of this tree that needs no external engine is the
TIP4P water cluster, so the one panel serves both. It spans (H2O)8, on the
10-body-or-fewer side of the removed rigid hot-body split, and (H2O)12 to
(H2O)21 on the other. The covalent and conformer head-to-heads need engines
that are not available here (GFN2, xtb, CREST, openconf) and are not in the gate.

## Method

- Seeds 2000 to 2011 (n = 12), every system, both arms. A hit is a quenched
  energy at or below the stored minimum plus 0.001. A miss is entered at the
  budget. Calls to the minimum are `evals_to_hit`, the potential calls that the
  driver counts up to the first hit.
- One session, serial, one process at a time, pinned to one core, every thread
  pool at one thread, in a private UTS namespace. The runner refuses to start
  unless the session hostname is the one it is told to expect.
- Each (system, seed) is a round. The two arms run back to back and the arm that
  goes first flips with the seed: A first on even seeds, B first on odd seeds.
- Cheapest systems first: TIP4P, Sutton-Chen, Morse, BLJ.
- No `--quick`, no warm-up run, no rescue run, no retuning after results. A run
  that failed or timed out would be recorded once and scored as a miss at the
  budget. 0 did.
- Before the session each driver of each arm ran once at a tiny call budget,
  and the BLJ energies of the seven stored structures were checked against
  their headers (all within 1e-10). Those runs are not in `rows.csv`.
- The session was stopped once and resumed from the recorded rows after 576 of
  the 600 runs, to raise the per-job timeout from 2 h to 12 h. The first BLJ98
  job was running and was discarded after about 47 minutes without a result. No
  recorded run was repeated and no run reached either timeout.
- The platform paused the machine twice in the first hour of the session, for
  about 13 and 7 minutes, during the jobs that wrote data rows 123 and 251 of
  `rows.csv` (tip4p21 seed 2001 arm B and sc108-55 seed 2005 arm B). Every time
  is a monotonic-clock interval, so a pause is in no wall, and those two walls
  are in the range of their neighbours.

## Context for the verdict

The rule is applied as given: any loss at p < 0.05 on hits or calls is FIX. It
reads 25 systems on two tests each, plus the pooled tests of the panels. If the
arms did not differ, each system test would flag B as worse with probability at
most 2.5%, so up to about 1.25 of the 50 system tests would flag B by chance.
The ties in hits and calls make the real figure lower. The timing columns are
not part of the rule.

## Rerun

Build the four drivers once per arm (`h2h_morse`, `h2h_seed` and `h2h_blj` with
`cargo build --release --example`, `h2h_sc` the same way with the `rgpot-ex`
feature), then run `scripts/no_motif_pins_gate/run_gate.py run` with the two
driver directories, one identity string per arm, the expected session hostname,
the rgpot library and a scratch directory, from a private UTS namespace on one
core. `run_gate.py summarize` and `panel_summary.py` rebuild the two summary
files from `rows.csv` and `panels.csv`. The runner resumes from `rows.csv` and
never repeats a recorded run.

## Files

- `rows.csv`: one row per run, appended as each run finished. `position` is 1
  for the arm that ran first in the round. `status` is `ok` for every row.
- `panels.csv`: driver, budget and reference energy of each system.
- `summary.csv`: the system tests, from `scripts/no_motif_pins_gate/run_gate.py summarize`.
- `panel_summary.csv`: the panel tests, from `scripts/no_motif_pins_gate/panel_summary.py`.
- `bin-sha256.txt`: sha256 of the eight drivers that ran, four per arm.
