# PR 211 r-next2 bit-identity gate

This protocol and its harness are committed before any timed row.

## Arms

Identity is the production parent of the single harness commit.
The harness adds this directory and one record line in each driver.
It does not change the library.

- A is `5abb4dedec6cd2e95e7b17472639529fd139b278`, tree
  `23bd3a7458d5b709e35a711389f9fb0f6a1f7264`.
- A2 is an independent release build of that same harness tree.
- B is `4a2c4a6021fdc4aaf61e00f265a1bf6bb8b0f3cc`, tree
  `b57a30db80d41fddce42b4509bc40d9004261779`.

B is the eleventh applied patch. Its parent is the tenth.
The three driver sources must be byte-identical across arms.
Each arm is built once, release, locked, in its own target directory.
Processes are fresh, serial, and pinned to one core, with every
thread variable set to 1. Quick mode is not used. No external
competitor executable, source, or result is built or run. The
Lennard-Jones cells call the anneal subcommand only.

## Cells

Seeds are 2000 through 2011. Even seeds run A, B, A2, B.
Odd seeds run B, A2, B, A. A and A2 have one row per seed.
B has two, and the two deterministic outcomes must agree.

TIP4P cells are tip4p9, tip4p10, tip4p14, tip4p16, and tip4p17.
Each uses `h2h_seed` with the recommended preset, budget 2000000,
and a stop at the table energy plus 0.001. Table energies are
-344.4352092012, -391.0226842125, -582.9911036211,
-681.1925945898, and -723.8077531390.

Lennard-Jones cells are lj38 and lj75, via the recommended preset,
budgets 400000 and 4000000, references -173.928427 and -397.492331.
Morse is m55r6, budget 1000000, reference -250.286609.
A hit is the driver hit: quenched best at most reference plus 0.001.

## Record line

The line carries `unconverged_records`, the FNV-1a 64-bit hash of
the exact bits of every recorded energy in order, the record count,
best to 17 significant digits, the exact best bits, hops, basins,
and visits. Recorded energies are the improvement entries, in order.
The hash offset is `cbf29ce484222325` and the prime is
`100000001b3`. Each energy contributes its eight little-endian
IEEE-754 bytes. `unconverged_records` is the refusal counter.

## Analysis

A row matches when the energy hash, best bits, hops, basins, visits,
and unconverged count all match A. Identical seeds are counted out
of 12. Both B repeats must match. The same count is reported for A2.

Outside time is wall minus time inside the potential, floored at
zero. B's seed value is the mean of its two rows. The ratio is that
value over A's. The reported ratio is the type-7 median of the 12
seed ratios. The 95 percent interval is the type-7 2.5 and 97.5
percentiles of 10000 medians from seed resampling. One generator,
seed 21102, advances in cell order, B/A then A2/A.

Fisher's exact test is one-sided: the chance that A has at least as
many hits as observed, given the summed hits and 12 seeds each.
Mann-Whitney is one-sided for a larger B score, with the tie
correction and a continuity correction of one half. A miss scores
budget plus one. B contributes the shared seed score once.

## Verdict

A cell passes only when every condition below holds.

- Every B row matches A, 12 seeds out of 12.
- Every A2 row matches A. A broken control is a failure.
- B loses no hit that A found, and the two B rows agree
  on the hit and on the call count.
- The B/A outside-time upper bound is at most 1.10.

The adjusted limit, the maximum of 1.10 and the A2/A upper bound,
is reported beside that rule and does not replace it.
A differing row names the first seed and the first recorded energy
that disagrees, and says whether the refusal counter rose.
The gate passes only when every cell passes.
