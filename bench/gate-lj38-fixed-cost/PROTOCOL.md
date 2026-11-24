# lj38 fixed-cost default-on gate protocol

Written before any data. Nothing here changes after the first run.

## Subjects

A is develop `9b65fda3972fd8c014bce886a3ea5c05956d64af`, tree
`9b65fda3^{tree}`.

B is `f9bbb5e86a67b7bbfaa629ac7378306f1f22cd1d`, tree
`510c087e8c6e4c9aeedae202204a7919cee95a60`: five linear commits on A.
The payload mbox replays them on A and gives that tree exactly, so the
tree is the identity of B.

A2 is a second, independent release build of A. In the row files it is
the arm `Ap`.

Bc is B plus a counter patch. It is not a timed arm. It only shows that
the changed code paths run.

The change touches `candidate_from_basis` (src/lattice.rs), the AVX2
Kuhn-Munkres and shortest-path code (src/assignment.rs), the class
blocks and op hashes (src/csm.rs), and drops `PackingBank`
(src/methods/cluster_hopping.rs and its packing_bank.rs). It adds a
newsfragment.

## Host

The changed code has AVX2 loops, so the session needs an AVX2 host.
The runner refuses to start without the `avx2` flag. The CPU model and
flags are recorded in sessions.csv.

## Arms and schedule

Cells run one at a time, serially, on core 0 (`taskset -c 0`). Every
environment variable whose name ends in `_NUM_THREADS` is one. Release
builds only. There is no quick mode, no warm-up, no retuning, no
rescue, no hand-picking and no relabelling. No GMIN run is part of this
gate.

Seeds start at 2000. Each cell runs palindromic blocks of two seeds: A,
B, A2 on the first seed, then A2, B, A on the second. Each arm has the
same mean position in every cell. Budgets and stop levels are identical
in every arm. All arms run on one VM in one session. A session that
cannot finish is sealed as a partial and is not extended on another VM.

## Cells

Default-preset cells (stop at published minimum plus 0.001):

| cell | n | seeds | budget | published minimum |
|---|---|---|---|---|
| lj38 | 24 | 2000-2023 | 400000 | -173.928427 |
| m38r14d | 12 | 2000-2011 | 400000 | -144.321054 |
| lj75 | 12 | 2000-2011 | 4000000 | -397.492331 |
| lj98 | 12 | 2000-2011 | 4000000 | -543.665361 |
| m80r6 | 12 | 2000-2011 | 4000000 | -378.333471 |
| au55 | 12 | 2000-2011 | 1000000 | -191.03085280 |
| hexamer | 12 | 2000-2011 | 500000 | -197.7805260034 |

That is 96 runs per arm and 288 runs in all.

## Measurements

Every row records the hit, the budget, the calls to the hit (or the
whole budget on a miss), the charged calls, wall seconds, seconds
inside potential calls, outside seconds per call, the best-energy bits,
the trajectory hash, and `RssAnon` in kB. Rows are append-only. A failed
row is kept and makes its cell INCONCLUSIVE. There are no replacement
runs.

Memory is the peak `RssAnon`, read in-process: a sampler thread reads
it every 10 ms for the whole run, and the end-of-run value is also
taken. The row holds the larger as the peak and the end value as
`rss_end_kb`. The sampler runs in every arm.

Outside time per call is (wall minus inside) divided by calls. Outside
time in total is wall minus inside. Both are reported for every cell,
side by side.

## Statistics

Random procedures use seed 20261123 and 10000 seed-level bootstrap
resamples. Every ratio is seed-paired B/A, summarised as a geometric
mean with a 95 percent percentile interval. The same is computed for
A2/A as the A/A band.

1. Bit identity, strict. For every seed of every cell, A and B must
   agree on the hit, the calls, the charged calls, the best-energy bits
   and the trajectory hash. Any difference is a FIX. A against A2 is
   checked the same way and a difference is reported as a note, because
   it would mean the run is not repeatable.
2. No hit loss. Fisher's exact test on hits (one-sided, B fewer) and a
   one-sided Mann-Whitney test on calls with every miss scored as
   budget + 1 (B larger). A hit loss is p < 0.05 in either test.
3. Outside time, default-on bar. The ratio B/A has upper bound at most
   1.10, for the per-call figure and for the total. Both versions are
   reported: the strict one, upper at most 1.10, and the band one,
   upper at most max(1.10, upper of A2/A). The verdict uses the band
   version. A regression is real only if the lower bound is above both
   1 and the upper of A2/A.
4. Memory. The peak RssAnon ratio B/A. The one-sided gate fails only if
   its lower bound exceeds max(1.10, upper of A2/A). Upper at most 1.10
   is also reported.
5. Early-stop artifact. A run that stops at a hit has fewer calls, so
   its per-call outside time carries a fixed cost over fewer calls.
   A cell with any early-stopped run is flagged. If the strict identity
   rule holds, the calls of A and B are equal and per-call and total
   ratios coincide; if they ever disagree, that is stated and neither is
   relabelled.
6. Builder smoke claim. The builder reports an lj38 per-call outside
   ratio of about 0.92 against A. It is unverified. The lj38 cell
   reports the ratio with its interval and whether the interval upper
   is below 1; the claim is not an input to the verdict.

## Path counter

Bc counts entries to `solve_shortest_paths_loop_avx2`,
`kuhn_munkres_loop_avx2`, `augment_tight_avx2`, `recall_op` in csm, and
`candidate_from_basis`. After the session, Bc runs lj38 seed 2000, lj38
seed 2001 and m38r14d seed 2000, serial on core 0. Each counter must be
nonzero in the sum, and each Bc run must equal the timed B row of the
same cell and seed on trajectory, energy bits, calls and charged calls.
A zero counter is reported as unexercised and does not become a PASS
for that path. The drop of `PackingBank` has no counter: the type no
longer exists in B.

## Verdict

Per cell: FIX if any rule above fails, INCONCLUSIVE if rows are missing
or errored, PASS-BAND if only the strict outside-time or memory bound is
missed but the band version holds, otherwise PASS. A FIX is never
relabelled as a PASS. The gate is PASS only if every cell is PASS or
PASS-BAND and the path counter check holds.

Energy agreement up to spectroscopic accuracy would be enough for a
converged or repeatable quantity, and differences near 1e-9 are not
flagged elsewhere. The identity fields in rule 1 are the default-parity
bar and stay strict.

## Known gaps

- No molecular cells. The change is in cluster code and the default
  cluster cells are where its paths run.
- No simple-cubic or TIP4P cell beyond the water hexamer: those run in
  other harnesses and are not in this one.
- The counter build is run after the session and is not timed.
