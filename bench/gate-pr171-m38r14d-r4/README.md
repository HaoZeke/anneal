# Gate for morse m38 rho 14, round 4

BASE is 61fa67cf380e46876b569da45e7cc091647a775b,
tree cff796c507a8fa1d0bfb28888c3570b61988055e.
Arm A is that tree. Arm A2 is a second release build
of the same tree, in its own worktree and target
directory. Arm B is
633d5ce5e8176ebd99cfe13f779a10752a059910, tree
a5a1423fb2c00475962770dd792f37944cf61174. The harness
checks those four ids before any timed row and stops
on a mismatch.

## Reach

The PR adds observed_basis_candidate
(src/lattice.rs:650 on the PR tree) and calls it from
run_full when a directed leave is due
(src/methods/cluster_hopping.rs:9252).
claim_observed_basis_probe
(src/methods/cluster_hopping.rs:2948) accepts a basin
only when the move library is not a rigid body, no
atoms are frozen, and no active region is set.
directed_leave follows escape_on_stall and the call
plateau (src/methods/cluster_hopping.rs:5858). The
climb predicate beside the probe also changed
(src/methods/cluster_hopping.rs:9300).

run_with_gradient (src/methods/cluster_hopping.rs:2653)
and run_with_energy_gradient
(src/methods/cluster_hopping.rs:2680) both enter
run_full (src/methods/cluster_hopping.rs:4199).
Config::recommended sets escape_on_stall
(src/methods/cluster_hopping/config.rs:1842).

h2h_gmin_proof builds Config::recommended and calls
run_with_gradient (examples/h2h_gmin_proof.rs:68,
examples/h2h_gmin_proof.rs:327). That is every
Lennard-Jones cell of the gmin panel. lj103 is
included: this tree's binary lists it, so the panel
resolver uses h2h_gmin_proof
(scripts/gmin_panel/run_panel.py:279).

h2h_morse does the same (examples/h2h_morse.rs:159,
examples/h2h_morse.rs:295). That is every Morse cell
of the gmin panel and of the morse panel, including
m38 at rho 14.

h2h_metal builds Config::recommended, adds species
labels for an alloy, and calls run_with_gradient
(examples/h2h_metal.rs:250,
examples/h2h_metal.rs:536). Au55 reaches the diff.

h2h_seed on the recommended arm calls
cluster_search::search_from (examples/h2h_seed.rs:24,
examples/h2h_seed.rs:276). search_from calls
run_with_energy_gradient
(src/methods/cluster_search.rs:1604).
cluster_search::search
(src/methods/cluster_search.rs:1479) calls
optimize_with_energy_gradient
(src/methods/cluster_search.rs:1521), which calls
run_with_energy_gradient
(src/methods/cluster_hopping.rs:10193 on the PR
tree). Both entry points reach the diff. The atomic
recommended preset is not a rigid body, so m38 at
rho 14 and LJ75 are also run through h2h_seed.

The hexamer does not reach those statements.
Config::for_tip4p installs a rigid-body library
(src/methods/cluster_hopping/config.rs:2314,
src/methods/cluster_hopping/config.rs:2321).
recommended_tip4p keeps that library
(src/methods/cluster_hopping/config.rs:2474) and
leaves escape_on_stall at the default false
(src/methods/cluster_hopping/config.rs:2216). The
probe returns false for a rigid body, and the climb
block is not entered. The cell is h2h_seed, model
tip4p, six molecules, recommended arm. It is a
no-regression guard. Its budget is the 40000-call
ledger in the hexamer search test
(src/potentials/tip4p.rs:839). Its reference is the
n = 6 row of data/tip4p_ccd.csv.

lj38, lj75, m80 at rho 6, and Au55 were named as
guards. Their drivers reach the diff, so they are
scored by the pass rule and are not labeled
no-regression guards.

Line numbers in src/lattice.rs and
src/methods/cluster_hopping.rs are the PR tree. The
examples and src/methods/cluster_search.rs match
BASE.

## Cells

Nine cell and driver pairs. Seeds are 2000 through
2023. Budgets and
references are the tuples in cells.py, copied from
scripts/gmin_panel/run_panel.py and
scripts/morse_gmin/run_panel.py, except the hexamer
budget cited above. A hit is a quenched energy at or
below the reference plus 0.001.

| cell | driver | budget | role |
| --- | --- | --- | --- |
| lj75 | h2h_gmin_proof | 4000000 | reaches |
| lj98 | h2h_gmin_proof | 4000000 | reaches |
| m55r14 | h2h_morse | 1000000 | reaches |
| m80r14 | h2h_morse | 4000000 | reaches |
| m30r14 | h2h_morse | 400000 | reaches |
| m38r6 | h2h_morse | 400000 | reaches |
| m38r14 | h2h_morse | 400000 | reaches |
| m38r14 | h2h_seed | 400000 | reaches |
| lj75 | h2h_seed | 4000000 | reaches |

h2h_seed is stopped at the reference plus 0.001, the
same window the panel ledgers pass to stop_below.

## Pass rule

This section is the rule. It is committed before any
timed row and is applied unchanged.

PASS only if, in every cell, all six hold.

1. B has the same evals_to_hit and best bits as A
   on every seed. This checks that the search before
   a hit is unchanged.
2. On every B hit, charged equals evals_to_hit.
3. B does not lose on hits. The exact two-sided
   paired McNemar test on per-seed hits does not
   show B worse at p < 0.05, and B's hit count is
   not below A's by more than A' differs from A in
   any cell.
4. The median charged-call ratio B/A is not above
   the A/A2 band, unless hits improve. The band is
   the maximum over cells of max(r, 1/r), where r
   is the median charged-call ratio A'/A.
5. The median of outside_s per charged call, as the
   ratio B/A, is at most 1.10. This cap is not
   waived.
6. Both B positions agree on hit, both hit scores,
   evals_to_hit, charged, and best bits.

A cell that fails a clause is named with that
clause. The gate fails if any cell fails any
clause. Bit identity of best_bits between A and B
is reported per cell. A non-zero exit makes the
whole gate inconclusive; it is never scored as a
miss.

## Definitions

Hits are paired by seed. b counts A miss and B hit.
c counts A hit and B miss. B is worse when c > b.
The exact two-sided McNemar probability sums the
binomial(n, 1/2) probabilities that are no larger
than the probability of min(b, c), with n = b + c.
No discordant pair gives p = 1. Clause 1 fails only
when B is worse and p < 0.05, or when the hit-count
deficit exceeds the allowance.

The hit allowance is the maximum, over every
measured cell, of the absolute difference between
the A2 hit count and the A hit count. The deficit
is the A hit count minus the B hit count.

A completed hit keeps its charged count. A
completed miss is ranked at max(charged, budget).
An error, a timeout, a non-zero exit, or a missing
row makes the result inconclusive. The median charged
count in clause 4, and the Wilcoxon signed-rank
test, use only a complete, conclusive set. Hits improve when the
B hit count is greater than the A hit count. That
waives clause 2 only.

Medians are Hyndman-Fan type 7.

outside_s per charged call uses the raw charged
count. The median is over the 24 seeds. If any seed
lacks a finite outside_s or a positive charged
count, clause 3 fails for that cell.

Clause 2 fails when the B/A median ratio is above
the band. A ratio equal to the band passes. A zero
or non-finite A median fails the clause. Clause 3
fails when the B/A median ratio is above 1.10. A
ratio within 1e-12 of the limit is treated as equal
to it.

Wilcoxon is two-sided on ranked charged counts, B
minus A. Zero differences are dropped. Absolute
differences take average ranks. The null puts an
independent sign on each rank. The p value is twice
the smaller tail, capped at 1. The reported W is
W+, the sum of ranks with B minus A positive.

Bootstrap intervals are type-7 percentiles 0.025
and 0.975 of 10000 paired resamples of the 24
seeds. The generator is random.Random(1712026),
one stream in cell order. The code draws six
bootstrap ratios: B/A charged, A2/A charged, B/A
calls, B/A wall, B/A outside, and A2/A outside.
The intervals are reported. The
pass rule uses the full-sample medians.

best_bits are the big-endian IEEE-754 bits of the
decimal the driver printed, parsed as binary64.
Panel drivers print eight digits after the point.
h2h_seed prints ten on the quenched best line. The
drivers do not print the unsounded value. A blank
best is not identical.

Panel hit flags are the driver's, computed on the
unsounded energy. h2h_seed is scored from
seen.lowest with the reference plus 0.001 window.
The printed best lags seen.lowest when the stop
fires mid-quench. Both seen.lowest and printed-best
scores are reported, including every difference.

The separate LJ75 check runs h2h_seed and
h2h_gmin_proof on the same seeds and reports both
scorings.

## Protocol

Arms of one seed run in A, B, B, A2 order. The two
B slots use the same release build. Every process
is pinned to core 0. Slot is the position, starting
at 1.

Peak RssAnon is sampled from the process tree and
is the only memory gate. ru_maxrss and peak VmHWM
are reported but never gate. An OOM kill keeps the
row and its exit code. A process that ends before
the row is stored is recorded with exit -9 and is
not rerun. The hang bound is 14400 seconds and an
exit 124 is inconclusive.
The bound is not a search budget.

The child environment sets these to 1:
OMP_NUM_THREADS, MKL_NUM_THREADS,
OPENBLAS_NUM_THREADS, RAYON_NUM_THREADS,
NUMEXPR_NUM_THREADS, BLIS_NUM_THREADS,
VECLIB_MAXIMUM_THREADS, GOTO_NUM_THREADS,
NUMBA_NUM_THREADS, JULIA_NUM_THREADS,
TBB_NUM_THREADS. OMP_DYNAMIC and MKL_DYNAMIC
are FALSE. Also passed: PATH, HOME, LANG,
LC_ALL, TMPDIR. No other variable is passed.

Pin and provenance checks run before the timed
process. wall_s, pot_s, and outside_s are the
driver's. h2h_gmin_proof does not print
outside_s, so that field is max(wall_s - pot_s,
0). There is no warm-up and no shortened budget.

The machine id on each row is the sha256 of the
platform machine-id file. The raw id is not
written down.

Builds use the 1.88.0 compiler, release, locked,
for h2h_gmin_proof, h2h_morse, and h2h_seed.
A and A2 do not share a target
directory. Compiler path prefixes are remapped
to fixed labels.

The preflight verifies revisions, trees, binaries,
the core pin, and harness self-tests before any
timed row. Each process log is first written with a
.tmp suffix, synced, moved to its final name, synced
again, and read back for its sha256. Rows are
append-only. A completed row is never rerun,
replaced, relabeled, rescued, or hand-picked.

## Prior round evidence

Filled from summary.csv after the measurement.
The rule above is not edited when this section
is filled.

Session e8ae91f267499713 wrote 1512 rows. Every
exit code is 0. No row is missing and no row is a
timeout. Three lanes ran on cores 0, 1, and 2.
Peak resident set on a row is 654032896 bytes.
The machine id on the rows is the sha256 of the
platform machine-id file.

Arms A and A' are separate release builds of the
BASE tree. Each driver has the same bytes in both
builds. Across all 504 seed pairs, A and A' match
on the hit, the charged count, and best_bits. The
hit allowance is 0. The charged A/A' band is
1.000000. Wall time still moves between A and A'.

Arm B is 03476abc24dc56e0620e2ce5508b5f1f21e8a3f1.
Its tree is
5d4ce286694a171f2507edc442d15fcfcb908113.

The gate fails. Six cells fail.

Clause 1 fails for m38r14 with h2h_seed. A has 23
hits and B has 19, a deficit of 4 against an
allowance of 0. The discordant counts are b = 0
and c = 4. The exact two-sided McNemar
probability is 0.125, so the p < 0.05 part of
clause 1 does not fire. The deficit does.

Clause 2 fails for lj98 with h2h_gmin_proof
(1.001278), m55r14 with h2h_morse (1.005916),
m80r14 with h2h_morse (1.008927), m30r14 with
h2h_morse (1.004557), m38r6 with h2h_morse
(1.000324), and m38r14 with h2h_seed (1.185919).
None of those cells improved the hit count, so
the band is not waived. Ranked charged counts
decide this clause. On the first five, the median
raw call count matches across arms and the ranked
median for B sits above the band. On m38r14 with
h2h_seed the raw-call ratio is 0.849362 and the
ranked ratio is 1.185919, because the four extra
misses are ranked at the budget.

Clause 2 passes for the other fifteen cells.
m38r14 with h2h_morse improves hits from 11 to
24, which waives clause 2. Its charged ratio is
0.106845.

Clause 3 fails for no cell. The largest outside
ratio B/A is m30r14 with h2h_morse at 1.038049.

Bit identity of best_bits, A against B, is in the
table. It is not a pass clause. The hexamer guard
matches on all 24 seeds. Its hits, charged counts,
and best_bits also match A against B.

summary.csv holds median calls, median wall time,
median outside per charged call, the B/A and A'/A
ratios, the bootstrap intervals, the Wilcoxon
signed-rank statistic, and the hit strings.
aa_band.csv holds the A/A' terms. The allowance
and the band are the same on every row.

| cell | driver | hits A/A'/B | b | c | p | deficit | charged B/A | outside B/A | bits | rules | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | h2h_gmin_proof | 23/23/24 | 1 | 0 | 1 | -1 | 0.355773 | 1.025565 | 23/24 | 1/1/1 | 1 |
| lj55 | h2h_gmin_proof | 24/24/24 | 0 | 0 | 1 | 0 | 1.000000 | 0.928786 | 24/24 | 1/1/1 | 1 |
| lj75 | h2h_gmin_proof | 7/7/7 | 0 | 0 | 1 | 0 | 1.000000 | 0.998277 | 24/24 | 1/1/1 | 1 |
| lj98 | h2h_gmin_proof | 16/16/16 | 0 | 0 | 1 | 0 | 1.001278 | 1.011999 | 24/24 | 1/0/1 | 0 |
| lj102 | h2h_gmin_proof | 11/11/11 | 0 | 0 | 1 | 0 | 1.000000 | 1.004865 | 24/24 | 1/1/1 | 1 |
| lj103 | h2h_gmin_proof | 13/13/13 | 0 | 0 | 1 | 0 | 1.000000 | 0.999579 | 24/24 | 1/1/1 | 1 |
| lj104 | h2h_gmin_proof | 11/11/11 | 0 | 0 | 1 | 0 | 1.000000 | 1.001231 | 24/24 | 1/1/1 | 1 |
| lj150 | h2h_gmin_proof | 18/18/18 | 0 | 0 | 1 | 0 | 1.000000 | 1.010154 | 24/24 | 1/1/1 | 1 |
| au55 | h2h_metal | 8/8/8 | 0 | 0 | 1 | 0 | 1.000000 | 1.003178 | 24/24 | 1/1/1 | 1 |
| m55r6 | h2h_morse | 24/24/24 | 0 | 0 | 1 | 0 | 1.000000 | 1.021379 | 24/24 | 1/1/1 | 1 |
| m55r14 | h2h_morse | 18/18/18 | 0 | 0 | 1 | 0 | 1.005916 | 1.037299 | 22/24 | 1/0/1 | 0 |
| m80r6 | h2h_morse | 24/24/24 | 0 | 0 | 1 | 0 | 1.000000 | 1.018344 | 24/24 | 1/1/1 | 1 |
| m80r14 | h2h_morse | 15/15/15 | 0 | 0 | 1 | 0 | 1.008927 | 1.028987 | 24/24 | 1/0/1 | 0 |
| m30r6 | h2h_morse | 24/24/24 | 0 | 0 | 1 | 0 | 1.000000 | 0.998750 | 24/24 | 1/1/1 | 1 |
| m30r14 | h2h_morse | 17/17/17 | 0 | 0 | 1 | 0 | 1.004557 | 1.038049 | 24/24 | 1/0/1 | 0 |
| m38r6 | h2h_morse | 24/24/24 | 0 | 0 | 1 | 0 | 1.000324 | 1.000623 | 24/24 | 1/0/1 | 0 |
| m38r14 | h2h_morse | 11/11/24 | 13 | 0 | 0.000244141 | -13 | 0.106845 | 0.938369 | 11/24 | 1/1/1 | 1 |
| m80r3 | h2h_morse | 24/24/24 | 0 | 0 | 1 | 0 | 1.000000 | 1.002976 | 24/24 | 1/1/1 | 1 |
| m38r14 | h2h_seed | 23/23/19 | 0 | 4 | 0.125 | 4 | 1.185919 | 1.024673 | 19/24 | 0/0/1 | 0 |
| lj75 | h2h_seed | 0/0/0 | 0 | 0 | 1 | 0 | 1.000000 | 0.996697 | 24/24 | 1/1/1 | 1 |
| hexamer | h2h_seed | 24/24/24 | 0 | 0 | 1 | 0 | 1.000000 | 0.987185 | 24/24 | 1/1/1 | 1 |

Per-seed hit strings, A then B, seeds 2000 through
2023.

| cell | driver | A | B |
| --- | --- | --- | --- |
| lj38 | h2h_gmin_proof | 111111111111111111101111 | 111111111111111111111111 |
| lj55 | h2h_gmin_proof | 111111111111111111111111 | 111111111111111111111111 |
| lj75 | h2h_gmin_proof | 010001001011000001100000 | 010001001011000001100000 |
| lj98 | h2h_gmin_proof | 111101101000101111110101 | 111101101000101111110101 |
| lj102 | h2h_gmin_proof | 110101011000010000010111 | 110101011000010000010111 |
| lj103 | h2h_gmin_proof | 111110000001011111000110 | 111110000001011111000110 |
| lj104 | h2h_gmin_proof | 000001110101010100101110 | 000001110101010100101110 |
| lj150 | h2h_gmin_proof | 101101101011111110101111 | 101101101011111110101111 |
| au55 | h2h_metal | 000001001011000000001111 | 000001001011000000001111 |
| m55r6 | h2h_morse | 111111111111111111111111 | 111111111111111111111111 |
| m55r14 | h2h_morse | 110110111011110110011111 | 110110111011110110011111 |
| m80r6 | h2h_morse | 111111111111111111111111 | 111111111111111111111111 |
| m80r14 | h2h_morse | 110100110010110111111010 | 110100110010110111111010 |
| m30r6 | h2h_morse | 111111111111111111111111 | 111111111111111111111111 |
| m30r14 | h2h_morse | 010110101111011011111101 | 010110101111011011111101 |
| m38r6 | h2h_morse | 111111111111111111111111 | 111111111111111111111111 |
| m38r14 | h2h_morse | 010000001101100001111110 | 111111111111111111111111 |
| m80r3 | h2h_morse | 111111111111111111111111 | 111111111111111111111111 |
| m38r14 | h2h_seed | 101111111111111111111111 | 101101110101111111101111 |
| lj75 | h2h_seed | 000000000000000000000000 | 000000000000000000000000 |
| hexamer | h2h_seed | 111111111111111111111111 | 111111111111111111111111 |

## Round four verdict

INCONCLUSIVE. The preflight passed revision, tree,
binary, pin, and self-tests, but did not exercise
the timing wrapper. All 864 append-only rows have
exit 127 because that wrapper was unavailable. No
row was rerun, replaced, relabeled, rescued, or
hand-picked.

No process reached a driver. Search equality,
hit charging, clauses 1 through 3 from round three,
both h2h_seed scorings, the LJ75 scoring comparison,
and the claimed m80r14 seed 2006 count are therefore
unverified. Peak RssAnon and VmHWM are zero because
no driver process started; ru_maxrss is unavailable.

| cell | driver | rows | exit | verdict |
| --- | --- | --- | --- | --- |
| lj75 | h2h_gmin_proof | 96 | 127 | inconclusive |
| lj98 | h2h_gmin_proof | 96 | 127 | inconclusive |
| m55r14 | h2h_morse | 96 | 127 | inconclusive |
| m80r14 | h2h_morse | 96 | 127 | inconclusive |
| m30r14 | h2h_morse | 96 | 127 | inconclusive |
| m38r6 | h2h_morse | 96 | 127 | inconclusive |
| m38r14 | h2h_morse | 96 | 127 | inconclusive |
| m38r14 | h2h_seed | 96 | 127 | inconclusive |
| lj75 | h2h_seed | 96 | 127 | inconclusive |

## Round four retry protocol

The 864 failed rows above remain unchanged. The
retry writes rows-r2.csv, sessions-r2.jsonl, and
hosts-r2.csv as new append-only files in a new
session.

The harness measures elapsed time and per-child
ru_maxrss directly. It samples RssAnon and VmHWM
from the process tree. RssAnon is the only memory
gate; ru_maxrss and VmHWM are reported and never
gate. The production command has no external timing
wrapper.

Before the campaign, preflight runs one real
cell-seed for every A, B, B2, and A2 driver
combination. It uses seed 1999, which is outside
the campaign. Each probe uses the exact production
affinity, environment, timeout, arguments, parser,
timing, and memory path. Preflight aborts unless
every exit is zero and the timing, RssAnon,
ru_maxrss, and VmHWM fields are populated.

The campaign remains seeds 2000 through 2023 in
A, B, B2, A2 order on core 0. It does not alter
the cells, budgets, references, scoring, bootstrap
stream, pass clauses, or round-three comparison
rules above.

## Round four retry verdict

FAIL. The production preflight passed all 12 probes.
Session 79ca48cebfe85ed5 then wrote 864 rows. Every
exit is zero, every cell has 24 seeds in all four
slots, and both B slots agree on every deterministic
field.

The A/A2 charged band is 1.000000. Clauses 1, 2,
and 3 pass in every cell. Four cells fail the
additional search or hit-charging invariants.

| cell | driver | hits A/A2/B | charged B/A | outside B/A | search mismatches | hit-charge mismatches | score differences B | rules | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj75 | h2h_gmin_proof | 7/7/7 | 1.000000 | 1.008795 | 0 | 0 | 0 | 1/1/1 | pass |
| lj98 | h2h_gmin_proof | 16/16/16 | 1.000000 | 1.005716 | 0 | 0 | 0 | 1/1/1 | pass |
| m55r14 | h2h_morse | 18/18/18 | 1.000000 | 1.042058 | 2 | 0 | 0 | 1/1/1 | fail |
| m80r14 | h2h_morse | 15/15/15 | 1.000000 | 1.050832 | 0 | 0 | 0 | 1/1/1 | pass |
| m30r14 | h2h_morse | 17/17/17 | 1.000000 | 1.034256 | 0 | 0 | 0 | 1/1/1 | pass |
| m38r6 | h2h_morse | 24/24/24 | 1.000000 | 1.006105 | 0 | 0 | 0 | 1/1/1 | pass |
| m38r14 | h2h_morse | 11/11/24 | 0.106436 | 0.932571 | 19 | 1 | 0 | 1/1/1 | fail |
| m38r14 | h2h_seed | 24/24/24 | 0.849222 | 1.010731 | 9 | 18 | 5 | 1/1/1 | fail |
| lj75 | h2h_seed | 24/24/24 | 1.000000 | 0.995462 | 3 | 24 | 24 | 1/1/1 | fail |

For h2h_seed m38r14, seen.lowest scores 24 hits
in every arm. Printed best scores 23 for A and A2,
differing at seed 2001. It scores 19 for B and B2,
differing at seeds 2001, 2004, 2008, 2010, and
2019.

For h2h_seed LJ75, seen.lowest scores 24 hits and
printed best scores zero in every arm; all 24 seeds
differ. On the same seeds, h2h_gmin_proof scores
seven hits in every arm. This is why h2h_seed is
judged from seen.lowest rather than printed best.

The m80r14 seed 2006 claim is verified in all four
slots: hit is one and evals_to_hit and charged are
both 3802858.

Peak RssAnon is 286130176 bytes. The corresponding
reported ru_maxrss and VmHWM are both 291151872
bytes. Only RssAnon is a memory gate.
