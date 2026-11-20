# Gate for morse m38 rho 14, round 3

BASE is 61fa67cf380e46876b569da45e7cc091647a775b,
tree cff796c507a8fa1d0bfb28888c3570b61988055e.
Arm A is that tree. Arm A' is a second release build
of the same tree, in its own worktree and target
directory. Arm B is
03476abc24dc56e0620e2ce5508b5f1f21e8a3f1, tree
5d4ce286694a171f2507edc442d15fcfcb908113. The harness
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

Twenty-one cell and driver pairs. Morse cells that
sit in both panels are measured once, through
h2h_morse. Seeds are 2000 through 2023. Budgets and
references are the tuples in cells.py, copied from
scripts/gmin_panel/run_panel.py and
scripts/morse_gmin/run_panel.py, except the hexamer
budget cited above. A hit is a quenched energy at or
below the reference plus 0.001.

| cell | driver | budget | role |
| --- | --- | --- | --- |
| lj38 | h2h_gmin_proof | 400000 | reaches |
| lj55 | h2h_gmin_proof | 1000000 | reaches |
| lj75 | h2h_gmin_proof | 4000000 | reaches |
| lj98 | h2h_gmin_proof | 4000000 | reaches |
| lj102 | h2h_gmin_proof | 4000000 | reaches |
| lj103 | h2h_gmin_proof | 4000000 | reaches |
| lj104 | h2h_gmin_proof | 4000000 | reaches |
| lj150 | h2h_gmin_proof | 4000000 | reaches |
| au55 | h2h_metal | 1000000 | reaches |
| m55r6 | h2h_morse | 1000000 | reaches |
| m55r14 | h2h_morse | 1000000 | reaches |
| m80r6 | h2h_morse | 4000000 | reaches |
| m80r14 | h2h_morse | 4000000 | reaches |
| m30r6 | h2h_morse | 400000 | reaches |
| m30r14 | h2h_morse | 400000 | reaches |
| m38r6 | h2h_morse | 400000 | reaches |
| m38r14 | h2h_morse | 400000 | reaches |
| m80r3 | h2h_morse | 4000000 | reaches |
| m38r14 | h2h_seed | 400000 | reaches |
| lj75 | h2h_seed | 4000000 | reaches |
| hexamer | h2h_seed | 40000 | no-regression guard |

h2h_seed is stopped at the reference plus 0.001, the
same window the panel ledgers pass to stop_below.

## Pass rule

This section is the rule. It is committed before any
timed row and is applied unchanged.

PASS only if, in every cell, all three hold.

1. B does not lose on hits. The exact two-sided
   paired McNemar test on per-seed hits does not
   show B worse at p < 0.05, and B's hit count is
   not below A's by more than A' differs from A in
   any cell.
2. The median charged-call ratio B/A is not above
   the A/A' band, unless hits improve. The band is
   the maximum over cells of max(r, 1/r), where r
   is the median charged-call ratio A'/A.
3. The median of outside_s per charged call, as the
   ratio B/A, is at most 1.10. This cap is not
   waived.

A cell that fails a clause is named with that
clause. The gate fails if any cell fails any
clause. Bit identity of best_bits between A and B
is reported per cell and is not a pass clause.

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
the A' hit count and the A hit count. The deficit
is the A hit count minus the B hit count.

A completed hit keeps its charged count. A
completed miss is ranked at max(charged, budget).
An error, a timeout, a non-zero exit, or a missing
row is ranked at 1000000000000. The median charged
count in clause 2, and the Wilcoxon signed-rank
test, both use this ranking. Hits improve when the
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
the smaller tail, capped at 1.

Bootstrap intervals are type-7 percentiles 0.025
and 0.975 of 10000 paired resamples of the 24
seeds. The generator is random.Random(1712026),
one stream in cell order, drawing the charged
ratio, the call ratio, the wall ratio, and the
outside ratio. The intervals are reported. The
pass rule uses the full-sample medians.

best_bits are the big-endian IEEE-754 bits of the
decimal the driver printed, parsed as binary64.
Panel drivers print eight digits after the point.
h2h_seed prints ten on the quenched best line. The
drivers do not print the unsounded value. A blank
best is not identical.

Panel hit flags are the driver's, computed on the
unsounded energy. h2h_seed is rescored from its
printed quenched best with the 0.001 window,
because its own flag uses a zero tolerance against
the stop level.

## Protocol

Arms of one seed run on one core. Even seeds run
A, then A', then B. Odd seeds run the reverse,
B, then A', then A. Slot is that position,
starting at 1.

Up to three lanes, pinned with taskset to cores
0, 1, and 2, one lane per core. If MemAvailable
is below 8 GiB when the lanes are chosen, one
lane is used. Peak resident set is the maximum
RSS of the wrapped process, in bytes. An OOM
kill keeps the row and its exit code. A process
that ends before the row is stored is recorded
with exit -9 and is not rerun. The hang bound is
14400 seconds, exit 124, and counts as a miss.
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
for h2h_gmin_proof, h2h_morse, h2h_metal, and
h2h_seed. A and A' do not share a target
directory. Compiler path prefixes are remapped
to fixed labels.

## Verdict

Filled from summary.csv after the measurement.
The rule above is not edited when this section
is filled.
