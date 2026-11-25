# PR 168 walklen3 gate, shard B

Measurement only. Library sources on the measured commits stay as they
are. This file is the rule, written before any row.

## Ancestry

Develop `7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec` (tree
`e5ebda63f9f0117226d50c13abf33243ccf452cc`) is not an ancestor of payload
`4ab2681ae6bf72d61c633197e88047f00cdd7efa` (tree
`62e3bca3895a3de6d9db9fe88e0ea09f3127ce18`), and the payload is not an
ancestor of develop. `git merge-tree --write-tree` of those two commits
exits 0 and writes tree `991d5afc6d7b7f7e6c392cb68aa351af1fc50179`. That
merged tree is the tree under measurement. A scratch merge of that tree,
first parent develop and second parent the payload, is
`8824564b2479f0917c1e4927cd5ac938a86c2ac0` on
`refs/scratch/walklen3-shardB-m`. The scratch ref is used only to build
M. It is not part of the result history. The row file starts empty on
this base.

## Arms

- A is develop `7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec`, tree
  `e5ebda63f9f0117226d50c13abf33243ccf452cc`.
- A2 is a second release build of that same commit and tree.
- M is `8824564b2479f0917c1e4927cd5ac938a86c2ac0`, tree
  `991d5afc6d7b7f7e6c392cb68aa351af1fc50179`.

Each arm has its own worktree and its own target directory. Release
build, locked. Toolchain `1.88.0`. One machine. The timed runs are
serial on core 0, with one thread for OpenMP, BLAS, and Rayon.

## Driver

Each arm runs the binary built from its own tree. The preset is the
recommended configuration. No external minimiser program is built or
run. Lennard-Jones budgets are the paper budgets, and the Lennard-Jones
75 budget is 4e6 calls. A seeded run stops at the published reference
plus 0.001, passed to six decimal places. The seeded driver then applies
a zero window on top of that stop.

```
lj75-seed  h2h_seed lj 75 0 4000000 SEED recommended -397.491331 PREFIX
lj38       h2h_seed lj 38 0 400000 SEED recommended -173.927427 PREFIX
m38r6      h2h_morse run m38r6 SEED
m55r14     h2h_morse run m55r14 SEED
au55-seed  h2h_metal run au55 SEED
ase-lj98   ase_panel lj98 4000000 SEED recommended
ase-lj104  ase_panel lj104 4000000 SEED recommended
```

`h2h_seed` has no gold potential, so `au55-seed` uses the seeded gold
driver, whose preset is recommended and whose budget is 1e6. Morse uses
its recommended driver, budgets 4e5 at 38 and 1e6 at 55, and the same
0.001 hit window. The two ASE cells use the recommended preset, not the
plain basin hop. A seeded usage line must contain

```
h2h_seed <lj|morse|tip4p> <n> <rho> <budget> <seed> <arm> <stop> <prefix>
```

The optional ninth argument on M is not passed.

## Memory

A's seeded driver does not read a resident set. The same sampler is
compiled into A, A2, and M, then the three source files are restored.
It reads `RssAnon` from `/proc/self/status` at search start, when the
basin count crosses a power of two, at search end, and at process exit,
and keeps the largest value. The reported peak is that value. M's own
seeded memory line is not the measurement.

The same insertion prints `gate_walks hops=` at search end. A walk is
one hop. Calls per walk are charged calls divided by hops.

## Cells and order

`lj75-seed` uses seeds 2000 through 2107 (108 rounds). The other cells
use seeds 2000 through 2011 (12 rounds). Cells run one after another in
this order: lj38, m38r6, m55r14, au55-seed, ase-lj98, ase-lj104,
lj75-seed. A round that cannot start before the session deadline is not
started. The deadline is four hours after the timed session starts.

ABBA, with the triple (A, M, A2) reversed in pairs. Round `i` is seed
`2000 + i`. The cycle is

```
0  A  M  A2
1  A2 M  A
2  M  A2 A
3  A  A2 M
4  A2 A  M
5  M  A  A2
```

Rounds 0 and 1, 2 and 3, and 4 and 5 are reversals. In each block of
six, each arm occupies each slot twice.

Rows and the session log are append-only, fsynced after every finished
arm. A failed run is not a row. A later process skips a finished arm
and does not rewrite earlier lines. A run shorter than 50 ms is
repeated and the outside time is the mean of those repetitions. The
repeats must agree on the hit, the calls, and the best energy.

## Scores

The hit flag is the driver's. A miss counts as budget plus one in the
calls-to-hit analysis. Outside time is `outside_s`. For a seeded hit,
that is the outside time to the hit, because the driver stops at the
threshold. RSS is the shared peak, in kilobytes.

Bootstrap: 10000 resamples, seed 20261123, `random.Random` started anew
for each interval, paired by seed. The interval is the type-7 percentile
at 2.5 and 97.5. Calls-to-hit reports the geometric mean of the paired
M/A ratios and the ratio of the medians, each with an interval. Outside
time and RSS report the round-paired geometric mean, M/A and A2/A. A
non-positive value drops that seed from that geometric mean only.

Hits use a one-sided Fisher exact test, the alternative that M has
fewer hits. Calls-to-hit use a one-sided Mann-Whitney test, tie-corrected
normal approximation with a continuity correction, the alternative that
M is larger, with a miss entered at budget plus one.

A trajectory file is the seeded driver's trace. It is bit-identical
when the file digest matches. Drivers that do not write a trace have no
trajectory file; the report says so and compares the printed best
energy.

## Verdict

A cell is complete only at its target round count, with a positive RSS
on every row.

Bars, on a complete cell:

- no hit loss: hits of M are at least hits of A
- no calls-to-hit loss: the median of M is at most the median of A
- outside-time M/A upper interval at most 1.10
- RSS M/A upper interval at most 1.10

A hit loss is real when the one-sided Fisher probability is below 0.05.
A calls, outside, or RSS regression is real only when its lower interval
is above 1 and above the A2/A upper interval. The calls interval in that
rule is the ratio of medians.

PASS when the cell is complete and every bar holds. FIX when the cell
is complete and at least one real regression is present. UNRESOLVED
when the cell is short of its target, or a bar fails and no regression
is real. A finished verdict is not rewritten. The geometric mean at
seeds 2000-2011 on lj75 is a check of the claim 0.932. It is not a bar.
