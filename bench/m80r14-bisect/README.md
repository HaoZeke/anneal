# M80 rho = 14 bisect

Morse M80 at rho = 14 (Cambridge 80G, -340.811371) at 4e6 calls,
`Config::recommended` through `examples/h2h_morse.rs`, seeds 2000-2011.
Develop `61fa67cf380e46876b569da45e7cc091647a775b` hits 6 of 12. The GMIN
table of the manuscript prints 9 of 12 for revision `1ce3ab64`.

`1ce3ab64` is in `bench/unreachable-revisions.csv`: no pushed branch holds
it. Its m80r14 rows in `bench/gmin-panel/rows.csv` have the same hit call
and best energy on every seed as the `c2de652d` rows in that file and the
`07f89cad` rows in `bench/paper-0120/rows.csv`. `07f89cad` is an ancestor
of develop, so it is the good end.

Every row here is from one 8-core Intel Xeon virtual machine, labelled
`xeon8-vm` in the host column, with rustc 1.100.0.
Each seed is one single-threaded process, and eight run at once, so
`wall_s` and `pot_s` are not comparable with the `bench` host. Calls and
energies do not depend on that: `07f89cad` reproduces the recorded
`bench/paper-0120` calls and energies on all twelve seeds.

```
bench/m80r14-bisect/run_rev.sh <rev> m80r14
bench/m80r14-bisect/run_rev.sh 61fa67cf <system> bench/m80r14-bisect/<arm>.patch
```

The script builds the driver at the revision and runs
`h2h_morse run <system> <seed>` (`h2h_gmin_proof` for `lj*`) for seeds
2000-2011. Per-seed rows are `rows.csv`.

## Bisect

Develop's first-parent history, then the second parent of the merge it
named. Good is the `07f89cad` trajectory and bad is the develop trajectory.
Every revision tested gave one of those two sets of rows exactly, seed by
seed.

| step | revision | hits | rows equal to |
| --- | --- | --- | --- |
| good end | `07f89cad` | 9/12 | |
| bad end | `61fa67cf` | 6/12 | |
| 1 | `8e50ff2e` | 9/12 | `07f89cad` |
| 2 | `6edf563a` | 9/12 | `07f89cad` |
| 3 | `bc16d35c` (merge #74) | 6/12 | `61fa67cf` |
| 4 | `7fda9e44` | 9/12 | `07f89cad` |
| 5 | `39068f4d` | 9/12 | `07f89cad` |
| 6 | `04f52150` | 9/12 | `07f89cad` |
| 7 | `0ae5bfe0` (first parent of #74) | 9/12 | `07f89cad` |
| 8 | `dc4e3776` | 9/12 | `07f89cad` |
| 9 | `f353eec4` | 9/12 | `07f89cad` |
| 10 | `1c1b1238` | 6/12 | `61fa67cf` |
| 11 | `20b94ada` | 9/12 | `07f89cad` |
| 12 | `628773d6` | 9/12 | `07f89cad` |

The first bad commit is `1c1b1238` ("fix: remove structure templates from
cluster search"). Its parent is `628773d6`.

The same commit carries the other two Morse rows of that table. The
`1ce3ab64` rows are the `628773d6` trajectory, and develop is the
`1c1b1238` trajectory:

| system | `1ce3ab64` (paper) | `628773d6` | `1c1b1238` | develop |
| --- | --- | --- | --- | --- |
| m80r14 | 9/12 | 9/12 | 6/12 | 6/12 |
| m80r6 | 2/12 | not run | 12/12 | 12/12 |
| m55r14 | 7/12 | 7/12 | 9/12 | 9/12 |

At m55r14, `628773d6` matches the `1ce3ab64` calls on all twelve seeds. At
m80r6, `1c1b1238` matches develop on all twelve. The M80 rho = 6 gain, the
M55 rho = 14 gain and the M80 rho = 14 loss are one change, and the GMIN
table's Morse rows labelled `1ce3ab64` are the search with the templates
still in it.

## Mechanism

`1c1b1238` removed the stall growth that, on a call plateau of a
non-icosahedral incumbent, regrew it from each source of
`lattice::Source::library()` (observed order, face-centred cubic,
hexagonal close-packed, icosahedral, simple cubic) and adopted the best
quench when it lowered the record.
`trace-628773d6.patch` prints those quenches at the parent. On m80r14 the
face-centred cubic cut is what puts the chain into the close-packed funnel:

| seed | call | incumbent | record | fcc cut quench |
| --- | --- | --- | --- | --- |
| 2001 | 15819 | -292.730911 | -294.479033 | -336.857378 |
| 2004 | 20193 | -279.306762 | -309.737667 | -336.857378 |
| 2007 | 21058 | -290.927940 | -299.435513 | -336.857378 |

A second face-centred cubic cut reaches -338.826826 on seeds 2001
(call 33113) and 2007 (call 29637). The move arms then descend to 80G:
at the parent the best energy of every arm on seed 2007 stays at or
above -338.828609 and on seed 2004 at or above -338.840315, and the run
hits at 148765 and 285912 calls.

Without the cut the chain picks its funnel itself. `diag-61fa67cf.patch`
prints the improvement record and the 555/421/422 common-neighbour
fractions (cutoff 1.35) of the best structure on develop:

| seeds | best | 555 | 421 | 422 | reached at call |
| --- | --- | --- | --- | --- | --- |
| hits 2001, 2003 | -340.811371 | 0.000 | 0.412 | 0.091 | 529680, 1226468 |
| 2004, 2005, 2011 | -340.387336 | 0.012 | 0.279 | 0.202 | 511157, 161915, 510410 |
| 2008 | -340.357131 | 0.012 | 0.284 | 0.205 | 2185165 |
| 2002, 2009 | -340.802563 | 0.000 | 0.494 | 0.000 | |

80G is close packed with a stacking fault. Four of the six misses enter
a twinned packing with fivefold common neighbours, 0.42 to 0.45 above
80G, and keep it for the rest of the 4e6 calls: the preset's stall
responses do not leave it. The other two misses end at
-340.802563, the untwinned face-centred cubic structure 0.0088 above 80G.
The parent misses there too, on seeds 2000, 2002 and 2011.

Of the other removed paths the trace instruments, the packing-bank return
and the surface hop never printed on seeds 2001, 2004 and 2007. For the
CSM reseed it marks only that the first gate condition was read, not
that a reseed ran. The traced hits at the parent go through the named
face-centred cubic cut, which a template-free search may not carry.

## Generic candidates

On develop, m80r14 at n = 12 with the driver's existing switches:

| arm | hits |
| --- | --- |
| recommended | 6/12 |
| `H2H_BRANCH=1` | 5/12 |
| `H2H_PLATEAU_HEAT=1` | 8/12 |
| `H2H_PAIR_SCALE=1` | 8/12 |

`observed-growth.patch` keeps the removed stall growth with only
`Source::Observed`, which reads the local order off the structure the
chain stands on and names no packing. On each directed leave of a
one-label cluster it quenches one regrowth and adopts it when it lowers
the record. `observed-growth-hold.patch` adds the removed hold: after a
growth fails at a record, the next one waits until the record drops.

| system | develop | observed growth | observed growth + hold |
| --- | --- | --- | --- |
| m80r14 | 6/12 | 10/12 | 8/12 |
| m80r6 | 12/12 | 12/12 | not run |
| m55r14 | 9/12 | 8/12 | not run |
| m55r6 | 12/12 | 12/12 | not run |
| LJ38 | 12/12 | 12/12 | not run |
| LJ75 | 5/12 | 1/12 | 4/12 |

Observed growth reaches the close-packed funnel without a template: on
seed 2004 it lifts the chain from a record of -311.520675 to -338.826826 at
call 80593. It also regrows the icosahedral order of an LJ75 incumbent.
Every LJ75 miss with it ends at -396.282249, and on seeds 2008 and 2011 it
ran about 770 regrowth quenches and adopted none, so the loss is calls
taken from the walk. With the hold, both systems are within one or two
hits of develop.

Two-sided Fisher p against develop, n = 12 each: observed growth 0.19 on
m80r14 and 0.16 on LJ75; with the hold 0.68 on m80r14 and 1 on LJ75; the
switches 0.68 or 1. The `07f89cad` 9/12 against 6/12 is itself p = 0.40.
No arm here moves m80r14 at this n without giving back LJ75, and none is
offered as a fix. The bisect and the mechanism stand on the trajectories,
which are deterministic, not on those counts.
