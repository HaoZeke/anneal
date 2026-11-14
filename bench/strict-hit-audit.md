# Strict hit audit

This is a report over committed `bench/**/rows.csv` files. It does not
rewrite the append-only records. A strict hit has a finite best energy and
`abs(best - reference) <= 0.001` in the row's energy unit.

No committed `hit=1` row has a non-finite best. The following nine rows
have `hit=1` but are outside the two-sided tolerance.

| file:line | system | code | seed | best | reference | signed gap |
|---|---|---|---:|---:|---:|---:|
| `bench/water-gmin/rows.csv:33` | tip4p12 | gmin | 2003 | -492.909247199 | -492.9082357122 | -0.0010114868 |
| `bench/water-gmin/rows.csv:37` | tip4p12 | gmin | 2005 | -492.909247767 | -492.9082357122 | -0.0010120548 |
| `bench/water-gmin/rows.csv:41` | tip4p12 | gmin | 2007 | -492.909247951 | -492.9082357122 | -0.0010122388 |
| `bench/water-gmin/rows.csv:43` | tip4p12 | gmin | 2008 | -492.909247227 | -492.9082357122 | -0.0010115148 |
| `bench/water-gmin/rows.csv:47` | tip4p12 | gmin | 2010 | -492.909247213 | -492.9082357122 | -0.0010115008 |
| `bench/tip4p-h2h/rows.csv:190` | tip4p15 | gmin-control | 2004 | -628.373520636 | -628.3722525077 | -0.0012681283 |
| `bench/covalent-h2h/gfn2/rows.csv:10` | si10 | anneal-general | 2009 | -492.59511796 | -460.68373809 | -31.91137987 |
| `bench/covalent-h2h/gfn2/rows.csv:74` | si13 | anneal-general | 2003 | -624.70446339 | -599.34546694 | -25.35899645 |
| `bench/covalent-h2h/gfn2/rows.csv:76` | si13 | anneal-general | 2005 | -617.18016907 | -599.34546694 | -17.83470213 |

## Covalent anomalies

### Si10 seed 2009: SCC false minimum

The row energy belongs to a reused-wavefunction SCC solution, not to the
geometry. `gfn2/scc-false-minimum/README.md` replays the run call for call:
call 621 gives -492.5951179571 eV in sequence but -449.0503535512 eV after
an engine reset, while standalone xtb gives -449.050334 eV. The fresh
energy is 11.63340441 eV above the -460.68373809 eV reference. The exact
geometry is retained in `gfn2/scc-false-minimum/calls.xyz`, and
`gfn2/false-hits.csv` records the classification. This is neither a new
minimum nor a reference error.

### Si13 seed 2003: the compact reference is not the global reference

This row is a reproducible lower minimum, not an SCC artifact. Its open
thirteen-member silicon ring is retained in `gfn2/minima/si13-ring.xyz`.
Fresh xtb 6.7.1 gives -22.957465589557 Eh, or -624.70443593 eV, within
0.00002746 eV of the row. Every atom has one or two neighbours, so the
geometry is an intact connected ring rather than an overlap or rejected
geometry. `gfn2/minima.csv` records the fresh score. The old
-599.34546694 eV compact minimum is therefore not a valid global reference
for the unconstrained Si13 search.

### Si13 seed 2005: broken, disconnected structure

The row is independently repeated at the same energy in
`gfn2/floor_gate.csv` and `floor-gate/gfn2.csv`, both at call 2481 of the
base arm, but the campaign did not retain its geometry. Replaying seed 2005
at the recorded `c1c02d16` revision with the same target reproduces the
failure mode: it stops at call 2948 on -617.68988566 eV. A fresh standalone
xtb score is -22.699684612803 Eh, or -617.68985851 eV, so this replay is
not an SCC false minimum.

The replay uses xtb 6.7.1 and an engine build with SHA-256
`13e46cd127e19413e41e53fa6b44d043f173b9f60063f602da55f81407c0c503`;
the archived row names a different engine build. That prevents a
call-for-call energy reproduction but does not affect the connectivity
classification below.

The replayed geometry is disconnected. At a 2.7 Å Si-Si neighbour cutoff,
twelve atoms each have degree two and one atom has degree zero; its shortest
pair is 2.161595 Å. The row is therefore evidence of a missing connectivity
gate, not a valid replacement reference. The exact historical geometry was
not archived, so the replay establishes the defect class rather than
bit-reproducing -617.18016907 eV. No geometry rejection was reported.
