# LJ38 catalogue epoch

`examples/lj38_catalogue_epoch.rs` logs one four-replica catalogue epoch
on LJ38. The full run launches from the v0.12.0 tag. A process with
`PAPER_SMOKE=1` is a smoke and writes a file whose name contains
`SMOKE`. It is not a paper table.

## Conflicts with FIXES.md

None. FIXES.md does not name this weight. IDEAS.md E3 is the formula
`log w = -r_E + 0.8 r_N + 0.6 r_S + 0.4 r_U`, clipped at 4 nats.
`SelectionCoefficients::default` is that formula.

## Selection

Energy, packing-histogram novelty, and inverse packing-family visits
are ranked inside the epoch. The fourth rank is the catalogue residual
score `variance / (1 + effort)` from `ResidualField::score`. Higher
uncertainty ranks higher. The coordinator passes that score into
`PopulationMember` on population submit and join.

The systematic rule is `a_k = min { j : C_j > (u + k) / R }` with
`u = 0.125` and `R = 4`. The family cap is 2, which is
`replicas.div_ceil(2)` for this population. Offspring in the log are
checked against that rule. When the uncapped assignment already
respects the cap, the logged parents are that assignment.

## Seeds and budget

Seeds 3100, 3101, 3102, and 3103. Those are the first four integers of
the registered LJ38 block in `paper/seeds.toml`. They are not a new
block.

The paper force total is 400000 Lennard-Jones evaluations, the LJ38
supporting budget, shared across the four replicas. Smoke caps each
replica at 200 evaluations. A capped quench is not reported as a
minimum.

The graph for this driver is a star: replica 0 linked to the other
three. Leaves and the centre do not share a posterior variance, so
`r_U` is not a tie. That is the field of this one logged epoch. It is
not a deposited transition network.

## Reproduction

Smoke:

```
PAPER_SMOKE=1 PAPER_OUT=lj38-catalogue-epoch-SMOKE.txt \
  cargo run --offline --example lj38_catalogue_epoch
```

Campaign shape, still not a tagged v0.12.0 result. Do not launch it
for a table:

```
cargo run --offline --example lj38_catalogue_epoch
```

The wall-time line `estimated_paper_wall_seconds` scales the smoke
quench time to 400000 evaluations. That quench time includes the four
packing observations. A hop is not in that rate.
