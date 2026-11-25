# (H2O)8 matched-budget protocol

## Recorded rows

`bench/h2o-octamer/results.csv` holds the GFN2 `(H2O)8` head-to-head.
`bench/h2o-octamer/README.md` states the rule: `recommended_molecular`,
a fixed 4000 charged calls on the anneal arm, native CREST with no wall,
and a hit within 1e-6 Eh of the lowest intact energy either arm found,
−40.681336013150 Eh.

The CREST arm in that file records a call count on every seed:

| seed | calls |
| --- | --- |
| 17000 | 127141 |
| 17001 | 128176 |
| 17002 | 130752 |
| 17003 | 123134 |
| 17004 | 151738 |
| 17005 | 126694 |

The median of those six counts is 127658.5. The maximum is 151738, on
seed 17004. That maximum is the budget for every anneal seed in this
rerun. CREST is not run again. Each CREST row in `rows.csv` keeps that
seed's own call count.

## Search

The driver is `examples/gfn2_mol_h2h`, the same binary the octamer rows
use (`scripts/mol_crest_h2h.py` names `gfn2_mol_h2h` for `h2o-8`).
It builds `Config::recommended_molecular` at
`examples/gfn2_mol_h2h.rs:474`. The preset is defined at
`src/methods/cluster_hopping/config.rs:2623`.

Cargo builds that example with feature `rgpot-ex`
(`Cargo.toml` example `gfn2_mol_h2h`). The SOAP packing key inside
`recommended_molecular` is compiled only with `featomic`, which this
binary does not enable.

The potential is the rgpot GFN2 handle, `libxtb_engine.so` at rgpot
`8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e`, linked to xtb 6.7.1
(`edcfbbe`). Energies from the handle are eV. The hartree factor is
27.211386245988, the factor in that engine and in
`src/catalog/molecular.rs:95`.

The stop threshold is the reference plus the tolerance, in eV:

`(−40.681336013150 + 1e-6) * 27.211386245988`.

The driver installs that value with `Ledger::stop_below`. The preset's
resolved configuration is unchanged. A hit is an intact geometry whose
recorded energy is at or below that threshold. `calls_to_hit` is the
first objective evaluation at or below it. A miss is stored as
budget + 1, which is 151739.

Seeds are 2000–2011, twelve starts, one process at a time, pinned to
one core. One thread is used inside the potential. This execution
finished 2000–2007 and did not start 2008–2011. `REPORT.md` records
why, and applies the comparisons below to those eight seeds.

## Comparison

Anneal and CREST do not share seeds, so the tests are unpaired.
The hit counts use a two-sided Fisher exact test. The call counts,
with a miss at budget + 1, use a two-sided exact Mann–Whitney test:
midranks, and the probability of a rank sum at least as far from the
null mean as the one observed.

The anneal hit rate has a Wilson 95% interval. The median of the finished
calls-to-hit has a bootstrap 95% interval: 10000 resamples with
replacement, generator seed 20261123, and the order statistics at
2.5% and 97.5% of the sorted medians.

The CREST call counts also define two earlier marks: the minimum and
the median of the six recorded counts. The report gives the fraction
of anneal seeds whose calls-to-hit are at or below each mark. A miss,
stored as budget + 1, is above both marks.
