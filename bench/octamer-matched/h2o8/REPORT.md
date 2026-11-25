# (H2O)8 matched budget

Anneal seeds 2000–2007 finished. Seeds 2008–2011 were not started.
The mean wall of a finished seed is 1539.9 s, so twelve seeds project
to 5.13 h of search time. After seed 2007, 1436 s remained of the
four-hour window, which is shorter than that mean. CREST was not rerun.

The code that ran is revision `9747c02ee673b41a584ed48e2422a27b15449c75`.
Binary `gfn2_mol_h2h`, feature `rgpot-ex`.

## Recorded rows

`bench/h2o-octamer/results.csv` is the GFN2 `(H2O)8` head-to-head, and
`bench/h2o-octamer/README.md` states its rule. The potential is the
rgpot GFN2 handle (`libxtb_engine.so`, rgpot
`8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e`, xtb 6.7.1 `edcfbbe`).
The reference is −40.681336013150 Eh and the tolerance is 1e-6 Eh.
The anneal arm in that file is capped at 4000 calls. The CREST arm
records a full run on each of seeds 17000–17005.

## Budget

The shared anneal budget is the maximum of those six CREST call counts:
151738, from seed 17004.

| seed | calls | energy / Eh | hit |
| --- | --- | --- | --- |
| 17000 | 127141 | −40.681335999128 | 1 |
| 17001 | 128176 | −40.681335990637 | 1 |
| 17002 | 130752 | −40.681335997000 | 1 |
| 17003 | 123134 | −40.681335987654 | 1 |
| 17004 | 151738 | −40.681335997331 | 1 |
| 17005 | 126694 | −40.681335977771 | 1 |

The minimum is 123134. The median is 127658.5. Each CREST row in
`rows.csv` keeps that seed's own call count. Anneal rows leave
`crest_calls` empty: seeds 2000–2007 are not those CREST seeds.

## Search

`examples/gfn2_mol_h2h` builds `Config::recommended_molecular` at
`examples/gfn2_mol_h2h.rs:474`. The preset is
`src/methods/cluster_hopping/config.rs:2623`. The example is built with
`rgpot-ex` (`Cargo.toml`). The threshold is the reference plus the
tolerance, converted with 27.211386245988 eV/Eh
(`src/catalog/molecular.rs:95`) and installed with `Ledger::stop_below`.
A hit is an intact geometry at or below that threshold. `calls_to_hit`
is the first objective evaluation there. A miss would be stored as
151739. No finished seed missed.

On every finished seed the hop used 150138 calls and closed on the
budget. The incumbent was still above the tolerance. The reserved 1600
polish calls then crossed the threshold, between the 31st and the 46th
of those calls. The recorded stop reason is therefore `budget`, and
`calls_to_hit` is that polish evaluation. Charged calls are 151738.
The best energy is −40.681336013361 Eh on every seed, 2.11e-10 Eh
below the reference, and the geometry is intact.

## Anneal seeds 2000–2007

| seed | hit | calls_to_hit | stop | best / Eh | wall / s |
| --- | --- | --- | --- | --- | --- |
| 2000 | 1 | 150182 | budget | −40.681336013361 | 1606.973612 |
| 2001 | 1 | 150174 | budget | −40.681336013361 | 1591.807295 |
| 2002 | 1 | 150169 | budget | −40.681336013361 | 1716.752373 |
| 2003 | 1 | 150170 | budget | −40.681336013361 | 1523.731298 |
| 2004 | 1 | 150184 | budget | −40.681336013361 | 1531.149134 |
| 2005 | 1 | 150178 | budget | −40.681336013361 | 1565.511328 |
| 2006 | 1 | 150172 | budget | −40.681336013361 | 1373.239084 |
| 2007 | 1 | 150169 | budget | −40.681336013361 | 1410.121805 |

## Comparison

The two lists are independent samples: eight anneal seeds and six CREST
seeds. CREST hit 6/6.

Anneal hit 8/8. Wilson 95% interval [0.6756, 1.0000].

Median calls to the hit: 150173. Bootstrap 95% interval, 10000
resamples, seed 20261123: [150169, 150182].

Fisher exact two-sided p on the hit counts: 1.

Mann–Whitney exact two-sided p on the call counts: 0.038961.
That is 117 of the 3003 assignments of the fourteen ranks.

Anneal seeds with `calls_to_hit` at or below the CREST minimum
(123134): 0/8. Wilson 95% interval [0.0000, 0.3244].

Anneal seeds with `calls_to_hit` at or below the CREST median
(127658.5): 0/8. Wilson 95% interval [0.0000, 0.3244].
