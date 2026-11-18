# General search controls for the GMIN losses

This directory records the same-session gate for the two losses in the GMIN
panel. All runs use the paper call budgets, seeds 2000--2011, hostname
`bench`, one process pinned to core 0, and the same starting coordinates for
paired arms. The Mackay cap and `mackay_cut` are absent from every anneal arm.
Times named `pot_s` are inside the potential; `outside_s` is the rest.

## Diagnosis

The committed panel puts M80 rho=6 in three funnels. Ten of twelve anneal runs
miss: six finish at -377.97249727, three at -376.25456752, and one at
-376.86294627. GMIN reaches -378.333471 on all twelve seeds, in 59,837 to
1,298,809 calls. Its median is 268,934 calls. Anneal's two hits are seeds 2004
at 865,177 calls and 2011 at 3,739,049 calls.

The Morse driver used the Lennard-Jones length unit although
`PairKind::length_scale()` documents the pair minimum as the natural unit.
After applying the pair-derived scale, five training seeds had aggregate
acceptance between 0.5910 and 0.6010. Hits and misses had the same acceptance:
seeds 2000, 2001 and 2003 hit, while 2002 and 2004 repeatedly returned to
-377.97249727. The stall is therefore a funnel and population problem, not
simple rejection.

GMIN's common input makes the contrasting policy explicit: `STEP 0.40 0.51`
adapts the Cartesian step toward 51% acceptance, and `NEWRESTART 40` starts
again after a short unproductive run. Anneal's preset accepts near 60% on the
wrong M80 funnel and responds to its 5,000-call quiet clock with an in-place
directed leave. It neither retunes every acceptance window nor preserves a
fingerprinted set of alternative funnels.

LJ150 has the analogous plateau at a larger scale. The seven misses in the
committed panel finish between -883.110588 and -885.282282, while every hit
reaches -893.310258. The prior fresh-branch experiment recovered six of those
seeds, but charged branch calls delayed five late hits elsewhere past their
budgets. A safe branch must respond to observed progress and preserve the live
walk's budget.

## Rejected screens

These screens were serial on the required host and core, with the Mackay paths
removed. They are negative results, not the gate.

| candidate | M80 rho=6 seeds | hits | failure |
| --- | --- | ---: | --- |
| pair-derived length scale | 2000--2004 | 3/5 | 2002, 2004 at -377.97249727 |
| scale, plateau heat, acceptance adaptation | 2000--2003 | 3/4 | traded away baseline seed 2003 |
| two replicas | 2000--2001 | 1/2 | seed 2001 at -377.97249727 |
| half cold, half adaptive population | 2000--2005 | 4/6 | 2004 and 2005 at -377.97249727 |

The half-and-half population is complementary on seeds 2000--2003, but is not
the requested fix: both members enter the same wrong funnel on 2004 and 2005.
It remains off.

## Gate

`rows.csv` and `summary.csv` are written by
`scripts/gmin_panel/run_fix_gate.py`. The fix is eligible to become the
default only when all thirteen baseline/fix pairs have no lost seed. Held-out
rows use seeds 3000--3011 for M80 rho=6 and LJ150 and include GMIN.

The full gate on reviewed base `628773d6` and candidate `2b987493` was
negative. Both arms used the same template ablation, whose patched base tree
is `a9cef902826cc328e28dd77d2fd95689686c4697`. M80 rho=6 improved from
0/12 to 6/12 (`p=0.01373`), but LJ150 remained 0/12. The candidate also lost
seeds on LJ75 (9/12 to 6/12), LJ98 (3/12 to 1/12), LJ103 (2/12 to 0/12),
and Au55 (3/12 to 2/12). It therefore remains opt-in.

The zero LJ150 baseline is itself material: removing all named builders and
template paths is substantially harder than merely removing the historical
Mackay cap. The adaptive population does not recover that loss at the paper
budget. `summary.csv` contains Fisher exact and tie-corrected Mann--Whitney
statistics plus median wall, potential, and outside-potential times for every
system. No manuscript claim was changed.
