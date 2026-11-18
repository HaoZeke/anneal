# Template-removal rerun deltas (append-only rows)

All reruns use seeds 2000--2011 (`n=12`) on `host=bench`, with appended rows keyed by `code_revision` and `harness_revision`.

## GMIN panel deltas

| System | Old (template-era) | New (template-free) |
| --- | --- | --- |
| M55, rho=6 | anneal 12/12 vs GMIN 12/12, Fisher p=1.0 | anneal 12/12 vs GMIN 12/12, Fisher p=1.0 |
| M55, rho=14 | anneal 7/12 vs GMIN 0/12, Fisher p=0.00457666 | anneal 7/12 vs GMIN 0/12, Fisher p=0.00457666 |
| M80, rho=6 | anneal 2/12 vs GMIN 12/12, Fisher p=6.73038e-05 | anneal 2/12 vs GMIN 12/12, Fisher p=6.73038e-05 |
| M80, rho=14 | anneal 9/12 vs GMIN 0/12, Fisher p=0.000336519 | anneal 9/12 vs GMIN 0/12, Fisher p=0.000336519 |
| Au55 | anneal 6/12 vs GMIN 1/12, Fisher p=0.0686499 | anneal 4/12 vs GMIN 1/12, Fisher p=0.31677 |

Current gmin-panel summary medians for changed rows:

- Au55 wall/in/out (anneal): 38.043 / 28.083 / 9.966 s.
- M55 rho=6 wall/in/out (anneal): 1.742 / 1.196 / 0.547 s.
- M55 rho=14 wall/in/out (anneal): 12.814 / 9.196 / 3.618 s.
- M80 rho=6 wall/in/out (anneal): 161.389 / 107.846 / 53.778 s.
- M80 rho=14 wall/in/out (anneal): 49.325 / 37.783 / 11.542 s.

Significant clusters in `bench/gmin-panel/summary.csv` (Fisher p < 0.05):

- Old template-era panel: 7 (LJ38, LJ75, LJ98, Au55, M55 rho=14, M80 rho=6, M80 rho=14)
- New template-free panel: 6 (LJ38, LJ75, LJ98, M55 rho=14, M80 rho=6, M80 rho=14)

## OCCAT and ASE Au55 deltas

- OCCAT (`bench/occat-h2h`): appended template-free anneal Au55 rows are 4/12 against OCCAT 0/12 (two-sided Fisher p=0.0931677).
- ASE Au55 (`bench/ase-au55`): anneal `recommended` remains 0/12; ASE BH 0/12; ASE MH 0/12 (Fisher p=1.0 against each arm).
- ASE Au55 median best energies above reference: anneal +0.05083548 eV, ASE BH +1.19646891 eV, ASE MH +0.73689833 eV.

## Why the two Au55 panels differ

Both anneal arms use a 1,000,000-call budget and the same
`-191.03085280` reference. The ASE `recommended` runs have a nominal
122.695098-second cap, but every one stopped on the call budget rather than
that cap. Thus the 4/12 against GMIN and 0/12 against ASE are not different
reference or effective-budget results: the GMIN-facing driver runs the
unscaled preset, while the ASE panel runs `recommended_with_scales` at the
Gupta length and energy scales.

| Seed | GMIN-facing hit/calls/best | ASE-facing hit/stop/charged/best |
| --- | --- | --- |
| 2000 | 0 / 1000000 / -190.98001732 | 0 / budget / 1000000 / -190.98001732 |
| 2001 | 0 / 1000000 / -191.00616498 | 0 / budget / 1000000 / -191.00616498 |
| 2002 | 0 / 1000000 / -191.00270211 | 0 / budget / 1000000 / -191.00367602 |
| 2003 | 0 / 1000000 / -191.00616498 | 0 / budget / 1000000 / -190.98001732 |
| 2004 | 0 / 1000000 / -191.00616498 | 0 / budget / 1000000 / -190.98001732 |
| 2005 | 1 / 313188 / -191.03085280 | 0 / budget / 1000000 / -190.97692561 |
| 2006 | 0 / 1000000 / -191.00616498 | 0 / budget / 1000000 / -190.97692561 |
| 2007 | 0 / 1000000 / -191.00616498 | 0 / budget / 1000000 / -190.98001732 |
| 2008 | 1 / 39878 / -191.03085280 | 0 / budget / 1000000 / -191.00270211 |
| 2009 | 0 / 1000000 / -190.98001732 | 0 / budget / 1000000 / -191.00270211 |
| 2010 | 1 / 78896 / -191.03085280 | 0 / budget / 1000000 / -190.98001732 |
| 2011 | 1 / 113461 / -191.03085280 | 0 / budget / 1000000 / -190.98331173 |

## Morse rerun identity

The old rows use revision `c2de652d`; the new rows use
`1ce3ab64`. Their hit counts and Fisher p-values happen to agree, but the
wall times do not. In particular the old M55 rho=6 cap quench finishes in
roughly 0.01 seconds, whereas the template-free searches take up to 5.52
seconds. The unchanged aggregate outcomes therefore do not represent copied
or bit-identical trajectories.

The wall times below are ordered by seed 2000--2011.

- M55 rho=6 old:
  `0.061560, 0.010639, 0.010869, 0.010276, 0.008137, 0.009523,`
  `0.009833, 0.007412, 0.013258, 0.008619, 0.011769, 0.007439`.
  New:
  `1.832575, 3.968768, 1.651991, 5.517872, 0.992372, 3.078680,`
  `0.126953, 1.943321, 2.699255, 0.840343, 0.151131, 0.547798`.
- M55 rho=14 old:
  `14.982468, 3.792062, 12.203148, 19.045816, 19.632807, 18.892036,`
  `18.211304, 7.983129, 12.182674, 2.179527, 3.009660, 18.684374`.
  New:
  `14.217777, 3.716057, 11.078522, 17.871176, 17.716379, 17.786544,`
  `17.828954, 7.722626, 11.409511, 2.091692, 2.871347, 17.725040`.
- M80 rho=6 old:
  `168.810125, 171.408946, 164.804626, 178.269596, 36.934321,`
  `172.331125, 168.137572, 169.352436, 169.001876, 163.138129,`
  `170.842585, 159.618860`.
  New:
  `162.420494, 158.267487, 157.174044, 160.358363, 33.625986,`
  `164.271055, 159.409597, 164.699698, 165.969554, 165.355919,`
  `169.904211, 153.800096`.
- M80 rho=14 old:
  `155.935033, 16.415339, 156.175105, 43.492577, 11.687988, 40.540960,`
  `36.497870, 6.172172, 67.822350, 144.639098, 56.246389, 160.146719`.
  New:
  `149.969474, 15.901948, 159.012142, 46.451742, 10.733651, 38.730232,`
  `36.003533, 5.947639, 70.822954, 154.646489, 52.198042, 150.132623`.

## ABCluster handoff

The five-patch handoff applies cleanly to `61b5c9a0`. It contains no Morse
systems. Its Au55 `anneal-recommended` arm is template-affected (6/12 in
the handed-off rows); the ABCluster arm itself is 0/12 and unchanged.

The template-free Au55 rerun is stored in
`bench/abcluster-template-free-rows.csv` at revision `1ce3ab64`, host
`bench`. Anneal recommended changes from 6/12 to 4/12 against the reused
ABCluster 0/12 rows (two-sided Fisher p changes from 0.01373 to 0.0931677);
anneal for_cluster remains 0/12 (Fisher p=1). Recommended median
wall/potential/outside times are 39.908/29.180/10.728 s and for_cluster
times are 36.617/31.759/4.807 s. Tie-corrected Mann--Whitney p-values
against ABCluster are 0.0366874 (calls), 0.0000365846 (wall),
0.0000365846 (potential), and 0.174853 (outside) for recommended; they are
1, 0.0000365846, 0.0000365846, and 0.0000365846 for for_cluster.

## Pin delta

The clock-free Au55 seeds 2000--2002 pin changes from
`b22e212181eff57e9412f780c40e85208b355d07` to
`1213f9cdf92a2dabfc8a51c83522086f8a9d7cdc`. The old pin depended on the
removed structure-template path.

The 2,500-call LJ38 short-search regression changes from best
-170.2604835583855163, 25 hops, 16 converged and 23 capped quenches to
-172.2981287221047921, 38 hops, 33 converged and 26 capped quenches. The
corresponding LJ98 regression is -520.3133998213386349, 10 hops, 10
converged and 7 capped quenches.
The longer LJ98 seed-2000 trajectory hash changes from
`6e277f8611c9988bbb95c72873fb5b044be414e9148ab9571b732f834e3dceb9`
to
`2f496356ddb82f2745cec803637ef90d415dd8307511834f57074051fedc5e3b`
because that trajectory used the removed structure-specific proposals.
The 16-seed LJ38 two-phase regression keeps the compacted hit mask
`0xffff`, while the plain-walk mask changes from `0x91e4` (7/16) to
`0x8505` (5/16).

## Notes on unavailable landed rows

- `bench/abcluster-h2h` is not yet on the landing base, so its data remain in
  the separate handoff rather than being copied into this patch stack.
