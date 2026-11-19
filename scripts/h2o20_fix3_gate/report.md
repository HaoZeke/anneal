# xtb force-sign gate

Arm A sha 5eb79cdcb0a986984dc7b487e5a3cc93224246f9 tree 7a2ba49b37241a8fe42b2c45c226374de6d92274.
Arm B sha 1c21c02bcb6cdff459b2425573bd0388686a3a99 tree 66d16d8d61c61a65f89ef455edc8919297dd6afd.

Arm A publishes the xtb buffer after negating it. The engine already
stores the physical force, and the potential negates that buffer once
to form the objective gradient, so the extra negation reverses the force.
Arm B publishes the engine force unchanged.

Both arms ran natively. No gradient shim was applied.
The H2O20 and crest-equal hit rules rescore the saved geometry with
xtb GFN2 vtight. CREST itself is not executed; H2O20 hits at or below
-101.749249454294 Eh by 1e-6 Eh, the campaign reference.
The crest-equal rows hit at or below -30.496506792187 Eh by 1e-7 Eh.
Hexamer prism hits are an intact cluster whose prism IRA RMSD is at most
0.598496 A, the half-gap cutoff at 0.598 A.
Lennard-Jones is dynamic lattice search at 400000 evaluations.
Morse is m38r6 at 400000 evaluations.
Gold is Sutton-Chen family 108 (sc108-38) at 400000 evaluations,
with forces from a second evaluation of the saved geometry.
That example does not print a force vector for Lennard-Jones or Morse;
those guards compare the printed energy, the hit, and the charged count.

Two-sided exact tests use Bonferroni alpha 0.008333 across the six cells.
Median intervals are percentile bootstrap, 10000 draws, seed 20261119.
The Mann-Whitney p-value is the exact rank-sum test.
A scoring cell fails when B is significantly worse on hits, energy, or cost.
Cost is worse only when the exact test says so and the median wall is outside the A/A band.
A guard fails when a printed energy differs by more than 1e-12, a hit or
charged count differs, a gold force component differs by more than 1e-12,
or the median wall is outside the A/A band.
The hexamer A/A wall band is applied to the xtb cells.
The Morse A/A wall band is applied to the guards.

## Slot check

Even seed indexes run A then B. Odd indexes run B then A.
Each AB slot therefore holds six A runs and six B runs.

```
slot arm pair count
h2o20 slot 0 arm A 6
h2o20 slot 0 arm B 6
h2o20 slot 1 arm A 6
h2o20 slot 1 arm B 6
hexamer slot 0 arm A 6
hexamer slot 0 arm B 6
hexamer slot 1 arm A 6
hexamer slot 1 arm B 6
crest-equal slot 0 arm A 6
crest-equal slot 0 arm B 6
crest-equal slot 1 arm A 6
crest-equal slot 1 arm B 6
lj slot 0 arm A 6
lj slot 0 arm B 6
lj slot 1 arm A 6
lj slot 1 arm B 6
morse slot 0 arm A 6
morse slot 0 arm B 6
morse slot 1 arm A 6
morse slot 1 arm B 6
au slot 0 arm A 6
au slot 0 arm B 6
au slot 1 arm A 6
au slot 1 arm B 6
```

## A/A wall band

hexamer: replica medians 9.63558 and 9.71908, band [9.63558, 9.71908]
morse: replica medians 0.440808 and 0.442955, band [0.440808, 0.442955]

## Cells

| cell | hits A | hits B | Fisher p | median energy A (95% CI) | median energy B (95% CI) | energy p | median wall A | median wall B | wall p | A/A band | median evals A | median evals B | evals p | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h2o20 | 0/12 | 0/12 | 1 | -101.726409086 (-101.729880211, -101.720459882) | -101.744273888 (-101.74576838, -101.742512913) | 7.396e-07 | 1071.54 | 1717.44 | 7.396e-07 | [1066.92, 1076.16] | 26400 | 28000 | 7.396e-07 | FAIL |
| hexamer | 0/12 | 12/12 | 7.396e-07 | inf (inf, inf) | -829.852227521 (-829.852227523, -829.85222751) | 7.396e-07 | 9.65745 | 18.2716 | 7.396e-07 | [9.63558, 9.71908] | 3000 | 4000 | 7.396e-07 | FAIL |
| crest-equal | 0/12 | 12/12 | 7.396e-07 | -30.4946632583 (-30.4959874145, -30.4923598001) | -30.4965068671 (-30.4965068673, -30.4965068654) | 7.396e-07 | 11.8164 | 18.3319 | 7.396e-07 | [11.7654, 11.8674] | 3021 | 4000 | 7.396e-07 | FAIL |
| lj | 1/12 | 1/12 | 1 | -173.1933475 (-173.252378, -173.134317) | -173.1933475 (-173.252378, -173.134317) | 1 | 1.40049 | 1.40092 | 0.3186 | [1.39709, 1.40389] | 400000 | 400000 | 1 | PASS |
| morse | 12/12 | 12/12 | 1 | -157.47710811 (-157.47710811, -157.47710811) | -157.47710811 (-157.47710811, -157.47710811) | 1 | 0.441495 | 0.442061 | 0.9774 | [0.440808, 0.442955] | 49070 | 49070 | 1 | PASS |
| au | 8/12 | 8/12 | 1 | -10117.2454073 (-10117.2454073, -10108.5598498) | -10117.2454073 (-10117.2454073, -10108.5598498) | 1 | 12.204 | 12.3716 | 0.9774 | [12.1744, 12.2337] | 311796 | 311796 | 1 | FAIL |

## Verdicts

h2o20: FAIL (wall,evals)
hexamer: FAIL (wall,evals)
crest-equal: FAIL (wall,evals)
lj: PASS (none)
morse: PASS (none)
au: FAIL (wall-band)

GATE FAIL

Failed cells: h2o20, hexamer, crest-equal, au
