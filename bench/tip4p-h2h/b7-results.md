# b7 generic rigid-body results

These results use release builds, seeds 0–5, and a 20,000-call budget.
The reference energy is read only by the driver as the stop threshold,
with the usual 0.001 kJ/mol tolerance. Search code does not read it.
Raw rows are under
`kits/tip4p-hits-b7-evidence/{b6-panel,peripheral-close}/`.

## Matched-budget hits

| Bodies | b6 hits | b7 hits | Change |
| ---: | ---: | ---: | ---: |
| 8 | 5/6 | 6/6 | +1 |
| 9 | 4/6 | 6/6 | +2 |
| 10 | 5/6 | 5/6 | 0 |
| 11 | 0/6 | 3/6 | +3 |
| 12 | 1/6 | 1/6 | 0 |
| 13 | 0/6 | 0/6 | 0 |
| 14 | 0/6 | 0/6 | 0 |
| 15 | 0/6 | 0/6 | 0 |
| 16 | 0/6 | 0/6 | 0 |
| 17 | 0/6 | 0/6 | 0 |
| 18 | 0/6 | 0/6 | 0 |
| 19 | 0/6 | 0/6 | 0 |
| 20 | 0/6 | 0/6 | 0 |
| 21 | 0/6 | 0/6 | 0 |

The three new 11-body hits are seed 0 at 19,132 calls, seed 3 at
10,724 calls, and seed 4 at 11,628 calls. No hit count falls at 8, 9,
or 10 bodies. Across 11–21 bodies the total changes from 1/66 to 4/66;
the 12-body count remains 1/6.

There is seed-level churn despite the nondecreasing counts. At 10 bodies,
b6 seed 0 is lost and b7 seed 4 is gained. At 12 bodies, b6 seed 4 is
lost and b7 seed 0 is gained. All other b6 hits remain hits.

## Move derived from the diagnosis

The diagnosis found that only about 237–256 fully quenched proposals fit
in the budget and that relocation accepts only 13.7–15.4% of draws.
The added move closes one missing centre contact before the expensive
quench:

1. The median nearest-neighbour distance is the observed contact scale.
2. Bodies with the fewest contacts at that scale form the loose set.
3. The most peripheral loose body is selected.
4. It moves toward its nearest non-contact until that contact reaches
   the observed scale.

The current geometry determines every choice. The move has no species,
body-count, motif, template, reference-energy, or fitted constant.

The initial novelty-acceptance experiment produced no 11–21 hits and
lost a 10-body hit. A symmetric random-body closure produced one 14-body
hit but reduced 10-body hits from 5/6 to 3/6. Moving one random body
produced one 11-body and one 14-body hit but reduced 10-body hits to
4/6. Random choice among least-coordinated bodies produced three larger
hits but reduced 9-body hits to 3/6. All were discarded. Selecting the
most peripheral least-coordinated body is the first generic variant
without a small-size loss.

## Non-rigid identity and overhead

The full outcome digest covers best and final energy bits, best and final
state bits, charged calls, hops, basins, screening and return counts, and
every improvement record. All 12 digests for `lj38`, `lj75`, `m38r6`,
and `m38r14`, seeds 0–2, match b6 bit for bit.

Across all 84 rigid rows, total outside time divided by charged calls is
2.328076 microseconds per call for b6 and 2.371825 for b7, an increase
of 0.043749 microseconds or 1.88%. The median per-row values are
2.123300 and 2.195650 microseconds per call.
