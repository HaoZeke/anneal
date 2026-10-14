The values-only loop of ``global_optimize``, which ``minimize`` runs, reads
the gain of each opening slice after the first from below the start, and the
gains it is held to, as log ratios, ``ln(before / after)``, while every value
the run has found is positive. The first slice from below the start is still
held to the gain of the slice from the start, both read as differences, and
from the first value at or below zero every gain, the earlier slices'
included, reads as a difference. A positive loss that falls by decades from
a far start, such as a squared energy error, no longer ends the opening on
its valley floor, held to the gains of the slices that came down the wall.
The ratios keep every slice that differences keep, so they change a run only
where differences would have ended its opening, after its first slice from
below the start, on a slice that left every value positive. Over 40 paired
seeds the
bar no longer leaves the shifted and rotated ellipsoids and Rosenbrock in 30
dimensions at 5000 evaluations, nor Rosenbrock's copies with 1e2 added and
multiplied by 1e-6, behind an opening that goes on while each slice
succeeds, and it still keeps shifted Ackley, Levy and rotated Levy in 30
dimensions at 5000 ahead of that opening (medians 5.1e-7 against 4.0e-6,
3.7e-11 against 1.5e-6 and 0.633 against 3.15). These comparisons with
0.11.1 take in this release's other changes to the loop as well as the
ratios.
