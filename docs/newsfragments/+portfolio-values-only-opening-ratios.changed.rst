The values-only loop of ``global_optimize``, which ``minimize`` runs, reads
the gain of each opening slice after the first from below the start, and the
gains it is held to, as log ratios, ``ln(before / after)``, while every value
the run has found is positive. The first slice from below the start is still
held to the gain of the slice from the start, both read as differences, and
from the first value at or below zero every gain, the earlier slices'
included, reads as a difference. A positive loss that falls by decades from
a far start, such as a squared energy error, no longer ends the opening on
its valley floor, held to the gains of the slices that came down the wall.
The ratios keep every slice that differences keep, so a run takes
another course only where differences would have ended its opening, after its
first slice from below the start, on a slice that left every value positive,
and there it no longer repeats the run of 0.11.1.
