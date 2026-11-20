# Template-Free2 Gate B

## Declared protocol

This gate compares two independent clean builds of the base tree (A and
A') with the candidate tree (B).  All three builds use compiler 1.88.0,
locked dependencies, release settings, one core, and one thread for every
threaded runtime.  There is no warm-up.  Seeds are 2000 through 2011.

Runs are serial.  For each cell, arm order rotates across three consecutive
seed pairs.  The odd seed reverses the preceding even seed's order.  This
places every arm in every slot four times.

The 38-atom budget is 400,000 calls, the 55-atom budget is 1,000,000
calls, and sizes of 75 atoms or more use 4,000,000 calls.  TIP4P and
silicon retain their committed gate budgets.  A hit is a finite best
energy no more than 0.001 above the committed reference.  Errors remain
in the rows as misses.

For each cell, B fails when any of the following is true:

1. B has fewer hits than A and the exact one-sided Fisher probability for
   a hit loss is below 0.05.
2. B has worse calls with a one-sided, tie-corrected Mann-Whitney
   probability below 0.05, its median call ratio over A is above 1.0, and
   that ratio is above the symmetric A/A' band
   `max(A'/A, A/A')`.  Misses rank as positive infinity in the test.
3. The ratio of the median per-row `outside_s / charged` for B over A is
   above 1.10 while the corresponding A'/A ratio lies in
   `[0.95, 1.05]`.

Every hit loss is reported even when it is not significant.  The runtime
activation count is opt-in output read after each trajectory.  For TIP4P
and silicon it must be zero, and B must match A seed by seed in both
`best_bits` and calls.

## Cell availability

The committed gate definitions provide TIP4P sizes 8 through 21.  The
committed paper panel provides Au55 and Stillinger-Weber Si10, Si13, and
Si20.

The reused gate names four BLJ controls, but neither that gate nor the
base tree contains their driver source.  They are therefore marked
missing rather than reconstructed.  A missing requested cell makes the
overall verdict fail.
