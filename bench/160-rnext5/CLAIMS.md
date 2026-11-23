# r-next5 analysis notes

The overall gate verdict is **FAIL**.

Every hit clause passed and there were no paired seeds where A hit and B
missed. The rigid diagnostic requirement passed with 51 shell events and
267400 recorded events across B's rigid-water rows.

The outside-cost clause failed in TIP4P9, TIP4P10, TIP4P14, TIP4P16, and
TIP4P17. Their paired B/A t-interval upper bounds were 1.301, 1.140,
1.272, 1.168, and 1.158.

The one-sided memory clause failed in LJ55 and M55 radius 6. LJ55 had
9/1/2 B-more/less/tied outcomes and one-sided probability 0.0107. M55
radius 6 had 9/2/1 and probability 0.0327.

## LJ150

LJ150 passed every registered clause. Its paired B/A outside-cost estimate
was 0.982 with t interval [0.959, 1.004]. The A2/A band was 0.998
[0.964, 1.031].

Seed 2000 reproduced byte-identical best-energy bits and equal charged
calls between A and B. Both had bits `c08bea7b687bbf69` and 162859 calls.

The claimed outside medians of 22.455 and 23.087 microseconds per call
were not reproduced. The observed A and B medians were 19.292 and 18.785
microseconds per call.
