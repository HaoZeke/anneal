# Diffusion phase walls

Hyndman-Fan type 7 gives every median below.
The five phase medians add up to the measured wall.
Those phases are setup, selection, potential-energy calls, synchronization, and allocation.
Three later columns split synchronization into the evaluation batch, the per-control polish, and the endgame wave.
Pool thread i is pinned to physical core i, wrapping after the last core.
Worker counts above the physical core count stay out of this table.

| workers | n | wall s | setup s | selection s | potential s | sync s | allocation s | batch sync s | polish sync s | wave sync s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 0.452942 | 0.000938 | 0.006347 | 0.445419 | 0.000070 | 0.000155 | 0.000032 | 0.000034 | 0.000005 |
| 2 | 6 | 0.256359 | 0.001106 | 0.006486 | 0.225164 | 0.023092 | 0.000240 | 0.001874 | 0.003227 | 0.017654 |
| 4 | 6 | 0.157801 | 0.001040 | 0.006641 | 0.116032 | 0.033789 | 0.000261 | 0.002075 | 0.004504 | 0.026969 |
| 8 | 6 | 0.097259 | 0.001156 | 0.006653 | 0.066413 | 0.022491 | 0.000288 | 0.002687 | 0.005643 | 0.013782 |
