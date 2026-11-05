# Diffusion phase walls

Hyndman-Fan type 7 gives every median in this table.
On each run, the five phase walls add up to that run's wall.
Those phases are setup, selection, potential-energy calls, synchronization, and allocation.
Three later columns split synchronization into the evaluation batch, the per-control polish, and the endgame wave.
Pool thread i is pinned to physical core i, wrapping after the last core.
Worker counts above the physical core count stay out of this table.

| workers | n | wall s | setup s | selection s | potential s | sync s | allocation s | batch sync s | polish sync s | wave sync s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 0.449129 | 0.001042 | 0.006369 | 0.441401 | 0.000066 | 0.000148 | 0.000031 | 0.000030 | 0.000004 |
| 2 | 6 | 0.255527 | 0.001036 | 0.006557 | 0.224315 | 0.023197 | 0.000236 | 0.001807 | 0.003498 | 0.017857 |
| 4 | 6 | 0.156836 | 0.001038 | 0.006572 | 0.115511 | 0.033246 | 0.000250 | 0.001835 | 0.004254 | 0.026949 |
| 8 | 6 | 0.090514 | 0.001543 | 0.006640 | 0.061371 | 0.020714 | 0.000291 | 0.002000 | 0.005170 | 0.013675 |
