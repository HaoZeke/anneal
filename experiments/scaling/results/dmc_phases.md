# Diffusion phase walls

Hyndman-Fan type 7 gives every median in this table.
On each run, the five phase walls add up to that run's wall.
Those phases are setup, selection, potential-energy calls, synchronization, and allocation.
Three later columns split synchronization into the evaluation batch, the per-control polish, and the endgame wave.
Pool thread i is pinned to physical core i, wrapping after the last core.
Worker counts above the physical core count stay out of this table.

| workers | n | wall s | setup s | selection s | potential s | sync s | allocation s | batch sync s | polish sync s | wave sync s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 0.452514 | 0.001078 | 0.006443 | 0.444680 | 0.000074 | 0.000149 | 0.000034 | 0.000033 | 0.000006 |
| 2 | 6 | 0.239127 | 0.001268 | 0.006592 | 0.226472 | 0.005222 | 0.000244 | 0.001539 | 0.003408 | 0.000182 |
| 4 | 6 | 0.135552 | 0.000921 | 0.006679 | 0.116426 | 0.011221 | 0.000259 | 0.001855 | 0.004403 | 0.004865 |
| 8 | 6 | 0.096052 | 0.000916 | 0.006946 | 0.064549 | 0.022022 | 0.000284 | 0.002326 | 0.005889 | 0.012853 |
