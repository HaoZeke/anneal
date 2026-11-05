# Weak scaling

Host CPU: Intel Xeon Processor, 8 cores.
GPU: none. CUDA: none.
Parallel tempering uses one process per worker.
The parent process draws every swap after each swap interval.
Population annealing and the diffusion population use threads.
Those threads call the native Lennard-Jones energy.
They do not take the interpreter lock.
Snakemake exports OMP_NUM_THREADS as 1.
The Rayon pools and the diffusion threads use the worker count.
Worker counts are 1, 2, 4, and 8.
Every published row uses at most 8 workers on this 8-core host.
A sixteen-worker row would share the cores and is omitted.
Each cell is six seeds, numbered 0 through 5.
The wall time is the median.
The IQR is the type 7 interquartile range of those six wall times.
Weak efficiency is the one-worker median divided by the cell median.
Parallel tempering keeps 2 replicas and 524290 charged evaluations on each worker.
Population annealing keeps 8 replicas and 524296 charged evaluations on each worker.
The diffusion population keeps 8 replicas and 262144 charged evaluations on each worker.
Atoms per cluster stay fixed, so the pair count of one evaluation stays fixed.
A multi-node path needs an MPI launcher or the bank RPC sockets,
one chain group or one population per rank, and an exchange of energies
at each swap or resampling step.
This host has no second node and no MPI library, so those rows are absent.

| family | workers | atoms | replicas | charged | median wall s | type 7 IQR s | weak efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|
| pt | 1 | 13 | 2 | 524290 | 0.334411 | 0.022259 | 1.000000 |
| pt | 2 | 13 | 4 | 1048580 | 0.400257 | 0.010201 | 0.835491 |
| pt | 4 | 13 | 8 | 2097160 | 0.402512 | 0.011356 | 0.830810 |
| pt | 8 | 13 | 16 | 4194320 | 0.563656 | 0.006234 | 0.593290 |
| population | 1 | 13 | 8 | 524296 | 0.300684 | 0.014117 | 1.000000 |
| population | 2 | 13 | 16 | 1048592 | 0.309041 | 0.003381 | 0.972956 |
| population | 4 | 13 | 32 | 2097184 | 0.329608 | 0.031387 | 0.912246 |
| population | 8 | 13 | 64 | 4194368 | 0.381900 | 0.002763 | 0.787336 |
| dmc | 1 | 13 | 8 | 262144 | 0.232455 | 0.001422 | 1.000000 |
| dmc | 2 | 13 | 16 | 524288 | 0.235264 | 0.000513 | 0.988063 |
| dmc | 4 | 13 | 32 | 1048576 | 0.242265 | 0.004645 | 0.959509 |
| dmc | 8 | 13 | 64 | 2097152 | 0.349850 | 0.029170 | 0.664442 |

HEAD: 9bd090b55a39da1a2ef474750a8f42d41e7a6efc
tree: 8b3d122f454b326de2789012605973f589ae29ce
