# Weak scaling

Host CPU: Intel Xeon Processor, 8 cores.
GPU: none. CUDA: none.
Parallel tempering uses one process per worker.
The parent process draws every swap after each swap interval.
Population annealing and the diffusion population use threads.
Those threads call the native Lennard-Jones energy.
The energy evaluation stays in native code.
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
Multi-node rows need a second host or an MPI library.
This host has neither, so those rows are absent.

| family | workers | atoms | replicas | charged | median wall s | type 7 IQR s | weak efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|
| pt | 1 | 13 | 2 | 524290 | 0.324728 | 0.003891 | 1.000000 |
| pt | 2 | 13 | 4 | 1048580 | 0.345876 | 0.026062 | 0.938857 |
| pt | 4 | 13 | 8 | 2097160 | 0.331309 | 0.002588 | 0.980135 |
| pt | 8 | 13 | 16 | 4194320 | 0.362702 | 0.055182 | 0.895303 |
| population | 1 | 13 | 8 | 524296 | 0.301126 | 0.008971 | 1.000000 |
| population | 2 | 13 | 16 | 1048592 | 0.308449 | 0.003243 | 0.976261 |
| population | 4 | 13 | 32 | 2097184 | 0.312533 | 0.004416 | 0.963501 |
| population | 8 | 13 | 64 | 4194368 | 0.334072 | 0.020218 | 0.901383 |
| dmc | 1 | 13 | 8 | 262144 | 0.236272 | 0.007235 | 1.000000 |
| dmc | 2 | 13 | 16 | 524288 | 0.236985 | 0.003098 | 0.996993 |
| dmc | 4 | 13 | 32 | 1048576 | 0.246501 | 0.017862 | 0.958504 |
| dmc | 8 | 13 | 64 | 2097152 | 0.282527 | 0.015918 | 0.836283 |

HEAD: baaf7f179a726e513a1455842658e4de3ddc9ef4
tree: 51a029e2a57b784a463f79a4f581ed99a59441b9
