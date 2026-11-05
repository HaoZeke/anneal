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
| pt | 1 | 13 | 2 | 524290 | 0.328648 | 0.004687 | 1.000000 |
| pt | 2 | 13 | 4 | 1048580 | 0.333972 | 0.000548 | 0.984059 |
| pt | 4 | 13 | 8 | 2097160 | 0.342690 | 0.011627 | 0.959024 |
| pt | 8 | 13 | 16 | 4194320 | 0.355212 | 0.038381 | 0.925217 |
| population | 1 | 13 | 8 | 524296 | 0.298442 | 0.013117 | 1.000000 |
| population | 2 | 13 | 16 | 1048592 | 0.305530 | 0.002612 | 0.976802 |
| population | 4 | 13 | 32 | 2097184 | 0.307793 | 0.008475 | 0.969620 |
| population | 8 | 13 | 64 | 4194368 | 0.310501 | 0.004607 | 0.961163 |
| dmc | 1 | 13 | 8 | 262144 | 0.231460 | 0.001446 | 1.000000 |
| dmc | 2 | 13 | 16 | 524288 | 0.235238 | 0.003726 | 0.983940 |
| dmc | 4 | 13 | 32 | 1048576 | 0.241782 | 0.001456 | 0.957309 |
| dmc | 8 | 13 | 64 | 2097152 | 0.275477 | 0.003188 | 0.840216 |

HEAD: 06c697bdd89dc0e2fcf8f2dc173a45843e6d7459
tree: 6cf68856299cd63e6a8bc51658dce005e893ff6c
