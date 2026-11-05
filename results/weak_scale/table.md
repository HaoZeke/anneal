# Weak scaling

Host CPU: Intel Xeon Processor, 8 cores.
GPU: none. CUDA: none.
Workers are threads in one process on this host.
Snakemake exports OMP_NUM_THREADS as 1.
The Rayon pools and the diffusion threads use the worker count.
Worker counts are 1, 2, 4, 8, and 16.
Sixteen threads share 8 cores.
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
| pt | 1 | 13 | 2 | 524290 | 0.322771 | 0.001348 | 1.000000 |
| pt | 2 | 13 | 4 | 1048580 | 0.329815 | 0.002622 | 0.978643 |
| pt | 4 | 13 | 8 | 2097160 | 0.421145 | 0.017405 | 0.766415 |
| pt | 8 | 13 | 16 | 4194320 | 0.487797 | 0.011387 | 0.661692 |
| pt | 16 | 13 | 32 | 8388640 | 0.789296 | 0.004308 | 0.408936 |
| population | 1 | 13 | 8 | 524296 | 0.295314 | 0.007089 | 1.000000 |
| population | 2 | 13 | 16 | 1048592 | 0.305168 | 0.002625 | 0.967710 |
| population | 4 | 13 | 32 | 2097184 | 0.311790 | 0.012381 | 0.947159 |
| population | 8 | 13 | 64 | 4194368 | 0.307402 | 0.003983 | 0.960679 |
| population | 16 | 13 | 128 | 8388736 | 0.627817 | 0.005351 | 0.470383 |
| dmc | 1 | 13 | 8 | 262144 | 0.226654 | 0.001601 | 1.000000 |
| dmc | 2 | 13 | 16 | 524288 | 0.230339 | 0.000880 | 0.984002 |
| dmc | 4 | 13 | 32 | 1048576 | 0.238804 | 0.002444 | 0.949120 |
| dmc | 8 | 13 | 64 | 2097152 | 0.259108 | 0.001191 | 0.874745 |
| dmc | 16 | 13 | 128 | 4194304 | 0.544750 | 0.004071 | 0.416069 |

HEAD: 3548427fbfaf4b70d82564c2ed0d479063b88fd2
tree: a43ccd5efa0c8138199a3b78b233e7876c99e93f
