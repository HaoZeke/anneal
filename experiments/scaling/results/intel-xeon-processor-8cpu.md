# Strong scaling on Intel Xeon Processor

Host: Intel Xeon Processor, 8 physical cores, 8 logical CPUs, 49342768 KiB memory (47.06 GiB).
GPU: none.
Fixed problem: Lennard-Jones, 98 atoms, 32 walkers, 400 generations, quench length 40, diffusion Monte Carlo budget 20000.
The worker count is the thread pool.
Published rows use at most 8 workers, the physical core count on this host.
Six repeats sit behind each row.
Hyndman-Fan type 7 supplies the median and the quartile gap.
Speedup divides the one-worker median by the row median.
Parallel efficiency divides that speedup by the worker count.
The Karp-Flatt column is the serial fraction from those unrounded medians.

| method | workers | n | median wall s | Q1 | Q3 | quartile gap | speedup | efficiency | Karp-Flatt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dmc_population | 1 | 6 | 0.452942 | 0.452219 | 0.454219 | 0.002000 | 1.0000 | 1.0000 | n/a |
| dmc_population | 2 | 6 | 0.256359 | 0.255201 | 0.257433 | 0.002232 | 1.7668 | 0.8834 | 0.131973 |
| dmc_population | 4 | 6 | 0.157801 | 0.156835 | 0.159371 | 0.002536 | 2.8703 | 0.7176 | 0.131188 |
| dmc_population | 8 | 6 | 0.097259 | 0.096059 | 0.118561 | 0.022503 | 4.6570 | 0.5821 | 0.102547 |
| population_annealing | 1 | 6 | 0.363087 | 0.359537 | 0.364776 | 0.005239 | 1.0000 | 1.0000 | n/a |
| population_annealing | 2 | 6 | 0.196063 | 0.192725 | 0.197259 | 0.004534 | 1.8519 | 0.9259 | 0.079980 |
| population_annealing | 4 | 6 | 0.116088 | 0.113640 | 0.121567 | 0.007927 | 3.1277 | 0.7819 | 0.092966 |
| population_annealing | 8 | 6 | 0.079123 | 0.078882 | 0.080282 | 0.001400 | 4.5889 | 0.5736 | 0.106191 |
| successive_halving | 1 | 6 | 0.307095 | 0.302098 | 0.310848 | 0.008750 | 1.0000 | 1.0000 | n/a |
| successive_halving | 2 | 6 | 0.157581 | 0.156916 | 0.158167 | 0.001252 | 1.9488 | 0.9744 | 0.026266 |
| successive_halving | 4 | 6 | 0.088078 | 0.087918 | 0.088284 | 0.000367 | 3.4866 | 0.8717 | 0.049080 |
| successive_halving | 8 | 6 | 0.065081 | 0.063967 | 0.067921 | 0.003954 | 4.7186 | 0.5898 | 0.099344 |
