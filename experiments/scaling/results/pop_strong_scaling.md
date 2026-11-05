# Strong scaling on this host

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
| dmc_population | 1 | 6 | 0.452514 | 0.450658 | 0.454504 | 0.003846 | 1.0000 | 1.0000 | n/a |
| dmc_population | 2 | 6 | 0.239127 | 0.238343 | 0.246392 | 0.008049 | 1.8924 | 0.9462 | 0.056884 |
| dmc_population | 4 | 6 | 0.135552 | 0.130838 | 0.145914 | 0.015076 | 3.3383 | 0.8346 | 0.066071 |
| dmc_population | 8 | 6 | 0.096052 | 0.095704 | 0.097727 | 0.002022 | 4.7112 | 0.5889 | 0.099728 |
| population_annealing | 1 | 6 | 0.363087 | 0.359537 | 0.364776 | 0.005239 | 1.0000 | 1.0000 | n/a |
| population_annealing | 2 | 6 | 0.196063 | 0.192725 | 0.197259 | 0.004534 | 1.8519 | 0.9259 | 0.079980 |
| population_annealing | 4 | 6 | 0.116088 | 0.113640 | 0.121567 | 0.007927 | 3.1277 | 0.7819 | 0.092966 |
| population_annealing | 8 | 6 | 0.079123 | 0.078882 | 0.080282 | 0.001400 | 4.5889 | 0.5736 | 0.106191 |
| successive_halving | 1 | 6 | 0.307095 | 0.302098 | 0.310848 | 0.008750 | 1.0000 | 1.0000 | n/a |
| successive_halving | 2 | 6 | 0.157581 | 0.156916 | 0.158167 | 0.001252 | 1.9488 | 0.9744 | 0.026266 |
| successive_halving | 4 | 6 | 0.088078 | 0.087918 | 0.088284 | 0.000367 | 3.4866 | 0.8717 | 0.049080 |
| successive_halving | 8 | 6 | 0.065081 | 0.063967 | 0.067921 | 0.003954 | 4.7186 | 0.5898 | 0.099344 |
