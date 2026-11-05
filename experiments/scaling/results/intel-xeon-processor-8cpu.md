# Strong scaling on Intel Xeon Processor

Host: Intel Xeon Processor, 8 physical cores, 8 logical CPUs, 49342768 KiB memory (47.06 GiB).
GPU: none.
Fixed problem: Lennard-Jones, 98 atoms, 32 walkers, 400 generations, quench length 40, diffusion Monte Carlo budget 20000.
The worker count is the thread pool.
Published rows use at most 8 workers, the physical core count on this host.
Six repeats.
The quartile is Hyndman-Fan type 7.
Speedup and parallel efficiency use the median wall time against one worker.
This is a scaling table.

| method | workers | n | median wall s | Q1 | Q3 | IQR | speedup | efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dmc_population | 1 | 6 | 0.451447 | 0.450374 | 0.453837 | 0.003463 | 1.0000 | 1.0000 |
| dmc_population | 2 | 6 | 0.255687 | 0.255004 | 0.256573 | 0.001569 | 1.7656 | 0.8828 |
| dmc_population | 4 | 6 | 0.166761 | 0.160953 | 0.172763 | 0.011810 | 2.7071 | 0.6768 |
| dmc_population | 8 | 6 | 0.095554 | 0.095115 | 0.096994 | 0.001879 | 4.7245 | 0.5906 |
| population_annealing | 1 | 6 | 0.363087 | 0.359537 | 0.364776 | 0.005239 | 1.0000 | 1.0000 |
| population_annealing | 2 | 6 | 0.196063 | 0.192725 | 0.197259 | 0.004534 | 1.8519 | 0.9259 |
| population_annealing | 4 | 6 | 0.116088 | 0.113640 | 0.121567 | 0.007927 | 3.1277 | 0.7819 |
| population_annealing | 8 | 6 | 0.079123 | 0.078882 | 0.080282 | 0.001400 | 4.5889 | 0.5736 |
| successive_halving | 1 | 6 | 0.307095 | 0.302098 | 0.310848 | 0.008750 | 1.0000 | 1.0000 |
| successive_halving | 2 | 6 | 0.157581 | 0.156916 | 0.158167 | 0.001252 | 1.9488 | 0.9744 |
| successive_halving | 4 | 6 | 0.088078 | 0.087918 | 0.088284 | 0.000367 | 3.4866 | 0.8717 |
| successive_halving | 8 | 6 | 0.065081 | 0.063967 | 0.067921 | 0.003954 | 4.7186 | 0.5898 |
