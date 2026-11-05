# Strong scaling on this host

Host: Intel Xeon Processor, 8 physical cores, 8 logical CPUs, 49342768 KiB memory (47.06 GiB).
GPU: none.
Fixed problem: Lennard-Jones, 98 atoms, 32 walkers, 400 generations, quench length 40, diffusion Monte Carlo budget 20000.
These worker counts exceed the physical core count on this host.
The rows record oversubscription, and they stay out of the published table.
Six repeats sit behind each row.
Hyndman-Fan type 7 supplies the median and the quartile gap.
Speedup divides the one-worker median by the row median.
Parallel efficiency divides that speedup by the worker count.
The Karp-Flatt column is the serial fraction from those unrounded medians.

| method | workers | n | median wall s | Q1 | Q3 | quartile gap | speedup | efficiency | Karp-Flatt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dmc_population | 16 | 6 | 0.110149 | 0.090130 | 0.118235 | 0.028105 | 4.1082 | 0.2568 | 0.192977 |
| dmc_population | 32 | 6 | 0.104920 | 0.099764 | 0.113373 | 0.013608 | 4.3130 | 0.1348 | 0.207081 |
| population_annealing | 16 | 6 | 0.091568 | 0.090359 | 0.095660 | 0.005301 | 3.9652 | 0.2478 | 0.202340 |
| population_annealing | 32 | 6 | 0.094303 | 0.093808 | 0.095361 | 0.001554 | 3.8502 | 0.1203 | 0.235845 |
| successive_halving | 16 | 6 | 0.079709 | 0.075599 | 0.082356 | 0.006757 | 3.8527 | 0.2408 | 0.210195 |
| successive_halving | 32 | 6 | 0.073330 | 0.072571 | 0.076633 | 0.004062 | 4.1878 | 0.1309 | 0.214231 |
