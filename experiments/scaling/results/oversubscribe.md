# Strong scaling on Intel Xeon Processor

Host: Intel Xeon Processor, 8 physical cores, 8 logical CPUs, 49342768 KiB memory (47.06 GiB).
GPU: none.
Fixed problem: Lennard-Jones, 98 atoms, 32 walkers, 400 generations, quench length 40, diffusion Monte Carlo budget 20000.
These worker counts exceed the physical core count on this host.
They are an oversubscription note.
They are not a published strong-scaling row.
Six repeats.
The quartile is Hyndman-Fan type 7.
Speedup and parallel efficiency use the median wall time against one worker.
This is a scaling table.

| method | workers | n | median wall s | Q1 | Q3 | IQR | speedup | efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dmc_population | 16 | 6 | 0.098596 | 0.097581 | 0.099527 | 0.001946 | 4.5788 | 0.2862 |
| dmc_population | 32 | 6 | 0.111565 | 0.108265 | 0.111716 | 0.003450 | 4.0465 | 0.1265 |
| population_annealing | 16 | 6 | 0.091568 | 0.090359 | 0.095660 | 0.005301 | 3.9652 | 0.2478 |
| population_annealing | 32 | 6 | 0.094303 | 0.093808 | 0.095361 | 0.001554 | 3.8502 | 0.1203 |
| successive_halving | 16 | 6 | 0.079709 | 0.075599 | 0.082356 | 0.006757 | 3.8527 | 0.2408 |
| successive_halving | 32 | 6 | 0.073330 | 0.072571 | 0.076633 | 0.004062 | 4.1878 | 0.1309 |
