# Strong scaling on Intel Xeon Processor

Host: Intel Xeon Processor, 8 logical CPUs, 49342768 KiB memory (47.06 GiB). GPU: none.
Fixed problem: Lennard-Jones, 98 atoms, 32 walkers, 400 generations, quench length 40, diffusion Monte Carlo budget 20000.
The worker count is the thread pool. Six repeats. The quartile is Hyndman-Fan type 7. Speedup and parallel efficiency use the median wall time against one worker.
This is a scaling table.

| method | workers | n | median wall s | Q1 | Q3 | IQR | speedup | efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dmc_population | 1 | 6 | 0.457903 | 0.457167 | 0.459006 | 0.001838 | 1.0000 | 1.0000 |
| dmc_population | 2 | 6 | 0.450438 | 0.449433 | 0.451631 | 0.002198 | 1.0166 | 0.5083 |
| dmc_population | 4 | 6 | 0.445834 | 0.445698 | 0.446064 | 0.000366 | 1.0271 | 0.2568 |
| dmc_population | 8 | 6 | 0.442662 | 0.441815 | 0.443558 | 0.001744 | 1.0344 | 0.1293 |
| dmc_population | 16 | 6 | 0.442234 | 0.442028 | 0.443223 | 0.001195 | 1.0354 | 0.0647 |
| dmc_population | 32 | 6 | 0.443777 | 0.442815 | 0.445127 | 0.002311 | 1.0318 | 0.0322 |
| population_annealing | 1 | 6 | 0.360252 | 0.359565 | 0.362555 | 0.002990 | 1.0000 | 1.0000 |
| population_annealing | 2 | 6 | 0.191487 | 0.190856 | 0.192287 | 0.001430 | 1.8813 | 0.9407 |
| population_annealing | 4 | 6 | 0.108857 | 0.108684 | 0.108929 | 0.000245 | 3.3094 | 0.8273 |
| population_annealing | 8 | 6 | 0.070040 | 0.069669 | 0.070466 | 0.000798 | 5.1435 | 0.6429 |
| population_annealing | 16 | 6 | 0.087112 | 0.086519 | 0.088578 | 0.002059 | 4.1355 | 0.2585 |
| population_annealing | 32 | 6 | 0.087798 | 0.087059 | 0.088842 | 0.001783 | 4.1032 | 0.1282 |
| successive_halving | 1 | 6 | 0.302386 | 0.299995 | 0.304696 | 0.004701 | 1.0000 | 1.0000 |
| successive_halving | 2 | 6 | 0.153453 | 0.153189 | 0.155224 | 0.002035 | 1.9705 | 0.9853 |
| successive_halving | 4 | 6 | 0.087354 | 0.086744 | 0.087966 | 0.001222 | 3.4616 | 0.8654 |
| successive_halving | 8 | 6 | 0.054339 | 0.053419 | 0.056634 | 0.003214 | 5.5648 | 0.6956 |
| successive_halving | 16 | 6 | 0.066502 | 0.065615 | 0.067096 | 0.001481 | 4.5470 | 0.2842 |
| successive_halving | 32 | 6 | 0.063735 | 0.063614 | 0.064637 | 0.001023 | 4.7444 | 0.1483 |
