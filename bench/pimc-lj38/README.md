# LJ38 path-integral annealing

Anneal `456be5ff` against Fiorentino and Marzari's replica-pinned path-integral annealing, one serial session. The driver is `scripts/pimc_lj38/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. The twelve (seed, arm) pairs are shuffled with schedule seed 20261113. The budget is the LJ38 entry of `PAPER_BUDGETS`, 400_000 force calls. A hit is a quenched energy within 0.001 of the Cambridge minimum, −173.928427. A miss is stored as the full budget. Wall, evaluation, and per-evaluation medians are Hyndman–Fan type 7. The ratio is the RPQA median wall divided by the anneal median wall. Inside microseconds are potential time divided by charged evaluations. Outside microseconds are the rest of the wall on the same denominator.

The RPQA code is the public tree `https://github.com/afiorent/i-pi` at `4e86f5fd` (`feature/rpqa-reset-momenta`). The paper's data-availability note points at GitHub on publication and at a Materials Cloud DOI that is still a placeholder. The runnable Lennard-Jones protocol in that tree is `examples/features/quantum_annealing/RPQA/LJ_cluster`: 16 beads, 50 K, a 1 fs Langevin step with `tau` 20 fs, `lambdaqkin` ramped from 19 to 0.01 with `sqrtscale`, a 5000-step delocalisation, and an LBFGS relax of every bead every 5000 steps. The cluster size here is 38. The initial geometry is the start `h2h_gmin_proof coords` writes for the same seed, scaled by their σ = 5.270446 bohr, and read as an angstrom xyz. One force client serves the unix socket. Every POSDATA is one charged evaluation, including the evaluations inside those LBFGS rounds. Each seed completed 4 pinning rounds inside the 400_000 calls. The run stops when a finished relax is within 0.001 ε of the Cambridge minimum, or when the call count reaches the budget. The pair potential is the driver's `LJ_functions` plus its volume tail. ε = 0.00367493 Ha.

| Size | Arm | Hits | Median evals | Median wall (s) | Ratio | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LJ38 | anneal | 6/6 | 10189.500000 | 0.109401 | 2705.599024 | 3.5884 | 5.4623 |
| LJ38 | rpqa | 0/6 | 400000.000000 | 295.993886 | 2705.599024 | 236.8139 | 503.1707 |

Per-seed rows are `seeds.csv` and `rows/lj38/<arm>-<seed>.row`.
