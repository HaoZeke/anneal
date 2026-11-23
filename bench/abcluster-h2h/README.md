Copyright (c) anneal developers

anneal against ABCluster 3.4 (artificial bee colony), LJ38, LJ55, LJ75,
LJ98, LJ104 and Gupta Au55, seeds 2000 to 2011, host bench.

## What ran

ABCluster is the official 3.4 Linux release from
http://www.zhjun-sci.com/abcluster.html, the executable download. It needs
no registration and no licence step. The tarball and `atom` checksums are in
`abcluster-source.txt`, and the runner refuses any other `atom`. The source
tarball needs a registration form and was not used. `atom` is a closed
binary, so nothing here rebuilds it.

ABCluster runs its own `atom` program with its built-in `LennardJones` and
`Gupta` force fields, unchanged. `scripts/abcluster_h2h/abc_shim.c` is an
`LD_PRELOAD` library that patches the entry of the force-field kernel and of
the local optimiser in memory, never on disk. It counts every kernel call,
reads `CLOCK_MONOTONIC` on entry to and return from the kernel, seeds the
program (ABCluster takes no seed) and stops the run at the first quench at
or below the reference plus 0.001, or at the budget.

ABCluster reports no call count. The count here is the number of kernel
calls, one energy and gradient per call, and is checked against a `gdb`
breakpoint count on the same run (`counter.csv`).

Anneal is `examples/h2h_abcluster.rs` with `Config::recommended` and with
`Config::for_cluster`, the same ledger and the same `Instant::now()` reads
around the potential as `examples/h2h_gmin_proof.rs`.

## Setup

- References: Cambridge Cluster Database, LJ38 -173.928427, LJ55
  -279.248470, LJ75 -397.492331, LJ98 -543.665361, LJ104 -582.086642.
  Au55 is -191.03085280, the reference `examples/h2h_metal.rs` uses for the
  same Gupta set. A hit is a quenched energy within 0.001 of the reference.
- Budgets: energy-and-gradient calls, the `PAPER_BUDGETS` of the Lennard-Jones
  paper map (400,000 at 38 atoms, 1,000,000 at 55, 4,000,000 at 75, 98 and
  104). Au55 has 1,000,000, as in `bench/metal-cluster-h2h`.
- ABCluster Lennard-Jones: sigma 1.4, epsilon 1, the manual's LJ38 example.
  ABCluster Gupta: the gold parameters of `Gupta::copper_gold`. Before any
  timed run, `atom` and anneal evaluate the same eight-decimal geometries
  (`parity.csv`, three per system, `atom` at sigma 1 for the Lennard-Jones
  systems; the energy is unchanged when the geometry and sigma are scaled
  together, and the search runs at sigma 1.4). The largest relative
  difference is below.
- ABCluster search settings: scout limit 5, amplitude L from 5.0 at 38 atoms
  grown with N^(1/3), population SN chosen per system from 30 and 100 on pilot
  seeds 1000 to 1002, which the campaign never uses (`pilot.csv`). The
  settings of every run are in the `setting` column of `seeds.csv`.
- Twelve seeds, 2000 to 2011, the same list on every arm. One VM, `unshare
  --uts` with hostname bench, one process at a time pinned to core 0, one
  thread. The (seed, arm) pairs inside a system are shuffled with a fixed
  seed.
- Rows are appended, never rewritten. Each carries `code_revision` and
  `host`. A miss is stored at the full budget for calls and as the whole run
  for seconds, so the medians and the rank tests use every seed. Medians are
  Hyndman-Fan type 7.
- Fisher is two-sided exact on the hit counts of each anneal arm against
  ABCluster. MWU is the Mann-Whitney U test, normal approximation with tie
  and continuity correction, on calls to first hit and on seconds to first
  hit.

## Results

<!-- BEGIN-SUMMARY -->
| System | Budget | Arm | Hits | Median calls to first hit | Median calls, hits only | Median s to first hit | Median s in potential | Median s outside | Fisher p | MWU p calls | MWU p time |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lj38 | 400,000 | ABCluster | 3/12 | 400,000 | 138,782 | 4.747 | 2.443 | 2.321 |  |  |  |
| lj38 | 400,000 | anneal recommended | 12/12 | 10,616 | 10,616 | 0.085 | 0.033 | 0.054 | 0.00034 | 2.2e-05 | 3.7e-05 |
| lj38 | 400,000 | anneal for_cluster | 10/12 | 268,312 | 216,577 | 1.263 | 0.746 | 0.517 | 0.012 | 0.023 | 0.00031 |
| lj55 | 1,000,000 | ABCluster | 12/12 | 7,814 | 7,814 | 0.178 | 0.109 | 0.069 |  |  |  |
| lj55 | 1,000,000 | anneal recommended | 12/12 | 12,242 | 12,242 | 0.157 | 0.066 | 0.092 | 1.000 | 0.403 | 1.000 |
| lj55 | 1,000,000 | anneal for_cluster | 12/12 | 74,779 | 74,779 | 0.626 | 0.447 | 0.179 | 1.000 | 0.00059 | 0.006 |
| lj75 | 4,000,000 | ABCluster | 0/12 | 4,000,000 | - | 149.6 | 98.319 | 51.315 |  |  |  |
| lj75 | 4,000,000 | anneal recommended | 12/12 | 1,409,230 | 1,409,230 | 27.340 | 13.651 | 13.592 | 7.4e-07 | 1e-05 | 3.7e-05 |
| lj75 | 4,000,000 | anneal for_cluster | 0/12 | 4,000,000 | - | 63.448 | 41.129 | 22.280 | 1.000 | 1.000 | 3.7e-05 |
| lj98 | 4,000,000 | ABCluster | 0/12 | 4,000,000 | - | 267.4 | 195.9 | 71.551 |  |  |  |
| lj98 | 4,000,000 | anneal recommended | 10/12 | 1,511,030 | 1,360,200 | 57.651 | 32.501 | 25.149 | 6.7e-05 | 0.00013 | 3.7e-05 |
| lj98 | 4,000,000 | anneal for_cluster | 7/12 | 2,827,143 | 1,614,641 | 85.048 | 58.493 | 26.556 | 0.005 | 0.003 | 3.7e-05 |
| lj104 | 4,000,000 | ABCluster | 0/12 | 4,000,000 | - | 245.5 | 175.8 | 69.882 |  |  |  |
| lj104 | 4,000,000 | anneal recommended | 9/12 | 2,328,723 | 794,846 | 81.709 | 46.054 | 35.248 | 0.00034 | 0.00038 | 3.7e-05 |
| lj104 | 4,000,000 | anneal for_cluster | 0/12 | 4,000,000 | - | 112.8 | 79.019 | 33.546 | 1.000 | 1.000 | 3.7e-05 |
| au55 | 1,000,000 | ABCluster | 0/12 | 1,000,000 | - | 70.431 | 62.270 | 8.162 |  |  |  |
| au55 | 1,000,000 | anneal recommended | 6/12 | 852,736 | 424,690 | 33.900 | 25.076 | 8.824 | 0.014 | 0.007 | 3.7e-05 |
| au55 | 1,000,000 | anneal for_cluster | 0/12 | 1,000,000 | - | 36.706 | 31.212 | 5.383 | 1.000 | 1.000 | 3.7e-05 |

Kernel parity: 18 geometries across 6 systems, largest relative energy difference between `atom` and anneal 2.905e-15 (limit 1e-08). Gold Gupta parameters: A, xi, d, p, q = 0.2061 1.79 2.884 10.229 4.036.
Call counter: shim count equals a debugger breakpoint count on 2 of 2 checked runs (au55, lj38).
ABCluster population size, chosen on pilot seeds 1000 to 1002: lj38 SN=30, lj55 SN=100, lj75 SN=30, lj98 SN=30, lj104 SN=30, au55 SN=30.

ABCluster revision: abcluster-3.4@80d5ceed91bf.
anneal revision: 1727ef4192add5827b56acb8b5fa011a6ef5ce51.
CPU: Intel(R) Xeon(R) Processor.
<!-- END-SUMMARY -->

`scripts/abcluster_h2h/summarize.py` writes `summary.csv` and the table
above from the rows. Run with `snakemake -s workflow/Snakefile
abcluster_h2h`.
