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
  timed run, `atom` and anneal evaluate the same eight-decimal geometries;
  the largest relative difference is in `parity.csv` and below.
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
<!-- END-SUMMARY -->

`scripts/abcluster_h2h/summarize.py` writes `summary.csv` and the table
above from the rows. Run with `snakemake -s workflow/Snakefile
abcluster_h2h`.
