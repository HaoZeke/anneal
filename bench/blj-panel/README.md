# A/B gate: binary Lennard-Jones patches against develop

Arm A is develop at `bc16d35cf13e71c2edaf343b80fd8c41368631d4`. Arm B is that
commit with the three patches of the binary Lennard-Jones series applied one
after the other with `git am`:

1. `feat: add a binary Lennard-Jones potential`
2. `fix: differentiate the species SOAP contraction and skip exact-zero products`
3. `fix: return a two-species plateau to a stored lower minimum`

B's tree is `e1b1087093f2be749d86fd3b3486cf27b038c4e3`. The commit hash of B
depends on the date `git am` ran; the tree does not. B's code was not changed.

The gate asks one question: does B lose to A anywhere on the hit panels the
repository already runs? B loses if any system has fewer hits with Fisher
p < 0.05, or a higher median of calls with Mann-Whitney p < 0.05. Otherwise B
passes. Every system with fewer hits is listed, significant or not.

## What runs

Twenty-six systems, seeds 2000-2011, twelve per arm, 624 runs.

| Panel | Systems | Driver |
|---|---|---|
| Binary Lennard-Jones, Doye-Meyer | blj105n38, blj105n55, blj115n38, blj115n55, blj130n38, blj130n55, blj130n98 | `h2h_blj` |
| Lennard-Jones | lj38, lj55, lj75, lj98, lj103, lj104 | `h2h_gmin_proof` |
| Morse | m38r6, m38r14, m80r6, m80r14 | `h2h_morse` |
| Sutton-Chen 12-6, 10-8, 9-6 at N=38, 55, 80 | sc126-38 to sc96-80, nine systems | `h2h_sc` |

The budgets are the paper budgets, `PAPER_BUDGETS` in `run_panel.py`: 4e5 calls
at N=38, 1e6 at N=55, and 4e6 at N=75 and above, except Sutton-Chen N=80
which is 1e6. The preset is `Config::recommended`. A hit is a quenched energy
at or below the Cambridge Cluster Database energy plus 0.001. A miss is the
budget. Each driver carries its own table of budgets and energies; the
preflight compares both arms' tables with `PANEL` in `run_panel.py`.

The binary Lennard-Jones driver, `h2h_blj.rs`, runs two species labels with no
diameters, so the search keeps the one-component preset and adds the
unlike-pair exchange. Develop at the base commit has the binary Lennard-Jones
potential, evaluated with its pair loop, so arm A runs the same panel and the
comparison is paired. In B the same potential dispatches to the pinned pair
kernel. The driver builds the potential with `BinaryLennardJones::new` and the
Doye-Meyer parameters (every epsilon 1, sigma_AA 1, sigma_AB the mean of
sigma_AA and sigma_BB), which is what `BinaryLennardJones::doye_meyer`
expands to on B. `build_arms.sh` copies this one source into both arms. The
Lennard-Jones, Morse and Sutton-Chen drivers are the ones each arm's own tree
carries; they are identical in A and B.

## Protocol

* Serial. One run at a time on core 0, `taskset -c 0`, inside
  `sudo unshare --uts` with hostname `bench`, one thread
  (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
  `RAYON_NUM_THREADS` are 1). Every other process is moved to the remaining
  cores before the first run.
* One fresh process per run. No warm-up run, no rescue rerun, no change of
  budget or setting after a result.
* For each seed the two arms run back to back. The order flips with the seed:
  A first on even seeds, B first on odd seeds.
* Rows are appended to `rows.csv` after every run and never rewritten. Each
  carries `code_revision`, `code_tree`, `host`, `cpu` and the session. A run
  already stored for the same system, arm, seed and revision is kept.
* A search that exits non-zero is stored as a miss at its budget with status
  `crash` and is not run again; three crashes in a row stop the session. A run
  whose output cannot be read, or that reports another system, seed, budget or
  revision, stops the session and stores nothing.
* The two arms are built in separate worktrees with one toolchain, one set of
  flags and one lockfile. `builds.txt` records the revisions, trees, compiler,
  binary hashes, the Sutton-Chen library and the harness.
* The Sutton-Chen driver loads `librgpot` built from the revision of the
  pair-kernel dependency in `Cargo.toml`. Both arms load the same file.
* The small preflight run by `run_panel.py --preflight` runs one seed of each
  driver at a budget of 2000 on each arm and stores nothing. It checks that
  both arms start, that the library loads and that every budget and energy
  matches the table.
* Scratch space comes from `PANEL_WORK`, `.bench/blj-panel` in the repository
  by default and git-ignored. No absolute path is built into any script.

## Statistics

Hits per system and arm, with Fisher's exact test, two-sided: the probability
of every table with the observed margins that is no more likely than the
observed one. Calls to the minimum, a miss entered at its budget, with the
median and the Mann-Whitney test, two-sided, the normal approximation with the
tie and continuity corrections; every value tied gives p = 1. Medians are
Hyndman-Fan type 7. Both tests agree with SciPy's `fisher_exact` and
`mannwhitneyu(method="asymptotic")` to machine precision on random inputs with
heavy ties.

Time inside the potential is `pot_s`, the time spent in calls to the potential
as the driver measures it. Time outside is the driver's search wall time minus
`pot_s`. Both are medians over the twelve seeds of each arm, and a crashed
run is left out of them.

Two arms are bit-identical on a system when, for every seed, they report the
same hit, calls, charged calls and best energy. The binary Lennard-Jones
driver also reports the IEEE-754 bits of the best energy; the other drivers
report it to eight decimals.

## Files

| File | Purpose |
|---|---|
| `run_panel.py` | The runner and the preflight. `PAPER_BUDGETS`, `PANEL`, `SEEDS`. |
| `summarize_panel.py` | Tables, statistics and the verdict, from `rows.csv`. |
| `enter_bench.sh` | Builds both arms and runs the panel in the namespace. |
| `build_arms.sh` | Worktree, driver injection and release build of each arm. |
| `build_rgpot.sh` | Builds `librgpot` for the Sutton-Chen driver. |
| `h2h_blj.rs` | The binary Lennard-Jones driver, one source for both arms. |
| `rows.csv` | One row per run, append-only. |
| `summary.csv` | One row per system, from the rows. |
| `pairs.csv` | One row per system and seed, A beside B. |
| `builds.txt` | Provenance of the binaries and the harness. |
| `benchmarks/blj_panel.py` | asv suite; `PAPER_BUDGETS` as in the runner. |

The Snakemake rule is `blj_panel`, and `bench/h2h.json` lists it.

## Reproduce

```sh
PANEL_B_REV=<commit with the three patches> snakemake -s workflow/Snakefile blj_panel
```

or `PANEL_B_REV=<commit> bash bench/blj-panel/enter_bench.sh`. `PANEL_A_REV`
defaults to the base commit above; `PANEL_B_TREE` makes the script check B's
tree. The build needs the pinned Rust toolchain, `meson`, `ninja` and a C++20
compiler. `bash bench/blj-panel/enter_bench.sh --summarize` rewrites
`summary.csv`, `pairs.csv` and the tables below from `rows.csv`.

## Results

<!-- results:begin -->
| System | Budget | A hits | B hits | Fisher p | A calls median | B calls median | Mann-Whitney p | Identical seeds |
|---|---|---|---|---|---|---|---|---|
| blj105n38 | 400,000 | 1/12 | 2/12 | 1 | 400000.0 | 400000.0 | 0.514 | 0/12 |
| blj105n55 | 1,000,000 | 12/12 | 12/12 | 1 | 204047.0 | 158055.0 | 0.341 | 0/12 |
| blj115n38 | 400,000 | 9/12 | 12/12 | 0.217 | 138863.5 | 82300.5 | 0.470 | 2/12 |
| blj115n55 | 1,000,000 | 0/12 | 0/12 | 1 | 1000000.0 | 1000000.0 | 1 | 0/12 |
| blj130n38 | 400,000 | 4/12 | 11/12 | 9.42e-03 | 400000.0 | 134072.5 | 0.080 | 0/12 |
| blj130n55 | 1,000,000 | 4/12 | 9/12 | 0.100 | 1000000.0 | 633042.0 | 0.347 | 0/12 |
| blj130n98 | 4,000,000 | 0/12 | 0/12 | 1 | 4000000.0 | 4000000.0 | 1 | 0/12 |
| lj38 | 400,000 | 12/12 | 12/12 | 1 | 127839.5 | 127839.5 | 1 | 12/12 |
| lj55 | 1,000,000 | 12/12 | 12/12 | 1 | 5722.0 | 5722.0 | 1 | 12/12 |
| lj75 | 4,000,000 | 5/12 | 5/12 | 1 | 4000000.0 | 4000000.0 | 1 | 12/12 |
| lj98 | 4,000,000 | 7/12 | 7/12 | 1 | 3166811.5 | 3166811.5 | 1 | 12/12 |
| lj103 | 4,000,000 | 6/12 | 6/12 | 1 | 3905072.5 | 3905072.5 | 1 | 12/12 |
| lj104 | 4,000,000 | 5/12 | 5/12 | 1 | 4000000.0 | 4000000.0 | 1 | 12/12 |
| m38r6 | 400,000 | 12/12 | 12/12 | 1 | 49438.5 | 49438.5 | 1 | 12/12 |
| m38r14 | 400,000 | 4/12 | 4/12 | 1 | 400000.0 | 400000.0 | 1 | 12/12 |
| m80r6 | 4,000,000 | 12/12 | 12/12 | 1 | 62677.0 | 62677.0 | 1 | 12/12 |
| m80r14 | 4,000,000 | 6/12 | 6/12 | 1 | 3901429.0 | 3901429.0 | 1 | 12/12 |
| sc126-38 | 400,000 | 12/12 | 12/12 | 1 | 113504.5 | 113504.5 | 1 | 12/12 |
| sc126-55 | 1,000,000 | 12/12 | 12/12 | 1 | 18379.5 | 18379.5 | 1 | 12/12 |
| sc126-80 | 1,000,000 | 5/12 | 5/12 | 1 | 1000000.0 | 1000000.0 | 1 | 12/12 |
| sc108-38 | 400,000 | 7/12 | 7/12 | 1 | 361031.5 | 361031.5 | 1 | 12/12 |
| sc108-55 | 1,000,000 | 5/12 | 5/12 | 1 | 1000000.0 | 1000000.0 | 1 | 12/12 |
| sc108-80 | 1,000,000 | 3/12 | 3/12 | 1 | 1000000.0 | 1000000.0 | 1 | 12/12 |
| sc96-38 | 400,000 | 12/12 | 12/12 | 1 | 50230.0 | 50230.0 | 1 | 12/12 |
| sc96-55 | 1,000,000 | 12/12 | 12/12 | 1 | 23157.0 | 23157.0 | 1 | 12/12 |
| sc96-80 | 1,000,000 | 7/12 | 7/12 | 1 | 989847.5 | 989847.5 | 1 | 12/12 |

| System | A in potential (s) | B in potential (s) | A outside (s) | B outside (s) | A wall (s) | B wall (s) | B/A wall |
|---|---|---|---|---|---|---|---|
| blj105n38 | 1.437 | 1.703 | 31.070 | 4.179 | 32.518 | 5.975 | 0.184 |
| blj105n55 | 1.588 | 1.460 | 12.611 | 0.685 | 14.543 | 2.291 | 0.158 |
| blj115n38 | 0.548 | 0.394 | 1.760 | 0.327 | 2.350 | 0.873 | 0.372 |
| blj115n55 | 7.481 | 8.851 | 258.773 | 25.919 | 266.235 | 34.775 | 0.131 |
| blj130n38 | 1.391 | 0.599 | 18.506 | 0.724 | 19.898 | 1.198 | 0.060 |
| blj130n55 | 7.428 | 5.589 | 112.527 | 9.466 | 120.115 | 15.353 | 0.128 |
| blj130n98 | 90.574 | 108.376 | 1463.161 | 58.895 | 1553.718 | 168.315 | 0.108 |
| lj38 | 0.299 | 0.285 | 0.340 | 0.324 | 0.639 | 0.609 | 0.952 |
| lj55 | 0.026 | 0.026 | 0.042 | 0.043 | 0.064 | 0.065 | 1.011 |
| lj75 | 34.135 | 34.004 | 24.800 | 24.985 | 58.822 | 59.189 | 1.006 |
| lj98 | 46.675 | 47.500 | 30.885 | 30.885 | 77.560 | 79.036 | 1.019 |
| lj103 | 61.862 | 62.140 | 36.846 | 37.642 | 99.341 | 100.818 | 1.015 |
| lj104 | 64.474 | 64.848 | 39.039 | 39.249 | 104.458 | 104.863 | 1.004 |
| m38r6 | 0.314 | 0.309 | 0.178 | 0.172 | 0.492 | 0.482 | 0.979 |
| m38r14 | 2.421 | 2.418 | 1.012 | 1.009 | 3.433 | 3.440 | 1.002 |
| m80r6 | 1.663 | 1.662 | 0.957 | 0.960 | 2.619 | 2.622 | 1.001 |
| m80r14 | 102.737 | 103.583 | 35.449 | 35.909 | 138.879 | 140.064 | 1.009 |
| sc126-38 | 4.193 | 4.219 | 0.258 | 0.264 | 4.429 | 4.462 | 1.007 |
| sc126-55 | 1.244 | 1.328 | 0.119 | 0.119 | 1.355 | 1.443 | 1.065 |
| sc126-80 | 125.886 | 125.864 | 11.162 | 11.427 | 137.077 | 137.790 | 1.005 |
| sc108-38 | 13.633 | 13.554 | 1.030 | 1.028 | 14.665 | 14.582 | 0.994 |
| sc108-55 | 65.788 | 65.948 | 4.952 | 5.040 | 71.296 | 71.525 | 1.003 |
| sc108-80 | 130.333 | 131.586 | 10.936 | 11.183 | 141.285 | 143.292 | 1.014 |
| sc96-38 | 1.864 | 1.864 | 0.119 | 0.120 | 1.983 | 1.984 | 1.000 |
| sc96-55 | 1.521 | 1.520 | 0.111 | 0.112 | 1.641 | 1.641 | 1.000 |
| sc96-80 | 126.842 | 126.551 | 10.000 | 10.035 | 136.706 | 136.860 | 1.001 |

Systems with any hit drop, significant or not: none.

`VERDICT: PASS`
<!-- results:end -->

## Reading the results

Per-seed results are bit-identical between A and B on every system that does
not use the binary Lennard-Jones potential: for each of the nineteen
Lennard-Jones, Morse and Sutton-Chen systems all twelve seeds agree in hit,
calls, charged calls and best energy. The patches concern the binary
Lennard-Jones potential, the species SOAP contraction and the plateau return
of the two-species search, so those systems check that nothing else moved. On
the seven binary Lennard-Jones systems the arms follow different
trajectories, which is what the patches are meant to do.

B has no system with fewer hits than A, so no system is listed as a hit drop.
On the binary systems B has more hits on four of the seven (blj105n38,
blj115n38, blj130n38, blj130n55), and the one difference with Fisher
p below 0.05 is blj130n38, 4 hits for A against 11 for B. No Mann-Whitney
p on calls is below 0.05. Neither arm hits blj115n55 or blj130n98 within the
budget.

## Provenance

`builds.txt` has the full record. In short:

* Arm A: develop at `bc16d35cf13e71c2edaf343b80fd8c41368631d4`, tree
  `5f746b218dee3774616745b2a84cced311ec4f10`.
* Arm B: A plus the three patches, commit
  `3192eaa63e2fe2619a482a6d1e19219d1c7db0a8`, tree
  `e1b1087093f2be749d86fd3b3486cf27b038c4e3`.
* Rust 1.88.0 (`rustc 1.88.0 (6b00bc388 2025-06-23)`), release profile,
  `--locked`, one target directory per arm.
* `librgpot` from `https://github.com/OmniPotentRPC/rgpot` at revision
  `8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e`, Meson release build with g++
  13.3.0, sha256
  `79489096fe0b7a243b2acd79b33a6a7ae7bb0e32516e21c2b414cfba134f4a05`.
* The harness at revision `565fde1ae7aab8b73b68739eb3b5106c8d2d9b23` with no
  tracked change; `builds.txt` has the hash of the runner, the driver and the
  scripts.
* One session, `20261118T140141Z`, 624 runs, each run once.

## What happened during the session

* Before the session, `run_panel.py --preflight` and one trial of the chain
  (sc96-38, seed 2000, both arms, output to a scratch directory) checked that
  both arms start and that the library loads. Neither is in `rows.csv`; the
  same pair is run again in the gate like every other seed.
* The machine was suspended and resumed by its host twice while a run was in
  progress: during run 76 (blj115n55, arm A, seed 2001) for about ten minutes
  and during run 81 (blj115n55, arm A, seed 2004) for about thirty-four. The
  `utc` stamps in `rows.csv` show these two gaps, 597 s and 2068 s longer than
  the wall time of the run, and no other. The drivers and the runner time with
  monotonic clocks, which do not count a suspension. The two wall times, 183.4
  s and 351.1 s, are inside the 48 s to 536 s that arm A takes on that system.
  The runner's own timing of the process agrees with them (183.44 s and 351.14
  s). The time inside the potential, 7.54 s and 7.40 s for 1,000,000 calls, is
  within 2% of the arm's median of 7.48 s. Neither run was repeated.
* No search crashed, no output was rejected, and no run was repeated.
