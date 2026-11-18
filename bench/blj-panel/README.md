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
No complete system yet.
<!-- results:end -->
