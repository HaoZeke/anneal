# GMIN panel record

Both arms of Au55, M55 and M80, on one core, from develop
`9837358ae1027067ff654fac8b48969020853452`, tree
`15032ef401fb681e230d27c7b315d89f7ac53e4e`. The harness that holds
`enter.sh`, `apply_patches.py` and `run_record.py` is commit
`e4341de058b73366f2d7b5f17995da283d57bad6`, tree
`63d1488f465fff1e65ce9b338641ef945fad75a9`. The record is the ref
`refs/audit/record/gmin-184`.
Its parent is `b8190ce61a64ff6cb61ee6bfc656b0657d78494d`. The develop
commit above is an ancestor of that record commit, and it is the
`harness_revision` the panel rows keep. Rust 1.88.0,
`cargo build --release --locked`, examples `h2h_metal` and
`h2h_morse`. One thread (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `RAYON_NUM_THREADS` and the other
`*_NUM_THREADS` values are 1). `taskset -c 0`. Serial.

Seeds 2000-2011. Even seeds run anneal then GMIN. Odd seeds run
GMIN then anneal, so the twelve seeds are ABBA repeated. Paper
budgets: 1000000 calls for Au55, M55 rho 6 and M55 rho 14;
4000000 calls for M80 rho 6 and M80 rho 14. A hit is a quenched
energy at or below the reference plus 0.001. A miss stops at the
budget. References: Au55 -191.03085280, M55 rho 6 -250.286609,
M55 rho 14 -220.646208, M80 rho 6 -378.333471, M80 rho 14
-340.811371.

GMIN is one build of the source tarball dated
`2026-11-21T03:01:22Z`. The file sha256 is

```
0d9917600ecda72b2176253d0e838bcb58ec6e38902ef7d0affc7e7e7eba8f39
```

The build is gfortran with ALIGN, MPI, OpenMP, sparse, Python,
internal BLAS and internal LAPACK off. The driver applies the
repository measurement patches that cap the evaluator and time
the Morse and Gupta potentials. It does not read a source
revision. The replica-index hunk in the cap patch does not match
this source, which already loops that index, so that hunk is
omitted.

Each system has its own data file in `inputs/`. Au55 is
`BGUPTAT 55` with the Darby gold row, container radius
11.829521, step 1.027741 and avoid 10.277408. Each Morse file
sets `MORSE` to that system's rho, 6.0 or 14.0, and states its
atom count. GMIN counts atoms from the coords file. The driver
writes those coords with that system's anneal binary and refuses
a line count other than 55 or 80 for that system. The numerical
block matches the panel (sloppy and tight convergence, step,
avoid, temperature 0.1 for gold and 1.0 for Morse). The driver
appends only `RANSEED`.

Rows are appended to `rows.csv` and never rewritten. A stored
seed is skipped, so a stopped run resumes from the rows already
there. Each row has the arm, system, rho, seed, budget, call
count, calls to the hit, best energy, hit, wall time, stop
reason (`hit` or `budget`), the anneal commit and tree, the
tarball date and file hash, and the hashed machine id. A row is
kept only when the call count is positive and the stop reason
is set.

```
bash record/gmin-184/enter.sh --tarball TARBALL
```

`TARBALL` is the file named by the sha256 above.
