# Binary Lennard-Jones outside-time gate

Question, declared before the measured session: an earlier comparison
reported an outside-time per charged call of 1.418 on `blj115n38` for
the shelf kit against develop, with the same hits and no second build
of develop. This gate asks whether that cost is real.

Three arms, one session, one runner, serial:

- `A` is develop `61fa67cf380e46876b569da45e7cc091647a775b`, library
  tree `cff796c507a8fa1d0bfb28888c3570b61988055e`.
- `A'` is a second clean release build of that same source, in its own
  worktree and its own target directory.
- `B` is `001e004eb8a8de51dd767ebe51ffc1988f4fbded`, library tree
  `809f59b4670937fb8e048739cba6c74b28ff5800`.

Develop does not contain the driver. Every arm is built from the same
driver text, `examples/h2h_blj.rs` at `d7fb268709d912682fc5dfd6422db8a211792daa`,
copied in for the release build and removed afterwards. The build is
`cargo build --release --locked --example h2h_blj` with rustc 1.88.0.
Only the library differs. Each row records that arm's library tree and
the sha256 of its binary.

Cells, references, and budgets are the driver's Cambridge header and
paper-list budget. A hit is a finite printed best at or below the
reference plus 0.001.

| Cell | Atoms | Budget | Reference |
| --- | --- | --- | --- |
| blj115n38 | 38 | 400000 | -187.3676867453 |
| blj115n55 | 55 | 1000000 | -289.2931853058 |
| blj105n55 | 55 | 1000000 | -283.1545313810 |

Seeds are 2000 through 2023, twenty-four per arm per cell. No other
seeds are added. The measured command does not set a budget override,
does not pass a quick flag, and does not drop a prefix.

Order. Cells run `blj115n38`, then `blj115n55`, then `blj105n55`.
Inside a cell the seeds run from 2000 upward. For seed `s` let
`k = s - 2000`. The forward cycle is `A A' B`, then `A' B A`, then
`B A A'`. Even `k` uses the forward triple; odd `k` reverses it.
Over each cell every arm occupies each of the three slots eight times.
The three arms of one cell and seed finish before the next triple
starts. Nothing measured runs beside them.

Each measured process is started with `taskset -c 0`. A sampled
affinity other than 0 is stored as an error miss. Thread counts
are set to 1 for `OMP_NUM_THREADS`, `OMP_THREAD_LIMIT`,
`MKL_NUM_THREADS`, `MKL_DOMAIN_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`GOTO_NUM_THREADS`, `BLIS_NUM_THREADS`, `RAYON_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMBA_NUM_THREADS`,
`TBB_NUM_THREADS`, `JULIA_NUM_THREADS`, and `POLARS_MAX_THREADS`.
`OMP_DYNAMIC` and `MKL_DYNAMIC` are `FALSE`. `machine_id_sha256` is
the sha256 of the raw bytes of the machine-id file. The row does not
record a machine name or a filesystem location.

`best` is the driver's printed field (eight digits after the decimal).
`best_bits` is the big-endian IEEE-754 binary64 of that parsed decimal,
as sixteen lowercase hex digits. It is the bits of the printed value.
`calls` is the driver's `evals_to_hit`. `outside_s` and `charged` are
the driver's fields.

A process that exits non-zero, prints no complete row, or shows no
increase in user-plus-system time for 30 minutes is an error row. Error
rows stay in the table as misses. Their timings are blank. A finished
row is never rewritten, dropped, or relabelled.

## Decision rule

This rule is fixed before the measured session.

For each cell, and for each finished row with a finite `outside_s` and
a positive finite `charged`, form `outside_s / charged`. The arm
summary is the Hyndman-Fan type 7 median of those ratios. For `n = 24`
that median is the mean of the 12th and 13th sorted values. `B/A` is
arm B's median divided by arm A's. `A'/A` is the same for `A'`.

The cost is `REAL` only when all three of the following hold:

- `B/A > 1.10`
- `A'/A` is inside the closed interval `[0.95, 1.05]`
- the two-sided Mann-Whitney p-value of B against A is strictly below
  0.05

Otherwise the verdict is `NOT-REAL`. If any arm has fewer than 24
finite ratios, if arm A's median is not positive, or if a p-value
required by the rule is undefined, the verdict is `UNRESOLVED`.

The Mann-Whitney statistic uses average ranks. Let `n` be the common
sample size, `R` the rank sum of B, and `U = R - n(n+1)/2`. With
`mu = n*n/2` and tie sizes `t_g`,

```
s = sqrt(n*n/12 * ((2n+1) - sum(t_g^3 - t_g) / ((2n)(2n-1))))
z0 = U - mu
z = (z0 - 0.5 * sign(z0)) / s
p = erfc(|z| / sqrt(2))
```

`sign(0) = 0`. This is the two-sided asymptotic test with the tie
correction and a continuity correction of one half. If `s = 0` the
p-value is undefined.

Hits are the row's hit bit. The Fisher p-value is the two-sided exact
test on the 2 by 2 table of hits and misses, summing every table with
the same margins whose probability is at most the observed table's.
It is reported for B against A and for A' against A. It is not part of
the `REAL` conjunction.

The calls comparison is a second Mann-Whitney test, same definition.
A miss, including an error row, is ranked at positive infinity. A hit
keeps its finite `calls`. The same ranking is used for A'.

Seed-by-seed agreement compares B with A on `best_bits` and on the
recorded `calls` value (the driver's number, not the infinity used in
the rank test). Agreement is reported as a count out of 24. It does
not change the verdict.
