# Move-choice knockout rerun, shard `move-b`

## What was run

- Block: Move choice (`Arm A`: `H2H_KNOCKOUT` unset, recorded as `stock`; `Arm B`: `H2H_KNOCKOUT=move_choice`).
- Systems: `lj98`, `lj102`, `lj103`, `lj104`, `lj150`.
- Seeds: `2000` through `2011` (12 seeds per system; 5 x 12 x 2 = 120 runs).
- Budget: `4000000` force calls per run (`PAPER_BUDGETS` in `scripts/gmin_panel/run_panel.py`).
- Hit criterion: best quenched energy `<= reference + 0.001` using the references in `SYSTEMS` of `scripts/gmin_panel/run_panel.py`; misses are counted at the full budget (`4000000`).
- Code under test: 13 commits applied on top of `dd6ff9273be30ab317bfd5508596f4beb19cb3a4`, producing git tree `b12293b9d637dd32417be3aa482302e0f30dd594`.
- Note on `revision` column in `rows.csv`: `revision` is the git tree id `b12293b9d637dd32417be3aa482302e0f30dd594` (not a commit SHA, because `git am` rewrites committer dates).
- Driver: `h2h_gmin_proof` built once in release mode (`cargo build --release --features rgpot-ex --example h2h_gmin_proof --example h2h_morse --example h2h_metal`), binary sha256 `aa42b5d60c1de031a5fdce8f620a9ae52ded46acf8ce35d6f776e70dd5891d8f`, source blob `6f4b277da12b9bd6ac4f2e022626210daeec0543`. Both arms ran the exact same binary and differed only by `H2H_KNOCKOUT`.
- Note on `best_bits`: big-endian IEEE-754 binary64 hex (`struct.pack('>d', float(best)).hex()`) of the 8-decimal `best` energy printed by `h2h_gmin_proof`.

## Host and execution protocol

- Host (`machine_id_sha256`): `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873` (full hardware and toolchain metadata in `hosts.csv`).
- Strictly serial execution, one process at a time, pinned to core 0 (`taskset -c 0`).
- Environment: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`; every other `H2H_*` and `ANNEAL_*` variable removed; `H2H_KNOCKOUT=move_choice` added only for Arm B.
- No warm-up exclusion, no `--quick`, no retries.

## Pairing and shared starts

- For each `(system, seed)`, the start geometry `start-<system>-<seed>.xyz` was generated once with `h2h_gmin_proof coords <system> <seed> start-<system>-<seed>.xyz` and passed as the 4th argument to both arms (`taskset -c 0 h2h_gmin_proof run <system> <seed> start-<system>-<seed>.xyz`).
- Note on `h2h_gmin_proof`: its `run` subcommand generates the deterministic seed start internally (`start_of`) and writes it to the 4th argument path rather than reading from it. The sha256 of every `start-<system>-<seed>.xyz` file was verified before Arm A, after the first arm, and after the second arm, confirming byte-identical start geometries across `coords`, Arm A, and Arm B for all 60 `(system, seed)` pairs (hashes recorded in `session.json`).
- Arm order was ABBA by seed: even seeds (`2000, 2002, 2004, 2006, 2008, 2010`) ran Arm A (`stock`) first then Arm B (`H2H_KNOCKOUT=move_choice`); odd seeds (`2001, 2003, 2005, 2007, 2009, 2011`) ran Arm B first then Arm A.
- Rows were appended to `rows.csv` in run order and never rewritten.

## Step 0 bit-identity verification

- Step 0(a): `cargo test --release --test knockouts` passed (3 passed, 0 failed); `pytest -v pytest/test_knockout_drivers.py` passed (1 passed, 0 failed).
- Step 0(b): `h2h_gmin_proof` built from `develop` (`dd6ff9273be30ab317bfd5508596f4beb19cb3a4`, sha256 `23210bd5608f168cbac2525f64cf1c26c2f60bd7eaa2fe0ddae9493d3944793d`) vs the `r5` binary with `H2H_KNOCKOUT` unset on seed `2000` at budget `4000000` matched bit-for-bit on `best`, `best_bits`, `charged`, `evals_to_hit`, `hit`, and start-file sha256 across all 5 systems (`lj98`, `lj102`, `lj103`, `lj104`, `lj150`). Full values are in `session.json`.

## Crashes

- None (all 120 runs exited 0 and passed run-count validation).

## Paired statistics (`scripts/paired_stats.py`)

Interval is Newcombe method-10 95% CI for `hit_rate(stock) - hit_rate(H2H_KNOCKOUT=move_choice)`.

| System | n | Arm A (`stock`) | Arm B (`move_choice`) | Both | Only A | Only B | Neither | Exact McNemar p | Newcombe 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lj98` | 12 | 7/12 | 9/12 | 7 | 0 | 2 | 3 | 0.5 | [-0.3992, +0.0942] |
| `lj102` | 12 | 6/12 | 7/12 | 3 | 3 | 4 | 2 | 1 | [-0.4427, +0.3068] |
| `lj103` | 12 | 6/12 | 8/12 | 4 | 2 | 4 | 2 | 0.6875 | [-0.4809, +0.2032] |
| `lj104` | 12 | 5/12 | 3/12 | 2 | 3 | 1 | 6 | 0.625 | [-0.1758, +0.4620] |
| `lj150` | 12 | 8/12 | 9/12 | 7 | 1 | 2 | 2 | 1 | [-0.3731, +0.2254] |
| `pooled` | 60 | 32/60 | 36/60 | 23 | 9 | 13 | 15 | 0.523467 | [-0.2155, +0.0870] |
