# Knockout rerun, shard move-a (move choice)

The move-choice knockout block on eight systems: LJ38, LJ55, LJ75, Au55,
and Morse M55 and M80 at rho 6 and 14. Seeds 2000-2011, two arms, 192
runs, all on one host in one session (`hosts.csv`).

- Arm A, `stock`: `H2H_KNOCKOUT` unset.
- Arm B, `H2H_KNOCKOUT=move_choice`: `arm_selection=ArmSelection::Uniform`,
  equation `eq:uniform` in place of `eq:thompson`.

## Code

Develop `dd6ff9273be30ab317bfd5508596f4beb19cb3a4` with the 13 commits of
the r5 science rewrite applied by `git am`. The `revision` column of
`rows.csv` is `b12293b9d637dd32417be3aa482302e0f30dd594`, which is a git
**tree** id (`git rev-parse HEAD^{tree}` of the r5 tip), not a commit id:
`git am` writes new committer dates, so the commit id differs from one
application to the next while the tree does not. `driver_blob` is the git
blob of `examples/<driver>.rs` in that tree.

One release build,

    cargo build --release --features rgpot-ex \
      --example h2h_gmin_proof --example h2h_morse --example h2h_metal

with rustc 1.88.0. Both arms ran the same three binaries; their sha256 is
in `session.txt` and in the `binary_sha256` column. Drivers:
`h2h_gmin_proof` for LJ38, LJ55 and LJ75, `h2h_metal` for Au55,
`h2h_morse` for the four Morse systems.

## Step 0, before any measured run

(a) `cargo test --release --test knockouts` passed (3 tests) and
`pytest pytest/test_knockout_drivers.py` passed (1 test).

(b) The same three examples were built on develop `dd6ff927` in a
separate worktree. For each of the eight systems, seed 2000 ran at its
paper budget with the develop binary and with the r5 binary, switch
unset, under the run protocol below. Both drivers accept `coords` and the
fourth argument, so both ran with the same start file. The best energy
bits and the charged calls match on all eight systems; the table is in
`session.txt` under `[step0b]`.

## Protocol

- Serial, one process at a time, `taskset -c 0`, with
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
  `RAYON_NUM_THREADS` set to 1 and every other `H2H_*` and `ANNEAL_*`
  variable removed, as `scripts/gmin_panel/run_panel.py::single_thread_env`
  does. Arm B adds only `H2H_KNOCKOUT=move_choice`.
- Budgets from `PAPER_BUDGETS` in `scripts/gmin_panel/run_panel.py`:
  400000 for LJ38; 1000000 for LJ55, Au55, M55 rho 6 and M55 rho 14;
  4000000 for LJ75, M80 rho 6 and M80 rho 14.
- Shared start: before the first measured run, the stock binary wrote one
  start per system and seed, `<driver> coords <system> <seed>
  start-<system>-<seed>.xyz`, and both arms got that file as the fourth
  argument, `taskset -c 0 <driver> run <system> <seed>
  start-<system>-<seed>.xyz`. The sha256 of each start is in
  `session.txt` under `[starts]`.
- `h2h_morse` reads the fourth argument as its start. `h2h_gmin_proof`
  and `h2h_metal` instead write the start they quench to that path, and
  that start is a function of the system and seed only. For every run the
  sha256 of the start file was taken before and after; it equals the
  shared start's sha256 on all 192 runs (`session.txt`, `[runs]`), so on
  those two drivers too both arms quenched the same start.
- Order: systems in the order above, seeds ascending, ABBA by seed: on
  even seeds arm A ran first, on odd seeds arm B. The `slot` of each run
  is in `session.txt`. No warm-up exclusion, no `--quick`, no retries.
- Hit: best quenched energy at or below the reference in `SYSTEMS` of
  `run_panel.py` plus 0.001. A miss is entered at the full budget in
  `evals_to_hit` and `charged`.

## Rows

`rows.csv` holds one row per system, arm and seed, appended as each run
finished and not rewritten, with the columns of
`paper/jctc/data/knockouts/templates/rows.csv` and three more: `arm`
(`stock` or the exact environment string), `binary_sha256`, and
`best_bits`. `pair` is `move-a` on every row. `outside_s` is the
driver's own column where it prints one (`h2h_metal`, `h2h_morse`) and
`wall_s - pot_s`, floored at 0, for `h2h_gmin_proof`, as `run_panel.py`
does. `machine_id` is the sha256 of the host's machine-id file.

The drivers print the best energy with eight decimals and nothing more
precise. `best_bits` is the IEEE-754 double of that printed value, in
big-endian hex. It is the same for two runs exactly when their printed
best is the same, and it is what step 0(b) compares.

Arm A here was run for this block alone; it is not shared with the other
knockout blocks.

## Crashes

None. All 192 runs exited 0 and printed one result row; every miss
reports the full budget in `evals_to_hit` and `charged`, and the hit
column agrees with the hit rule on every row.

## Statistics

`stats.csv` is the output of

    python3 paper/jctc/data/knockouts/move-a/stats.py \
      paper/jctc/data/knockouts/move-a/rows.csv

which pairs the arms by system and seed and applies
`scripts/paired_stats.py`: hits per arm, the paired outcome table, the
two-sided exact McNemar test, and the Newcombe method-10 95% interval of
the stock hit rate less the knockout hit rate. The `all` line pools the
96 pairs.

| System | Stock hits | Knockout hits | Both | Stock only | Knockout only | Neither | McNemar p | Newcombe 95% |
|---|---|---|---|---|---|---|---|---|
| lj38 | 12/12 | 12/12 | 12 | 0 | 0 | 0 | 1 | -0.2425 to 0.2425 |
| lj55 | 12/12 | 12/12 | 12 | 0 | 0 | 0 | 1 | -0.2425 to 0.2425 |
| lj75 | 5/12 | 6/12 | 2 | 3 | 4 | 3 | 1 | -0.4427 to 0.3068 |
| au55 | 4/12 | 5/12 | 2 | 2 | 3 | 5 | 1 | -0.4115 to 0.2718 |
| m55r6 | 12/12 | 12/12 | 12 | 0 | 0 | 0 | 1 | -0.2425 to 0.2425 |
| m55r14 | 9/12 | 8/12 | 5 | 4 | 3 | 0 | 1 | -0.3202 to 0.4554 |
| m80r6 | 12/12 | 12/12 | 12 | 0 | 0 | 0 | 1 | -0.2425 to 0.2425 |
| m80r14 | 7/12 | 8/12 | 4 | 3 | 4 | 1 | 1 | -0.4471 to 0.3111 |
| all | 73/96 | 75/96 | 61 | 12 | 14 | 9 | 0.845019 | -0.1266 to 0.0854 |

## Files

- `rows.csv`: the 192 rows.
- `hosts.csv`: the host, in the format of
  `paper/jctc/data/hosts/3b11c160/hosts.csv`.
- `session.txt`: binary sha256s, step-0 results, start sha256s, and per
  run the arm, slot, exit status, start sha256 before and after, and UTC
  start and end.
- `stats.py`, `stats.csv`: the paired statistics above.
