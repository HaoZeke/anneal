# no-motif SOAP gate

This gate compares arm A (baseline `bc16d35cf13e71c2edaf343b80fd8c41368631d4`)
and arm B (recorded in `rows.csv` as `e3f617afaea225941feb36511fa79a481bd92048`)
in one serial process:

- `taskset -c 0`
- one thread (`OMP`, `MKL`, `OPENBLAS`, `NUMBA`, `RAYON` = `1`)
- no pre-warm
- fresh process per run
- per-seed interleaving with flipped arm order each seed

Rows append to `rows.csv` and include `code_revision` and `host`.

## Arm revision provenance

`config.yaml` points `a_tree` and `b_tree` at external worktrees
(`../anneal-arm-a` and `../anneal-arm-b`), and `run_gate.py` records
`git -C <tree> rev-parse HEAD` as `code_revision` in `rows.csv` (discarding
the `HEAD^{tree}` field printed by the binary; neither `work/` nor `logs/` is
committed).

- Arm A (`bc16d35cf13e71c2edaf343b80fd8c41368631d4`) is in the public
  repository (`Merge pull request #74 from HaoZeke/bench/land1`).
- Arm B (`e3f617afaea225941feb36511fa79a481bd92048`) is not in the public
  repository, and no tree SHA or run log is stored in `rows.csv` or the
  repository. The candidate published branch is `origin/fix/no-motif-soap` at
  `8c41eddbd778eda912b24acaaff3efe755493e4c` (`refactor: drop the fivefold
  template from the search`, parent `bc16d35cf13e71c2edaf343b80fd8c41368631d4`,
  tree `f8e4f5249d4fe561f407e7488e73c5d92a38cb6a`, committed `2026-11-18T14:30:37Z`,
  44 minutes before session `20261118T151421Z`), which replaced the fivefold
  residual with `step_descriptor` / `step_away_outliers`. Because
  `e3f617afaea225941feb36511fa79a481bd92048` and its tree object are absent,
  arm B cannot be pinned to `8c41eddbd778eda912b24acaaff3efe755493e4c` or
  proven tree-identical (it may have been an unpushed local rewrite/amend or a
  local commit built on top of it).

## Command source by system

`command-source.csv` records the active source/signature for each system.

- **NOTES commands** (`anneal-no-motif-soap-NOTES.md`):
  - LJ: `lj38`, `lj55`, `lj75`, `lj98`, `lj103`, `lj104` (`n=12`)
  - Morse: `m55r6`, `m55r14`, `m80r6`, `m80r14` (`n=12`)
  - Au: `au55` (`n=12`)
  - Sutton-Chen: `sc126-38`, `sc126-55`, `sc126-80`, `sc108-38`, `sc108-55`,
    `sc108-80`, `sc96-38`, `sc96-55`, `sc96-80` (`n=6`, 2000-2005)
- **Repo runner budgets**:
  - binary-LJ systems from `scripts/metal_h2h/run_darby.py::PAPER_BUDGETS`

Unsupported systems are listed in `unsupported.csv` (for this capture:
M75, surface/slab, and `sutton-chen/all` because `RGPOT_SC_LIB` was not
provided, plus any budget-probe incompatibilities).

## Superseded rows

When a system's command signature changes after rows were already recorded,
`superseded.csv` appends a restart marker. Analysis filters out rows before
that marker for the affected `(panel, system)` pair.

## Run

```bash
snakemake -s bench/no-motif-soap-gate/Snakefile --cores 1 --config notes=<notes-file>
```

Inside the bench namespace:

```bash
sudo -n unshare --uts sh -c 'hostname bench; exec sudo -u "$(id -un)" snakemake -s bench/no-motif-soap-gate/Snakefile --cores 1 --config notes=<notes-file>'
```

## Gate rule (`summarize.py`)

`summarize.py` evaluates every system in `summary.csv` at `ALPHA = 0.05` across
four directional loss criteria (`GATE_LOSS_CRITERIA`):

1. `lose_hit_sig`: `hits_b < hits_a` and two-sided Fisher exact `fisher_p < 0.05`
   (`hits`)
2. `lose_call_sig`: `median_calls_b > median_calls_a` and tie-corrected
   Mann-Whitney normal approximation `mannwhitney_p_calls < 0.05`
   (`calls to hit`)
3. `lose_outside_sig`: `median_outside_s_b > median_outside_s_a` and
   `mannwhitney_p_outside_s < 0.05` (`time outside the potential`)
4. `lose_wall_sig`: `median_wall_s_b > median_wall_s_a` and
   `mannwhitney_p_wall_s < 0.05` (`wall time`)

The top-level line in `verdict.txt` is derived from `summary.csv` using three
mutually exclusive outcomes:

- `LOSE`: at least one system has `1` for any criterion in `GATE_LOSS_CRITERIA`;
  the `verdict:` line names every triggered criterion and every matching system
  with its sample size and p-value.
- `INCOMPLETE`: no criterion in `GATE_LOSS_CRITERIA` triggers and at least one
  declared system has `status != "ok"` (or `summary.csv` is empty).
- `PASS`: every declared system has `status == "ok"` and no criterion in
  `GATE_LOSS_CRITERIA` triggers.

Incomplete systems (`status != "ok"`) are split into two labelled lists with
reasons taken from `unsupported.csv`, `declared-n.csv`, `rows.csv`, and
`summary.csv`:

- `not run (never run; unsupported)`: declared systems matched in
  `unsupported.csv` (exact `(panel, system)` or panel-wide `(panel, all)`).
- `not run (skipped or stopped for outside-potential cost)`: supported declared
  systems that stopped mid-run or were skipped after the cost stop and gate
  loss.

## Outputs

- `declared-n.csv`: per-system declared seed count
- `command-source.csv`: per-system command source and signature
- `superseded.csv`: append-only restart markers
- `rows.csv`: append-only per-run rows
- `summary.csv`: per-system hits, calls, p-values, medians
- `seed-identity.csv`: per-seed A/B bit-identity
- `deltas.csv`: every system with any hit drop, call increase, outside-potential
  time increase, or wall-time increase
- `verdict.txt`: `PASS` / `LOSE` / `INCOMPLETE` verdict and breakdown under the
  declared gate rule
