# no-motif SOAP gate

This gate compares arm A (baseline `bc16d35c`) and arm B (the no-motif SOAP
patch) in one serial process:

- `taskset -c 0`
- one thread (`OMP`, `MKL`, `OPENBLAS`, `NUMBA`, `RAYON` = `1`)
- no pre-warm
- fresh process per run
- per-seed interleaving with flipped arm order each seed

Rows append to `rows.csv` and include `code_revision` and `host`.

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
M75 and surface/slab, plus any budget-probe incompatibilities).

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

## Outputs

- `declared-n.csv`: per-system declared seed count
- `command-source.csv`: per-system command source and signature
- `superseded.csv`: append-only restart markers
- `rows.csv`: append-only per-run rows
- `summary.csv`: per-system hits, calls, p-values, medians
- `seed-identity.csv`: per-seed A/B bit-identity
- `deltas.csv`: every system with any hit drop or call increase
- `verdict.txt`: PASS/LOSE under the gate rule
