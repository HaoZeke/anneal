# generic-packings-r1 gate

Independent full-panel performance gate for the commit that restores the
constant-free fcc, polyicosahedral, Mackay and axial packings as default-on
candidates charged to the budget (tree
`cb901209d99ad6d4653008a793182a64b22e6ac7`, one commit on develop
`61fa67cf380e46876b569da45e7cc091647a775b`).

## Arms

| arm | revision | tree | role |
| --- | --- | --- | --- |
| D | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | develop |
| G | the candidate commit | `cb901209d99ad6d4653008a793182a64b22e6ac7` | candidate |
| Dp | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | second independent build of develop (A/A) |
| P | `628773d6837ea2d959a27256ff4163b21162fd4d` | `dd98579835b57f4f0c86588f7c9cb2a3cd7a9691` | parent of `1c1b1238`, pre-regression reference |

Each arm is its own worktree and its own `cargo +1.88.0 build --release
--locked` target directory, with its own examples. D and G carry
byte-identical `h2h_gmin_proof`, `h2h_morse`, `h2h_metal` and `h2h_sc`
sources, so D against G differs only in the library. P uses its own
examples, as the 1c1b1238 attribution did. `h2h_sc` is built with
`--features rgpot-ex` and loads one Sutton-Chen library for all arms, built
from rgpot `8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e`, the revision
`Cargo.toml` pins for `rgpot-pair`.

## Panel

Every cell of the paper head-to-head tables that the anneal drivers run, at
those tables' `PAPER_BUDGETS` and reference energies (24 cells):

- `scripts/gmin_panel/run_panel.py`: lj38, lj55, lj75, lj98, lj102, lj103,
  lj104, lj150, au55, m55r6, m55r14, m80r6, m80r14;
- `scripts/morse_h2h/run_campaign.py`, cells not above: m38r14, m38r6;
- `scripts/metal_h2h/run_sc.py`: sc126-38, sc126-55, sc126-80, sc108-38,
  sc108-55, sc108-80, sc96-38, sc96-55, sc96-80.

`run_gate.py` loads those tables from the harness files and refuses to run
unless every arm's `budget <system>` prints the same budget and reference.

## Protocol

Seeds 2000-2011, n = 12 per arm per cell. The command is `run <system>
<seed>` with no `H2H_*` or `ANNEAL_*` variable in the environment: the
budget is the only stop besides reaching the reference, there is no wall
cap, no quick mode and no warm-up run. One serial session, one process
under `taskset -c 0`, every thread-count variable (`OMP`, `OMP_THREAD_LIMIT`,
`MKL`, `OPENBLAS`, `GOTO`, `BLIS`, `VECLIB`, `NUMEXPR`, `NUMBA`, `RAYON`,
`TBB`) at 1. Cells run in the order above; within a cell, seeds in order;
within a seed the arms run D, G, Dp, P on even seed offsets and P, Dp, G, D
on odd ones, so every arm has the same mean slot. Each row records the
session id, a sha256 of the machine id, the cpu model, the arm's revision
and tree, and the sha256 of the binary and of `run_gate.py`.

## Declared rules

Declared before the run.

- Hit: a finite best at or below the reference plus 0.001.
- Calls: `evals_to_hit`; a miss is the full budget and ranks worse than
  every hit.
- PASS needs every one of the following in every cell:
  1. no hit loss: G has at least as many hits as D. The two-sided Fisher
     exact p on G against D is reported for every cell, and every cell where
     G has fewer hits is listed;
  2. no calls loss: not (two-sided Mann-Whitney p < 0.05 on calls with G
     ranked worse than D). The test is the harness's normal approximation
     with the tie and continuity corrections;
  3. outside time per call: per row `outside_s / charged`; the ratio of the
     Hyndman-Fan type-7 medians G/D is at most 1.10, or within the A/A band,
     that is at most max(Dp/D, D/Dp) of the same medians.
- A cell that cannot run for any arm is reported as not run; nothing is
  substituted for it, and the verdict cannot be PASS.
- Reported but not gating: per cell, how many of P's hit seeds G also hits,
  with G, P and D hit counts, and the totals.

## Run

```bash
python3 bench/generic-packings-r1-gate/run_gate.py --self-test
taskset -c 0 python3 -u bench/generic-packings-r1-gate/run_gate.py \
  --rows bench/generic-packings-r1-gate/rows.csv \
  --arm D=<D tree> --arm G=<G tree> --arm Dp=<Dp tree> --arm P=<P tree> \
  --bin-root <dir with D-bin/ G-bin/ Dp-bin/ P-bin/> --logs <log dir> \
  --sc-lib <librgpot.so>
python3 bench/generic-packings-r1-gate/run_gate.py --summarize \
  --rows bench/generic-packings-r1-gate/rows.csv \
  --summary bench/generic-packings-r1-gate/summary.csv
```
