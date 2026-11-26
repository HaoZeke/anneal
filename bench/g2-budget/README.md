# Copyright (c) anneal developers

# G2 budget-stop campaign

This file pre-registers the campaign. No seed has been run.

The systems and the call budgets are the paper budgets. One oracle call is one energy and force evaluation of one distinct geometry.

| system | oracle calls |
| --- | --- |
| lj38 | 400000 |
| lj55 | 1000000 |
| lj98 | 4000000 |
| lj104 | 4000000 |
| au55 | 1000000 |

Seeds are 2000-2011, twelve seeds. Each seed is serial, on one core. There is no wall cap.

## Arms

`ase_bh` is budget-stopped on lj38, lj55, lj98, lj104, and au55.

`ase_mh` is budget-stopped on lj38 and lj55. On lj98, lj104, and au55 it stops at a fixed 12 h process-CPU cap, 43200 s.

`anneal` is `Config::for_cluster`. It runs on all five systems at the full paper budget. On lj98, lj104, and au55 it uses the same 43200 s process-CPU cap as `ase_mh`.

The cap is the constant `CPU_CAP_S` in `scripts/g2_budget/protocol.py`. `workflow/Snakefile` loads that constant for rules `g2_budget` and `g2_budget_cell`. A Snakemake config value cannot replace it. The workflow rejects `cpu_cap_s`, `cpu_cap`, `cap_s`, `g2_cpu_cap_s`, and `g2_budget_cpu_cap_s`.

## Pairs

A comparison uses the same stop rule on both arms. A capped run is never paired with an uncapped run.

On lj38 and lj55 every arm is budget-stopped. The pairs are `ase_bh` with `ase_mh`, `ase_bh` with `anneal`, and `ase_mh` with `anneal`.

On lj98, lj104, and au55 the pair is `ase_mh` with `anneal`. Both arms use the CPU cap. `ase_bh` on those systems is budget-stopped and is not paired with either capped arm.

## Rows

A seed that reaches the reference writes `stop=hit`. A seed that spends the call budget writes `stop=budget`. A seed that hits the CPU cap before the budget is a censored miss at the cap. That row writes `stop=cap`, `calls_reached`, `fraction_of_budget` equal to `calls_reached/budget`, the best energy, CPU seconds, wall seconds, and the revision. The best energy is the lowest energy among the seed's segments.

A production `stop=cap` row records summed CPU of at least 43200 s. That CPU is at most the summed wall, with a tolerance of a millionth of a second.

Checkpoint and resume are allowed only for capped seeds. CPU seconds are the sum of the process CPU across segments. Wall seconds are the sum of the segment walls. A budget-stopped seed has one segment.

Long seeds run one per VM. Rule `g2_budget_cell` is one seed.

Rows are appended to `bench/ase-<system>/rows-ljfix-budget.csv`. Shapes are appended to `bench/ase-<system>/shapes-ljfix-budget.jsonl`. The shape is written first. A later attempt that finds the shape and not the row writes the row when it is the same record. These files are separate from the scope-1 `rows-ljfix*.csv` tables.

## Scoring

Fisher is the two-sided exact test on hits. MWU is the two-sided Mann-Whitney U test on calls-to-hit, with the tie correction and the continuity correction.

A `stop=hit` row contributes `calls_reached` to the calls-to-hit statistic. A `stop=cap` row, and any other non-hit, contributes `budget+1`.

## Why the cap is 12 h

The scope-1 projections are lower bounds, because ASE MH duplicate checks grow about as budget squared.

LJ98 minima hopping is at least 22 h per seed. LJ104 minima hopping is at least 25 h per seed. Au55 minima hopping is at least 1.9-4.3 h per seed. Basin hopping fits in hours.

The cap is 43200 s of process CPU for minima hopping and for_cluster on lj98, lj104, and au55. LJ98 and LJ104 minima hopping are above 12 h per seed on these lower bounds. A seed that reaches the cap before the call budget is a censored miss.

## Amendment A1 (post-registration)

This section is added after 1308d867. It does not change the five preregistered systems or their arms.

Amendment A1 covers lj75 and lj150 on `ase_mh` only. `ase_bh` and `anneal` are rejected for both systems. Seeds are 2000-2011. The runner is `scripts/g2_budget/run_cell.py`, the same runner as the preregistered cells. The stop rule is the budget-or-cap rule `ase_mh` already uses on lj98, lj104, and au55: a seed stops on a hit, on the call budget, or at the 43200 s process-CPU cap, whichever comes first. A capped seed resumes from its checkpoint. CPU seconds are the sum of the segments.

The call budgets are 4_000_000 for lj75 and 4_000_000 for lj150. The lj75 budget is `PAPER_BUDGETS` in `scripts/ase_lj75/run_one.py`, which `scripts/ase_nonlj/cpu_cap_spec.py` loads for the capped-shard harness. The lj150 budget is `SYSTEMS` in `scripts/ase_panel/run_one.py`. The reference energies are the Cambridge Cluster Database values already listed in `scripts/gmin_panel/run_panel.py`: -397.492331 for lj75 and -893.310258 for lj150. The same lj75 reference is `REFERENCE` in `scripts/ase_lj75/run_one.py`. The same lj150 reference is the lj150 entry of `scripts/ase_panel/run_one.py`.

Rows are appended to `bench/ase-lj75/rows-ljfix-budget.csv` and `bench/ase-lj150/rows-ljfix-budget.csv`. Shapes are appended to the matching `shapes-ljfix-budget.jsonl` files. The row columns stay the preregistered columns. The system label is lj75 or lj150. Each shape carries `amendment` `A1`. These cells are reported separately from the five preregistered systems. They are not in `rule g2_budget`. `rule g2_amend_a1` runs them.

## Runner setup

A fresh checkout needs the Python extension and the panel evaluator before the runner tests can import a search.

```
pip install maturin
maturin develop --release
```

The documented equivalent from the install notes is:

```
pip install "maturin>=1.5,<2"
maturin develop --release --features python
```

Au55 cells also need the panel evaluator:

```
cargo build --release --example asg_eval
export ASG_EVAL="$PWD/target/release/examples/asg_eval"
```
