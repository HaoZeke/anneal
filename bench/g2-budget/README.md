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

A seed that reaches the reference writes `stop=hit`. A seed that spends the call budget writes `stop=budget`. A seed that hits the CPU cap before the budget is a censored miss at the cap. That row writes `stop=cap`, `calls_reached`, `fraction_of_budget` equal to `calls_reached/budget`, the best energy, CPU seconds, wall seconds, and the revision.

Checkpoint and resume are allowed only for capped seeds. CPU seconds are the sum of the process CPU across segments. Wall seconds are the sum of the segment walls. A budget-stopped seed has one segment.

Long seeds run one per VM. Rule `g2_budget_cell` is one seed.

Rows are appended to `bench/ase-<system>/rows-ljfix-budget.csv`. Shapes are appended to `bench/ase-<system>/shapes-ljfix-budget.jsonl`. These files are separate from the scope-1 `rows-ljfix*.csv` tables.

## Why the cap is 12 h

The scope-1 projections are lower bounds, because ASE MH duplicate checks grow about as budget squared.

LJ98 minima hopping is at least 22 h per seed. LJ104 minima hopping is at least 25 h per seed. Au55 minima hopping is at least 1.9-4.3 h per seed. Basin hopping fits in hours.

The cap is 43200 s of process CPU for minima hopping and for_cluster on lj98, lj104, and au55. LJ98 and LJ104 minima hopping are above 12 h per seed on these lower bounds. A seed that reaches the cap before the call budget is a censored miss.
