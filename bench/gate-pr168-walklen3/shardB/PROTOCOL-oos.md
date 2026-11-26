# PR 168 walklen3 gate, shard B, out of sample

Measurement only. This file is the rule for the out-of-sample round,
written before any out-of-sample row. The in-sample report is left as
it was scored.

## Arms

- A is develop `7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec`, tree
  `e5ebda63f9f0117226d50c13abf33243ccf452cc`.
- M is scratch merge `8824564b2479f0917c1e4927cd5ac938a86c2ac0`, tree
  `991d5afc6d7b7f7e6c392cb68aa351af1fc50179`, the clean merge of the
  base with payload `4ab2681ae6bf72d61c633197e88047f00cdd7efa`.
- O payload `fd5449766b4f1f693de2245933afa609a93a5c6b` has tree
  `491819895e46a8fccbfc2449c18868780d1a090b`, three commits on
  `52af8e47`. `git merge-tree --write-tree` of the base and that payload
  exits 0 and writes tree `7f2625597a167d3c7841b2266eec844997e53266`.
  O is scratch merge `c96807c54e8cc602322255c75be61425c7bf722a` of that
  tree, first parent the base and second parent the payload, on
  `refs/scratch/walklen3-shardB-o`.

The scratch refs are used only to build. They are not part of the
result history. A, M, and O use the same recommended preset, the same
budgets, the same stop, and the same release build flags. The same
resident-set sampler is compiled into each arm and the sources are
restored. No external minimiser program is built or run.

## Seeds

Seeds 2000-2011 are the in-sample record in `rows.csv`. They are not
inputs to this round. Out-of-sample seeds start at 2012.

- `lj75-seed` uses seeds 2012 through 2119 (108 rounds) and runs first.
- `ase-lj98`, `ase-lj104`, `au55-seed`, `lj38`, `m38r6`, and `m55r14`
  use seeds 2012 through 2023 (12 rounds), in that order.

Rows are append-only in `rows-oos.csv`. A run shorter than 50 ms is
repeated under the same rule as the in-sample round.

## Order

Serial, one core, one thread. Round `i` is seed `2012 + i`. The cycle is

```
0  A M O
1  O M A
2  M A O
3  O A M
4  M O A
5  A O M
```

## Scores

Each arm reports hits over n with a Wilson 95 percent interval, the
median calls-to-hit with a miss entered at budget plus one, the median
outside time, and the median resident set.

Pairs are O/M, O/A, and M/A. The first arm is the challenger. Hits use
a one-sided Fisher exact test, the alternative that the challenger has
fewer hits. Calls, outside time, and RSS use a one-sided Mann-Whitney
test, tie-corrected, the alternative that the challenger is larger.
The median ratio is the challenger's median divided by the reference
median. Its interval is 10000 paired resamples, seed 20261123, a fresh
generator for each interval, type-7 percentiles at 2.5 and 97.5.

## Verdict

A comparison is complete only when every arm has the target round count
and every row has a positive RSS.

Bars, on a complete comparison:

- no hit loss: hits of the challenger are at least hits of the reference
- no calls-to-hit loss: the median of the challenger is at most the median of the reference
- outside-time median-ratio upper interval at most 1.10
- RSS median-ratio upper interval at most 1.10

A hit loss is real when the one-sided Fisher probability is below 0.05.
A calls, outside, or RSS regression is real only when the bar fails and
the lower interval is above 1. This round has no second build of the base.

PASS when the comparison is complete and every bar holds. FIX when it
is complete and at least one real regression is present. UNRESOLVED
when it is short of its target, or a bar fails and no regression is
real. A finished verdict is not rewritten. The three comparisons are
M versus A, O versus A, and O versus M.
