# m38r14d A/B gate

Arm A is revision `61fa67cf`. Arm B is the same revision plus
`fix: rebuild stalled packing from observed bases`, whose source tree is
`1defbf0898a6ef9654c1f5783b25f57acb9cc498`.

The pass rule was committed before the first timed run in `pass-rule.md`.
The campaign uses paper budgets, ABBA order within each cell, core 0, and
one thread for each supported runtime. The same seeds and generated starts
are used for both arms. LJ38 and LJ98 use 24 seeds per arm; every other cell
uses 12.

The priority campaign ran in two resumable runner sessions, `p1-s1` and
`p1-s2`. The session identifier is stored on every raw row and session event.
Rows are append-only and a completed `(band, cell, arm, seed)` key is never
rerun.

Files:

- `rows.csv`: raw append-only timing and outcome rows.
- `sessions.jsonl`: affinity, machine hash, binary, elapsed time, and session
  metadata for each row.
- `verdict.csv`: per-cell statistics and gate result.
- `progress-p1.json`: priority-campaign completion count.
