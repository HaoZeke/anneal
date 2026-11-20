# TIP4P stall-vs-develop and PR #160 gate: files and layout

The pre-registered gate rules and bars are in
`gates/anneal-tip4p-stall-vs-develop-gate.md`. The harness is
`scripts/tip4p_h2h/gate_stall_vs_develop.py`, adapted from the four-arm
b11 gate harness to run `TIP4P9` through `TIP4P21` on seeds `2000` to
`2023` in a balanced 4x4 Williams order and the thirteen-system
non-rigid identity panel on `A`, `B` and `C`.

## Files

- `rows.csv`: every scored row of both campaigns (`1,248` rigid rows across
  `13` cells, `24` seeds and `4` arms, plus `225` non-rigid rows across
  `25` system-driver pairs, `3` seeds and `3` arms = `1,473` rows).
- `session.json`: committed session record with the hashed machine id,
  CPU 0 pinning evidence (`taskset -p` output and `Cpus_allowed_list`
  from `/proc/<pid>/status`), the thread environment variables, the
  verified commit, tree and binary SHA-256 digests of arms `A`, `A2`,
  `B` and `C`, and peak `RssAnon` and `ru_maxrss` per arm.
- `summary.json`: per-cell and pooled metrics, exact McNemar tests for
  `B` vs `A`, `C` vs `A` and `C` vs `B`, `A2/A` agreement, non-rigid
  identity checks, RSS summaries, and rule verdicts produced by
  `gate_stall_vs_develop.py score`.

## Row columns

- `campaign`: `rigid` or `nonrigid`.
- `seed_set`: `primary` for seeds `2000` to `2011`, `second` for seeds
  `2012` to `2023`, empty for `nonrigid`.
- `driver`: `h2h_seed`, `h2h_gmin_proof`, `h2h_morse`, or `h2h_metal`.
- `cell`: `tip4p9` through `tip4p21`, or one of the thirteen non-rigid
  panel systems.
- `seed`, `arm` (`A`, `A2`, `B`, `C`), `label` (`develop`, `develop-2`,
  `stall`, `pr160`), `round`, `slot`, `commit`, `tree`, `machine_id`,
  `core`.
- `budget`, `reference`, `hit`, `evals_to_hit`, `calls`, `charged`,
  `wall_s`, `pot_s`, `outside_s`.
- `best` (lowest returned objective energy, scored against `reference +
  0.001`), `search_best`, `lowest`, `geometry_energy`, `hops`, `basins`.
- `start_sha`, `trace_sha`, `geometry_sha`: SHA-256 digests of the
  written start, trace and best-geometry files.
- `rss_anon_kb`, `ru_maxrss_kb`: peak `RssAnon` from
  `/proc/<pid>/status` and child `ru_maxrss` from `wait4` in KiB
  (reported only, never gated).
- `pause_gap_s`: seconds between consecutive log end timestamps not
  accounted for by the run's own `wall_s`.
