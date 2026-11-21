# Basin lookup bound and symmetrise trigonometry, r6

There are three arms, all built with Rust 1.88 and `--features rgpot-ex`, each from its own worktree into its own target directory:

- **A** is develop `9837358ae1027067ff654fac8b48969020853452`.
- **A2** is the same commit, built a second time from a second worktree. Its binaries differ from A's only in embedded paths. The two give the A/A band.
- **R6** is `4d2fa0f47698cda5c658c98f9e78d08b5017816b`. That is develop with the block-sum bound on the basin lookup (`64093db7`) and one sine and cosine per symmetry rotation (`4d2fa0f4`).

`run_arms.py` runs them one at a time with `taskset -c 0` and every thread pool at one, on seeds 2000 to 2011 and `PAPER_BUDGETS`. Within a seed the arm order rotates with `seed % 3`.

- A hit is `energy_hits` against the reference minimum.
- Medians are Hyndman–Fan type 7, with a miss counted as infinite calls.
- Outside time is `wall_s - pot_s`. The outside cost per call for a cell is the type-7 median of `outside_s / charged` over its twelve runs.

Every row of `rows.csv` names the commit its binary was built from, the binary's sha256 and the host label. It also records `started_unix` and `elapsed_s`, the runner's own wall-clock span of the run. A pause of the host would show as `elapsed_s` well above `wall_s`. The largest difference in the campaign is 0.155 s, so no run spanned a pause.

`raw/` holds the driver output of every run. The runner writes `<cell>-<seed>-<arm>.log`, and the files are renamed to `.txt` here because the repository ignores `*.log`. These are development numbers from one host, not the benchmark gate.

## Identity

All 84 R6 rows and all 84 A2 rows match the A row of the same cell and seed on `calls_to_hit`, `charged`, `best_bits` and `hit`, with 0 differences. Both changes alter how a result is computed, not the result. The library tests compare lookups and registrations against a verbatim copy of the old walk, and the symmetrise deviation against a copy of the per-point rotation.

## Outside time per call

| Cell | Hits | Median calls | A (µs) | A2 (µs) | R6 (µs) | A2/A | R6/A | R6/A2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m55r6 | 12/12 | 22596 | 4.968 | 4.942 | 4.821 | 0.995 | 0.970 | 0.976 |
| lj38 | 12/12 | 127839.5 | 2.500 | 2.468 | 2.411 | 0.987 | 0.964 | 0.977 |
| m55r14 | 9/12 | 451561.5 | 4.501 | 4.373 | 4.392 | 0.971 | 0.976 | 1.004 |
| lj98 | 7/12 | 3166811.5 | 9.256 | 9.208 | 8.699 | 0.995 | 0.940 | 0.945 |
| lj102 | 6/12 | miss | 10.143 | 10.087 | 9.585 | 0.994 | 0.945 | 0.950 |
| m80r14 | 6/12 | miss | 9.242 | 9.194 | 8.218 | 0.995 | 0.889 | 0.894 |
| lj75 | 5/12 | miss | 6.056 | 6.031 | 5.673 | 0.996 | 0.937 | 0.941 |

Hits and median calls are the same in all three arms. The A/A ratio lies between 0.971 and 0.996.

- On lj98, lj102, m80r14 and lj75, R6/A is 0.889 to 0.945, below the band on both A and A2. m80r14 gains most, at 11 percent.
- On m55r6 and lj38, R6/A is 0.970 and 0.964, just below the band. R6/A2 is 0.976 and 0.977.
- On m55r14, R6/A is 0.976, inside the band, and R6/A2 is 1.004. That is no measurable change.
