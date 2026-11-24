# PR 186 restack same-session gate protocol

Written before any data. Nothing here changes after the first run.

## Subjects

A is develop `9b65fda3972fd8c014bce886a3ea5c05956d64af`, tree
`9b65fda3^{tree}`.

B is `c8581d69319a2f960c25ecb5f0261feb34a739b3`, tree
`184736e2f596126ef71baca23d0ecdf756aaf77c`. It is the merge of the
PR head `9e126b9ed90b504b64accce2fb64d5550f6da737` with A. B is
built from a squash commit on A with exactly that tree.

A2 is a second, independent release build of A. In the row files it is
the arm `Ap`.

## Arms and schedule

Cells run one at a time, serially, on core 0 (`taskset -c 0`). Every
environment variable whose name ends in `_NUM_THREADS` is one. Release
builds only. There is no quick mode, no warm-up, no retuning, no
rescue, no hand-picking and no relabelling. No GMIN run is part of
this gate.

Seeds are 2000 through 2011, n = 12 per arm per cell. Each cell runs
palindromic blocks of two seeds: A, B, A2 on the first seed, then A2,
B, A on the second. Each arm has the same mean position in every cell.
Budgets and stop levels are identical in every arm. All arms run on
one machine, one session.

## Cells

Molecular cells (recommended molecular preset, paired conformer hit
rule, budgets are the per-seed anneal budgets of the paired conformer
benchmark, with the cell median where a seed has none): rq3, 915,
xld, fluvastatin, z34, amprenavir, lopinavir, lorlatinib.

Default-preset cells (stop at published minimum plus 0.001):

| cell | budget | published minimum |
|---|---|---|
| lj38 | 400000 | -173.928427 |
| m38r14d | 400000 | -144.321054 |
| lj75 | 4000000 | -397.492331 |
| lj98 | 4000000 | -543.665361 |
| m80r6 | 4000000 | -378.333471 |
| au55 | 1000000 | -191.03085280 |
| hexamer | 500000 | -197.7805260034 |

## Measurements

Every row records the hit, the budget, the calls to the hit (or the
whole budget on a miss), wall seconds, seconds inside potential
calls, outside seconds per call, and `RssAnon` in kB read in-process
while the search state is alive. Rows are append-only. A failed row is
kept and makes its cell INCONCLUSIVE. There are no replacement runs.

Outside time per call is (wall minus inside) divided by calls. Outside
time in total is wall minus inside. Both are reported for every cell.

## Statistics

Random procedures use seed 20261123 and 10000 seed-level bootstrap
resamples. Every ratio is seed-paired B/A, summarised as a geometric
mean with a 95 percent percentile interval. The same is computed for
A2/A as the A/A band.

1. No hit loss. Fisher's exact test on hits (one-sided, B fewer) and a
   one-sided Mann-Whitney test on calls with every miss scored as
   budget + 1 (B larger). A hit loss is p < 0.05 in either test.
2. Outside time. The per-call ratio B/A has upper bound at most 1.10.
   Both versions are reported: the strict one, upper at most 1.10, and
   the band one, upper at most max(1.10, upper of A2/A). The verdict
   uses the band version. A regression is real only if the lower bound
   is above both 1 and the upper of A2/A.
3. Memory. The RssAnon ratio B/A. The one-sided gate fails only if its
   lower bound exceeds max(1.10, upper of A2/A). Upper at most 1.10 is
   also reported.
4. Early-stop artifact. A run that stops at a hit has fewer calls, so
   its per-call outside time carries a fixed cost over fewer calls.
   A cell with any early-stopped run is flagged, and the per-call and
   total outside ratios are always printed side by side. If they
   disagree, that is stated and neither is relabelled.
5. Default-preset cells also compare trajectory hash, best-energy
   bits, calls and charged calls between A and B seed by seed. These
   strict fields stay as the default-parity bar. Agreement of
   converged or repeatable energies up to spectroscopic accuracy is a
   pass for everything else, and differences near 1e-9 are not flagged.

## Verdict

Per cell: FIX if any rule above fails, INCONCLUSIVE if rows are
missing or errored, PASS-BAND if only the strict outside-time or
memory bound is missed but the band version holds, otherwise PASS. A
FIX is never relabelled as a PASS. The gate is PASS only if every cell
is PASS or PASS-BAND.

## Known gaps

The harness has no build-only path counter. The lazy basin probe
reaches B and A alike through develop, so the default-preset cells and
the molecular cells are the evidence that B moved nothing.

## Addendum (before data)

1. Environment and build deviations from README steps 3-4:
   - `rustup toolchain install 1.88.0 --profile minimal` was run
     because only Rust 1.83.0 was pre-installed.
   - `scripts/openconf_h2h/setup_env.sh` defaults `MAMBA_EXE` and
     `MAMBA_ROOT_PREFIX` to a non-writable system prefix, so both were
     pointed to a writable scratch directory alongside
     `OPENCONF_H2H_ENV`.
   - `maturin` is installed inside `$OPENCONF_H2H_ENV/bin`, so step 4
     runs with `$OPENCONF_H2H_ENV/bin` prepended to `PATH`.
   - `rdff_count.c` compiled cleanly with `gcc -O2 -shared -fPIC ... -ldl`.
2. Arm tree checks:
   - Each arm checkout has `anneal-186-restack-harness.mbox` applied
     on top of its base (`HEAD^{tree}` on B is `a839b9a9...` with
     `HEAD~1^{tree}` = `184736e2...`; `HEAD~1^{tree}` on A/A2 is
     `9b65fda3^{tree}`). `run_gate.py` verifies both the underlying
     arm tree and the harness-applied tree, and tracks `PROTOCOL.md`
     in `HARNESS_FILES`.
   - `guard_trace.rs` from the updated harness is copied into
     `examples/guard_trace.rs` of each arm before its release build.
3. Peak in-process `RssAnon` (`peak_rss_anon_kb`):
   - `guard_trace.rs` samples procfs self status `RssAnon:` at search
     start, every 1024 potential calls (outside `PesClock`), and at
     search end while `outcome` is alive, emitting both `rss_anon_kb`
     and `peak_rss_anon_kb`.
   - The molecular worker wraps `anneal.cluster_search` to sample
     `RssAnon:` at search start, every 256 energy calls, and at search
     end, recording both `rss_anon_kb` and `peak_rss_anon_kb`.
   - The one-sided memory gate (`lower > max(1.10, A2/A upper)`) is
     evaluated on both `rss_anon_kb` and `peak_rss_anon_kb`.
4. Nonzero path counter for B's conf-miss code path:
   - The restacked B payload adds the `ledger` breakdown dict to
     `anneal.cluster_search` in `src/python.rs` (absent on A and A2).
   - The molecular worker records `b_path_fired` (`1` when `ledger` is
     returned with `spent > 0`, `0` otherwise), `b_path_quenches`, and
     `b_path_spent`. `analyze.py` reports the per-cell count and flags
     any molecular cell with zero count as testing timing only.
5. Default-preset parity reporting:
   - `analyze.py` also records `max_abs_dE` from `energy_bits` between
     B and A alongside the bit-level `identical` check.
