# PR 186 restack same-session gate summary

Overall verdict: **PASS** (15/15 cells `PASS`, 0 `PASS-BAND`, 0 `FIX`, 0 `INCONCLUSIVE`, 540/540 runs exit 0).

## Subjects and session

- `A` = `A2` = develop `9b65fda3972fd8c014bce886a3ea5c05956d64af` (tree `a49a66d1422cfefa28012fd12e6729bc4e879d5f`), built independently in separate target directories.
- `B` = `A` + restack payload (`184736e2f596126ef71baca23d0ecdf756aaf77c`); harness-applied B tree `a839b9a99afb155d3ebb0e3e2a42fa086f9c72ba`.
- Schedule: 15 cells (8 molecular, 7 default-preset), `n = 12` seeds (`2000-2011`), arms `A`, `B`, `A2` (`Ap`), palindromic blocks `A, B, A2, A2, B, A` per pair, 540 serial runs pinned to core 0, `*_NUM_THREADS=1`, release builds, no quick mode, no GMIN.
- Bootstrap: 10000 seed-paired resamples, seed `20261123`.

## Gate results by cell

| cell | n | verdict | hits A/B/A2 | p Fisher | p MWU | out/call B/A [lo,hi] | A2/A hi | out total B/A [lo,hi] | end RSS B/A [lo,hi] | peak RSS B/A [lo,hi] | B path | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 12 | PASS | 12/12/12 | 1 | 0.5115 | 0.9448 [0.8445,1.007] | 1.06 | 0.9448 [0.8445,1.007] | 1.005 [0.9999,1.01] | 1.005 [0.9999,1.01] | 12/12 | |
| 915 | 12 | PASS | 8/8/8 | 0.6666 | 0.5115 | 0.9888 [0.9704,1.004] | 1.003 | 0.9888 [0.9704,1.004] | 0.9998 [0.9991,1] | 0.9998 [0.9991,1] | 12/12 | |
| xld | 12 | PASS | 10/10/10 | 0.705 | 0.5115 | 0.9946 [0.9799,1.007] | 1.002 | 0.9946 [0.9799,1.007] | 1 [0.9999,1.001] | 1 [0.9999,1.001] | 12/12 | |
| fluvastatin | 12 | PASS | 6/6/6 | 0.6579 | 0.5115 | 0.999 [0.9805,1.019] | 1.004 | 0.999 [0.9805,1.019] | 0.9965 [0.9911,1] | 0.9965 [0.9911,1] | 12/12 | |
| z34 | 12 | PASS | 1/1/1 | 0.7609 | 0.5115 | 0.9851 [0.951,1.013] | 1.011 | 0.9851 [0.951,1.013] | 1 [0.9999,1.001] | 1 [0.9999,1.001] | 12/12 | |
| amprenavir | 12 | PASS | 1/1/1 | 0.7609 | 0.5115 | 0.9988 [0.9845,1.01] | 1.001 | 0.9988 [0.9845,1.01] | 1.002 [0.9996,1.006] | 1.002 [0.9996,1.006] | 12/12 | |
| lopinavir | 12 | PASS | 1/1/1 | 0.7609 | 0.5115 | 1.011 [0.9914,1.031] | 1.04 | 1.011 [0.9914,1.031] | 1.003 [0.9973,1.009] | 1.003 [0.9973,1.009] | 12/12 | |
| lorlatinib | 12 | PASS | 7/7/7 | 0.6599 | 0.5115 | 0.9933 [0.9751,1.009] | 1.012 | 0.9933 [0.9751,1.009] | 0.9963 [0.9889,1.004] | 0.9963 [0.9889,1.004] | 12/12 | |
| lj38 | 12 | PASS | 12/12/12 | 1 | 0.5115 | 0.9615 [0.8964,1.01] | 1.002 | 0.9615 [0.8964,1.01] | 0.9992 [0.9982,1] | 0.9992 [0.9982,1] | 0/12 | early-stop artifact possible |
| m38r14d | 12 | PASS | 12/12/12 | 1 | 0.5115 | 1.015 [0.9941,1.053] | 1.004 | 1.015 [0.9941,1.053] | 1 [0.9993,1.001] | 1 [0.9993,1.001] | 0/12 | early-stop artifact possible |
| lj75 | 12 | PASS | 5/5/5 | 0.6599 | 0.5129 | 1.012 [0.991,1.035] | 1.034 | 1.012 [0.991,1.035] | 1 [1,1] | 1 [1,1] | 0/12 | early-stop artifact possible |
| lj98 | 12 | PASS | 7/7/7 | 0.6599 | 0.512 | 1.005 [0.9917,1.018] | 1.024 | 1.005 [0.9917,1.018] | 0.9999 [0.9997,1] | 0.9999 [0.9997,1] | 0/12 | early-stop artifact possible |
| m80r6 | 12 | PASS | 12/12/12 | 1 | 0.5115 | 1.01 [0.9919,1.03] | 1.054 | 1.01 [0.9919,1.03] | 0.9999 [0.9992,1.001] | 0.9999 [0.9992,1.001] | 0/12 | early-stop artifact possible |
| au55 | 12 | PASS | 4/4/4 | 0.6666 | 0.5137 | 0.983 [0.9567,1.004] | 1.034 | 0.983 [0.9567,1.004] | 1 [1,1] | 1 [1,1] | 0/12 | early-stop artifact possible |
| hexamer | 12 | PASS | 12/12/12 | 1 | 0.5115 | 0.8382 [0.7419,0.9416] | 0.9138 | 0.8382 [0.7419,0.9416] | 1.001 [0.9973,1.005] | 1.001 [0.9973,1.005] | 0/12 | early-stop artifact possible |

## Gate-by-gate findings

1. **No hit loss (Fisher + MWU, miss = budget + 1)**: **PASS** across all 15 cells. Hit counts match identically across `A`, `B`, and `A2` on every cell (`min p_fisher = 0.6579`, `min p_mwu = 0.5115`).
2. **Outside time (per-call and total)**: **PASS** across all 15 cells. Both strict (`UB <= 1.10`) and band (`UB <= max(1.10, A2/A hi)`) hold on every cell (`max opc_hi = 1.053` on `m38r14d`, `max tot_hi = 1.053`), with zero real regressions (`opc_regression = False`, `tot_regression = False` everywhere). On all 7 default-preset cells, early-stopped runs occur (`8` to `24` runs per cell), and per-call and total ratios agree (`calls_ratio = 1.0` everywhere).
3. **Memory (`rss_anon_kb` and `peak_rss_anon_kb`)**: **PASS** across all 15 cells. Both end-of-search and peak-sampled `RssAnon` ratios have upper CI bounds `<= 1.0101` (strict `<= 1.10` and band `<= max(1.10, A2/A hi)` both satisfied; `rss_fail = False` and `peak_rss_fail = False` everywhere).
4. **B conf-miss path counter**: **PASS** (`12/12` runs nonzero on all 8 molecular cells). Total `quenches` / `spent` recorded on B's `ledger` path per cell: `rq3` `19056` / `591079`, `915` `7740` / `287540`, `xld` `12963` / `508291`, `fluvastatin` `13260` / `472108`, `z34` `16125` / `524659`, `amprenavir` `9820` / `455169`, `lopinavir` `11298` / `542388`, `lorlatinib` `14149` / `500593` (`0/12` on A and A2 as expected). Default-preset guard cells execute `guard_trace` directly (`0/12`).
5. **Default-preset parity (B vs A)**: **PASS** across all 7 default-preset cells (`lj38`, `m38r14d`, `lj75`, `lj98`, `m80r6`, `au55`, `hexamer`). Every seed matches A bit-for-bit on `trajectory`, `energy_bits`, `calls`, and `charged` (`identical = True`, `max_abs_dE = 0.0`).
