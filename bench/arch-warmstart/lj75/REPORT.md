# LJ75 Warm-Start vs Cold-Start Report

## Environment and Provenance

- Base revision: `3ccb2ad4fb930270f0fe95568ff09a5d0127d81b` (`develop`)
- Source invariant: `git diff -- src/` is empty.
- Hardware: `8` logical cores (`nproc = 8`, `Intel(R) Xeon(R) Processor`), pinned to core `0` via `taskset -c 0`.
- Total oracle-call budget: `4_000_000` (`PAPER_BUDGETS["lj75"]` in `scripts/paper_0120/run_panel.py:47`).
- Target threshold: `-397.492331 + 1.0e-3 = -397.491331` (`scripts/paper_0120/run_panel.py:57-58`).
- Configuration: single-chain `Config::communicating(75)` via `cluster_hopping::run_with_gradient`.

## Corpus Construction (`seeds 1000..=1011`)

- Runs: `12` single-chain runs at full `4_000_000` budget (no early stop).
- Corpus hit rate: `8/12` (`0.667`, 95% CI `[0.417, 0.917]`), median calls-to-hit `2,340,492` (`[324,347, 4,000,001]`), median full-budget wall `89.201 s`, median peak `RssAnon` `232,348 KiB`.
- Stored minima: `12` frames recorded via `MinimaCorpus::record` (`1` final `best_state` per corpus seed, as `cluster_hopping::run_with_gradient` leaves `Ledger::quench_boundaries()` empty at base revision): `8` at `-397.4923309829`, `2` at `-396.2822488256`, and `2` at `-396.2385122152`.
- Corpus size: `372,864` bytes total (`data.mdb` `339,968` bytes `<= 5 MB`, committed under `bench/arch-warmstart/lj75/corpus/`).
- `data.mdb` SHA-256: `e1f8d74d760f81fbb2906253b6c1983056bc840b6d8c1a803da8550e745623d1`.

## Paired Evaluation (`cold` vs `warm`, `seeds 2000..=2011`)

Evaluation runs used `ledger.stop_below(-397.491331)` and ABBA seed ordering (`i % 4 in {0, 3}` ran `cold` then `warm`; `i % 4 in {1, 2}` ran `warm` then `cold`). `warm` initialized every seed from the lowest stored sub-target minimum in `MinimaCorpus` (`warm_seed_energy = -396.2822488256`). Misses are coded as `4_000_001` in `calls_to_hit`. Confidence intervals use 10,000 percentile bootstrap resamples (PRNG seed `20261123`).

| Arm | Hits | Hit Rate [95% CI] | Calls-to-Hit Med [95% CI] | Wall-to-Hit Med s [95% CI] | Wall Med s [95% CI] | CPU Med s [95% CI] | Peak RssAnon Med KiB |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| `cold` | 10/12 | 0.833 [0.583, 1.000] | 2,852,494 [1,293,692, 3,731,920] | 59.179 [12.686, 81.114] | 65.671 [32.619, 90.985] | 65.66 [32.61, 90.97] | 165,736 |
| `warm` | 7/12 | 0.583 [0.333, 0.833] | 3,333,865 [1,815,420, 4,000,001] | 52.308 [35.740, 73.890] | 79.059 [46.193, 98.052] | 79.04 [46.18, 98.03] | 182,066 |

### Hypothesis Tests

- Paired contingency (`warm` vs `cold`): `both = 6`, `warm_only = 1`, `cold_only = 4`, `neither = 1`.
- Two-sided Fisher's exact test (hit rate): `p = 0.370709`.
- Two-sided exact McNemar test (paired hits): `p = 0.375`.
- Two-sided tie-corrected Mann-Whitney U test (`calls_to_hit`): `U = 87.0`, `p = 0.396695`.

## Interpretation

Seeding from the lowest sub-target minimum (`-396.2822488256`, the deep competing non-global funnel where all non-hitting runs stall) did not improve over `cold` random-cluster starts (`7/12` vs `10/12` hits; median `3.33M` vs `2.85M` oracle calls; all tests non-significant at `n = 12`). On LJ75's double-funnel landscape, initializing directly inside a deep competing basin requires overcoming the inter-funnel barrier before reaching `-397.492331`, whereas cold starts sample multiple funnels before relaxing into a deep trap.
