# no-size-constants audit and replacement report

## Removed size/system-specific special cases

1. **`Config::recommended_with_scales` LJ-size branch**
   - Removed: `if n_points < 98 { surfaces = [TwoPhase::relative(0.7, beta)] }`
   - Now: compacted-surface arm is installed for all sizes.

2. **`Config::communicating` LJ-size branch**
   - Removed: `if n_points < 98 { surfaces = [TwoPhase::relative(0.7, 1.0)] }`
   - Now: communicating uses the same surface arm at all sizes.

3. **LJ98 shelf guard in growth retry**
   - Removed: `growth_held(..., n_points) => stall_growth_waits && (n_points < 98 || skipped < GROWTH_RETRY_LEAVES)`
   - Now: `growth_held(... ) => stall_growth_waits && skipped < GROWTH_RETRY_LEAVES` for every system.

4. **Motif-gated layer slide/slip trigger**
   - Removed: close-packed trigger keyed on motif fractions (`frac_421 > frac_422`, `frac_421 > frac_555`) and `fivefold_plateau`.
   - Now: trigger uses **general plateau machinery** (`stalled(cfg, quiet, longest_quiet)`) plus **fingerprint diversity** (`occupied_unseen_share <= 0.10`).

5. **Motif-threshold source selection in prior motif reseed**
   - Removed: source inclusion rules keyed on fixed motif thresholds (`frac_421 < 0.20`, `frac_422 < 0.15`, `frac_555 < 0.20`).
   - Now: source inclusion keys on **fingerprint diversity** (`occupied_unseen_share <= 0.10`), independent of motif class thresholds.

6. **Motif-threshold twin proposal in prior motif reseed**
   - Removed: twin branch gated by decahedral motif signature (`frac_422 >= 0.15 && frac_421 < 0.20`).
   - Now: twin proposal gating follows the same **fingerprint diversity** threshold (`occupied_unseen_share <= 0.10`).

## Validation

### Unit tests

- Ran: `cargo +1.88.0 test`
- Result: **pass**

### Quick smoke run (seed 2000, recommended, budget 400000)

Command form:
`SEED_OFFSET=2000 cargo +1.88.0 run --release --features bank-rpc --example lj_cluster_search -- <N> 400000 1 rec`

| System | Before hit | Before calls | After hit | After calls |
|---|---:|---:|---:|---:|
| LJ38 | 1/1 | 400000 | 1/1 | 400000 |
| LJ75 | 1/1 | 400000 | 1/1 | 400000 |
| LJ98 | 0/1 | 400000 | 0/1 | 400000 |

