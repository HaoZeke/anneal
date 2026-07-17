# What is new: Budget-Feasible Window Temperature (BFWT / D11)

## Claim (honest, local)

**New algorithm:** a single Metropolis temperature law that clamps the D6 design
point `T_des = θ⋆·gap/d` (θ⋆=1/2) into the intersection of

- **D6 descent ceiling** `T_hi = 2·gap/d` (θ < 2 for positive expected state gain on the ES-sphere model), and
- **D7 escape floor** `T_lo = b̂ / log(B + e)` (Kramers-style budgeted escape of barrier proxy `b̂` within remaining work units `B`),

with explicit mode tags when the design is raised, capped, or forced to escape because the window is empty.

```text
if T_lo < T_hi:  T = clamp(T_des, T_lo, T_hi)
else:            T = T_lo   # escape_forced
```

Load-bearing identity: window nonempty ⇔ `b̂ · d < 2 · gap · log(B + e)`.

## Not claimed

- Not dual-annealing / GSA (no visiting distribution redesign).
- Not field SOTA on CUTEst or global optimization benchmarks.
- Not that `b̂` is estimated optimally; it is an external proxy input.
- Not a replacement for multi-start / portfolio when the window is empty (escape_forced is a deliberate fallback, not a proof of escape).

## Recoveries / relations

| Case | Behavior |
|------|----------|
| `b̂ → 0` | `T → T_des` (GPMD recovered) |
| Design inside window | `mode=design` |
| Escape floor binds | `mode=escape_floor` |
| Window empty | `mode=escape_forced`, `T = T_lo ≥ T_hi` |

## Evidence paths

| Artifact | Path |
|----------|------|
| Design study | `SCRATCH/paper_repo_study.md` (this goal) / study of D6, D7, SciPy dual_annealing |
| Derive + gates | `proofs/d11_budget_feasible_temp.py` → `D11_DERIVE_OK` |
| Ship | `src/methods/bfwt.rs` (`budget_feasible_temp`, `bfwt_optimize`) |
| Unit tests | `cargo test --lib bfwt` (6 passed on rg.terra) |
| Python | `anneal.bfwt_optimize(..., barrier_hat=...)` |

## Design method (mirrored from paper repos)

1. Read load-bearing identities in published-style proof modules (D6 gain window, D7 budgeted escape).
2. Identify the unfilled composition (ceiling ∩ floor + design interior).
3. Emit a pure law + modes; gates (symbolic + numeric) before shipping the optimizer loop.
4. Ship with recovery of the prior special case (GPMD) as a regression gate.
