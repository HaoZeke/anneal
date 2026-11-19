# Follow-up gate `gmin-losses2`: LJ75 root cause and pair-scaled candidate

This directory records the follow-up gate (`fix2`) on top of
`bench/gmin-losses/`, comparing the same template-ablated reviewed base
`628773d6` (patched tree `a9cef902826cc328e28dd77d2fd95689686c4697`) against
the non-regressing `fix2` arm on all 13 paper systems (seeds 2000--2011,
`taskset -c 0`, `unshare --uts` with hostname `bench`).

## Root cause of LJ75 seeds 2000 and 2001 losses in `gmin-losses`

Diagnostic runs of `baseline`, `fix`, `heat+adapt` (`budgeted_population=0`),
`heat_only`, and `adapt_only` on LJ75 seeds 2000 and 2001 identified two
interacting causes:

1. **Up-front 50/50 budget truncation (`run_budgeted_population`)**:
   `fix` ran the cold baseline configuration for `first_budget = 2,000,000`
   calls and discarded both its coordinates and its `BasinBias` to restart
   from `start` with `adaptive_cfg`.
   - **Seed 2000**: `baseline` and `fix` are bit-identical through 3,222 full
     quenches and 22,559 accepted transitions up to hop 41,842 (`2,000,000`
     calls, incumbent `-396.19164886` reached at hop 35,613, call 1,714,681).
     They first diverge at transition 22,559 / hop 41,842 when `first_ledger`
     stops `cold_cfg`. `baseline` continues from that state and hits
     `-397.49233098` at hop 70,613 (call 3,309,546).
   - **Seed 2001**: `baseline` and `fix` are bit-identical through 2,819 full
     quenches and 23,267 accepted transitions up to hop 43,550 (`2,000,000`
     calls, incumbent `-396.19164886` reached at hop 32,388, call 1,490,429).
     They first diverge at quench 2,819 / hop 43,550 (`2,000,000` calls, where
     `first_ledger` interrupts the quench at `-387.156780` versus baseline's
     `-390.229633` at call 2,000,201). `baseline` continues and hits
     `-397.49233098` at hop 59,013 (call 2,700,862).
   The same 2M cutoff also caused all other late baseline losses in
   `gmin-losses` (LJ75 2005/2007/2011, LJ98 2000/2005, LJ103 2002/2004,
   Au55 2000/2001, and M80 rho=6 seed 2000 at 3,397,404).

2. **Adversarial feedback between `plateau_heat` and `AcceptDial`**:
   Even with `budgeted_population=0` (full 4M calls), `heat+adapt` still
   missed both LJ75 seed 2000 (`-396.19164886`, first diverging at quench 87 /
   call 33,175) and seed 2001 (`-395.47274521`, first diverging at quench 59 /
   call 20,639). `plateau_heat` raised `temperature = dial.temperature() * heat`
   up to `4x`, pushing acceptance above 0.75; `AcceptDial` observed those heated
   acceptances, clamped `dial.temperature()` down to `0.25x`, and inflated
   `acceptance_step = (cfg.temperature / dial.temperature()).sqrt()` to `2.0x`.
   With `plateau_heat=0`, `adapt_only` hit both seed 2000 (2,202,662 calls) and
   seed 2001 (1,759,788 calls) and tied LJ75 at 9/12, though unconditional
   40-hop retuning still lost seeds on LJ75 (2007/2009), LJ98, LJ103, and Au55.

## Fix in `gmin-losses2`

1. Guard `dial.observe(accept)` so `AcceptDial` only adapts on unheated hops
   (`heat <= 1.0 + 1e-12`).
2. `fix2` enables `Config::recommended_for_pair` (`H2H_PAIR_SCALE=1`) alone with
   `H2H_PLATEAU_HEAT=0`, `H2H_ACCEPTANCE_ADAPT=0`, and `H2H_POPULATION=0`. This
   preserves every Lennard-Jones and Gupta trajectory (including LJ75 9/12)
   bit-for-bit while giving Morse its pair-minimum length scale.
