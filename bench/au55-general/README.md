# Au55 misses and a general plateau response

Au55 (Gupta) was the weakest headline row at 6/12 on 98dcfb31. This
directory holds why its missed seeds miss, the levers screened against
that, the one general change kept (`Config::plateau_heat`), and its
same-session gate against 98dcfb31.

## Files

- `diagnosis-98dcfb31.txt`: `scripts/au55_general/diagnose.py` on the
  hop, adoption and improvement traces (`ANNEAL_HOP_TRACE`,
  `ANNEAL_ACC_TRACE`, `ANNEAL_IMP_TRACE`) of 98dcfb31, Au55 seeds
  2000-2011, misses first.
- `screen-knobs.patch`, `screen-rows.csv`, `screen-summary.csv`: the
  screens. The patch is the environment switches added to 98dcfb31 for
  them and is not part of any commit's code. Rows carry the variant, its
  switches, 98dcfb31, the patch sha1 and `host` `vm-unpinned`: six runs at
  a time on cores 1-6 of the development VM, outside the bench namespace,
  so their clocks are not comparable to the gate's.
- `rows.csv`, `summary.csv`: the gate, appended by
  `scripts/au55_general/run_gate.py` with `code_revision` and `host`.
- `pins.txt`, `pins-default-on.txt`: the pins with the change off and on.

## Why the misses miss

98dcfb31, seeds 2000-2011: hits 2000, 2004, 2005, 2009, 2010, 2011.
Five misses end on the same minimum, -191.00616498 (0.0247 above the
reference), reached at 64k-470k calls; 2001 ends on -191.00270211,
reached at 986k after a run on -190.98.

- The chain is not on the incumbent. Acceptance is 0.7-0.9, about 93% of
  accepted hops stand on an unquenched 25-step screen, and the chain's
  energy sits a median 0.35-0.6 eV above the incumbent; after the final
  record 0.5-2% of the hops stand within 0.05 eV of it (10% of the 346
  in 2001).
- Moves: `dls-kernel` gives almost every record; `single`, `soap`,
  `sym`, `surface`, `burst` and `all` take the rest of the hops.
- Population: one chain. Per window of charged calls it offers about a
  hundred distinct quenched minima and stands on 1500-2000 distinct
  states, so it does not collapse onto one basin.
- Stall gates: the call plateau (5000 calls) fires the climb every
  110-130 hops, 175-196 times a miss, about a sixth of the budget; orbit
  completion runs 230-350 times. No symmetrise, tabu trail or restart.
- Every hit is an incumbent finished by the close-packed layer slide:
  -190.983312 or -191.003676 (-191.007805 in 2010) as a new incumbent,
  then stacking slip and layer slide to the reference (the improvement
  trace; the hop trace ends on the incumbent). A miss reaches
  -191.006165 first, which is lower than the first two; after that, -190.983312 and -191.003676 recur 0-5
  times in the rest of the run and are never an incumbent again, so
  nothing finishes them.

## Screens

Au55 at 1e6 calls; LJ at the paper budgets. Hits over seeds:

| variant | Au55 |
| --- | --- |
| 98dcfb31 | 11/24, 30/48, 35/72 (76/144) |
| temperature x0.1, x0.3 | 1/24, 6/24 |
| quenched trials only may be adopted | 6/24 |
| measured energy scale | 2/24 |
| plateau restart from the lowest / a random one of 8 distinct minima | 7/24, 4/24 |
| bias height x0.04, x0.2, x4 | 11/24, 11/24, 14/24 |
| patience 50000 calls | 13/24 |
| no climb (`escape_on_stall` off) | 11/24, 25/48 |
| temperature x2, x4, x8, x30 | 14/24, 13/24 + 30/48, 14/24, 14/24 |
| hop temperature x4 (Metropolis, exchange, soft mode) and no climb | 54/72, 42/72 (96/144) |
| the same with the climb only off the plateau, backed off after a fruitless climb, retired, or without growth or activation | 38-45/72 |
| `plateau_heat` | 41/72 |

Cooling, quenched-only acceptance, elite restarts and a measured energy
scale lose. A hotter walk with the climb removed gains Au55
(96/144 against 76/144 on seeds 2000-2143), and nothing short of removing
the climb keeps that. Without the climb LJ38 goes from 10.6k to 188k calls
(11/12): its minimum is reached through the climb. That combination is
not a general change.

A hotter walk alone is: it leaves LJ38 and Au55 where they were and
takes LJ75 and LJ98 from about 1.4-1.5M calls to 0.34-0.63M. Heated from
the start it costs LJ55 (12.2k to 14.7k); heated only once a call
plateau passes it leaves LJ38 and LJ55 alone until their first plateau.
That is `plateau_heat`.

## The change

`Config::plateau_heat`: the clock that drives the plateau climb doubles
the hop temperature each time it runs out, up to four times the preset,
and a new lowest minimum returns it to the preset. The factor multiplies
the biased Metropolis test, the exchange kernel's Boltzmann weights and
the thermal soft-mode amplitude. The climb is unchanged. A replica
ladder sets its own rung temperatures and does not heat. With the flag
off every factor is one and the trajectory is the old one bit for bit.
The h2h examples and `h2h_seed` take `H2H_PLATEAU_HEAT=1` or `0`.

## Gate

`snakemake -s workflow/Snakefile au55_general` with
`GATE_HEAT_REV=3c78e32d` and `RGPOT_SC_LIB` (rgpot 8efdd246,
`librgpot.so` sha256 `08815bf0...`): both revisions built in the same
session (cargo 1.88.0, `--locked`), run serially under `sudo unshare
--uts` with hostname bench, `taskset -c 0`, one thread, base then heat
for each seed. Fisher exact on hits, tie-corrected Mann-Whitney on calls
to the minimum (misses at the budget); medians of wall, potential and
outside-potential seconds.

| system | n | hits 98dcfb31 / heat | Fisher p | median calls | MW p | potential s | outside s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Au55 | 24 | 11 / 12 | 1 | 1.00M / 0.96M | 0.68 | 26.5 / 25.6 | 8.9 / 7.9 |
| LJ38 | 12 | 12 / 12 | 1 | 10.6k / 10.6k | 0.98 | 0.027 / 0.027 | 0.042 / 0.043 |
| LJ55 | 12 | 12 / 12 | 1 | 12.2k / 14.1k | 0.91 | 0.056 / 0.064 | 0.075 / 0.085 |
| LJ75 | 12 | 12 / 12 | 1 | 1.41M / 0.67M | 0.069 | 11.9 / 5.7 | 11.8 / 8.2 |
| LJ98 | 12 | 10 / 12 | 0.48 | 1.51M / 0.37M | 0.0016 | 21.6 / 5.2 | 16.0 / 3.7 |
| LJ104 | 12 | 9 / 10 | 1 | 2.33M / 1.01M | 0.47 | 37.7 / 16.4 | 28.9 / 12.0 |
| LJ150 | 12 | 5 / 11 | 0.027 | 4.00M / 1.44M | 0.010 | 134.6 / 48.6 | 89.1 / 34.2 |
| M80 rho=6 | 12 | 2 / 3 | 1 | 4.00M / 4.00M | 0.60 | 105.5 / 105.6 | 53.1 / 53.3 |
| M80 rho=14 | 12 | 9 / 6 | 0.40 | 1.34M / 3.97M | 0.16 | 35.5 / 104.8 | 11.0 / 42.2 |
| sc108-55 | 12 | 6 / 5 | 1 | 0.51M / 1.00M | 0.87 | 34.4 / 68.3 | 2.7 / 5.4 |

The base rows repeat the screen's 98dcfb31 rows seed for seed, and the
heat rows repeat the `plateau_heat` screen rows (the knob in
`screen-knobs.patch`) seed for seed, on every seed both ran.

Per call the potential time moves by under 1% in every row, so its
differences are the call differences. Outside-potential time per call
rises on LJ75 (8.4 to 12.5 microseconds) and M80 rho=14 (9.4 to 11.2) and
moves by under 8% elsewhere.

## Decision

`plateau_heat` loses nowhere significantly, so it is on by default
(the commit after the gate). It does not fix Au55: 12/24 against 11/24,
and 41/72 against 41/72 in the screen. It is a gain on the large
Lennard-Jones clusters (LJ150 hits, LJ98 calls). The largest adverse
move is M80 rho=14, 6/12 against 9/12 (Fisher 0.40, Mann-Whitney 0.16);
that is the row to watch, and `plateau_heat = false` or
`H2H_PLATEAU_HEAT=0` restores the old walk.

What the Au55 screens do show: the misses are the chain committing to
-191.006165 before it meets the minima the layer slide finishes, and the walk
that avoids that is a hotter one without the plateau climb. The climb is
what LJ38 needs, so a general rule that drops it has to know when it is
not paying; backing it off and retiring it after an unproductive climb
both kept it too long for Au55 here.

## Pins

With the change off (`H2H_PLATEAU_HEAT=0` on the default-on commit, and
the flag's default on 3c78e32d), `scripts/covalent_h2h/pins.sh` prints
sha1 `a22093e707faa91d683813fa7cc507706c5265e6` and `h2h_metal run
au55` seeds 2000-2002, clock columns removed, sha1
`b22e212181eff57e9412f780c40e85208b355d07`, the values on 98dcfb31.
With the default on the pins move (`pins.txt`; the full output is
`pins-default-on.txt`), and the Au55 rows of seeds 2000-2002 equal the
gate's heat rows. `tests/lj_trajectory_pin.rs` passes with the default
on; the committed `bench/generic-alloy/g2-pins.txt` and `g3-pins.txt` are
unchanged.

## Checks

On the delivered head: `cargo +1.88.0 fmt --all -- --check`, `cargo
+1.88.0 clippy --all-targets -- -D warnings`, `cargo +1.88.0 test
--release` (2214 passed), `pytest pytest/` and the ChemFit tests and
example on a wheel built from it.
