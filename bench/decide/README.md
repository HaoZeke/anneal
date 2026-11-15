# Hyperband walkers and the discounted move bandit

Three opt-in switches for the cluster hop, measured against develop. None
of them is on by default.

- `hyperband`: `Config::walkers = WalkerSchedule::Hyperband { eta: 3 }`.
  Hyperband brackets of short walkers, with the best third of each rung
  kept by successive halving. A survivor is ranked by its best energy
  minus its gain over the second half of its last segment, and a walker
  that sits in the same minimum as a better one is skipped. The ledger is
  the same one, and every call of every walker is charged.
- `thompson` and `ducb`: `Config::arm_selection = ArmSelection::Thompson`
  or `ArmSelection::DiscountedUcb`. One discounted bandit chooses among
  the move kernels, the angular move, the continuous-symmetry reseed and
  the plateau leave. The reward is the record improvement per charged
  call, and the discount is 0.995. It replaces the reseed patience, the
  Ih gate and the plateau restart gate.

The bench drivers take the switches from `H2H_DECIDE`, as
`Config::apply_decisions` parses it. With the switches off,
`scripts/covalent_h2h/pins.sh` prints the same pins as develop.

## First pass

Seeds 2000 through 2011, the paper budgets, one serial session on core 0
under `unshare --uts` with hostname `bench`. `develop` is built at the
switches' parent `86c3be01`. That commit is develop `ec825179` plus the
row runner and the LJ76, LJ77 and LJ103 rows of the LJ driver's system
table, so its search is develop's. The three switched arms are built at
`c0a558a3`.

Calls are the force calls to the hit, with a miss counted at the full
budget. The median and IQR are type-7 quartiles. Fisher is two-sided
exact on hits. Mann-Whitney is two-sided on calls, with the normal
approximation and tie and continuity corrections. Pot and outside are the
medians of the seconds spent inside the potential and outside it.

| System | Arm | Hits | Fisher p | Median calls | IQR | Mann-Whitney p | Pot (s) | Outside (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ38 | develop | 12/12 | | 10615.5 | 3591.2 | | 0.026 | 0.043 |
| LJ38 | hyperband | 12/12 | 1 | 10084.5 | 4902.2 | 0.75 | 0.025 | 0.034 |
| LJ38 | thompson | 12/12 | 1 | 3066.5 | 3588.2 | 0.00038 | 0.009 | 0.011 |
| LJ38 | ducb | 12/12 | 1 | 1662.0 | 911.0 | 3.7e-05 | 0.007 | 0.007 |
| Au55 | develop | 6/12 | | 852736.0 | 572700.8 | | 22.499 | 7.875 |
| Au55 | hyperband | 9/12 | 0.40 | 585688.5 | 606481.0 | 0.30 | 15.617 | 4.238 |
| Au55 | thompson | 0/12 | 0.014 | 1000000.0 | 0.0 | 0.0071 | 26.877 | 5.408 |
| Au55 | ducb | 0/12 | 0.014 | 1000000.0 | 0.0 | 0.0071 | 29.203 | 5.841 |
| TIP4P 20 | develop | 6/12 | | 1686249.5 | 1376627.5 | | 14.832 | 4.213 |
| TIP4P 20 | hyperband | 2/12 | 0.19 | 2000000.0 | 0.0 | 0.11 | 17.457 | 5.986 |
| TIP4P 20 | thompson | 1/12 | 0.069 | 2000000.0 | 0.0 | 0.048 | 17.522 | 5.535 |
| TIP4P 20 | ducb | 2/12 | 0.19 | 2000000.0 | 0.0 | 0.059 | 18.115 | 5.623 |
| M80 ρ=6 | develop | 2/12 | | 4000000.0 | 0.0 | | 105.001 | 54.791 |
| M80 ρ=6 | hyperband | 2/12 | 1 | 4000000.0 | 0.0 | 0.96 | 104.393 | 49.877 |
| M80 ρ=6 | thompson | 7/12 | 0.089 | 3109186.0 | 3268465.0 | 0.026 | 83.299 | 21.435 |
| M80 ρ=6 | ducb | 8/12 | 0.036 | 726672.0 | 3791876.8 | 0.0075 | 20.052 | 6.814 |
| SC108-55 | develop | 6/12 | | 511393.5 | 986769.8 | | 34.023 | 2.685 |
| SC108-55 | hyperband | 8/12 | 0.68 | 797099.5 | 970755.0 | 0.61 | 54.207 | 4.706 |
| SC108-55 | thompson | 3/12 | 0.40 | 1000000.0 | 180334.5 | 0.27 | 69.642 | 3.340 |
| SC108-55 | ducb | 0/12 | 0.014 | 1000000.0 | 0.0 | 0.0071 | 79.410 | 3.924 |
| LJ75 | develop | 12/12 | | 1409229.5 | 983622.0 | | 11.744 | 11.761 |
| LJ75 | hyperband | 11/12 | 1 | 1361962.0 | 1048705.8 | 0.58 | 11.352 | 10.729 |
| LJ75 | thompson | 8/12 | 0.093 | 3506636.0 | 2029384.0 | 0.040 | 29.591 | 60.592 |
| LJ75 | ducb | 4/12 | 0.0013 | 4000000.0 | 1269711.2 | 0.037 | 38.060 | 83.831 |
| LJ98 | develop | 10/12 | | 1511030.5 | 1985194.5 | | 21.442 | 16.117 |
| LJ98 | hyperband | 10/12 | 1 | 1697108.0 | 1453736.0 | 0.66 | 24.194 | 16.951 |
| LJ98 | thompson | 9/12 | 1 | 903991.5 | 2031530.5 | 0.37 | 12.945 | 6.603 |
| LJ98 | ducb | 7/12 | 0.37 | 3043486.5 | 2881027.0 | 0.36 | 49.855 | 28.100 |
| LJ103 | develop | 8/12 | | 2293133.0 | 2608443.8 | | 36.118 | 24.439 |
| LJ103 | hyperband | 7/12 | 1 | 1235047.0 | 3654810.0 | 0.39 | 19.400 | 14.836 |
| LJ103 | thompson | 4/12 | 0.22 | 4000000.0 | 963054.2 | 0.10 | 63.426 | 33.951 |
| LJ103 | ducb | 0/12 | 0.0013 | 4000000.0 | 0.0 | 0.0011 | 72.295 | 41.159 |

Hit totals over the 96 runs: develop 62, hyperband 61, thompson 44,
ducb 33.

No arm meets the land criterion: no system may lose hits at Fisher
p < 0.05, the hit total has to rise, and no median may rise. Hyperband
loses no system significantly. Its total falls by one, and its median
rises on TIP4P 20, SC108-55 and LJ98. Both bandits find LJ38 in under a
third of develop's calls, and on M80 at ρ=6 they hit 7 and 8 times out of
12 against 2. Both also lose Au55 at Fisher p = 0.014, and D-UCB loses
SC108-55, LJ75 and LJ103 as well. So no arm went on to the full panel
or to 24 seeds, and every switch stays off.

`rows.csv` is appended and never rewritten. Its `baseline` rows were
taken at `c2110583`, a commit on the older develop `65557416`, and are
kept only as provenance. `summary.csv` is `summarise.py` over the table
against `develop`.

    snakemake -s workflow/Snakefile decide_rows decide_summary
