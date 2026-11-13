# Structural priors ablation

`ANNEAL_STRUCTURAL_PRIORS=0` turns off every path that builds a structure from a known packing or chooses a move by classifying one (`Config::structural_priors`). Priors on is the unset default. Each cell below is hits out of six, the median calls to hit with a miss counted as the full budget, the median wall, and the median seconds inside and outside the potential. Medians are Hyndman–Fan type 7. `p` is the two-sided Fisher exact test of the on and off hit counts.

Seeds are 2000–2005, serial, `taskset -c 0`, hostname `bench` under `sudo unshare --uts`, working directory `/work`, at the harness budgets: 4e5 at 38 atoms, 1e6 at 55, 4e6 at 75 and above, and 2e6 for TIP4P. `scripts/priors_ablation/enter_bench.sh` runs the anneal arms and `enter_gmin.sh` runs GMIN where no recorded row existed. `summarize.py` writes `summary.csv` and `table.md` from `rows.csv` and `gmin-rows.csv`.

Each binary scores its own hit within 0.001 of its Cambridge table: `h2h_gmin_proof`, `h2h_morse` (M80 at ρ = 6 is 80B, −378.333471), `h2h_metal`, and `h2h_seed` for TIP4P (−116.5904186160, −305.5183221676 and −628.3722525077 kJ/mol).

The LJ38, LJ75, LJ98 and LJ104 rows, and the single LJ150 `general_only` row, ran at `8bad9cc4`, whose `src/` is `58bc2f83` on develop `3cab6d7e`. Every other row ran at `d341f62a` on develop `51523528`. Rebuilt at `51523528`, the binary reproduces the calls to hit of 19 earlier rows exactly: all twelve LJ38 runs, plus seven LJ75, LJ98 and LJ104 runs.

`general_only` also turns off the n < 98 size gates. The head-to-head binaries quench with their own L-BFGS, so the two-phase surface those gates install is never read. The stall-growth retry rule only acts when stall growth runs, and priors off removes that. On every seed `general_only` gave the same calls as priors off, so it is left out of the table.

GMIN is `scripts/gmin_h2h/build_gmin.sh` at revision `aac1548a`, with the Morse timer, gfortran 13.3.0, and Ubuntu reference BLAS and LAPACK. That build does not time `NEWTIP`, so the TIP4P GMIN inside and outside cells are blank. The TIP4P GMIN arm is the `gmin-control-same-start` setup of `bench/tip4p-h2h`, run from the starts this session's priors-on arm wrote. Where this session did not run GMIN, the GMIN cell is a recorded panel, and so is the recorded priors-on cell (`summary.csv` names each file and commit). `~` marks times computed from per-call medians.

| system | GMIN | anneal priors on | anneal priors off | p (on vs off) | recorded priors on |
|---|---|---|---|---|---|
| lj38 | 4/6 · 389399 · 3.655 s · 2.685 / 0.970 | 6/6 · 10190 · 0.093 s · 0.035 / 0.059 | 6/6 · 192322 · 1.340 s · 0.604 / 0.736 | 1.000 | 6/6 · 10190 |
| lj75 | 0/6 · 4000000 · 101.314 s · 86.352 / 14.950 | 6/6 · 979290 · 27.420 s · 11.735 / 15.685 | 2/6 · 4000000 · 88.363 s · 47.075 / 41.288 | 0.061 | 6/6 · 979290 |
| lj98 | 0/6 · 4000000 · 166.002 s · 147.430 / 18.591 | 5/6 · 1387168 · 58.210 s · 25.363 / 32.847 | 4/6 · 2520676 · 85.859 s · 46.506 / 39.352 | 1.000 | 5/6 · 1387168 |
| lj104 | 2/6 · 4000000 · 185.040 s · 165.317 / 19.674 | 6/6 · 1812258 · 79.080 s · 38.674 / 40.405 | 2/6 · 4000000 · 148.684 s · 83.540 / 64.051 | 0.061 | 6/6 · 1812258 |
| lj150 | 4/6 · 3651960 · 337.591 s · 312.353 / 25.238 | 1/6 · 4000000 · 338.878 s · 173.902 / 164.975 | 2/6 · 4000000 · 315.689 s · 172.654 / 137.244 | 1.000 | 1/6 · 4000000 |
| m55r6 | 6/6 · 23758 · 0.406 s · 0.361 / 0.045 | 6/6 · 233 · 0.004 s · 0.004 / 0.001 | 6/6 · 16064 · 0.356 s · 0.253 / 0.103 | 1.000 | 6/6 · 232 |
| m80r6 | 6/6 · 193538 · 7.895 s · 7.080 / 0.815 | 6/6 · 748774 · 17.343 s · 10.213 / 7.538 | 6/6 · 91487 · 3.796 s · 2.850 / 0.946 | 1.000 | no row |
| m55r14 | 0/6 · 1000000 · 16.923 s · 15.239 / 1.656 | 4/6 · 537181 · 12.726 s · 7.787 / 4.939 | 5/6 · 271130 · 5.572 s · 3.874 / 1.790 | 1.000 | 5/6 · 446461 |
| m80r14 | 0/6 · 4000000 · 136.620 s · 128.280 / 8.356 | 2/6 · 4000000 · 192.176 s · 124.729 / 66.787 | 3/6 · 3986686 · 186.423 s · 122.262 / 64.161 | 1.000 | 4/6 · 2387370 |
| au55 | 0/6 · 1000000 · 59.565 s · ~56.656 / ~2.910 | 3/6 · 919460 · 45.738 s · 32.550 / 13.515 | 0/6 · 1000000 · 45.769 s · 34.633 / 11.083 | 0.182 | 0/6 · 1000000 |
| cu6au32 | 0/6 · 400000 · 12.228 s · ~11.330 / ~0.903 | 6/6 · 400000 · 7.218 s · 6.692 / 0.525 | 0/6 · 400000 · 8.200 s · 6.770 / 1.482 | 0.002 | 6/6 · 400000 |
| cu27au28 | 0/6 · 1000000 · 60.735 s · ~57.790 / ~2.946 | 6/6 · 1000000 · 39.406 s · 37.337 / 2.065 | 0/6 · 1000000 · 42.990 s · 37.300 / 6.198 | 0.002 | 6/6 · 1000000 |
| tip4p4 | 6/6 · 210 · 0.001 s · – / – | 6/6 · 278 · 0.001 s · 0.000 / 0.000 | 6/6 · 278 · 0.001 s · 0.000 / 0.000 | 1.000 | 6/6 · 278 |
| tip4p8 | 5/6 · 240163 · 1.999 s · – / – | 6/6 · 6526 · 0.027 s · 0.016 / 0.011 | 6/6 · 6526 · 0.025 s · 0.015 / 0.010 | 1.000 | 6/6 · 6526 |
| tip4p15 | 0/6 · 2000000 · 50.017 s · – / – | 6/6 · 1158631 · 9.453 s · 7.207 / 2.246 | 6/6 · 1158631 · 9.795 s · 7.442 / 2.352 | 1.000 | 6/6 · 1158631 |

On Cu6Au32 and Cu27Au28 the hitting energy is not among the walk's recorded improvements, so the ledger never stops below the reference and `h2h_metal` reports the full charge as the calls to hit, as in the recorded spine and Mackay rows. At ρ = 14 this session's priors-on rows match the `770121ac` baseline on M55 (4/6, 537181) and M80 (2/6). The recorded 5/6 and 4/6 are the `c1fe8188` panel.
