# The floored move bandit against develop

`floor` is `H2H_DECIDE=ducb,floor`: discounted UCB over the move kernels,
the angular relocation, the symmetry reseed and the plateau leave, held
to develop's schedule. Develop's reseed, leave and angular gates still
fire on their own clocks, and every kernel keeps half the draws develop's
allocator would give it. The bandit allocates the remaining draws. It is
off by default and does not land.

## Gate

One serial session on core 0, under `unshare --uts` with hostname
`bench`, at the paper budgets. Seeds 2000 through 2011 on every panel
system. Seeds 2012 through 2023 were added on Au55, M80 at ρ=6, SC108-55,
LJ75 and LJ103. `develop` is built at `f44e14e3`, which is develop
`91f835fe` plus the row runner and the LJ76, LJ77 and LJ103 rows of the
LJ driver's system table, so its search is develop's. `floor` is built
at `47afa4c0`.

Calls are the force calls to the hit, with a miss counted at the full
budget. The median and IQR are type-7 quartiles. Fisher is two-sided
exact on hits. Mann-Whitney is two-sided on calls, with the normal
approximation and tie and continuity corrections. Pot and outside are
the medians of the seconds spent inside the potential and outside it.

    python3 scripts/decide_bench/summarize.py --revision f44e14e3 --revision 47afa4c0

| System | n | develop hits | floor hits | Fisher p | develop calls | floor calls | floor IQR | Mann-Whitney p | develop pot / outside (s) | floor pot / outside (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ38 | 12 | 12 | 12 | 1 | 10615.5 | 1874.0 | 1218.8 | 3.7e-05 | 0.026 / 0.043 | 0.008 / 0.008 |
| LJ55 | 12 | 12 | 12 | 1 | 12242.0 | 15591.5 | 24936.5 | 0.80 | 0.056 / 0.074 | 0.082 / 0.080 |
| Au55 | 24 | 11 | 2 | 0.0078 | 1000000.0 | 1000000.0 | 0.0 | 0.0043 | 26.355 / 9.028 | 30.377 / 6.173 |
| TIP4P 15 | 12 | 7 | 9 | 0.67 | 1481619.5 | 795687.5 | 1206282.0 | 0.26 | 7.913 / 2.799 | 4.744 / 1.590 |
| TIP4P 20 | 12 | 6 | 4 | 0.68 | 1686249.5 | 2000000.0 | 823173.8 | 0.50 | 14.924 / 4.205 | 19.356 / 5.825 |
| M80 ρ=3 | 12 | 12 | 12 | 1 | 761438.5 | 232260.5 | 156523.5 | 0.0020 | 20.068 / 8.730 | 7.126 / 1.556 |
| M80 ρ=6 | 24 | 5 | 7 | 0.74 | 4000000.0 | 4000000.0 | 228257.0 | 0.64 | 110.331 / 57.670 | 121.498 / 33.725 |
| SC108-55 | 24 | 9 | 1 | 0.010 | 1000000.0 | 1000000.0 | 0.0 | 0.0075 | 70.756 / 6.195 | 80.512 / 4.098 |
| SC96-80 | 12 | 7 | 0 | 0.0046 | 981181.0 | 1000000.0 | 0.0 | 0.0029 | 153.579 / 12.862 | 152.627 / 6.729 |
| SC108-80 | 12 | 0 | 3 | 0.22 | 1000000.0 | 1000000.0 | 244863.0 | 0.079 | 154.216 / 13.597 | 154.654 / 6.098 |
| SC126-80 | 12 | 6 | 0 | 0.014 | 828059.5 | 1000000.0 | 0.0 | 0.0071 | 122.429 / 11.283 | 146.184 / 6.414 |
| LJ75 | 24 | 24 | 5 | 7.4e-09 | 1323864.0 | 4000000.0 | 0.0 | 1.4e-06 | 12.672 / 13.346 | 37.399 / 101.352 |
| LJ76 | 12 | 12 | 7 | 0.037 | 980823.0 | 2810075.0 | 2473630.2 | 0.022 | 9.795 / 9.029 | 26.494 / 54.929 |
| LJ77 | 12 | 11 | 1 | 0.00011 | 1251915.5 | 4000000.0 | 0.0 | 0.00044 | 12.517 / 12.096 | 40.202 / 68.910 |
| LJ98 | 12 | 10 | 7 | 0.37 | 1511030.5 | 3229376.0 | 2506992.8 | 0.25 | 24.668 / 18.583 | 53.883 / 30.500 |
| LJ102 | 12 | 9 | 2 | 0.012 | 894801.0 | 4000000.0 | 0.0 | 0.0062 | 15.583 / 12.179 | 72.052 / 39.775 |
| LJ103 | 24 | 15 | 7 | 0.041 | 2478125.0 | 4000000.0 | 1994630.5 | 0.13 | 44.094 / 31.171 | 74.037 / 39.405 |
| LJ104 | 12 | 9 | 2 | 0.012 | 2328723.0 | 4000000.0 | 0.0 | 0.0062 | 41.735 / 31.930 | 74.489 / 40.048 |
| LJ150 | 12 | 5 | 12 | 0.0046 | 4000000.0 | 602280.0 | 2209836.8 | 0.0038 | 142.662 / 97.393 | 23.752 / 9.557 |

Hit totals over the 288 runs: develop 182 and floor 105.

`floor` does not pass the gate. At 24 seeds the M80 ρ=6 gain of the
unfloored bandit is gone (7 against 5, p = 0.74), and Au55 drops from 11
to 2 (p = 0.0078). Ten systems lose hits at p < 0.05: Au55, SC108-55,
SC96-80, SC126-80, LJ75, LJ76, LJ77, LJ102, LJ103 and LJ104. It wins
LJ150 (12 against 5) and reaches LJ38 and M80 at ρ=3 in a fifth and a
third of develop's calls. Keeping develop's gates and half its kernel
draws does not keep develop's hit rate, so the floor does not protect the
rare moves the bandit starves.

`rows.csv` is appended and never rewritten. It also holds the earlier
sessions: `baseline` at `c2110583` on develop `65557416`; `develop`,
`hyperband`, `thompson` and `ducb` at `86c3be01` and `c0a558a3` on develop
`ec825179`; and an unfinished `develop` first pass at `1d74bd64` on develop
`fde4f076`. Those rows are provenance only. `summary.csv` is the command
above.

    snakemake -s workflow/Snakefile decide_gate decide_summary --config decide_revisions="f44e14e3 47afa4c0"
