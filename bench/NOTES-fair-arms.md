# Fair arms

GMIN on the paper panel was rerun. Anneal was not: its inputs did not
change, and every anneal row below is the stored campaign at
`c2de652dca0bc75e2b0cb4afffac91fd656404e4`, harness
`c2de652dca0bc75e2b0cb4afffac91fd656404e4`, host `bench`. The fair GMIN
rows use driver `GMIN fair` and code revision
`aac1548a8aec13b53ad12a8d6904625081685d2e` (the Wales SVNREV). Rows whose
driver is `GMIN` are the earlier tutorial-file runs and are not scored.

Seeds 2000--2011. A hit is a quenched energy at or below the reference
plus 0.001. A miss is entered at the call budget. Both arms start from
the geometry `coords` writes for that seed. One process, `taskset -c 0`,
`OMP_NUM_THREADS=1`, hostname `bench` under `unshare --uts`.

The ASE call-budget rerun has not replaced a published result. Tables
`tab:ase` and `tab:ase-morse` stay on the wall-capped rows until every
requested seed of both arms has a call-budget row. Those `\rerun{5}`
fences stay.

## What changed in the GMIN arm

The binary is the public tarball
`http://www-wales.ch.cam.ac.uk/svn.tar.bz2`, sha256
`a972c29b081a74235286bf3e8755c8811da2f7e2a4faab81ffb41570d6d2fe05`,
SVNREV `aac1548a8aec13b53ad12a8d6904625081685d2e`, built by
`scripts/gmin_panel/build_gmin.sh` with
`-DWITH_OPENMP=ON -DBLA_VENDOR=OpenBLAS` and ALIGN, MPI, SPARSE, PYTHON,
MYBLAS, and MYLAPACK off. This build links
`/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so`
(gfortran 13.3.0). The binary sha256 is
`73ef457f0b2db3a0196ed19b390ed1c46db7301d48bace14357844c2d377acbc`.
OpenMP is enabled in the build. Each run still uses one thread.

`scripts/gmin_panel/symmetrise-jp.patch` passes the run index into the
three live `TESTSYMOP` calls in `symmetry.f90`. Without it `SYMMETRISE`
indexes `ATMASS` with an undefined run and segfaults. The patch changes
no tolerance and no step.

Each size takes the block in `scripts/gmin_panel/inputs.py`. Sources:

- LJ38: OptBench `gmin-csm`, Stevenson and Wales, 6 Aug 2014. `T=3.0`,
  `NEWRESTART 30`, `AVOID 1.25 18`, `STEP 0.43 0.52`, `SLOPPYCONV 0.002`,
  `PERMDIST 0.005`, `SYMMETRISECSM`.
- LJ55: no OptBench file. Oakley section 4.1, `NEWRESTART 60`. Tolerances
  from the OptBench LJ38 symmetrise block. `T=3.2`, `AVOID 1.25 20`,
  `STEP 0.43 0.52`, `SYMMETRISE`.
- LJ75: OptBench `gmin`, Wales, 9 Jan 2014. `T=3.5`, `NEWRESTART 4`,
  `AVOID 3.8 12`, `STEP 0.43 0.51`, `SYMMETRISE`. Oakley's optimum is
  about 5.
- LJ98: OptBench `gmin-symmetrise`, Wales, 3 Oct 2022. `T=1.3`,
  `NEWRESTART 6`, `STEP 0.46 0.47`, `FIXBOTH`, `PERMDIST 0.01`,
  `SLOPPYCONV 0.007`, `SYMMETRISE`. No `AVOID` line.
- LJ102, LJ103, LJ104: the LJ98 block. Oakley section 5, the core-orbit
  tolerances transfer.
- LJ150: OptBench `gmin-symmetrise`, Wales, 3 Oct 2022. `T=1.0`,
  `NEWRESTART 7`, `STEP 0.415 0.45`, `FIXBOTH`, `PERMDIST 0.01`,
  `SLOPPYCONV 0.007`, `SYMMETRISE`.
- Morse: GMIN `MORSE`, Doye and Wales, Faraday Trans. 93, 4233 (1997).
  `rho=6` uses the Lennard-Jones block of that size. `m80r6` uses the
  LJ55 core-orbit block with `NEWRESTART 75` (Oakley LJ74, section 4.1).
  `rho=14` uses the LJ75 OptBench block. Each block appends `MORSE <rho>`.
- Au55 is the reviewed Gupta block and was not retuned. Radius
  `11.829521` from `io1.f90`, `(2 + (3n/17.77153175)^(1/3)) * r0` with
  `r0=2.884` and `n=55`. `STEP 1.027741 0.51` and `AVOID 10.277408 20 F`
  are the LJ tutorial 0.40 and 4.0 times `r0/2^(1/6)`. `F` leaves
  `AVOIDRESEEDT` false. `TEMPERATURE 0.1` eV. The Darby row is
  `0.2061 10.229 4.036 1.79 2.884`.

The earlier Au55 rows (driver `GMIN`, session `20261117T035457Z`) used
`RADIUS 50`, `TEMPERATURE 1.0`, `STEP 0.40 0.51`, and `NEWRESTART 40`
without `AVOID ... F`. They stay in `rows.csv` and are not scored. The
fair Au55 rows are session `20261117T173638Z`. On this OpenBLAS binary
GMIN hits Au55 on 0 of 12. A different binary, in the review that fixed
the input, hit 1 of 12. This note reports 0 of 12.

Lennard-Jones fair rows are session `20261117T161421Z`. Morse fair rows
are session `20261117T173638Z`. Anneal is session `20261117T035457Z` on
every system, so a paired comparison has two sessions.

## LJ98 campaign check

`scripts/gmin_panel/lj98_check.py` ran seeds 0--63 at a ceiling of
`1e6` calls with the published LJ98 symmetrise block and the panel
coordinate writer. The campaign coordinate blobs are not in this tree.
0 of 64 starts match an OptBench file that is not here. Result: 62 of
64. Misses: seed 10, best `-542.693520123`, wall 42.890 s; seed 37,
best `-543.642901271`, wall 42.784 s. Rows:
`bench/gmin-panel/lj98-campaign-check.csv`, driver
`GMIN fair lj98-campaign`. This is not the historical campaign. The
Supporting Information sentence that the GMIN arm hits 64 of 64 is that
other campaign and is still fenced.

## Hits, old against new

Anneal hits are unchanged. Old GMIN is the tutorial-file driver. New
GMIN is `GMIN fair`. Fisher is two-sided exact. Mann-Whitney is
two-sided, normal approximation, tie correction and continuity
correction. A blank wall Mann-Whitney is not estimated: a miss whose
wall lies inside the other arm's hit walls censors the comparison, and
so does a miss on both arms.

| system | anneal | GMIN old | GMIN new | Fisher old | Fisher new | MW calls old | MW calls new |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 12/12 | 7/12 | 12/12 | 0.0372671 | 1 | 3.38156e-05 | 0.112273 |
| lj55 | 12/12 | 12/12 | 12/12 | 1 | 1 | 0.750832 | 0.402504 |
| lj75 | 12/12 | 0/12 | 12/12 | 7.39602e-07 | 0.000123346 | 1.02689e-05 | 0.000123346 |
| lj98 | 10/12 | 0/12 | 12/12 | 6.73038e-05 | 0.478261 | 0.000125309 | 0.00293945 |
| lj102 | 9/12 | 5/12 | 8/12 | 0.213757 | 1 | 0.0933694 | 0.930134 |
| lj103 | 8/12 | 3/12 | 7/12 | 0.0995327 | 1 | 0.0474499 | 0.656485 |
| lj104 | 9/12 | 6/12 | 5/12 | 0.400323 | 0.213757 | 0.163434 | 0.0632174 |
| lj150 | 5/12 | 9/12 | 12/12 | 0.213757 | 0.00457666 | 0.168122 | 0.000505483 |
| au55 | 6/12 | 0/12 | 0/12 | 0.01373 | 0.01373 | 0.0070541 | 0.0070541 |
| m55r6 | 12/12 | 12/12 | 12/12 | 1 | 1 | 3.65846e-05 | 3.65846e-05 |
| m55r14 | 7/12 | 0/12 | 0/12 | 0.00457666 | 0.00457666 | 0.00285559 | 0.00285559 |
| m80r6 | 2/12 | 12/12 | 9/12 | 6.73038e-05 | 0.0122781 | 3.11691e-05 | 0.00343134 |
| m80r14 | 9/12 | 0/12 | 0/12 | 0.000336519 | 0.000336519 | 0.000383392 | 0.000383392 |

This setting is ahead on lj102 (9 against 8), lj103 (8 against 7),
lj104 (9 against 5), au55 (6 against 0), m55r14 (7 against 0), and
m80r14 (9 against 0). The lead is significant on au55, m55r14, and
m80r14. It is not significant on lj102, lj103, and lj104.

This setting loses on lj98 (10 against 12, Fisher 0.478261), lj150
(5 against 12, Fisher 0.00457666), and m80r6 (2 against 9, Fisher
0.0122781). The ties are lj38, lj55, lj75, and m55r6, 12 against 12.

## Calls and wall, old against new

Medians are Hyndman-Fan type 7. A miss is the full budget. Anneal
medians did not change. Times are seconds. `pot` is inside the
potential and `out` is the rest of the wall. The GMIN columns are the
ones that moved.

| system | calls ann | calls GMIN old | calls GMIN new | wall ann | wall GMIN old | wall GMIN new | pot GMIN old | pot GMIN new | out GMIN old | out GMIN new | MW wall old | MW wall new |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 10615.5 | 389399 | 3388.5 | 0.075305 | 3.657605 | 0.036660 | 2.673021 | 0.019641 | 0.979026 | 0.018304 | 3.65846e-05 | 0.506721 |
| lj55 | 12242 | 12764.5 | 8070.5 | 0.141765 | 0.236408 | 0.121182 | 0.191486 | 0.097257 | 0.044923 | 0.027334 | 0.312321 | 0.930987 |
| lj75 | 1409229.5 | 4000000 | 70388 | 27.018620 | 119.928500 | 3.672120 | 102.850343 | 1.576805 | 16.951687 | 2.059408 | 3.65846e-05 | 0.00294565 |
| lj98 | 1511030.5 | 4000000 | 370350 | 41.585683 | 182.536000 | 16.378500 | 161.915162 | 14.740441 | 20.311312 | 1.638059 | 3.65846e-05 | 0.0261968 |
| lj102 | 894801 | 4000000 | 2764598 | 27.445406 | 194.437000 | 127.002500 | 173.823487 | 114.980929 | 20.613512 | 12.021571 | 0.0120228 |  |
| lj103 | 2293133 | 4000000 | 1929800.5 | 66.187612 | 204.897500 | 91.998450 | 183.075672 | 83.635650 | 21.660352 | 8.395714 | 0.000477682 |  |
| lj104 | 2328723 | 3696118 | 4000000 | 69.305803 | 193.584500 | 191.982500 | 173.577432 | 174.411237 | 20.007068 | 17.506963 | 0.00725995 |  |
| lj150 | 4000000 | 3129518.5 | 123671.5 | 237.227586 | 326.093500 | 11.353210 | 302.138621 | 10.761351 | 23.954880 | 0.591859 | 0.285476 | 0.000505483 |
| au55 | 852736 | 1000000 | 1000000 | 31.522419 | 62.948550 | 60.539700 | 59.823898 | 58.076846 | 3.142188 | 2.462854 | 3.65846e-05 |  |
| m55r6 | 563 | 37047.5 | 24190 | 0.010055 | 0.751413 | 0.469402 | 0.636854 | 0.390811 | 0.114559 | 0.078591 | 3.65846e-05 | 3.65846e-05 |
| m55r14 | 730604 | 1000000 | 1000000 | 13.592808 | 18.898200 | 18.875450 | 16.321503 | 15.784014 | 2.576696 | 3.074779 | 0.0165749 |  |
| m80r6 | 4000000 | 268934 | 1378958 | 168.906001 | 10.690900 | 68.281450 | 9.582060 | 45.984060 | 1.108840 | 22.297390 | 6.00576e-05 |  |
| m80r14 | 1338932.5 | 4000000 | 4000000 | 49.869483 | 152.698000 | 166.789500 | 138.296631 | 134.091386 | 14.429569 | 32.493834 | 0.0165749 |  |

Anneal's median wall is lower on 7 of the 13 clusters and higher on
lj38 (0.075305 against 0.036660), lj55 (0.141765 against 0.121182),
lj75 (27.018620 against 3.672120), lj98 (41.585683 against 16.378500),
lj150 (237.227586 against 11.353210), and m80r6 (168.906001 against
68.281450). On m55r6 the fair GMIN hits fall in 6337 to 48269 calls.
The old range was 4594 to 103829.

## ASE

Basin hopping and minima hopping now stop at the paper call budget.
`cap_s` is empty on a call-budget row. `deadline` is unset. The
recommended arm still uses its stored wall cap. A system's published
rows stay on the wall cap until every system of that table has a
call-budget row for both arms, so a partial rerun does not move
`tab:ase` or `tab:ase-morse`.

The rerun is seeds 2000--2011 of lj38, lj55, lj98, lj104, lj150, and
au55, then m38r6, m38r14, m55r6, and m55r14, both `ase_bh` and `ase_mh`,
revision `61b5c9a01b2002223f9c1e7024863b112001ebf7`, hostname `bench`,
core 0. Anneal, recommended, and for_cluster rows are reused.

LJ38 basin hopping has finished 12 of 12 call-budget seeds and is not
in the published table yet: 0 of 12, every miss charged 400000 calls,
walls 5.327 to 5.756 s. The published row is still 0 of 12 at the wall
cap of 4.654 s.

## Reproduce

GMIN, from a checkout that contains these scripts. The build skips a
rebuild when `/work/gmin-panel/bin/GMIN` is already the binary above.
`enter_bench.sh` itself calls `unshare --uts`, sets the hostname to
`bench`, and pins the search to core 0. It leaves a stored anneal row
in place. It leaves a stored `GMIN fair` row in place. A row whose
driver is `GMIN` does not count, so the same command reruns that seed.

```
bash scripts/gmin_panel/build_gmin.sh
bash scripts/gmin_panel/enter_bench.sh
bash scripts/gmin_panel/enter_bench.sh --summarize --tex
```

The LJ98 check has to run under hostname `bench` and `taskset -c 0`.
Seeds already in the CSV are skipped.

```
sudo unshare --uts sh -c 'hostname bench; exec taskset -c 0 python3 scripts/gmin_panel/lj98_check.py --bin /work/gmin-panel/bin --rows bench/gmin-panel/lj98-campaign-check.csv --svnrev aac1548a8aec13b53ad12a8d6904625081685d2e'
```

ASE, one seed. `enter_bench.sh` itself calls `unshare --uts` and
`taskset -c 0`. For basin hopping and minima hopping it sets
`ASE_PANEL_WALL_CAP` empty, so the arm stops at the call budget. It
reruns a seed whose stored row has a wall cap, and leaves a call-budget
row in place. The Morse script reads `ASE_NONLJ_REVISION`, or `HEAD`
when that is unset; the binaries of this rerun are
`61b5c9a01b2002223f9c1e7024863b112001ebf7`.

```
ASE_PANEL_REVISION=61b5c9a01b2002223f9c1e7024863b112001ebf7 \
  bash scripts/ase_panel/enter_bench.sh --system lj38 --arm ase_bh --seed 2000
ASE_NONLJ_REVISION=61b5c9a01b2002223f9c1e7024863b112001ebf7 \
  bash scripts/ase_nonlj/enter_bench.sh --system m38r6 --arm ase_bh --seed 2000
```

Summarise only after every requested seed has a call-budget row:

```
python3 scripts/ase_panel/summarize.py --all --tex
python3 scripts/ase_nonlj/summarize.py --tex
python3 scripts/paper_audit.py
```
