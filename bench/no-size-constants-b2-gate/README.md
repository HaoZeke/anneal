# No-size-constants-b2 regression gate

Paired head-to-head comparison of Arm A (`bc16d35cf13e71c2edaf343b80fd8c41368631d4`, `develop` baseline) against Arm B (`0cc5db69373c4790f828c18dc2dfe75faa3db79d`, `bc16d35c` plus the five `no-size-constants-b2` patches, tree `8ec90c12015ea780662cbe27409c62d04d952bbb`).

## Protocol

- **Recommended/default presets and paper budgets**:
  - **LJ** (`lj38` at `400_000`; `lj75`, `lj98`, `lj103`, `lj104` at `4_000_000`): `h2h_seed` (`cluster_search::search_from`, `Config::recommended`), hit tolerance `0.001`.
  - **Binary-LJ**: `unsupported` (`bc16d35c` has no binary-LJ `bench/` panel or inputs).
  - **Morse** (`m75r3`, `m75r6`, `m75r10`, `m75r14`, `m80r3`, `m80r6`, `m80r10`, `m80r14` at `4_000_000`): `h2h_seed` (`cluster_search::search_from`, `Config::recommended`), existing range values `rho in {3, 6, 10, 14}` and Cambridge references from `examples/fixtures/morse_ccd.csv`, hit tolerance `0.001`.
  - **TIP4P** (`tip4p8`, `tip4p12`, `tip4p15`, `tip4p16`, `tip4p20`, `tip4p21`, `tip4p25` at `2_000_000`): `h2h_seed` (`Config::recommended_tip4p`), Cambridge headers from `data/tip4p_ccd.csv` for `8..21` and pooled session minimum for `25`, hit tolerance `0.001` kJ/mol.
  - **Surface/slab** (`cuh2_fcc_slab` at `2_500`, `cuh2_fcc_slab_h6` at `5_000`): `slab_adsorption` (`recommended` arm) against `rgpot` CuH2 EAM, pooled minimum scoring (`0.001` eV / `0.0001` eV).
  - **Conformer set** (`rq3`, `915`, `xld`, `fluvastatin`, `z34`, `amprenavir`, `lopinavir`, `lorlatinib` from `bench/openconf-h2h/molecules.csv` and `budgets.csv`): `anneal.cluster_search(..., recommended=True)` on RDKit MMFF94, call cap `2 * anneal_budget`, scored via MMFF94 polish and heavy-atom RMSD (`<= 0.25` A, `|dE| <= 0.5` kcal/mol) against the pooled lowest valid conformer per molecule.
- **Seeds & execution**: `n = 12` (`2000..2011`) on every supported system, one serial session inside `unshare --uts` (`host = bench`), `taskset -c 0`, single-threaded (`OMP`/`OPENBLAS`/`MKL`/`RAYON` threads = `1`), fresh process per run, arms interleaved per seed with order flipped each seed (`A, B` on even seeds; `B, A` on odd seeds), append-only `rows.csv`.
- **Re-summarizing**: `snakemake -s bench/no-size-constants-b2-gate/Snakefile --cores 1`

## Verdict: `B LOSES`

Three systems show statistically significant regressions (`p < 0.05`):

- `tip4p12`: median calls increased from `41418.5` (A) to `161734.0` (B), Mann-Whitney `U = 112.0`, `p = 0.0225761` (hits `12/12 -> 12/12`, Fisher `p = 1`).
- `tip4p20`: hits dropped from `6/12` (A) to `0/12` (B), Fisher `p = 0.01373`; median calls increased from `1686249.5` (A) to `2000000.0` (B), Mann-Whitney `U = 108.0`, `p = 0.0070541`.
- `rq3`: hits dropped from `12/12` (A) to `4/12` (B), Fisher `p = 0.00134608`; median calls increased from `12738.0` (A) to `56914.0` (B), Mann-Whitney `U = 111.5`, `p = 0.0217639`.

### Systems with any hit drop or call increase (including non-significant)

- `lj38`: hits `12/12 -> 12/12` (`p = 1`), median calls `6357.0 -> 12924.5` (`p = 0.272552`).
- `lj98`: hits `10/12 -> 12/12` (`p = 0.478261`), median calls `1321410.0 -> 1306968.5` (rank sum `U = 76.0 > 72`, `p = 0.839826`).
- `m75r6`: hits `12/12 -> 12/12` (`p = 1`), median calls `39942.5 -> 44315.0` (`U = 61.0`, `p = 0.54437`).
- `m75r10`: hits `12/12 -> 12/12` (`p = 1`), median calls `34286.5 -> 39462.0` (`p = 0.285476`).
- `m75r14`: hits `12/12 -> 12/12` (`p = 1`), median calls `36120.5 -> 36236.0` (`p = 0.340779`).
- `tip4p12`: hits `12/12 -> 12/12` (`p = 1`), median calls `41418.5 -> 161734.0` (`p = 0.0225761`, significant).
- `tip4p20`: hits `6/12 -> 0/12` (`p = 0.01373`, significant), median calls `1686249.5 -> 2000000.0` (`p = 0.0070541`, significant).
- `tip4p25`: hits `1/12 -> 0/12` (`p = 1`), median calls `2000000.0 -> 2000000.0` (`U = 78.0`, `p = 0.359317`).
- `rq3`: hits `12/12 -> 4/12` (`p = 0.00134608`, significant), median calls `12738.0 -> 56914.0` (`p = 0.0217639`, significant).
- `915`: hits `7/12 -> 4/12` (`p = 0.413649`), median calls `20330.0 -> 25776.0` (`p = 0.244349`).
- `xld`: hits `10/12 -> 5/12` (`p = 0.0893795`), median calls `19412.5 -> 44988.0` (`p = 0.0802164`).
- `fluvastatin`: hits `5/12 -> 2/12` (`p = 0.370709`), median calls `43258.0 -> 43258.0` (`U = 92.0`, `p = 0.161038`).
- `z34`: hits `1/12 -> 0/12` (`p = 1`), median calls `46026.0 -> 46026.0` (`U = 78.0`, `p = 0.359317`).
- `amprenavir`: hits `1/12 -> 0/12` (`p = 1`), median calls `39154.0 -> 39154.0` (`U = 78.0`, `p = 0.359317`).
- `lorlatinib`: hits `4/12 -> 1/12` (`p = 0.31677`), median calls `43676.0 -> 43676.0` (`U = 88.0`, `p = 0.207632`).

## Per-system summary

| Panel | System | Status | Hits A | Hits B | Fisher p | Med Calls A | Med Calls B | MW p | In (s) A / B | Out (s) A / B | Bit-ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj | lj38 | measured | 12/12 | 12/12 | 1 | 6357.0 | 12924.5 | 0.272552 | 0.015 / 0.032 | 0.068 / 0.100 | 1/12 |
| lj | lj75 | measured | 12/12 | 12/12 | 1 | 44176.0 | 22986.5 | 0.340779 | 0.417 / 0.204 | 0.637 / 0.471 | 0/12 |
| lj | lj98 | measured | 10/12 | 12/12 | 0.478261 | 1321410.0 | 1306968.5 | 0.839826 | 18.900 / 18.626 | 17.970 / 18.135 | 0/12 |
| lj | lj103 | measured | 8/12 | 12/12 | 0.0931677 | 2581470.5 | 43484.5 | 0.000189942 | 40.946 / 0.682 | 42.711 / 0.828 | 0/12 |
| lj | lj104 | measured | 9/12 | 12/12 | 0.217391 | 1136745.5 | 82809.0 | 0.000243152 | 18.395 / 1.324 | 19.549 / 1.424 | 0/12 |
| binary-lj | binary_lj | unsupported | - | - | - | - | - | - | - | - | - |
| morse | m75r3 | measured | 11/12 | 12/12 | 1 | 767869.5 | 303229.5 | 0.236584 | 17.965 / 7.062 | 7.280 / 3.101 | 0/12 |
| morse | m75r6 | measured | 12/12 | 12/12 | 1 | 39942.5 | 44315.0 | 0.54437 | 0.930 / 1.032 | 0.655 / 0.610 | 0/12 |
| morse | m75r10 | measured | 12/12 | 12/12 | 1 | 34286.5 | 39462.0 | 0.285476 | 0.798 / 0.919 | 0.544 / 0.664 | 0/12 |
| morse | m75r14 | measured | 12/12 | 12/12 | 1 | 36120.5 | 36236.0 | 0.340779 | 0.841 / 0.844 | 0.562 / 0.590 | 0/12 |
| morse | m80r3 | measured | 9/12 | 12/12 | 0.217391 | 1897020.0 | 1044449.0 | 0.402094 | 50.268 / 27.838 | 20.285 / 12.315 | 0/12 |
| morse | m80r6 | measured | 12/12 | 12/12 | 1 | 238328.0 | 57880.5 | 0.0781226 | 6.318 / 1.538 | 2.906 / 0.886 | 2/12 |
| morse | m80r10 | measured | 3/12 | 6/12 | 0.400323 | 4000000.0 | 3813261.5 | 0.219445 | 104.575 / 99.401 | 50.601 / 61.828 | 0/12 |
| morse | m80r14 | measured | 8/12 | 10/12 | 0.640405 | 3477864.0 | 2197559.0 | 0.154043 | 92.295 / 58.279 | 43.856 / 27.445 | 0/12 |
| tip4p | tip4p8 | measured | 12/12 | 12/12 | 1 | 6339.0 | 4272.5 | 0.298592 | 0.012 / 0.008 | 0.008 / 0.005 | 1/12 |
| tip4p | tip4p12 | measured | 12/12 | 12/12 | 1 | 41418.5 | 161734.0 | 0.0225761 | 0.152 / 0.603 | 0.072 / 0.289 | 0/12 |
| tip4p | tip4p15 | measured | 7/12 | 9/12 | 0.666846 | 1481619.5 | 829988.0 | 0.556404 | 7.895 / 4.543 | 2.766 / 1.767 | 0/12 |
| tip4p | tip4p16 | measured | 2/12 | 4/12 | 0.640405 | 2000000.0 | 2000000.0 | 0.569211 | 11.972 / 11.913 | 4.327 / 3.700 | 0/12 |
| tip4p | tip4p20 | measured | 6/12 | 0/12 | 0.01373 | 1686249.5 | 2000000.0 | 0.0070541 | 14.895 / 17.830 | 4.180 / 5.528 | 0/12 |
| tip4p | tip4p21 | measured | 0/12 | 0/12 | 1 | 2000000.0 | 2000000.0 | 1 | 19.357 / 19.410 | 6.052 / 6.088 | 0/12 |
| tip4p | tip4p25 | measured | 1/12 | 0/12 | 1 | 2000000.0 | 2000000.0 | 0.359317 | 28.470 / 28.692 | 8.898 / 8.838 | 0/12 |
| slab | cuh2_fcc_slab | measured | 0/12 | 0/12 | 1 | 2500.0 | 2500.0 | 1 | 1.466 / 1.440 | 130.675 / 129.886 | 12/12 |
| slab | cuh2_fcc_slab_h6 | measured | 0/12 | 0/12 | 1 | 5000.0 | 5000.0 | 1 | 2.130 / 2.087 | 352.463 / 349.726 | 12/12 |
| conformer | rq3 | measured | 12/12 | 4/12 | 0.00134608 | 12738.0 | 56914.0 | 0.0217639 | 1.310 / 1.317 | 1.016 / 3.183 | 0/12 |
| conformer | 915 | measured | 7/12 | 4/12 | 0.413649 | 20330.0 | 25776.0 | 0.244349 | 0.488 / 0.490 | 0.421 / 2.237 | 0/12 |
| conformer | xld | measured | 10/12 | 5/12 | 0.0893795 | 19412.5 | 44988.0 | 0.0802164 | 1.330 / 1.319 | 0.991 / 4.684 | 0/12 |
| conformer | fluvastatin | measured | 5/12 | 2/12 | 0.370709 | 43258.0 | 43258.0 | 0.161038 | 1.291 / 1.283 | 0.964 / 3.545 | 0/12 |
| conformer | z34 | measured | 1/12 | 0/12 | 1 | 46026.0 | 46026.0 | 0.359317 | 1.597 / 1.590 | 1.188 / 2.452 | 0/12 |
| conformer | amprenavir | measured | 1/12 | 0/12 | 1 | 39154.0 | 39154.0 | 0.359317 | 1.586 / 1.585 | 1.234 / 6.697 | 0/12 |
| conformer | lopinavir | measured | 0/12 | 1/12 | 1 | 44336.0 | 44336.0 | 0.359317 | 2.893 / 2.903 | 2.309 / 4.545 | 0/12 |
| conformer | lorlatinib | measured | 4/12 | 1/12 | 0.31677 | 43676.0 | 43676.0 | 0.207632 | 1.054 / 1.050 | 0.828 / 2.852 | 0/12 |

On both `cuh2_fcc_slab` and `cuh2_fcc_slab_h6`, Arm A (`bc16d35c`) already records `0/12` converged minima (`converged 0`), and Arm B is `12/12` bit-identical (`trace_sha256`, `charged`, and `best` match on all 12 seeds).
