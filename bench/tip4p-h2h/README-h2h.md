# TIP4P head-to-head, anneal against GMIN, twelve seeds

The head-to-head cited in the manuscript is run r2c, record branch
`record/tip4p-h2h-r2` (commit `cdae978c` on `40c5b461`, tree `f7ee9adc`).
Its per-seed rows are `rows-h2h-r2.csv`. Anneal is `e1a69a57` and GMIN is
`aac1548a`. Twelve seeds, 2000–2011, a budget of 2_000_000 charged
evaluations at every size, serial, core 0, one machine. Both arms open the
same molecular centres. One charged evaluation is one entry to the potential
on either side. A hit is a quenched energy at or below the Cambridge Cluster
Database TIP4P header plus 0.001 kJ/mol. The database lists headers for
n ≤ 21. (H₂O)₂₅ is not in the database, so run r2c leaves it unscored.

Fisher's exact test on hits and the Mann–Whitney test on calls to a hit are
two-sided and Bonferroni-corrected with m = 4, one comparison for each scored
size. A miss is ranked at the budget in the calls test. The wall-time
Mann–Whitney test is a per-call cost and is not a search result.

GMIN is built from source by `scripts/gmin_h2h/build_gmin.sh` at revision
`aac1548a`, gfortran 13.3.0, reference BLAS/LAPACK, with NEWTIP, TIPID 4,
temperature 2 kJ/mol held fixed, step 0.40 / 0.51 held fixed, and the
Lennard-Jones container (`examples/h2h_gmin_tip4p.data`). Medians are
Hyndman–Fan type 7. `h2h-summary.csv` is an earlier summary and is not run r2c.

## Both codes reproduce the Cambridge global minimum

Before the campaign each code takes a single point at the published
centre/angle-axis block (`scripts/tip4p_h2h/gm_radius.py --gmin`). The
anneal single point also backs the `cambridge_headers_match` unit test.
Numbers are `h2h-verify.csv`.

| Size | Cambridge header (kJ/mol) | anneal − header | GMIN − header |
| --- | --- | --- | --- |
| (H₂O)₈ | −305.5183221676 | +1.5e-05 | −6.0e-04 |
| (H₂O)₁₅ | −628.3722525077 | −7.6e-06 | −1.3e-03 |
| (H₂O)₂₀ | −872.9887544952 | −1.1e-05 | −1.8e-03 |
| (H₂O)₂₁ | −916.7074302940 | −9.7e-05 | −1.9e-03 |

Anneal reproduces every header to 1e-4 kJ/mol. GMIN reads a constant amount
lower, growing with the pair count from 0.6 to 1.9 mJ/mol: its `DEFTIP4`
Coulomb prefactor rounds where `GminTip4p` does not. Both land on the same
global minimum; the next distinct minimum is more than 0.1 kJ/mol away, so the
offset never changes which cage a run is scored at.

## Panel

Rows are `rows-h2h-r2.csv`, run r2c. `best` is the lowest of the twelve; the
median is beside it. `p(F)` is Fisher's exact test on the hit counts and
`p(calls)` is the exact Mann–Whitney test on calls to a hit, both after the
Bonferroni correction above. (H₂O)₂₅ is unscored, so its hit and p cells are
blank. Wall is a per-call cost.

| Size | anneal | GMIN | anneal best / median | GMIN best / median | wall (a/g s) | p(F) | p(calls) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (H₂O)₈  | 12/12 | 8/12 | −305.5177 / −305.5176 | −305.5189 / −305.5189 | 0.022 / 1.952 | 0.37 | 3.0e-06 |
| (H₂O)₁₅ | 7/12  | 0/12 | −628.3716 / −628.3713 | −622.8593 / −619.9326 | 11.667 / 48.016 | 0.018 | 0.018 |
| (H₂O)₂₀ | 6/12  | 0/12 | −872.9879 / −868.8666 | −860.7014 / −851.4433 | 21.541 / 83.630 | 0.055 | 0.055 |
| (H₂O)₂₁ | 0/12  | 0/12 | −915.7735 / −914.3685 | −903.7597 / −895.5929 | 26.312 / 81.097 | 1 | 1 |
| (H₂O)₂₅ | unscored | unscored | −1113.035 / −1103.345 | −1087.334 / −1082.127 | 35.730 / 107.109 | -- | -- |

Run r2c hits (H₂O)₈ on 12 of 12 and GMIN on 8 of 12. The calls-to-hit p is
3.0e-06 and Fisher's p is 0.37. It hits (H₂O)₁₅ on 7 of 12 against 0 of 12,
p = 0.018, and (H₂O)₂₀ on 6 of 12 against 0 of 12, p = 0.055, which is not
significant. Neither arm hits (H₂O)₂₁. On the unscored (H₂O)₂₅ the best and
median energies are −1113.035 / −1103.345 kJ/mol for anneal and −1087.334 /
−1082.127 kJ/mol for GMIN.

The median wall is a per-call cost. It is lower for anneal on all five sizes,
by a factor of 3.00 to 87.1. That factor is not a search result.

At (H₂O)₈ both arms find the same global-minimum cage. GMIN's best reads
1.2 mJ/mol below anneal's there, which is the potential offset above plus
GMIN's quench settling past the early-stop energy that anneal halts on; it is
the same cage, not a lower one. Anneal finds that cage on all twelve seeds and
GMIN on eight. Fisher's p on that split is 0.37 after the Bonferroni
correction, so the size is not a significant hit win. The calls-to-hit p is
3.0e-06.

## (H₂O)₂₁ budget sweep

(H₂O)₂₁ is the one Cambridge size neither arm reaches in 2 M calls, so it is
run again on anneal develop's default preset at 2 M, 4 M and 8 M charged
evaluations, the same twelve seeds and the same serial arm. The driver is
`scripts/tip4p_h2h/enter_budget.sh`, the Snakemake rule is
`tip4p21_budget_h2h`, and the asv class is `Tip4p21BudgetHeadToHead`. GMIN at
a (budget, seed) pair is deterministic: the 2 M GMIN rows repeat the panel's
best and median energies, and their median wall is 75.15 s against the
panel's 81.097 s. Summary rows are `summary-budget21.csv`
and per-seed rows are `rows-budget21.csv`. `p(U)` is the two-sided
Mann–Whitney U test on the per-seed best energies; the hit counts are 0/12 on
both arms at every budget, so the Fisher exact p is 1.000 throughout.

| Budget | anneal | GMIN | anneal best / median | GMIN best / median | wall (a/g s) | p(U) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 M | 0/12 | 0/12 | −915.774 / −914.368 | −903.760 / −895.593 | 24.70 / 75.15 | 3.2e-05 |
| 4 M | 0/12 | 0/12 | −915.774 / −914.889 | −905.330 / −897.861 | 49.48 / 150.16 | 3.2e-05 |
| 8 M | 0/12 | 0/12 | −915.774 / −915.143 | −905.330 / −899.538 | 102.45 / 299.73 | 3.2e-05 |

Four times the budget does not reach the global minimum on either arm. Anneal's
best holds at −915.7735 kJ/mol, 0.93 kJ/mol short of the −916.7074 header,
from 2 M through 8 M: the same seed plateaus in another oxygen cage with as
many hydrogen bonds (next section), and more calls from that incumbent do not
leave it. The median does climb,
from −914.37 to −915.14, as the slower seeds reach that plateau. GMIN's best
improves from −903.76 to −905.33 and then holds, 11.4 kJ/mol short. Anneal's
per-seed best distribution stays below GMIN's at every budget, Mann–Whitney p
3.2e-05, at a third of the wall. Lifting (H₂O)₂₁ is left to later work; the
default preset is unchanged here.

## (H₂O)₂₁ misses at 2 M
Each of the twelve 2 M rows above is rerun with the panel's stop target and
its final structure compared with the Cambridge minimum
(`paper/jctc/data/system_visuals/ccd/TIP4P/xyz/TIP4P-21.xyz`). The oxygen
framework is the O–O contact graph at 3.0 Å, which falls in the empty 2.9–3.1
Å band of every structure here. The proton topology is the directed
hydrogen-bond graph (H···O below 2.5 Å, O–H···O at least 130°). Both graphs
are tested for isomorphism with the minimum's graphs. `o_rmsd_gm_a` is the
oxygen RMSD after the optimal permutation, rotation and reflection. The rings
are chordless cycles of the contact graph. The budget columns come from the
search's own counters: `accepted` sums the arm accepts, `rejected` is hops
minus accepted, and `quench_return/known/new` is the minima-hopping visit
classification of every quench. Per-seed rows are `diag-h2o21.csv`.
`scripts/tip4p_h2h/diag21.py run --binary <h2h_seed> --runs <dir>` writes the
twelve runs, and `diag21.py diag --runs <dir>` writes the rows from them.

Every miss is an oxygen-framework miss. No seed's contact graph is isomorphic
to the minimum's, so none is a proton isomer of the right cage. The twelve
seeds end in eleven different frameworks: seeds 2000 and 2006 share one basin,
F1. The plateau at −915.7735 (seed 2007, 0.934 kJ/mol short) has the same 37
contacts and 37 hydrogen bonds as the minimum and the same coordination counts
(11 three-, 9 four- and 1 five-coordinate). But its rings are 16/3/9 four-,
five- and six-membered, against the minimum's 14/7/10, at 0.99 Å oxygen RMSD.
That is a different cage, not a missing bond. The other seeds fall in two
families. 2000, 2001, 2002, 2003, 2006 and 2009 are pentagon-rich with 34 to
36 contacts. 2004, 2005 and 2010 are square-rich, and 2008 and 2011 have ten four- and
ten five-membered rings, all arranged differently from 2007.

A hop costs 105 to 184 calls, nearly all of them in its quench. Of those
quenches, 61 to 87% fall back into the basin they started from, and only 20 to
47 per run are new to the minima-hopping classifier. Most accepts are those
returns: the translation, rotation and all-body kick arms show about 100%
acceptance and almost never change the basin. Relocation, the one move that
rewires the framework, is accepted on 0.5 to 4.4% of its draws. The Thompson
allocator pays a rigid arm only for a new incumbent, so its draws stay near
uniform. The best stops improving between 1% (seed 2007) and 90% (seed 2009)
of the budget.

General mechanisms tried against that, n = 12 at 2 M with the same stop target
(`summary-h2o21-mechanisms.csv`):

| Variant | (H₂O)₂₁ hits / median | (H₂O)₁₅ | (H₂O)₂₀ | (H₂O)₈ |
| --- | --- | --- | --- | --- |
| baseline | 0/12 / −914.368 | 7/12 / −628.371 | 6/12 / −868.867 | 12/12 |
| fresh rigid restart at every stall | 0/12 / −914.123 | | | |
| fresh rigid restart at every other stall | 0/12 / −914.552 | | | |
| Metropolis temperature from basin-changing acceptance, factor cap 4 | 0/12 / −912.563 | | | |
| the same, cap 24 at every size | 0/12 / −913.079 | | | |
| arm reward for an accepted basin change | 1/12 / −911.748 | | | |
| arm reward for an accepted downhill basin change | 1/12 / −914.117 | 6/12 / −626.124 | 3/12 / −863.852 | 12/12 |

The restarts keep the bias and replace the stored-minimum stall escape with a
fresh random rigid cluster. They change neither the hits nor the median beyond
noise. Raising the Metropolis temperature only raises the return fraction and
lowers the median. Paying the allocator for basin changes moves draws off the
returning arms and reaches the minimum once in twelve seeds (2010 at 735,589
calls, 2011 at 215,380 for the downhill form). That is the first anneal hit on
(H₂O)₂₁. The downhill form loses one seed at (H₂O)₁₅ and three at (H₂O)₂₀, so
no variant is made default. Baseline (H₂O)₈, ₁₅ and ₂₀ rows are the same arm
run on f0dfa206; seeds 2000 at 15 and 2001 at 20 reproduce on 3141987e call
for call. The variants were scratch builds that are not committed. The
summary is `diag21.py mechanisms` over their run directories, one
`variant:n:dir` argument per row in table order.

## Novel-basin reward and basin-archive taboo

The misses above spend 61 to 87% of their quenches returning to the basin they
left, and the allocator pays a rigid arm only for a new incumbent. Two general
mechanisms address that. Each is tried alone and the two are tried together,
at n = 12 on every chart, with nothing set by size.

- **(a) Novel-basin reward.** An arm is paid when its quench lowers the
  incumbent, or when the quench's fingerprint is not yet in this run's basin
  archive (`bias.locate` finds no seat). On the Bernoulli allocator that is
  the reward. On the depth allocator (Lennard-Jones and Sutton-Chen
  `recommended`), a revisit earns the shallowest depth any arm has reached so
  far in the run, instead of its own depth. Stride-skipped uphill proposals
  have no lookup and keep the old reward.
- **(b) Basin-archive taboo.** For each archive seat, count the quenches from
  that seat that land on a basin already in the archive, and reset the count
  when one lands on a new basin. The proposal step from that seat is
  multiplied by √(1 + count / k). Here k is the run's own trials per accept,
  hops over accepted hops.

TIP4P runs are `h2h_seed` at 2 M calls with the panel's stop target. LJ38 and
LJ75 are `h2h_gmin_proof` at the paper budgets from the GMIN-shared starts.
sc96-80 is `h2h_sc` at 1 M calls. All runs are serial on one core. Every run is
on df550674: the baseline on develop, the variants on scratch builds. The
same baseline seeds give identical calls and best on ca6adf99, but only the
clock changes there, so the times are those of df550674. Per-seed rows are
`rows-novel-basins.csv`. Hits, medians, median time inside and outside the
potential, Fisher p and the lost and gained seeds are in
`summary-novel-basins.csv`. Both are written from the run logs by
`scripts/tip4p_h2h/novel_basins.py rows` and `summary`.

| Variant | (H₂O)₂₁ | (H₂O)₂₀ | (H₂O)₁₅ | (H₂O)₈ | LJ38 | LJ75 | sc96-80 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0/12 −914.368 | 6/12 −868.867 | 7/12 −628.371 | 12/12 | 12/12 | 10/12 | 7/12 |
| (a) | 0/12 −912.261 | 0/12 −862.394 | 8/12 −628.371 | 12/12 | 12/12 | 11/12 | 7/12 |
| (b) | 0/12 −913.767 | 1/12 −864.080 | 8/12 −628.371 | 12/12 | 12/12 | 12/12 | 7/12 |
| (a)+(b) | 1/12 −913.679 | 2/12 −862.095 | 5/12 −623.877 | 12/12 | 12/12 | 12/12 | 5/12 |

The baseline (H₂O) counts are the anneal arm of run r2c.
Water cells give hits and the median best in kJ/mol. None of the three is made
default. (a) and (b) do not raise (H₂O)₂₁, and the median falls. (a)+(b)
reaches the minimum once (seed 2011) with a lower median. Every variant loses
five or six of the six (H₂O)₂₀ hits: Fisher p 0.014 for (a), 0.069 for (b) and
0.19 for (a)+(b). (a)+(b) also loses three (H₂O)₁₅ seeds and four sc96-80
seeds. LJ75 gains one or two seeds under each variant (Fisher p 0.48 for (b)
and (a)+(b)), LJ38 and (H₂O)₈ stay at 12/12, and sc96-80 under (a) or (b) loses
three seeds and gains three.
