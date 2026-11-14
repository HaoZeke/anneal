# TIP4P head-to-head, anneal against GMIN, twelve seeds

Anneal develop `f511db94` against Wales's GMIN on rigid-body TIP4P
basin-hopping. The timed anneal binary was built from `7da10659`; develop's
only search change since then, the cov3 ranged-trial reach bound, uses a
pair's measured reach, which a Coulomb/Lennard-Jones tail like TIP4P does not
have, so `pair_range` is unset and the rigid trajectory is unchanged. Three
rows rebuilt on `f511db94` reproduce their `7da10659` best, evaluations and
hit bit-for-bit, including the (H₂O)₂₅ seed 2010 best of −1113.0346568571.
Twelve seeds, 2000–2011, a budget of 2_000_000 charged
evaluations at every size (the paper ceiling for (H₂O)₂₀), serial, `taskset
-c 0`, one session under `sudo unshare --uts` with hostname `bench`, working
directory `/work`. The driver is `scripts/tip4p_h2h/enter_h2h.sh`, the
Snakemake rule is `tip4p_h2h`, and the asv suite is
`bench/tip4p-h2h/benchmarks/tip4p_h2h.py`.

Both arms open the same molecular centres: `h2h_seed` writes the start it
draws for a seed and GMIN reads it as its `coords`. One charged evaluation is
one entry to the potential on either side. A hit is a quenched energy at or
below the reference plus 0.001 kJ/mol. The references are the Cambridge
Cluster Database TIP4P headers, which the database lists for n ≤ 21; (H₂O)₂₅
is not in the database, so its hit is scored against the lowest energy either
arm reached in the session and no early stop is armed.

GMIN is built from source by `scripts/gmin_h2h/build_gmin.sh` at revision
`aac1548a`, gfortran 13.3.0, reference BLAS/LAPACK, with NEWTIP, TIPID 4,
temperature 2 kJ/mol held fixed, step 0.40 / 0.51 held fixed, and the
Lennard-Jones container (`examples/h2h_gmin_tip4p.data`). `pot_s` is time
inside the potential; `scripts/tip4p_h2h/newtip-timer.patch` brackets NEWTIP
the way the public patch already brackets the Lennard-Jones kernel.
`outside_s` is the rest of the wall. Medians are Hyndman–Fan type 7.

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

Rows are `h2h-summary.csv`. `best` is the lowest of the twelve; the median is
beside it. `p(F)` is the two-sided Fisher exact test on the hit counts; `p(U)`
is the two-sided Mann–Whitney U test on the per-seed best energies.

| Size | anneal | GMIN | anneal best / median | GMIN best / median | wall (a/g s) | pot (a/g s) | p(F) | p(U) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (H₂O)₈  | 12/12 | 8/12 | −305.5177 / −305.5176 | −305.5189 / −305.5189 | 0.020 / 1.697 | 0.012 / 1.355 | 0.093 | 0.166 |
| (H₂O)₁₅ | 9/12  | 0/12 | −628.3716 / −628.3714 | −622.859 / −619.933 | 8.01 / 40.81 | 6.12 / 36.97 | 0.00034 | 3.2e-05 |
| (H₂O)₂₀ | 5/12  | 0/12 | −872.9879 / −864.394 | −860.701 / −851.443 | 22.17 / 71.13 | 17.61 / 66.39 | 0.037 | 5.3e-05 |
| (H₂O)₂₁ | 0/12  | 0/12 | −915.774 / −914.294 | −903.760 / −895.593 | 24.89 / 76.37 | 19.21 / 71.46 | 1.000 | 3.2e-05 |
| (H₂O)₂₅ | 1/12  | 0/12 | −1113.035 / −1103.345 | −1087.334 / −1082.127 | 33.86 / 105.23 | 26.23 / 99.89 | 1.000 | 3.2e-05 |

GMIN beats anneal nowhere. Anneal has at least as many hits at every size and
a lower per-seed best-energy distribution at every size from 15 up, with the
Mann–Whitney p at or below 5.3e-05. At (H₂O)₂₁ neither arm reaches the global
minimum in 2 M calls, but anneal stops 0.93 kJ/mol short of it against GMIN's
13 kJ/mol; at (H₂O)₂₅ anneal reaches −1113.03 kJ/mol against GMIN's −1087.33.
Anneal's median wall is 3.1 to 85× shorter, and most of each wall on both
sides is time inside the potential.

At (H₂O)₈ both arms find the same global-minimum cage. GMIN's best reads
1.2 mJ/mol below anneal's there, which is the potential offset above plus
GMIN's quench settling past the early-stop energy that anneal halts on; it is
the same cage, not a lower one. Anneal finds that cage on all twelve seeds and
GMIN on eight, so anneal wins the size on hits. Because no arm beats anneal,
there is no accompanying fix.

Per-seed rows and logs stay in `/work`; only the summary and verification
rows are committed.

## (H₂O)₂₁ budget sweep

(H₂O)₂₁ is the one Cambridge size neither arm reaches in 2 M calls, so it is
run again on anneal develop's default preset at 2 M, 4 M and 8 M charged
evaluations, the same twelve seeds and the same serial arm. The driver is
`scripts/tip4p_h2h/enter_budget.sh`, the Snakemake rule is
`tip4p21_budget_h2h`, and the asv class is `Tip4p21BudgetHeadToHead`. GMIN at
a (budget, seed) pair is deterministic, so the rows the panel above and the
4 M/8 M runs already stored are reused. Summary rows are `summary-budget21.csv`
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

Every miss is an oxygen-framework miss. No seed's contact graph is isomorphic
to the minimum's, so none is a proton isomer of the right cage. The twelve
seeds end in eleven different frameworks: seeds 2000 and 2006 share one basin,
F1. The plateau at −915.7735 (seed 2007, 0.934 kJ/mol short) has the same 37
contacts and 37 hydrogen bonds as the minimum and the same coordination counts
(11 three-, 9 four- and 1 five-coordinate). But its rings are 16/3/9 four-,
five- and six-membered, against the minimum's 14/7/10, at 0.99 Å oxygen RMSD.
That is a different cage, not a missing bond. The other seeds fall in two
families. 2000, 2001, 2002, 2003, 2006 and 2009 are pentagon-rich with 34 to
36 contacts. 2004, 2005, 2008, 2010 and 2011 are square-rich but arranged
differently.

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
for call.
