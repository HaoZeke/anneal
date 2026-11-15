# (H₂O)₁₆ and (H₂O)₂₁ misses, and the rigid side-walk hand-back

The `water_gmin` panel on develop leaves (H₂O)₁₆ at 2/12 and (H₂O)₂₁ at 0/12.
This directory holds the structures behind those misses, the step that fails,
and the same-session panel of the fix against develop `7192e9f8`. The rows
are in `bench/water-gmin/rows.csv` under each arm's `code_revision`; the
GMIN rows are the `water_gmin` ones, reused because every start sha1 matches.

## The misses are other oxygen frameworks

`scripts/tip4p_h2h/diag_water.py structures` on develop's lowest geometries,
`diag-h2o16-develop.csv` and `diag-h2o21-develop.csv`. The contact graph
joins oxygens closer than 3.1 Å; in every geometry here the first-shell pairs
end by 2.85 Å and the next begin at 3.22 Å.

| (H₂O)₁₆ | Energy (kJ/mol) | Seeds | H-bonds | Dangling OH | Donor/acceptor | Rings 4/5/6/7/8 | R_g (Å) | Asphericity | O RMSD to GM (Å) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM (F0) | −681.1926 | 2 | 28 | 4 | 8 DDAA, 4 DDA, 4 DAA | 16/0/12/0/16 | 3.660 | 0.340 | 0 |
| F2 | −676.1859 | 4 (+1 at −675.509) | 26 | 6 | 4 DDAA, 6 DDA, 6 DAA | 9/4/0/20/0 | 3.537 | 0.179 | 1.02 |
| F3 | −674.0058 | 3 | 26 | 6 | 4 DDAA, 6 DDA, 6 DAA | 9/4/0/20/0 | 3.537 | 0.170 | 1.31 |
| F1 | −672.26, −671.53 | 2 | 26 | 6 | 4 DDAA, 6 DDA, 6 DAA | 9/4/4/4/15 | 3.52 | 0.15–0.16 | 1.08–1.09 |

The minimum is three cubes stacked face to face: four square layers and
twelve square side faces make its sixteen 4-rings. The two recurring misses
are rounder cages, R_g 0.12 Å smaller and half the asphericity, built from
nine 4-rings, four 5-rings and twenty 7-rings. Neither shares the minimum's
oxygen framework, so neither is a proton isomer of it. The proton
arrangement differs with the framework: two fewer hydrogen bonds, four of
the eight fourfold DDAA waters become DDA or DAA, and two more OH groups
dangle. F2 and F3 have the same ring and donor/acceptor counts but are not
isomorphic.

(H₂O)₂₁ misses the same way. All twelve develop seeds end in a framework
other than the minimum's, eleven distinct ones; eleven are one to three
hydrogen bonds short of its 37, with DDAA waters turned DDA or DAA. Seed
2007 at −915.7735 kJ/mol has the minimum's 37 bonds and census, including
its single DDAAA water, and differs in the rings: sixteen 4-rings and three
5-rings against fourteen and seven, O RMSD 0.99 Å.

## The transition is a climb the walker does not make

On (H₂O)₁₆ seed 2002 the walk's quenches return to the basin they left on
15_798 of 19_023 hops, and only 18 reach a minimum not seen before. The
three collective arms are accepted on 99.9% of draws because they come back
to the same minimum. The acceptance on a rigid chart is Metropolis at the
preset 2 kJ/mol on the quenched energy, and the orientation scale it steers
is capped at the preset step, so neither grows when the walker keeps
returning.

Where a run reaches the minimum, the last move is made inside the stall
side walk, from a cage 4 to 17 kJ/mol above the minimum, by a single-body
arm. A temporary print in a scratch build of `92bc5805` with the hand-back
below and the twist arm of the screen (not committed) shows, on four seeds,
`rtrans1` from −666.85 to −680.75 kJ/mol and `rcycle` from −664.13, −676.71
and −677.13 to −680.76 and −681.16. From F2 at −676.19 the chain must accept
several rises of up to 12 kJ/mol in a row at 2 kJ/mol, which the main
walker does not. The side walk opens with a turn of the outer half of the
cluster by three orientation steps and walks on from that quench.

Two things kept that from counting. The side walk's quenches go through
`relax` and its state is local, so a lower minimum it reaches is neither
recorded nor handed to the walker: on seeds 2002 and 2011 the first side
walk reached −676.1859 kJ/mol with the ledger at −667.71 and −671.61, and
both runs ended at −672.2605 (the same scratch print, without the twist). And the side walk ran once per run.

## The fix

`fix(hop)`: the lowest quench of the shell-turn walk and of the archive walk
is recorded, and the walker continues from it when it is below the
walker's own minimum by the rigid new-lowest margin (0.05 kJ/mol). The
shell-turn walk comes round after each further quiet stretch of the same
patience (40 windows of `restart_patience`) instead of once per run. No
step, size or water constant is added; the change is on the rigid-body path
only.

Variants screened on scratch builds of `92bc5805` (seeds 2000–2011, 2M
calls, (H₂O)₁₆ unless noted) and not kept:

| Variant | (H₂O)₁₆ | (H₂O)₂₀ |
| --- | --- | --- |
| develop | 2/12 | 6/12 |
| orientation scale allowed to 4× the preset step | 3/12 | |
| hand-back only | 3/12 | 6/12 |
| scale 4× and hand-back | 2/12 | |
| cap twist arm (rotate the bodies on one side of a random plane, k in [n/4, n/2]) | 3/12 | 1/12 |
| cap twist and hand-back | 6/12 | 2/12 |
| hand-back and repeated side walk (kept) | 5/12 | 6/12 |
| the same with the twist arm | 6/12 | |
| the same with side-walk legs a quarter as long | 7/12 | 4/12 |

The twist arm and the shorter legs cost (H₂O)₂₀ hits; the kept variant is
the one that loses none.

## Same-session panel against develop `7192e9f8`

`scripts/tip4p_h2h/enter_water_fix.sh`, Snakemake rule `water_fix`: the
`water_gmin` harness on a `7192e9f8` worktree and then on this branch,
back to back in one session, serial, `taskset -c 0`, `sudo unshare --uts`
with hostname `bench`, working directory `/work`, twelve seeds, 2_000_000
calls, stop at the Cambridge header plus 0.001 kJ/mol. Rows are
`compare.csv`; `p(F)` is the two-sided Fisher exact test on hits and `p(U)`
the two-sided Mann–Whitney U test on calls, branch against develop. Times
are medians in seconds.

| Size | develop | branch | GMIN | calls develop / branch | wall develop / branch | out develop / branch | p(F) | p(U) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (H₂O)₈ | 12/12 | 12/12 | 8/12 | 6_339 / 6_339 | 0.020 / 0.020 | 0.008 / 0.008 | 1.00 | 1.00 |
| (H₂O)₁₂ | 12/12 | 12/12 | 5/12 | 41_418.5 / 41_418.5 | 0.224 / 0.220 | 0.070 / 0.070 | 1.00 | 1.00 |
| (H₂O)₁₆ | 2/12 | 5/12 | 0/12 | 2_000_000 / 2_000_000 | 15.9 / 14.6 | 4.16 / 2.90 | 0.37 | 0.28 |
| (H₂O)₂₀ | 6/12 | 6/12 | 0/12 | 1_686_249.5 / 1_811_777 | 18.7 / 18.8 | 4.09 / 3.09 | 1.00 | 0.98 |
| (H₂O)₂₁ | 0/12 | 1/12 | 0/12 | 2_000_000 / 2_000_000 | 25.0 / 23.0 | 6.01 / 4.11 | 1.00 | 0.32 |

The branch loses no hit at any size and gains three at (H₂O)₁₆ and one at
(H₂O)₂₁, the first (H₂O)₂₁ hit of these panels: seed 2007 goes on from the
37-bond −915.7735 cage to the minimum's framework (`diag-h2o21-branch.csv`).
No difference is significant at twelve seeds. At 8 and 12 every row is the
same call for call: no side walk starts before a 200_000-call quiet stretch.
Time outside the potential falls by about 30% from 16 up, where the side
walk now takes a larger share of the calls. Against GMIN the
branch has p(F) 0.037 at (H₂O)₁₆, where develop had 0.48.

The change is on by default: it wins at 16 and 21 and regresses nowhere,
and the atomic and Morse traces are untouched (below).

## Other potentials and checks

`identity.txt` is `scripts/generic_alloy/trace_identity.sh` on `7192e9f8`
and on the fix commit, serial on one core. All 63 rows are the same: LJ 38,
55, 75–77, 98, 102–104 and 150, Morse 80 at ρ = 6 and 14 (`h2h_seed` traces
and `h2h_morse` rows), (H₂O)₈, ₁₂ and ₂₀ at 100_000 calls, seeds 2000–2002,
and Au55 `h2h_metal` seeds 2000–2011. The traces hash every new lowest
energy with its call index and the lowest geometry.

ChemFit: `pytest/test_chemfit*.py` 1005 passed and 4 skipped, and
`examples/chemfit_anneal.py` gives best loss 2.347665788572983.
`cargo clippy --all-targets -- -D warnings` is clean. `cargo test --release`
passes except `the_dual_orbit_core_is_opt_in` and
`a_held_core_cut_is_stamped_with_the_probe_call_that_found_it`, which fail
the same way on `7192e9f8`. `cargo fmt --check` reports only `src/soap.rs`
and `tests/surface_hop.rs`, the same as on `7192e9f8`.
