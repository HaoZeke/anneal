# (H₂O)₁₆ and (H₂O)₂₁ misses, and the rigid side-walk hand-back

The `water_gmin` panel on develop leaves (H₂O)₁₆ at 2/12 and (H₂O)₂₁ at 0/12.
This directory holds the structures behind those misses, the step that fails,
the side-walk hand-back that addresses it, and same-session panels of the
hand-back against develop `7192e9f8`, `fde4f076` and `91f835fe`. The
hand-back does not reach significance at 36 seeds, so it ships off by
default as `Config::rigid_side_walk_hand_back`. The rows are in
`bench/water-gmin/rows.csv` under each arm's `code_revision`; the GMIN rows
are the `water_gmin` ones, reused because every start sha1 matches.

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
only. After the 36-seed panels below, `refactor(hop)` puts it behind
`Config::rigid_side_walk_hand_back`, off on every preset, and
`ANNEAL_PROFILE_SET=rigid_side_walk_hand_back=1` turns it on in
`globalopt_profile`.

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

## Same-session panel against develop `91f835fe`, 36 seeds at 16 and 21

`scripts/tip4p_h2h/enter_water_fix.sh 91f835fe… panel`, Snakemake rule
`water_fix`: the `water_gmin` harness on a `91f835fe` worktree and then on
this branch at `259628af`, where the hand-back is on by default, back to
back in one session, serial, `taskset -c 0`, `sudo unshare --uts` with
hostname `bench`, working directory `/work`, 2_000_000 calls, stop at the
Cambridge header plus 0.001 kJ/mol. (H₂O)₈, ₁₂ and ₂₀ take seeds 2000–2011,
(H₂O)₁₆ and ₂₁ seeds 2000–2035; GMIN is the first twelve. Rows are
`compare.csv`; `p(F)` is the two-sided Fisher exact test on hits, `p(U)` the
two-sided Mann–Whitney U test on calls and `p(out)` the same test on time
outside the potential per charged call, branch against develop. Times are
medians in seconds, "in" inside the potential and "out" outside it.

| Size | Seeds | develop | branch | GMIN | calls develop / branch | in develop / branch | out develop / branch | p(F) | p(U) | p(out) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (H₂O)₈ | 12 | 12/12 | 12/12 | 8/12 | 6_339 / 6_339 | 0.012 / 0.012 | 0.0078 / 0.0079 | 1.00 | 1.00 | 0.065 |
| (H₂O)₁₂ | 12 | 12/12 | 12/12 | 5/12 | 41_418.5 / 41_418.5 | 0.149 / 0.150 | 0.070 / 0.070 | 1.00 | 1.00 | 0.64 |
| (H₂O)₁₆ | 36 | 14/36 | 19/36 | 0/12 | 2_000_000 / 1_697_248.5 | 11.7 / 10.1 | 3.74 / 2.66 | 0.34 | 0.48 | 0.0001 |
| (H₂O)₂₀ | 12 | 6/12 | 6/12 | 0/12 | 1_686_249.5 / 1_811_777 | 14.7 / 15.8 | 4.10 / 3.09 | 1.00 | 0.98 | 0.12 |
| (H₂O)₂₁ | 36 | 1/36 | 2/36 | 0/12 | 2_000_000 / 2_000_000 | 19.0 / 19.2 | 5.90 / 4.34 | 1.00 | 0.57 | <1e-6 |

At (H₂O)₁₆ the hand-back gains seeds 2003, 2005, 2011, 2016, 2019 and 2025
and loses 2023; at (H₂O)₂₀ it gains 2000 and loses 2002; at (H₂O)₂₁ it adds
2007 to 2012. At 8 and 12 every row is the same call for call: no side walk
starts before a 200_000-call quiet stretch. Time outside the potential per
call falls at 16 and 21, where the repeated side walk takes a larger share
of the calls. No hit count differs significantly.

The same panel against `fde4f076` (branch `b5289b57` before the rebase,
rows under those revisions) gave the same hit counts and median calls at
every size:
14/36 against 19/36 at (H₂O)₁₆ and 1/36 against 2/36 at (H₂O)₂₁. The
twelve-seed panel against `7192e9f8` (branch `e1c98f8c`) had 2/12 against
5/12 and 0/12 against 1/12.

## Off by default

Neither (H₂O)₁₆ nor (H₂O)₂₁ is significant at 36 seeds, so the hand-back is
opt-in: `Config::rigid_side_walk_hand_back`, false on every preset and
absent from the resolved JSON while false. With it false a search is
develop's: the shell-turn side walk runs once per run and no side-walk
minimum is handed back.

## Identity and checks

All on the opt-in HEAD against `91f835fe`, `enter_water_fix.sh … identity`,
serial on one core under `unshare --uts` with hostname `bench`.
`identity.txt` is `scripts/generic_alloy/trace_identity.sh`: all 63 rows are
the same, LJ 38, 55, 75–77, 98, 102–104 and 150, Morse 80 at ρ = 6 and 14
(`h2h_seed` traces and `h2h_morse` rows), (H₂O)₈, ₁₂ and ₂₀ at 100_000
calls, seeds 2000–2002, and Au55 `h2h_metal` seeds 2000–2011. The traces
hash every new lowest energy with its call index and the lowest geometry.
`identity-h2o16.txt` hashes the same trace for (H₂O)₁₆ at 2_000_000 calls,
where the side walks run, on seeds 2002, 2011, 2027 and 2031: all four are
the same. The revision named is the opt-in commit; the doc commit after it
changes a comment only.

ChemFit: `pytest/test_chemfit*.py` 1005 passed and 4 skipped, and
`examples/chemfit_anneal.py` gives best loss 2.347665788572983.
`cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` are clean,
and `cargo test --release` passes, 182 result blocks with no failure.
