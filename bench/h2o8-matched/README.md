# (H2O)8 matched anneal arm

Matched rerun of the anneal arm of the GFN2 (H2O)8 head-to-head, on develop `4feb0699`. The preset is `recommended_molecular`, defined at `src/methods/cluster_hopping/config.rs:2653`. `examples/gfn2_mol_h2h.rs` builds that preset and does not apply another overlay.

## Definition

`bench/h2o8-matched/run.py:25-28` is the CREST record:

- target `-40.681336013150` Eh
- tolerance `1e-6` Eh
- threshold `reference + tolerance`, written `-40.681335013150` Eh
- budget `151738` energy-and-gradient evaluations

A hit is one sided: the energy in hartree is at or below `reference + tolerance`. The search stops on the evaluation that crosses. A miss is stored as `calls_to_hit = 151739`. Seeds are 2000 through 2011, one process at a time.

The count is one `value_and_gradient` (`src/methods/cluster_search.rs:341`). That is the same unit as a CREST energy-and-gradient evaluation. `charged` is that count. A refused overlap or a lost bond returns before `Ledger::charge` (`cluster_search.rs:627`) and does not call the potential. The driver leaves the run when `charged`, the ledger spend, and the number of potential calls disagree.

## Start and GFN2 settings

The monomer is the GFN2-xTB 6.7.1 vtight water with oxygen at the origin, `examples/gfn2_mol_h2h.rs:47` (`WATER`). Seed `N` repacks those eight rigid monomers with `StdRng::seed_from_u64` (`gfn2_mol_h2h.rs:472`). The CREST arm reads that same `start.xyz`: `scripts/mol_crest_h2h.py:258` copies it into the CREST directory, and `mol_crest_h2h.py:305` invokes `crest start.xyz --gfn2 -T 1 --legacy` with one thread. `mol_crest_h2h.py:39` maps `h2o-8` onto `gfn2_mol_h2h`.

The anneal potential is `RgpotObjective::xtb` on a 60 Å box (`gfn2_mol_h2h.rs:537`). The handle is GFN2 method `3`, accuracy `0.01`, at most `500` SCF iterations (`src/catalog/molecular.rs:83`, `:86`, `:89`), electronic temperature 300 K, charge 0, uhf 0 (`examples/common/rgpot_eindir.rs:909`). The evaluator is xtb 6.7.1, revision `edcfbbe`.

## Polish

`recommended_molecular` sets `polish_on_improve` to false (`config.rs:2659`), so `polish_improved_answer` does not run. The molecular answer polish does run and is charged against the budget. `molecular_answer_reserve` (`src/methods/cluster_search.rs:1888`) holds up to a quarter of the remaining budget, and `finish_molecular_answers` (`cluster_search.rs:1899`) spends it through `polish_answer_walled_at` (`cluster_search.rs:1965`, `STOP` is false, so that pass does not stop on the target by itself). Each evaluation is charged in `charge_or_refuse_overlap` (`cluster_search.rs:627`) before `evaluated_pair`. `set_answer_polish_calls` (`cluster_search.rs:1939`) records how many of those charges fell in the polish. They sit inside `Ledger::spent`, so they count in `calls_to_hit` when the crossing is one of them. `answer_gradient` stays `0.0` (`config.rs:2365`), so `polish_recorded_answer` does not grant calls past the budget.

With `--eh`, the evaluation that crosses raises the ledger stop after it has been counted. The next `Ledger::charge` (`src/methods/cluster_hopping.rs:1081`) returns false, and the objective does not call the potential again. `charged` equals the number of potential calls, including answer-polish calls up through the hit.

## Rows

`rows.csv` is written by the Snakefile after seeds 2000-2011. The hit count and the median `calls_to_hit` are filled once those rows exist.
