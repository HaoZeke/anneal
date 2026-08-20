# Blueprint — occupancy retire identities

Goal: every occupancy bound in the Wiley SI has a Lean 4 declaration
that `lake build` accepts. Same Lea contract as the d-SEAMS 2.0
supplement (`VIDA-NYU/Lea`): a labelled statement and a matching
declaration in `Hop/OccupancyGt.lean`.

## Assembly

- [x] `leftover_hatch` (Eq. leftover-gt-hatch)
- [x] `leftover_hatch_nonneg`, `leftover_hatch_pos_iff`
- [x] `packing_n1_zero`, `packing_stays_saturated`
- [x] `hatch_stable_iff_next`, `hatch_next_one_fifth`
- [x] `visits_floor_three_fifths`, `production_min_visits` (`n_min = 20`)
- [x] `fes_factor_ge_one`, `fes_factor_eq_one_iff`, `fes_scale_invariant`
- [x] `mixed_threshold_bw`, `copy_collapse_mixed`
- [x] `certify_min_samples` (`2 × 2 × 4 = 16`)
- [x] `code_cut_is_cheeger` (Fiedler `c < λ₂` ⇒ Cheeger `c² < 2λ₂`)
- [x] `familyFloor` cases (disconnected / Cheeger / no spectrum)
- [x] retire Boolean conjuncts (CatalogSaturated does not retire)
