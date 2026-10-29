# Good-Turing water arm

Selector `good_turing`. The search flags are
`Config::recommended_molecular`. The stop is separate.

A validated quench is a visit only when species-aware IRA assigns it to a
named isomer inside the gate. That isomer is the visit's witness.
Off-target quenches are not visits. The descriptor is the sorted
oxygen-oxygen distances. The basin radius is 0.1 angstrom on that
vector. That radius is the arm's census radius. It is not the isomer
assignment cutoff. The witness merges near-duplicate minima of one
isomer that the radius splits.

The checkpoint retires when a witnessed target meets Good's next-step
hatch and Esty's one-sided 95% upper bound under the production
unseen-mass ceiling of 1/5 (`good_turing_target_stop_at`). The
anonymous visit floor is not that rule. `WATER_WITNESS_TRIPLETON=1`
raises the ceiling to 1/3 and retires only when the incumbent is itself
inside the gate. The trace records visits, singletons, doubletons, the
hatch, the Esty bound, and the witness against force calls.

Runs on gfn2, pbe, and petmad. Seeds 0-31. Budget 4000, or earlier if
the witnessed target is found. Smoke command in
`water_hexamer.METHODS-DETAIL.md`.
