# Good-Turing water arm

Selector `good_turing`. The search flags are
`Config::recommended_molecular`. The stop is separate.

Each validated quench is one visit. The descriptor is the sorted
oxygen-oxygen distances. The basin radius is 0.1 angstrom on that
vector. That radius is the arm's census radius. It is not the isomer
assignment cutoff.

The production rule is unchanged: at least 20 visits and unseen mass
`N1/n` below 1/5. The checkpoint retires when `BasinCensus::is_saturated`
is true. The trace records unseen mass against force calls. The
estimator is not replaced.

Runs on gfn2, pbe, and petmad. Seeds 0-31. Budget 4000, or earlier if
the census saturates. Smoke command in
`water_hexamer.METHODS-DETAIL.md`.
