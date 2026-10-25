# Derived water arm

Selector `derived`. Built on `Config::recommended_molecular`.

`budget_window` on. `bayes_screen` on. `bayes_threshold` is
`(R - S) / (2R - S)` from the configuration's screen and relax steps.
The atomic `Config::derived` preset is not substituted, and the
molecular groups stay.

Runs on gfn2, pbe, and petmad. Seeds 0-31. Budget 4000. Smoke command
in `water_hexamer.METHODS-DETAIL.md`. Hit rates for this stack are not
claimed.
