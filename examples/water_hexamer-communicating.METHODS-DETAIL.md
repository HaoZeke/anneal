# Communicating water arm

Selector `communicating`. Built on `Config::recommended_molecular`.

Flags set to the communicating channel, with the molecular groups kept:
`depth_reward` off, `allocate_moves` on, `return_screen` on,
`orbit_complete_on_new` on, `shared_deposits` 0, visit policy
Recognition, `jump_on_stall` off. The atomic two-phase surface is not
copied. `Config::communicating` itself is an atomic preset and is not
substituted.

Runs on gfn2, pbe, and petmad. Seeds 0-31. Budget 4000. Smoke command
in `water_hexamer.METHODS-DETAIL.md`.
