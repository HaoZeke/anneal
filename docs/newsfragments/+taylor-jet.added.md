A potential handle answers validated Taylor jets along a line: `rgpot_potential_taylor_jet` writes enclosures `lo_k <= D^k V(x)[v^k] / k! <= hi_k` from a jet registered with `rgpot_potential_set_taylor_jet` (`TaylorJet` and `serve_taylor_jet` on the Rust side).
A handle without a jet answers the new status `RGPOT_NO_JET` (6), and a jet that meets an undecided branch or leaves a function's domain answers `RGPOT_JET_BRANCH` (7) or `RGPOT_JET_DOMAIN` (8); nothing is differenced in their place.
The ABI minor is 4.
