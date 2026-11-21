Remove the public ``anneal_core::xtb_gradient::store_as_force`` helper. The
xtb engine already returns force, so callers should pass that buffer to rgpot
without another sign change.
