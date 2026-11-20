A contract test checks that the root crate, the wheel crate and ``Cargo.lock``
name one ``rgpot-pair`` revision. Cargo resolves two revisions side by side
and links both copies of the kernels, so a bump that misses a manifest is
caught here and not in a built wheel.
