<!-- Copyright 2023--present rgpot developers -->

Fixed the `RgpotEngineAbiStamp` minor-version rule in `engine_abi.org` to match
`rgpot_engine_abi_compatible` and `RgpotEngineAbiStamp::serves`: a host built
against minor `N` accepts an engine of the same major and layout revision whose
minor is at most `N` (rejecting plugins with a newer minor than the host's).
