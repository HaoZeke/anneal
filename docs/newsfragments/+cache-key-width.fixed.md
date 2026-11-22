<!-- Copyright 2023--present rgpot developers -->

Persistent result-cache keys are a version byte followed by a 64-bit hash in little-endian order, independent of the width of `size_t`. The engine ABI reference documents that the cache has no size cap and is bounded by removing its directory.
