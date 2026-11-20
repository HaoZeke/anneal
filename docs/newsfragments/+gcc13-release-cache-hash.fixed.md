<!-- Copyright 2023--present rgpot developers -->
Fixed strict-aliasing undefined behavior in `rgpot::Potential::cacheKey` under GCC 13 `-O3` Release builds by defining `XXH_FORCE_MEMORY_ACCESS 0` before including `<xxhash.h>` and copying the hashed scalar keys into byte arrays via `std::memcpy`.
