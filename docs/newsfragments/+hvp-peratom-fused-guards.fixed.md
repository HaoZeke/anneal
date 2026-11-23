<!-- Copyright 2023--present rgpot developers -->

Direct `forcePerAtom` calls, the C ABI Hessian-vector entry points, and `cellVolume` now reject missing or non-finite inputs before invoking kernels or publishing stress, while `FusedCache` no longer stores validation failures.
