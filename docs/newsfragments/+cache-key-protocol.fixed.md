<!-- Copyright 2023--present rgpot developers -->

Persistent result-cache keys preserve the order of the potential type, parameter fingerprint, and geometry fields. Scalar and batched callers reuse the same results under optimized builds. An existing database retains its old entries and recomputes each geometry on its first visit using the new key format.
