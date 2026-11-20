<!-- Copyright 2023--present rgpot developers -->

The capability handshake, in Rust and in C++, treats a schema id
with an optional `0x` prefix, in either case, as the same Cap'n
Proto file id. The C++ check uses one helper for that comparison.
