<!-- Copyright 2023--present rgpot developers -->

`meson setup -Dwith_rust_core=true` now works in a clean build directory. The
Rust library is built by a target that ninja runs before anything links it, so
setup no longer needs a prior cargo run. With the fast pair list on,
`rgpot-core` and the pair list are linked as one archive, `librgpot_bundle.a`,
so the Rust runtime is defined once. The EAM table free function also catches
a panic and records it in `rgpot_last_error` instead of unwinding into the
host.
