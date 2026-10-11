The documentation build writes the Rust API reference. sphinx-rustdocgen
1.2.1 failed on two redundant ``#[path]`` module attributes, which the build
logged without failing, so the reference named pages that were never made.
The attributes are gone and the crate builds as before.
