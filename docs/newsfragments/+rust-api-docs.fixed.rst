The documentation build writes the Rust API reference. sphinx-rustdocgen
1.2.1 failed on two redundant ``#[path]`` module attributes, which the build
logged without failing, so no page was written for the crate and the
reference's contents named a page that was never built. The attributes,
which named the files rustc finds without them, are gone. Each module page is
titled by the module's path within the crate, such as ``catalog::packing``,
which tells it from the crate's own ``packing`` module.
