The Rust API page is generated again. The docs build installs
``sphinx-rustdocgen`` 1.3.0 with ``#[path]`` resolved from the declaring
file's directory. Wrapped code spans, trailing literals, and LaTeX bars
in the generated pages are RST. Orphan method, communication, regime,
and escape pages are in the contents, and the eindir inventory is no
longer fetched.
