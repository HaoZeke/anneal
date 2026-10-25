``cargo fmt --check`` passes with stable rustfmt on the workspace member.
Vendored ``nng`` stays out of that check: its ``rustfmt.toml`` asks for
nightly options, the ``ignore`` key is unstable on stable, and
``cargo fmt --all`` would still walk the path dependency. The vendored
sources are not reformatted.
