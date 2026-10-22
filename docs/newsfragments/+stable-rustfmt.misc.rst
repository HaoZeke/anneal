``cargo fmt --all --check`` passes with stable rustfmt. The vendored nng
crate keeps nightly rustfmt options that stable rustfmt does not apply,
so the sources are formatted the way the lint job checks them.
