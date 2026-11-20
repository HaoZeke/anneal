The C ABI default helpers now derive the box budget from
``BoxEnsembleConfig::default()``. ``anneal_box_config_default`` and
``anneal_box_options_default`` therefore stay in parity with the Rust
default budget instead of carrying a separate hard-coded value.
