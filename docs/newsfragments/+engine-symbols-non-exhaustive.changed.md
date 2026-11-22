<!-- Copyright 2023--present rgpot developers -->

`EngineSymbols` is now `#[non_exhaustive]`, so a symbol added to the engine ABI no longer breaks code that builds one.
Build it with `EngineSymbols::new`, which takes the four required symbols, and add each optional symbol with a `with_*` method such as `with_set_callbacks`.
The optional `model_state`, `set_callbacks` and `hint` symbols this release adds are set the same way.
