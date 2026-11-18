`rgpot-core` loads any library that exports the generic engine C ABI
(`rgpot/engine_c_abi.h`) from Rust. The `engine` feature adds
`rgpot_core::engine::EnginePotential`: it opens the library, checks both the
ABI major and `rgpot_engine_abi_stamp` (major, minor and layout revision),
creates an instance from the backend's configuration message, and answers
energy and forces per geometry, or per batch when the library exports
`rgpot_engine_force_batch`. A positive return code is a recoverable
`EngineError`, and a negative one poisons the handle and returns a fatal
`EngineError` with the engine's own message without re-entering native code on
later calls. `EnginePotential::into_potential` wraps the instance as a
`PotentialImpl`.
