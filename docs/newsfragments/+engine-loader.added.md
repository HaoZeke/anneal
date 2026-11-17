`rgpot-core` loads any library that exports the generic engine C ABI
(`rgpot/engine_c_abi.h`) from Rust. The `engine` feature adds
`rgpot_core::engine::EnginePotential`: it opens the library, refuses another
ABI major, creates an instance from the backend's configuration message, and
answers energy and forces per geometry, or per batch when the library exports
`rgpot_engine_force_batch`. A positive return code is a recoverable
`EngineError`, a negative one a fatal one, with the engine's own message.
`EnginePotential::into_potential` wraps the instance as a `PotentialImpl`.
