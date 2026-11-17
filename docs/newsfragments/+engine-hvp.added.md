Load the optional `rgpot_engine_hvp` and `analytic_hvp` capability through
`rgpot-core::engine::EnginePotential`. Engine-backed potentials use the
existing `Capabilities.hessianVector` and `Potential.calculateHvp` RPC path;
an engine without the analytic operation returns
`RGPOT_ENGINE_ERR_UNSUPPORTED` and never falls back to finite differences.
