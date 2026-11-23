<!-- Copyright 2023--present rgpot developers -->

The generic engine C ABI adds the optional `rgpot_engine_unload_policy` export with `RGPOT_ENGINE_UNLOAD_RESIDENT` and `RGPOT_ENGINE_UNLOAD_SAFE`, and bumps `RGPOT_ENGINE_ABI_MINOR` and `ENGINE_ABI_MINOR` to 7. A plugin of minor 7 or later that returns `RGPOT_ENGINE_UNLOAD_SAFE` can be unloaded with the new `unsafe` `EnginePotential::unload(path)` once no instance uses it. Every other plugin stays loaded, and `unload` refuses it with `EngineError::Unsupported`. Plugins of minor 0 to 6 still load.
