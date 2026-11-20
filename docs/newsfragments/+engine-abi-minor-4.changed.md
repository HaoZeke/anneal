The generic engine C ABI minor is now 4, in `RGPOT_ENGINE_ABI_MINOR` and
`ENGINE_ABI_MINOR`, for the additive `rgpot_engine_hvp` entry point and the
trailing `RgpotEngineCaps.analytic_hvp` capability flag. An engine at minor 4
serves a host built against minor 4 or newer; a host built against minor 3
refuses that newer engine.
