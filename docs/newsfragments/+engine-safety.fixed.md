`EnginePotential` now validates `rgpot_engine_abi_stamp` (major, minor and
layout revision) on load and poisons its handle after a negative return code so
subsequent calls never re-enter a failed native engine, and `rgpot_engine_hvp`
rejects overlapping `direction` and `hv_out` buffers with
`RGPOT_ENGINE_ERR_INVALID`.
