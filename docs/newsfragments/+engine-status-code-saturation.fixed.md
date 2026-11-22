<!-- Copyright 2023--present rgpot developers -->

The documentation of `EngineStatus::code` no longer promises an exact round trip for every status.
A status built from a return code round-trips exactly, while a failure built directly with a magnitude beyond the range of `c_int` saturates to `c_int::MAX` or `c_int::MIN`.
