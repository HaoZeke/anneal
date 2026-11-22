<!-- Copyright 2023--present rgpot developers -->

`EngineSession` no longer returns a false success when a refused callback state that the engine kept registered fails.
A panic in the oracle, acquisition or model-update callback of such a state is now collected like one from the registered callbacks and reported as a fatal failure.
The same holds for `set_callbacks`, `hint` and the force calls.
