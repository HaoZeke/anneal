<!-- Copyright 2023--present rgpot developers -->

`EngineSession::set_callbacks` no longer frees a callback context the engine may still hold.
A successful replacement installs the new callbacks before the old ones are dropped, and a panic in the old callbacks' destructor is caught and reported as a fatal failure instead of unwinding out of the call.
The callbacks of a refused replacement are kept alive until the engine is destroyed, because an engine built before the header forbade it may have retained them.
The `rgpot_engine_set_callbacks` header and reference now state the lifetime of the previous and the supplied `ctx` after a successful and a refused call.
