<!-- Copyright 2023--present rgpot developers -->

A fatal engine failure is no longer hidden behind a recoverable one in `EngineSession`.
A callback or transform failure and the code the engine returns are judged together, and a fatal failure from either side wins.
The first fatal failure, with its message, is the one that poisons the handle, so every later call returns the identical `EngineError::Fatal`.
A later recoverable callback failure no longer replaces an earlier panic, and a fatal hint whose error buffer is empty reports a message naming its code instead of an empty one.
