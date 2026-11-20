Cache and parameter contract tests no longer hard-code shared temporary paths.
They now use per-test temporary locations, which avoids cross-test collisions in
parallel runs and works across sandboxed environments.
