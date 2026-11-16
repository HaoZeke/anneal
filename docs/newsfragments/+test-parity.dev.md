`CppCore/tests/registry.txt` lists every test with the features that gate it,
its kind and its options, and meson and CMake both register their tests from
it.
CMake now registers each test meson does except those only meson can run, whose
lines give the reason: in the build the `thread_limit` CI job tests, that is 99
of meson's 107 tests, where CMake registered 31.
CMake's unit tests link OpenMP as meson's do, and its ABI symbol tests run their
script through `sh`.
`CuH2PotTest`, `CapnpAdapterTest` and `BridgeStressTest` take meson's names,
`CuH2test`, `RPCTest` and `RPCClientTest`.
Each build registers `test_list_parity`, which fails when the build's tests or
their options differ from the registry.
`tools/check_test_parity.sh` configures the plain, the RPC and cache, and the
client-only builds with meson and CMake, without building them, and fails when
the two register different tests.
The `test_parity` CI job runs it, and `ci_gate` waits on that job.
