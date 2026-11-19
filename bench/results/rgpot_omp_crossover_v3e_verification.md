# OpenMP crossover v3e test verification

The earlier aggregate claim of 92 passed plus 2 expected failures is replaced
by these per-leg results:

|Leg|Result|
|---|---|
|Meson, RPC and cache enabled|135 passed, 2 expected failures (137 total)|
|Meson, RPC and cache disabled|112 passed, 2 expected failures (114 total)|
|CMake CTest, RPC and cache enabled|124 passed (124 total)|

The Meson legs used these common configure options:

```text
-Dwith_tests=True
-Dwith_examples=False
-Dwith_xtensor=False
-Dwith_eigen=False
-Dpure_lib=False
```

The enabled leg added `-Dwith_rpc=true -Dwith_cache=true`; the disabled leg
added `-Dwith_rpc=false -Dwith_cache=false`. Both were built with
`meson compile` and run with `meson test --print-errorlogs`.

The CMake CTest leg used:

```text
-DCMAKE_CXX_COMPILER_LAUNCHER=ccache
-DCMAKE_C_COMPILER_LAUNCHER=ccache
-DCMAKE_BUILD_TYPE=Release
-DRGPOT_BUILD_TESTS=True
-DRGPOT_BUILD_EXAMPLES=False
-DRGPOT_WITH_XTENSOR=OFF
-DRGPOT_WITH_EIGEN=OFF
-DRGPOT_WITH_RPC=ON
-DRGPOT_WITH_CACHE=ON
-DRGPOT_PURE_LIB=OFF
```

It was built with `cmake --build` and run with
`ctest --output-on-failure`.

The crossover checks are `test_omp_crossover_v3_provenance.py`,
`test_omp_crossover_v3_provenance_cases.py`,
`test_omp_crossover_v3_report.py`, and
`test_omp_crossover_v3_analysis.py`. A bare provenance check from the
repository root is:

```text
python3 bench/test_omp_crossover_v3_provenance.py
```

These checks are entries in `CppCore/tests/registry.txt`, which the existing
Meson and CMake CI matrix jobs consume. A separate workflow invocation would
duplicate the same checks, so none is added.
