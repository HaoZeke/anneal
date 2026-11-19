# OpenMP crossover v3e test verification

These counts are environment-specific.
`bench/record_omp_crossover_v3_verification.py` parsed them from the
Meson and CTest logs of the legs below. Another compiler, gfortran,
rustc, ASE, pytest, or MPI install can register a different set. The
configure arguments are the ones that produced each log, and the
resolved values are the ones the build recorded.

## Toolchain

|Item|Value|
|---|---|
|C compiler|gcc (conda-forge gcc 14.4.0-7) 14.4.0|
|C++ compiler|g++ (conda-forge gcc 14.4.0-7) 14.4.0|
|gfortran|yes, GNU Fortran (conda-forge gcc 14.4.0-7) 14.4.0|
|rustc|rustc 1.98.1 (48a229cea 2026-09-01)|
|ASE|no|
|pytest|no|
|MPI|yes, Open MPI 5.0.11|
|Meson|1.12.1|
|CMake|cmake version 3.31.8|
|Ninja|1.13.2|

## Counts

|Leg|Summary|
|---|---|
|Meson, rpc and cache on|Ok 139, Expected Fail 2, Fail 0, Unexpected Pass 0, Skipped 0, Ignored 0, Timeout 0 (141 total)|
|Meson, rpc and cache off|Ok 116, Expected Fail 2, Fail 0, Unexpected Pass 0, Skipped 0, Ignored 0, Timeout 0 (118 total)|
|Meson, rpc and cache on, MPI enabled|Ok 164, Expected Fail 2, Fail 0, Unexpected Pass 0, Skipped 0, Ignored 0, Timeout 0 (166 total)|
|CMake, rpc and cache on|153 passed, 0 failed (153 total)|

## Meson, rpc and cache on

Configure, build, and test:

```text
meson setup \
  build/verify-meson-on \
  -Dwith_tests=True \
  -Dwith_examples=False \
  -Dwith_xtensor=False \
  -Dwith_eigen=False \
  -Dpure_lib=False \
  -Dwith_rpc=true \
  -Dwith_cache=true

meson compile \
  -C \
  build/verify-meson-on

meson test \
  -C \
  build/verify-meson-on \
  --print-errorlogs
```

Resolved features are parsed from the setup log and the
host system. Configured is the option value.

|Option|Configured|Resolved|
|---|---|---|
|with_fortran_pots|auto|enabled|
|with_fast_nlist|auto|enabled|
|with_mpi|disabled|disabled|
|with_ipi|auto|enabled|

|Result|Count|
|---|---:|
|Ok|139|
|Expected Fail|2|
|Fail|0|
|Unexpected Pass|0|
|Skipped|0|
|Ignored|0|
|Timeout|0|
|Total|141|

## Meson, rpc and cache off

Configure, build, and test:

```text
meson setup \
  build/verify-meson-off \
  -Dwith_tests=True \
  -Dwith_examples=False \
  -Dwith_xtensor=False \
  -Dwith_eigen=False \
  -Dpure_lib=False \
  -Dwith_rpc=false \
  -Dwith_cache=false

meson compile \
  -C \
  build/verify-meson-off

meson test \
  -C \
  build/verify-meson-off \
  --print-errorlogs
```

Resolved features are parsed from the setup log and the
host system. Configured is the option value.

|Option|Configured|Resolved|
|---|---|---|
|with_fortran_pots|auto|enabled|
|with_fast_nlist|auto|enabled|
|with_mpi|disabled|disabled|
|with_ipi|auto|enabled|

|Result|Count|
|---|---:|
|Ok|116|
|Expected Fail|2|
|Fail|0|
|Unexpected Pass|0|
|Skipped|0|
|Ignored|0|
|Timeout|0|
|Total|118|

## Meson, rpc and cache on, MPI enabled

Configure, build, and test:

```text
meson setup \
  build/verify-meson-mpi \
  -Dwith_tests=True \
  -Dwith_examples=False \
  -Dwith_xtensor=False \
  -Dwith_eigen=False \
  -Dpure_lib=False \
  -Dwith_rpc=true \
  -Dwith_cache=true \
  -Dwith_mpi=enabled

meson compile \
  -C \
  build/verify-meson-mpi

meson test \
  -C \
  build/verify-meson-mpi \
  --print-errorlogs
```

Resolved features are parsed from the setup log and the
host system. Configured is the option value.

|Option|Configured|Resolved|
|---|---|---|
|with_fortran_pots|auto|enabled|
|with_fast_nlist|auto|enabled|
|with_mpi|enabled|enabled|
|with_ipi|auto|enabled|

|Result|Count|
|---|---:|
|Ok|164|
|Expected Fail|2|
|Fail|0|
|Unexpected Pass|0|
|Skipped|0|
|Ignored|0|
|Timeout|0|
|Total|166|

## CMake, rpc and cache on

Configure, build, and test:

```text
cmake -S \
  . \
  -B \
  build/verify-cmake-on \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_BUILD_TYPE=Release \
  -DRGPOT_BUILD_TESTS=True \
  -DRGPOT_BUILD_EXAMPLES=False \
  -DRGPOT_WITH_XTENSOR=OFF \
  -DRGPOT_WITH_EIGEN=OFF \
  -DRGPOT_WITH_RPC=ON \
  -DRGPOT_WITH_CACHE=ON \
  -DRGPOT_PURE_LIB=OFF

cmake --build \
  build/verify-cmake-on \
  --parallel \
  8

ctest --test-dir \
  build/verify-cmake-on \
  --output-on-failure
```

CMake runs find_package(MPI) even when the configure
arguments do not name MPI. RGPOT_HAVE_MPI is ON only when
the compile check succeeds. Cache values below were checked
against the configure arguments. Fortran and RGPOT_HAVE_MPI
are parsed from the CMake setup log. with_fast_nlist is a
Meson option.

|Cache variable|Value|
|---|---|
|CMAKE_CXX_COMPILER_LAUNCHER|ccache|
|CMAKE_C_COMPILER_LAUNCHER|ccache|
|CMAKE_BUILD_TYPE|Release|
|RGPOT_BUILD_TESTS|True|
|RGPOT_BUILD_EXAMPLES|False|
|RGPOT_WITH_XTENSOR|OFF|
|RGPOT_WITH_EIGEN|OFF|
|RGPOT_WITH_RPC|ON|
|RGPOT_WITH_CACHE|ON|
|RGPOT_PURE_LIB|OFF|

|Detection|Value|
|---|---|
|Fortran|GNU 14.4.0|
|RGPOT_HAVE_MPI|ON|

|Result|Count|
|---|---:|
|Passed|153|
|Failed|0|
|Total|153|

The crossover checks are `test_omp_crossover_v3_provenance.py`,
`test_omp_crossover_v3_provenance_cases.py`,
`test_omp_crossover_v3_report.py`, `test_omp_crossover_v3_analysis.py`,
and `test_omp_crossover_v3_verification.py`. A bare provenance check
from the repository root is:

```text
python3 bench/test_omp_crossover_v3_provenance.py
```

These checks are entries in `CppCore/tests/registry.txt`, which the
Meson and CMake test runs consume.
