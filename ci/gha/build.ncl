# Audit/reference mirror of orchestrator build/rust/bridge jobs (not exported as live workflow).
# Live: .github/workflows/ci-orchestrator.yml
let S = import "lib/steps.ncl" in
let P = S.pins in

let meson_build_test =
  S.run_pixi m%"
  %{S.darwin_env_prelude}
  meson setup bbdir \
    ${POTCTL_MESON_EXTRA:-} \
    -Dwith_tests=True \
    -Dwith_examples=False \
    -Dwith_xtensor=False \
    -Dwith_eigen=False \
    -Dwith_rpc=${{ matrix.rpc }} \
    -Dwith_cache=${{ matrix.cache }} \
    -Dpure_lib=False
  meson compile -C bbdir
  meson test -C bbdir --print-errorlogs
  ccache -s
"%
  & {
    name = "Build & Test (Meson)",
    "if" = "${{ matrix.sys == 'meson' }}",
  }
in

let cmake_build_test =
  S.run_pixi m%"
  CMAKE_EXTRA=""
  %{S.darwin_env_prelude}
  if [ "${{ runner.os }}" = "macOS" ]; then
    CMAKE_EXTRA="-DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_OSX_SYSROOT=${SDKROOT}"
    if [ -n "${CONDA_PREFIX:-}" ]; then
      CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DCMAKE_BUILD_RPATH=${CONDA_PREFIX}/lib -DCMAKE_INSTALL_RPATH=${CONDA_PREFIX}/lib"
    fi
    if [ -n "${FC:-}" ]; then
      CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_Fortran_COMPILER=${FC}"
    fi
  fi
  cmake -S . -B build \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_TYPE=Release \
    -DRGPOT_BUILD_TESTS=True \
    -DRGPOT_BUILD_EXAMPLES=False \
    -DRGPOT_WITH_XTENSOR=OFF \
    -DRGPOT_WITH_EIGEN=OFF \
    -DRGPOT_WITH_RPC=${{ matrix.rpc }} \
    -DRGPOT_WITH_CACHE=${{ matrix.cache }} \
    -DRGPOT_PURE_LIB=OFF \
    $CMAKE_EXTRA
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ccache -s
"%
  & {
    name = "Build & Test (CMake)",
    "if" = "${{ matrix.sys == 'cmake' }}",
  }
in

let rpc_integration =
  S.run_pixi m%"
  pixi r -e rpctest \
  python tests/rpc_integ.py --server-bin ${{ matrix.sys == 'meson' && 'bbdir/CppCore/potserv' || 'build/potserv' }}
"%
  & {
    name = "RPC Integration Test",
    "if" = "${{ matrix.rpc == 'true' && matrix.sys == 'meson' && matrix.os == 'ubuntu-latest' }}",
  }
in

let windows_meson_build_test =
  S.run_bash m%"
  set -euo pipefail
  # MSVC lib.exe must win over flang's llvm-ar (see win-flang-msvc.ini).
  export AR=lib
  pixi run -e devbld meson setup bbdir \
    --buildtype=release \
    --native-file %{P.win_flang_msvc_native_file} \
    -Dwith_tests=true \
    -Dwith_examples=false \
    -Dwith_xtensor=false \
    -Dwith_eigen=true \
    -Dwith_rpc=true \
    -Dwith_fortran_pots=enabled \
    -Dwith_cache=false \
    -Dpure_lib=false
  pixi run -e devbld meson compile -C bbdir
  pixi run -e devbld meson test -C bbdir --print-errorlogs
"%
  & { name = "Build & Test (Meson / MSVC + Flang kernels)" }
in

let nodownload_export =
  S.run_bash m%"
  set -euo pipefail
  rm -rf /tmp/rgpot-src
  mkdir -p /tmp/rgpot-src
  git archive HEAD | tar -x -C /tmp/rgpot-src
  # A conda-forge source build sees exactly this: tracked files, no .git, no
  # checked-out wrap subprojects.
  test ! -e /tmp/rgpot-src/.git
"%
  & { name = "Export tracked tree (git archive, no .git)" }
in

let nodownload_build_test =
  S.run_pixi m%"
  cd /tmp/rgpot-src
  meson setup bb \
    --wrap-mode=nodownload \
    -Dwith_tests=true \
    -Dwith_examples=false \
    -Dwith_xtensor=false \
    -Dwith_eigen=false \
    -Dwith_rpc=false \
    -Dwith_cache=false \
    -Dwith_fast_nlist=disabled \
    -Dpure_lib=true
  meson compile -C bb
  meson test -C bb --print-errorlogs
"%
  & { name = "Configure + build + test with --wrap-mode=nodownload" }
in

let fast_nlist_off_build_test =
  S.run_pixi m%"
  meson setup bbdir \
    -Dwith_tests=true \
    -Dwith_examples=false \
    -Dwith_xtensor=false \
    -Dwith_eigen=false \
    -Dwith_rpc=false \
    -Dwith_cache=false \
    -Dwith_fast_nlist=disabled \
    -Dpure_lib=true
  meson compile -C bbdir
  meson test -C bbdir --print-errorlogs
"%
  & { name = "Build and test without the fast pair list" }
in

let thread_limit_build =
  S.run_pixi m%"
  meson setup bbdir \
    -Dwith_tests=true \
    -Dwith_examples=false \
    -Dwith_xtensor=false \
    -Dwith_eigen=false \
    -Dwith_rpc=false \
    -Dwith_cache=false
  meson compile -C bbdir
"%
  & { name = "Build", id = "build" }
in

let thread_limit_test = fun limit =>
  S.run_pixi "meson test -C bbdir --print-errorlogs --logbase limit%{limit}"
  & {
    name = "Test with OMP_THREAD_LIMIT=%{limit}",
    "if" = "${{ !cancelled() && steps.build.outcome == 'success' }}",
    env = { OMP_THREAD_LIMIT = limit },
  }
in

{
  name = "Build Matrix",
  on = S.on_main_pr,
  concurrency = {
    group = "${{ github.workflow }}-${{ github.ref }}",
    "cancel-in-progress" = true,
  },
  # build_and_test: no in-tree NWChem; dlopens libnwchemc. Frontend + stub always compile.
  # Suite nwchem energy/grad needs external libnwchemc; stub probe always runs.
  jobs = {
    "ci-tools" = S.job_ci_tools_os_matrix [P.ubuntu, P.macos],
    build_and_test = {
      needs = ["ci-tools"],
      name = "${{ matrix.sys }} / ${{ matrix.os }} [RPC:${{ matrix.rpc }} CACHE:${{ matrix.cache }}]",
      "runs-on" = "${{ matrix.os }}",
      env = S.ccache_env,
      strategy = {
        "fail-fast" = false,
        matrix = {
          os = [P.ubuntu, P.macos],
          sys = ["meson", "cmake"],
          rpc = ["true", "false"],
          cache = ["true", "false"],
        },
      },
      steps =
        [S.checkout_full, S.ensure_potctl, S.setup_pixi "devbld rpctest" "devbld", S.ccache_step_matrix_sys]
        @ [meson_build_test, cmake_build_test, rpc_integration],
    },
    # Single explicit Windows leg, deliberately NOT folded into the
    # os x sys x rpc x cache matrix above (that would take 16 legs to 24).
    # meson only, MSVC + Flang kernels + Cap'n Proto RPC (WIN32_LEAN_AND_MEAN).
    # Native file forces ar=lib so link.exe accepts flang COFF objects.
    build_windows_meson = {
      name = "meson / windows-2022 [MSVC flang RPC:true CACHE:false]",
      "runs-on" = P.windows,
      steps = [
        S.checkout_minimal,
        # No activate-environment: vs2022_win-64/vswhere rewrite PATH and
        # setup-pixi refuses non-append activation. Use `pixi run -e devbld`.
        # environments is space-separated (comma forms one invalid name).
        S.setup_pixi "devbld" false,
        S.activate_msvc,
        windows_meson_build_test,
      ],
    },
    # Offline rehearsal for the conda-forge source build: configure, build and
    # test the `git archive` tree with --wrap-mode=nodownload, so the default
    # feature set is proven to need no wrap subproject.
    #
    # Scope: the Fortran pot kernels are in-tree (CppCore/rgpot/fortran, vesin
    # vendored under third_party), so this job builds them from the archived
    # tree with no wrap involved. The wraps stay unexercised by construction:
    # eigen (with_eigen), xxhash (with_cache), potentials-schema (with_rpc),
    # fmt (pure_lib).
    nodownload_rehearsal = {
      name = "offline tarball build (nodownload / ubuntu)",
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_minimal,
        S.setup_pixi_default "default",
        nodownload_export,
        nodownload_build_test,
      ],
    },
    # Cargo is present on this runner. The option still has to build and
    # test the C++ pair scan, which is what wheels and conda get.
    build_fast_nlist_off = {
      name = "meson / ubuntu [fast nlist off]",
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_minimal,
        S.setup_pixi "devbld" "devbld",
        fast_nlist_off_build_test,
      ],
    },
    # Under OMP_THREAD_LIMIT every region gets fewer threads than it asks
    # for. One build, then the suite at a limit of one and of two.
    thread_limit = {
      name = "meson / ubuntu [OMP_THREAD_LIMIT 1, 2]",
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_minimal,
        S.setup_pixi "devbld" "devbld",
        thread_limit_build,
        thread_limit_test "1",
        thread_limit_test "2",
      ],
    },
    rust_tests = {
      needs = ["ci-tools"],
      name = "Rust Core Tests (${{ matrix.os }})",
      "runs-on" = "${{ matrix.os }}",
      strategy = {
        "fail-fast" = false,
        matrix = { os = [P.ubuntu, P.macos] },
      },
      steps = [
        S.checkout_full,
        S.ensure_potctl,
        S.setup_pixi "devbld" "devbld",
        {
          name = "Install cargo-nextest",
          uses = P.install_action,
          with = { tool = "cargo-nextest" },
        },
        { name = "potctl unit tests", run = "cargo test -p potctl" },
        S.run_pixi m%"
          cargo nextest run -p rgpot-core
          cargo nextest run -p rgpot-core --all-features
        "%
        & { name = "Run Rust tests (nextest)" },
        S.run_pixi m%"
          cargo test --doc -p rgpot-core
        "%
        & { name = "Run Rust doc-tests" },
      ],
    },
    client_bridge_stress = {
      needs = ["ci-tools"],
      name = "Client Bridge Stress Test (Linux)",
      "runs-on" = P.ubuntu,
      env = {
        CCACHE_DIR = "${{ github.workspace }}/.ccache",
        CCACHE_COMPILERCHECK = "content",
      },
      steps = [
        S.checkout_full,
        S.ensure_potctl,
        S.setup_pixi "devbld rpctest" "devbld",
        S.ccache_step_matrix_sys,
        S.run_pixi m%"
          echo "--- Building Server (Heavy) ---"
          meson setup bbdir_server \
                -Dwith_rpc=true \
                -Dwith_tests=false \
                -Dwith_examples=false
          meson compile -C bbdir_server
        "%,
        S.run_pixi m%"
          echo "--- Building Client Bridge (Light) ---"
          cmake -S . -B build_client \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_BUILD_TYPE=Release \
                -DRGPOT_RPC_CLIENT_ONLY=ON \
                -DRGPOT_BUILD_TESTS=ON
          cmake --build build_client -j
        "%,
        S.run_pixi m%"
          echo "--- Starting Server ---"
          ./bbdir_server/CppCore/potserv 12345 LJ &
          SERVER_PID=$!
          sleep 2
          echo "--- Running Client Stress Tests ---"
          ctest --test-dir build_client --output-on-failure
          echo "--- Teardown ---"
          kill $SERVER_PID
        "%,
      ],
    },
  },
}
