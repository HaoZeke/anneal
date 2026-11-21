# Audit/reference mirror of orchestrator potentials_* jobs (not exported as live workflow).
# Live: .github/workflows/ci-orchestrator.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
let torch_ensure_script = m%"
set -euo pipefail
PY="$(command -v python)"
install_missing_pkgs() {
  # Only install packages whose purelib trees are missing. Avoid
  # --reinstall of torch: it can drop the cmake share/ files meson needs.
  local need=()
  local purelib
  purelib="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  [[ -d "${purelib}/metatensor/include" ]] || need+=("metatensor-torch>=0.8.4,<0.9")
  [[ -d "${purelib}/metatomic/torch" ]] || need+=("metatomic-torch>=0.1.9,<0.2")
  [[ -f "${purelib}/vesin/include/vesin.h" ]] || need+=("vesin>=0.6.0,<0.7" "vesin-torch>=0.6.0,<0.7")
  if ((${#need[@]} == 0)); then
    echo "metatomic/metatensor/vesin purelib trees already present"
    return 0
  fi
  echo "Installing missing wheels: ${need[*]}"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" --no-cache "${need[@]}"
  else
    "$PY" -m ensurepip --upgrade || true
    "$PY" -m pip install --no-cache-dir "${need[@]}"
  fi
}
reinstall_torch_cpu() {
  echo "Reinstalling CPU torch wheel (libtorch missing)"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" --reinstall --no-cache \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.10,<2.11"
  else
    "$PY" -m ensurepip --upgrade || true
    "$PY" -m pip install --force-reinstall --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.10,<2.11"
  fi
}
PURELIB="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# CUDA torch wheels make find_package(Torch) fail without a toolkit;
# CI has no CUDA, so always prefer the CPU wheel for meson/cmake.
TORCH_VER="$("$PY" -c "
import re, pathlib, sysconfig
vf = pathlib.Path(sysconfig.get_paths()['purelib']) / 'torch' / 'version.py'
if not vf.is_file():
    print('')
else:
    m = re.search(r\"^__version__\\s*=\\s*['\\\"]([^'\\\"]+)\", vf.read_text(), re.M)
    print(m.group(1) if m else '')
" 2>/dev/null || true)"
if [[ ! -f "${PURELIB}/torch/lib/libtorch_global_deps.so" ]] || \
   [[ "${TORCH_VER}" == *"+cu"* ]] || [[ "${TORCH_VER}" == *"+cuda"* ]]; then
  echo "Forcing CPU torch (current version: '${TORCH_VER:-missing}')"
  reinstall_torch_cpu
  PURELIB="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
fi
install_missing_pkgs
"$PY" - <<'PY'
import pathlib, re, sys, sysconfig
purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
vf = purelib / "torch" / "version.py"
m = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)", vf.read_text(), re.M)
if not m:
    print(f"torch version not found in {vf}", file=sys.stderr)
    sys.exit(5)
tver = ".".join(m.group(1).split("+")[0].split(".")[:2])
torch_share = purelib / "torch" / "share" / "cmake"
required = [
    purelib / "torch" / "lib" / "libtorch_global_deps.so",
    torch_share,
    purelib / "metatensor" / "include",
    purelib / "metatensor" / "torch" / f"torch-{tver}" / "include",
    purelib / "metatomic" / "torch" / f"torch-{tver}" / "include",
    purelib / "vesin" / "include" / "vesin.h",
    purelib / "vesin" / "lib",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    print("missing C++ layout paths:", *missing, sep="\n  ", file=sys.stderr)
    for child in sorted(purelib.iterdir())[:50]:
        print(" ", child.name, file=sys.stderr)
    sys.exit(6)
print("C++ metatomic layout ok; torch", tver, "python", sys.version.split()[0])
print("torch cmake", torch_share)
print("vesin.h", purelib / "vesin" / "include" / "vesin.h")
PY
TORCH_LIB="${PURELIB}/torch/lib"
TORCH_CMAKE="${PURELIB}/torch/share/cmake"
{
  echo "TORCH_LIB=${TORCH_LIB}"
  echo "TORCH_CMAKE=${TORCH_CMAKE}"
  echo "PURELIB=${PURELIB}"
  echo "LD_LIBRARY_PATH=${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
  echo "CMAKE_PREFIX_PATH=${TORCH_CMAKE}:${CMAKE_PREFIX_PATH:-}"
} >> "$GITHUB_ENV"
"%
in
let mta_build_script = m%"
set -euo pipefail
PY="$(command -v python)"
PURELIB="${PURELIB:-$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}"
TORCH_LIB="${TORCH_LIB:-${PURELIB}/torch/lib}"
TORCH_CMAKE="${TORCH_CMAKE:-${PURELIB}/torch/share/cmake}"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${TORCH_CMAKE}:${CMAKE_PREFIX_PATH:-}"
echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
meson setup bbdir-mta \
  -Dwith_tests=true \
  -Dwith_examples=false \
  -Dwith_metatomic=true \
  -Dwith_rpc=false \
  -Dwith_cache=false \
  --buildtype=debug
meson compile -C bbdir-mta
meson test -C bbdir-mta --print-errorlogs
ccache -s
"%
in
{
  name = "Potentials matrix",
  on = {
    push = { branches = ["main"] },
    pull_request = { branches = ["main", "develop"] },
    workflow_dispatch = {},
  },
  concurrency = S.concurrency_ref "potentials",
  jobs = {
    "ci-tools" = S.job_ci_tools_os_matrix [P.ubuntu],
    tight_binding = {
      needs = ["ci-tools"],
      name = "xtb+tblite (meson / ${{ matrix.os }})",
      "runs-on" = "${{ matrix.os }}",
      "timeout-minutes" = 25,
      strategy = {
        "fail-fast" = false,
        matrix = { os = [P.ubuntu] },
      },
      env = S.ccache_env,
      steps = [
        S.checkout_full,
        S.ensure_potctl,
        S.setup_pixi "tbbld" "tbbld",
        S.ccache_step "ccache-pot-tb",
        {
          name = "Meson build & test (xtb + tblite)",
          shell = "pixi run -e tbbld bash -e {0}",
          run = m%"
            meson setup bbdir-tb \
              -Dwith_tests=true \
              -Dwith_examples=false \
              -Dwith_xtb=true \
              -Dwith_tblite=true \
              -Dwith_rpc=false \
              -Dwith_cache=false \
              --buildtype=debug
            meson compile -C bbdir-tb
            meson test -C bbdir-tb --print-errorlogs
            ccache -s
          "%,
        },
      ],
    },
    # Compile-only. conda-forge tblite 0.6.0 (gfortran_ha02e54d_0) segfaults in
    # __dftd4_model_d4_MOD_get_polarizabilities on a GFN2 water single point,
    # including `tblite run`. The [tblite] tests stay on tight_binding
    # (tbbld, tblite 0.4.0).
    tblite_06 = {
      needs = ["ci-tools"],
      name = "tblite 0.6.0 compile (meson / ubuntu)",
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 10,
      env = S.ccache_env,
      steps = [
        S.checkout_full,
        S.ensure_potctl,
        S.setup_pixi "tblite06bld" "tblite06bld",
        S.ccache_step "ccache-pot-tb06",
        {
          name = "Compile TBLitePot against tblite 0.6",
          shell = "pixi run -e tblite06bld bash -e {0}",
          run = m%"
            set -euo pipefail
            meson setup bbdir-tb06 \
              -Dwith_tests=true \
              -Dwith_examples=false \
              -Dwith_tblite=true \
              -Dwith_rpc=false \
              -Dwith_cache=false \
              --buildtype=debug
            grep -E 'tblite found: YES 0\.6' bbdir-tb06/meson-logs/meson-log.txt
            meson compile -C bbdir-tb06 tblite_test
          "%,
        },
      ],
    },
    metatomic = {
      needs = ["ci-tools"],
      name = "metatomic+vesin (meson / ubuntu)",
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 30,
      env = {
        CCACHE_DIR = "${{ github.workspace }}/.ccache",
        CCACHE_COMPILERCHECK = "content",
        CCACHE_MAXSIZE = "800M",
        TORCH_DEVICE = "cpu",
      },
      steps = [
        S.checkout_full,
        S.ensure_potctl,
        S.setup_pixi "metatomicbld" "metatomicbld",
        S.ccache_step "ccache-pot-mta",
        {
          name = "Ensure torch + metatomic C++ layout",
          shell = "pixi run -e metatomicbld bash -e {0}",
          run = torch_ensure_script,
        },
        {
          name = "Meson build & test (metatomic)",
          shell = "pixi run -e metatomicbld bash -e {0}",
          run = mta_build_script,
        },
      ],
    },
  },
}
