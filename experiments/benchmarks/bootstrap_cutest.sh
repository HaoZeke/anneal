#!/usr/bin/env bash
# bootstrap_cutest.sh: clone + build the CUTEst stack into .bench/
# so pycutest can find it. Prints the MASTSIF / SIFDECODE / CUTEST /
# ARCHDEFS values expected by pycutest.
#
# Run from the anneal repo root via:
#   pixi run -e verify bash experiments/benchmarks/bootstrap_cutest.sh
#
# First-run cost: ~5-10 min (compiles 100+ Fortran objects). Subsequent
# runs are no-ops.

set -euo pipefail

BENCH_DIR="${PIXI_PROJECT_ROOT:-$(pwd)}/.bench"
mkdir -p "$BENCH_DIR"

ARCHDEFS="$BENCH_DIR/ARCHDefs"
SIFDECODE="$BENCH_DIR/SIFDecode"
CUTEST="$BENCH_DIR/CUTEst"
MASTSIF="$BENCH_DIR/sif"

clone_or_pull() {
  local url="$1"; local dst="$2"
  if [ -d "$dst/.git" ]; then
    echo "[bootstrap] $dst exists, skipping clone"
  else
    echo "[bootstrap] Cloning $url -> $dst"
    git clone --depth 1 "$url" "$dst"
  fi
}

clone_or_pull https://github.com/ralna/ARCHDefs.git    "$ARCHDEFS"
clone_or_pull https://github.com/ralna/SIFDecode.git   "$SIFDECODE"
clone_or_pull https://github.com/ralna/CUTEst.git      "$CUTEST"
clone_or_pull https://bitbucket.org/optrove/sif.git    "$MASTSIF"

export ARCHDEFS SIFDECODE CUTEST MASTSIF
export MYARCH="pc64.lnx.gfo"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[bootstrap] ERROR: required command not found on PATH: $1" >&2
    exit 1
  fi
}

has_required_paths() {
  local path
  for path in "$@"; do
    if [ ! -e "$path" ]; then
      return 1
    fi
  done
}

require_command meson
require_command ninja

# Build SIFDecode + CUTEst via Meson (the ARCHDefs path is the legacy
# install shim; pycutest needs the Meson-installed decoder and CUTEst libs).
build_meson() {
  local pkg="$1"; local src="$2"
  shift 2
  if [ -f "$src/builddir/build.ninja" ]; then
    echo "[bootstrap] $pkg already configured"
  else
    echo "[bootstrap] Configuring $pkg with Meson"
    (cd "$src" && meson setup builddir --prefix="$src/install")
  fi
  if has_required_paths "$@"; then
    echo "[bootstrap] $pkg already installed"
  else
    echo "[bootstrap] Building $pkg"
    (cd "$src" && meson compile -C builddir && meson install -C builddir)
    if ! has_required_paths "$@"; then
      echo "[bootstrap] ERROR: $pkg install did not create required files:" >&2
      printf '  %s\n' "$@" >&2
      exit 1
    fi
  fi
}

build_meson SIFDecode "$SIFDECODE" \
  "$SIFDECODE/install/bin/sifdecoder"
build_meson CUTEst "$CUTEST" \
  "$CUTEST/install/lib/libcutest_single.a" \
  "$CUTEST/install/lib/libcutest_double.a"

cat <<EOF

[bootstrap] Done. Set the following env vars in your shell:
  export ARCHDEFS=$ARCHDEFS
  export SIFDECODE=$SIFDECODE/install
  export CUTEST=$CUTEST/install
  export MASTSIF=$MASTSIF
  export MYARCH=$MYARCH
  export PYCUTEST_CACHE=$BENCH_DIR/cache

Then test:
  pixi run -e verify python -c "import pycutest; print(pycutest.find_problems(constraints='U', n=[2,5])[:3])"
EOF
