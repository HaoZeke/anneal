#!/usr/bin/env bash
# bootstrap_cutest.sh: clone + build the CUTEst stack into .bench/
# so pycutest can find it. Sets MASTSIF / SIFDECODE / CUTEST / ARCHDEFS
# in the current shell (source this script, do not exec).
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

# Build SIFDecode + CUTEst via Meson (the ARCHDefs path is the legacy
# install shim; modern pycutest needs Meson-built libs).
build_meson() {
  local pkg="$1"; local src="$2"
  if [ -f "$src/builddir/build.ninja" ]; then
    echo "[bootstrap] $pkg already configured"
  else
    echo "[bootstrap] Configuring $pkg with Meson"
    (cd "$src" && meson setup builddir --prefix="$src/install" >/dev/null 2>&1)
  fi
  if [ ! -f "$src/install/lib/libsifdecode.so" ] && [ ! -f "$src/install/lib/libcutest.so" ]; then
    echo "[bootstrap] Building $pkg"
    (cd "$src" && meson compile -C builddir >/dev/null 2>&1 && \
                  meson install -C builddir >/dev/null 2>&1) || \
      echo "[bootstrap] WARNING: $pkg build had non-fatal errors; continuing"
  else
    echo "[bootstrap] $pkg already installed"
  fi
}

build_meson SIFDecode "$SIFDECODE"
build_meson CUTEst    "$CUTEST"

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
