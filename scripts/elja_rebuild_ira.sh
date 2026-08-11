#!/usr/bin/env bash
# Rebuild libira.so on Elja without FlexiBLAS.
# Uses IRA's in-tree lap_local and OHPC gcc 12, with rpath so compute
# nodes load libgfortran/libquadmath without a module.
set -euo pipefail
GCC=${GCC_ROOT:-/opt/ohpc/pub/compiler/gcc/12.4.0}
export PATH="${GCC}/bin:${PATH}"
IRA=${IRA_ROOT:-$HOME/ira}
GCCLIB=${GCCLIB:-${GCC}/lib64}
if [[ ! -d $IRA/src ]]; then
  echo "missing $IRA/src" >&2
  exit 1
fi
if [[ ! -e $GCCLIB/libgfortran.so.5 || ! -e $GCCLIB/libquadmath.so.0 ]]; then
  echo "missing gfortran/quadmath under $GCCLIB" >&2
  exit 1
fi
if [[ -e $IRA/lib/libira.so ]]; then
  cp -a "$IRA/lib/libira.so" "$IRA/libira.so.flexiblas-bak"
fi
echo "gfortran=$(gfortran --version | head -1)"
cd "$IRA/src"
make clean
# Empty LIBLAPACK compiles lap_local/lap.f instead of -llapack (FlexiBLAS).
# No -march=native: login is Sapphire Rapids; AMD compute nodes SIGILL on that.
FFLAGS="-fPIC -cpp -O3 -ffree-line-length-512 -funroll-loops"
export FFLAGS
LIBLAPACK= make shlib FFLAGS="$FFLAGS"
# Bake gcc 12 into DT_RUNPATH so HQ tasks do not need LD_LIBRARY_PATH.
gfortran -o "$IRA/lib/libira.so" -shared \
  -J"$IRA/include" -I"$IRA/include" \
  "$IRA/src/Obj"/*.o \
  -Wl,-soname,libira.so \
  -Wl,-rpath,"$GCCLIB"
ln -sfn "$IRA/lib/libira.so" "$IRA/src/libira.so"
echo "=== ldd libira.so ==="
ldd "$IRA/lib/libira.so"
if ldd "$IRA/lib/libira.so" | grep -q "not found"; then
  echo "libira still has unresolved deps" >&2
  exit 1
fi
if ldd "$IRA/lib/libira.so" | grep -q flexiblas; then
  echo "libira still needs FlexiBLAS" >&2
  exit 1
fi
echo "IRA_OK $IRA/lib/libira.so"
