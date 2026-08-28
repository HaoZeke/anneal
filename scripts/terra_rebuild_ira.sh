#!/usr/bin/env bash
# Rebuild libira.so on a Terra Slurm allocation using in-tree lap_local.
# Does not link MKL. Requires SLURM_JOB_ID.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_rebuild_ira.sh: run under srun, not on $(hostname)" >&2
  exit 1
fi
IRA=${IRA_ROOT:-$HOME/ira}
if [[ ! -d $IRA/src ]]; then
  echo "missing $IRA/src" >&2
  exit 1
fi
echo "host=$(hostname) job=$SLURM_JOB_ID"
gfortran --version | head -1
rm -rf "$IRA/src/Obj"
mkdir -p "$IRA/src/Obj" "$IRA/include" "$IRA/lib"
FFLAGS="-fPIC -cpp -O3 -ffree-line-length-512 -funroll-loops"
cd "$IRA/src"
LIBLAPACK= make shlib FFLAGS="$FFLAGS"
gfortran -o "$IRA/lib/libira.so.new" -shared \
  -J"$IRA/include" -I"$IRA/include" \
  "$IRA/src/Obj"/*.o \
  -Wl,-soname,libira.so
mv -f "$IRA/lib/libira.so.new" "$IRA/lib/libira.so"
ln -sfn "$IRA/lib/libira.so" "$IRA/src/libira.so"
echo "=== ldd libira.so ==="
ldd "$IRA/lib/libira.so"
if ldd "$IRA/lib/libira.so" | grep -F "not found" >/dev/null; then
  echo "libira still has unresolved deps" >&2
  exit 1
fi
if ldd "$IRA/lib/libira.so" | grep -F mkl >/dev/null; then
  echo "libira still needs MKL" >&2
  exit 1
fi
echo "IRA_OK $IRA/lib/libira.so"
