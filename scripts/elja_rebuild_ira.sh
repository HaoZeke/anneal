#!/usr/bin/env bash
# Rebuild libira.so on an Elja compute node. Login-node gfortran
# -march=native is Sapphire Rapids and SIGILLs on amd-compute.
# Uses IRA's in-tree lap_local and OHPC gcc 12, with rpath so tasks
# load libgfortran/libquadmath without a module.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_rebuild_ira.sh: run under srun, not on $(hostname)" >&2
  exit 1
fi
GCC=${GCC_ROOT:-/opt/ohpc/pub/compiler/gcc/12.4.0}
export PATH="${GCC}/bin:${PATH}"
IRA=${IRA_ROOT:-$HOME/ira}
GCCLIB=${GCCLIB:-${GCC}/lib64}
SYS=${IRA_SYSROOT:-$HOME/ira/sysroot}
if [[ ! -e $SYS/crti.o ]]; then
  echo "missing $SYS/crti.o; run scripts/elja_stage_sysroot.sh on the login node" >&2
  exit 1
fi
export LIBRARY_PATH="${SYS}:${GCCLIB}:/usr/lib64:${LIBRARY_PATH:-}"
echo "host=$(hostname) job=$SLURM_JOB_ID"
lscpu | grep -E "Model name|Vendor ID" || true
if [[ ! -d $IRA/src ]]; then
  echo "missing $IRA/src" >&2
  exit 1
fi
if [[ ! -e $GCCLIB/libgfortran.so.5 || ! -e $GCCLIB/libquadmath.so.0 ]]; then
  echo "missing gfortran/quadmath under $GCCLIB" >&2
  exit 1
fi
echo "gfortran=$(gfortran --version | head -1)"
# Do not `make clean` the live lib/: HQ tasks keep the inode open and
# NFS leaves a .nfs* that makes `rm -rf lib` fail.
rm -rf "$IRA/src/Obj"
mkdir -p "$IRA/src/Obj" "$IRA/include" "$IRA/lib"
# Empty LIBLAPACK compiles lap_local/lap.f instead of -llapack (FlexiBLAS).
# Portable ISA: Intel and AMD compute share this .so.
FFLAGS="-fPIC -cpp -O3 -ffree-line-length-512 -funroll-loops -B${SYS}"
cd "$IRA/src"
LIBLAPACK= make shlib FFLAGS="$FFLAGS"
# Relink with rpath; write a new inode so mapped tasks keep the old file.
gfortran -B"$SYS" -o "$IRA/lib/libira.so.new" -shared \
  -J"$IRA/include" -I"$IRA/include" \
  "$IRA/src/Obj"/*.o \
  -Wl,-soname,libira.so \
  -Wl,-rpath,"$GCCLIB"
mv -f "$IRA/lib/libira.so.new" "$IRA/lib/libira.so"
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
