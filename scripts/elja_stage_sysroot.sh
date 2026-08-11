#!/usr/bin/env bash
# Copy login-node glibc startfiles. Compute images have libm.so.6 but
# no crti.o / libm.so linker script, so gfortran and rustc cannot link.
set -euo pipefail
SYS=${IRA_SYSROOT:-$HOME/ira/sysroot}
mkdir -p "$SYS"
src=/usr/lib64
missing=0
for f in crti.o crt1.o crtn.o Scrt1.o libc.so libm.so libpthread.so libdl.so librt.so; do
  if [[ ! -e $src/$f ]]; then
    echo "missing $src/$f (run this on the login node)" >&2
    missing=1
    continue
  fi
  cp -a "$src/$f" "$SYS/"
done
if [[ $missing -ne 0 ]]; then
  exit 1
fi
echo "SYSROOT_OK $SYS"
