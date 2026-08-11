#!/usr/bin/env bash
# Compute images have the glibc DSOs but no startfiles and no
# libm.so / libc.so linker scripts (those pull libc_nonshared.a).
# Copy crti from login and write scripts that only GROUP the DSOs.
set -euo pipefail
SYS=${IRA_SYSROOT:-$HOME/ira/sysroot}
mkdir -p "$SYS"
src=/usr/lib64
missing=0
for f in crti.o crt1.o crtn.o Scrt1.o; do
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
# rust-lld does not expand GNU ld GROUP scripts. Unversioned names
# must be real files or symlinks to the compute DSOs.
ln -sfn /lib64/libm.so.6 "$SYS/libm.so"
ln -sfn /lib64/libc.so.6 "$SYS/libc.so"
ln -sfn /lib64/libpthread.so.0 "$SYS/libpthread.so"
ln -sfn /lib64/libdl.so.2 "$SYS/libdl.so"
ln -sfn /lib64/librt.so.1 "$SYS/librt.so"
ln -sfn /lib64/libutil.so.1 "$SYS/libutil.so"
ln -sfn /opt/ohpc/pub/compiler/gcc/12.4.0/lib64/libgcc_s.so.1 "$SYS/libgcc_s.so"
echo "SYSROOT_OK $SYS"
