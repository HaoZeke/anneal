#!/usr/bin/env bash
# Compute images have the glibc DSOs but no startfiles and no
# libm.so / libc.so linker scripts (those pull libc_nonshared.a).
# Copy crti from login and write scripts that only GROUP the DSOs.
set -euo pipefail
SYS=${IRA_SYSROOT:-$HOME/ira/sysroot}
mkdir -p "$SYS"
src=/usr/lib64
missing=0
for f in crti.o crt1.o crtn.o Scrt1.o libc_nonshared.a libmvec_nonshared.a; do
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
ln -sfn /lib64/libm.so.6 "$SYS/libm.so"
ln -sfn /lib64/libpthread.so.0 "$SYS/libpthread.so"
ln -sfn /lib64/libdl.so.2 "$SYS/libdl.so"
ln -sfn /lib64/librt.so.1 "$SYS/librt.so"
ln -sfn /lib64/libutil.so.1 "$SYS/libutil.so"
ln -sfn /opt/ohpc/pub/compiler/gcc/12.4.0/lib64/libgcc_s.so.1 "$SYS/libgcc_s.so"
# GNU ld script so Scrt1.o can resolve __libc_csu_* from libc_nonshared.a.
cat >"$SYS/libc.so" <<EOF
OUTPUT_FORMAT(elf64-x86-64)
GROUP ( /lib64/libc.so.6 $SYS/libc_nonshared.a AS_NEEDED ( /lib64/ld-linux-x86-64.so.2 ) )
EOF
echo "SYSROOT_OK $SYS"
