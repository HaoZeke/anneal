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
write_group() {
  local name=$1
  local dso=$2
  cat >"$SYS/$name" <<EOF
OUTPUT_FORMAT(elf64-x86-64)
GROUP ( $dso )
EOF
}
write_group libm.so /lib64/libm.so.6
write_group libc.so /lib64/libc.so.6
write_group libpthread.so /lib64/libpthread.so.0
write_group libdl.so /lib64/libdl.so.2
write_group librt.so /lib64/librt.so.1
echo "SYSROOT_OK $SYS"
