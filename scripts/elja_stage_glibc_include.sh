#!/usr/bin/env bash
# Login node only. s-normal compute has glibc DSOs and no headers.
# nng-sys cmake needs stdint.h.
set -euo pipefail
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  echo "elja_stage_glibc_include.sh: run on the login node, not $(hostname)" >&2
  exit 1
fi
DEST=${ELJA_GLIBC_INCLUDE:-$HOME/elja-glibc-include}
if [[ ! -f /usr/include/stdint.h ]]; then
  echo "missing /usr/include/stdint.h on $(hostname)" >&2
  exit 1
fi
mkdir -p "$DEST"
rsync -a --delete /usr/include/ "$DEST/"
test -f "$DEST/stdint.h"
echo "GLIBC_INCLUDE_OK $DEST"
