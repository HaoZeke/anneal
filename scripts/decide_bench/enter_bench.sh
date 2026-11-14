#!/usr/bin/env bash
# Copyright (c) anneal developers
# Build the panel binaries at HEAD and append raw rows on hostname bench.
#
#   enter_bench.sh <variant> [<variant> ...] [--system NAME] [--seed N]
#
# Variants: baseline, hyperband, thompson, ducb, adapt. Rows go to
# bench/decide/rows.csv; finished (variant, system, seed) rows are kept.
# The Sutton-Chen potential is rgpot's librgpot.so (RGPOT_SC_LIB).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${DECIDE_WORK:-/work/decide}
LIB=${RGPOT_SC_LIB:-/work/rgpot/bdir-xtb/CppCore/librgpot.so}
cargo_home=${CARGO_HOME:-$HOME/.cargo}

if [ "$#" -lt 1 ]; then
  echo "usage: enter_bench.sh <variant> ... [--system NAME] [--seed N]" >&2
  exit 2
fi

revision=$(git -C "$ROOT" rev-parse HEAD)
BIN=$WORK/bin-$revision
if [ ! -x "$BIN/h2h_sc" ]; then
  mkdir -p "$WORK/target" "$WORK/rustc-tmp" "$BIN"
  (
    cd "$ROOT"
    export CARGO_TARGET_DIR=$WORK/target
    export TMPDIR=$WORK/rustc-tmp
    export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo"
    cargo +1.88.0 build --release --locked --features rgpot-ex \
      --example h2h_gmin_proof --example h2h_metal --example h2h_morse \
      --example h2h_seed --example h2h_sc -j 4
  )
  for name in h2h_gmin_proof h2h_metal h2h_morse h2h_seed h2h_sc; do
    cp -a "$WORK/target/release/examples/$name" "$BIN/$name"
  done
  printf '%s\n' "$revision" > "$BIN/anneal-head"
fi
cp -a "$HERE/run_rows.py" "$BIN/run_rows.py"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

user=$(id -un)
exec sudo -n unshare --uts bash -c \
  'hostname bench; exec sudo -u "$0" env RGPOT_SC_LIB="$1" LD_LIBRARY_PATH="$2" \
     PATH=/usr/bin:/bin python3 -u "$3/run_rows.py" --bindir "$3" \
     --head-file "$3/anneal-head" --scratch "$4/scratch" --root "$5" "${@:6}"' \
  "$user" "$LIB" "$(dirname "$LIB")" "$BIN" "$WORK" "$ROOT" "$@"
