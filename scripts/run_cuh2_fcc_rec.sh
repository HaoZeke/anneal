#!/usr/bin/env bash
# Serial recommended CuH2 slab on the FCC fixture.
# Budget 2500, seeds 0 through 7, one process at a time.
set -euo pipefail

cd "$(dirname "$0")/.."
unset CARGO_TARGET_DIR
msrv=$(awk -F'"' '/^rust-version = / { print $2; exit }' Cargo.toml)
case $msrv in
  *.*.*) tool=$msrv ;;
  *.*) tool=${msrv}.0 ;;
  *) echo "Cargo.toml has no rust-version" >&2; exit 1 ;;
esac
export RUSTUP_TOOLCHAIN=$tool
lib=${1:-engines/librgpot_cuh2.so}
con=examples/fixtures/cuh2_fcc_slab.con
outdir=paper/jctc/logs
receipt=$outdir/cuh2-hardware-receipt.txt

if [[ ! -s $lib ]]; then
  echo "missing $lib" >&2
  exit 1
fi
if [[ ! -f $con ]]; then
  echo "missing $con" >&2
  exit 1
fi
if [[ ! -s engines/rgpot.sha ]]; then
  echo "missing engines/rgpot.sha" >&2
  exit 1
fi

capture() {
  printf '%s\n' "$1"
  printf 'script\n'
  printf 'scripts/run_cuh2_fcc_rec.sh\n'
  printf 'host\n'
  hostname
  printf 'boot_id\n'
  cat /proc/sys/kernel/random/boot_id
  printf 'uptime\n'
  uptime
  printf 'cpuinfo_mhz\n'
  awk -F: '/^[Cc]pu MHz/ { gsub(/^[ \t]+/, "", $2); print $2 }' /proc/cpuinfo
  printf 'lscpu\n'
  lscpu
}

mkdir -p "$outdir"
{
  capture start
  printf 'rustc\n'
  rustc -V
  printf 'cargo\n'
  cargo -V
  printf 'gfortran_v\n'
  gfortran -v
  printf 'gpp_v\n'
  g++ -v
} >"$receipt" 2>&1

target=$(
  cargo metadata --format-version 1 --no-deps |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])'
)
cargo build --locked --release \
  --features rgpot-ex,featomic,bank-rpc \
  --example slab_adsorption
bin=$target/release/examples/slab_adsorption
if [[ ! -x $bin ]]; then
  echo "missing $bin" >&2
  exit 1
fi

for seed in 0 1 2 3 4 5 6 7; do
  SEED_OFFSET=$seed RGPOT_CUH2_LIBRARY=$lib \
    "$bin" "$con" 2500 1 recommended \
    >"$outdir/cuh2_fcc_rec_${seed}.out"
  rm -f "best_slab_eindir_s${seed}.xyz"
done

{
  capture end
  printf 'rgpot\n'
  cat engines/rgpot.sha
  printf 'library_sha256\n'
  sha256sum "$lib" | awk '{print $1}'
} >>"$receipt"
