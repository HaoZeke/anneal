#!/usr/bin/env bash
# Compare nng catalog carrier RTT against the TCP vat on packing-gt-stop.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "terra_nng_rpc_rtt.sh: run under srun/sbatch" >&2
  exit 1
fi
export PATH="${HOME}/.cargo/bin:/usr/bin:${PATH}"
NNG_ROOT=${NNG_ROOT:-$HOME/build/anneal-nng-rpc/src}
TCP_ROOT=${TCP_ROOT:-$HOME/build/anneal-tcp-rtt/src}
NNG_TARGET=${NNG_TARGET:-$HOME/build/anneal-nng-rpc/target}
TCP_TARGET=${TCP_TARGET:-$HOME/build/anneal-tcp-rtt/target}
TCP_TEST="$TCP_ROOT/tests/catalog_carrier_rtt.rs"
cleanup_tcp_test() {
  if [[ -f $TCP_TEST ]]; then
    if command -v rtrash >/dev/null; then
      rtrash -f "$TCP_TEST"
    else
      mv -f "$TCP_TEST" /tmp/catalog_carrier_rtt.rs.bak
    fi
  fi
}
trap cleanup_tcp_test EXIT
echo "host=$(hostname) job=$SLURM_JOB_ID"

echo "=== nng pair vs tcp loopback (same binary) ==="
export CARGO_TARGET_DIR="$NNG_TARGET"
cd "$NNG_ROOT"
cargo test --features bank-rpc --lib nng_rpc::tests::pair_roundtrip -- --nocapture --exact
cargo test --features bank-rpc --lib nng_rpc::tests::tcp_loopback -- --nocapture --exact

echo "=== nng catalog 200 snapshots ==="
cargo test --features bank-rpc --test catalog_carrier_rtt -- --nocapture

echo "=== TCP catalog 200 snapshots (packing-gt-stop vat) ==="
mkdir -p "$TCP_ROOT/tests"
cp -f "$NNG_ROOT/tests/catalog_carrier_rtt.rs" "$TCP_TEST"
export CARGO_TARGET_DIR="$TCP_TARGET"
cd "$TCP_ROOT"
cargo test --features bank-rpc --test catalog_carrier_rtt -- --nocapture
echo RTT_OK
