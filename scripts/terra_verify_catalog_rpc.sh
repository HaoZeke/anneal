#!/usr/bin/env bash
set -euo pipefail

catalog_rpc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${catalog_rpc_root}"

cargo test --locked --release --features bank-rpc \
  --test catalog_policy \
  --test catalog_proposals \
  --test cooperative_ledger \
  --test catalog_protocol \
  --test catalog_rpc \
  --test catalog_rpc_faults \
  --test cooperative_search \
  --test feynman_kac_population

cargo test --locked --release --features bank-rpc --lib residual_field
cargo test --locked --release --features bank-rpc --lib swarm::tests

catalog_rpc_pixi="${PIXI_BIN:-}"
if [[ -z "${catalog_rpc_pixi}" ]]; then
  catalog_rpc_pixi="$(command -v pixi || true)"
fi
if [[ -z "${catalog_rpc_pixi}" && -x "${HOME}/.pixi/bin/pixi" ]]; then
  catalog_rpc_pixi="${HOME}/.pixi/bin/pixi"
fi
if [[ ! -x "${catalog_rpc_pixi}" ]]; then
  echo "terra_verify_catalog_rpc.sh: set PIXI_BIN to an executable Pixi binary" >&2
  exit 127
fi
"${catalog_rpc_pixi}" run -e verify pytest \
  proofs/tests/test_catalog_differential.py \
  -q
