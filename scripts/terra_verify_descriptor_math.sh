#!/usr/bin/env bash
set -euo pipefail

descriptor_math_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${descriptor_math_root}"

cargo test --locked --release \
  --test descriptor_invariance \
  --test descriptor_pullback \
  --test census_calibration \
  --test basin_census \
  --test basin_catalog \
  --test catalog_proposals \
  --test feynman_kac_population

descriptor_math_pixi="${PIXI_BIN:-}"
if [[ -z "${descriptor_math_pixi}" ]]; then
  descriptor_math_pixi="$(command -v pixi || true)"
fi
if [[ -z "${descriptor_math_pixi}" && -x "${HOME}/.pixi/bin/pixi" ]]; then
  descriptor_math_pixi="${HOME}/.pixi/bin/pixi"
fi
if [[ ! -x "${descriptor_math_pixi}" ]]; then
  echo "terra_verify_descriptor_math.sh: set PIXI_BIN to an executable Pixi binary" >&2
  exit 127
fi
"${descriptor_math_pixi}" run -e verify pytest \
  proofs/tests/test_descriptor_pullback.py \
  proofs/tests/test_catalog_differential.py \
  -q
