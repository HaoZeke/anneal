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

descriptor_math_pixi="${PIXI_BIN:-pixi}"
"${descriptor_math_pixi}" run -e verify pytest \
  proofs/tests/test_descriptor_pullback.py \
  proofs/tests/test_catalog_differential.py \
  -q
