#!/usr/bin/env bash
# Build potserv and the fake NWChem and CPMD engines, export their paths,
# and run the RPC tests through potctl. The same variables are exported
# again by `potctl ci rpc-integ` for the processes it starts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="${RGPOT_BUILD_DIR:-$ROOT/bbdir}"

case "$(uname -s)" in
  Darwin) libext=dylib ;;
  *) libext=so ;;
esac

potserv="$BUILD/CppCore/potserv"
nwchem="$BUILD/CppCore/libnwchemc_fake_engine.${libext}"
cpmd="$BUILD/CppCore/libcpmdc_fake_engine.${libext}"

if [[ ! -x "$potserv" || ! -f "$nwchem" || ! -f "$cpmd" ]]; then
  if [[ ! -f "$BUILD/build.ninja" ]]; then
    meson setup "$BUILD" \
      -Dwith_tests=true \
      -Dwith_rpc=true \
      -Dwith_examples=false
  fi
  meson compile -C "$BUILD" potserv nwchemc_fake_engine cpmdc_fake_engine
fi

export RGPOT_POTSERV="$potserv"
export NWCHEMC_LIBRARY="$nwchem"
export RGPOT_NWCHEMC_ENGINE="$nwchem"
export RGPOT_NWCHEM_ENGINE="$nwchem"
export CPMDC_LIBRARY="$cpmd"
export RGPOT_CPMDC_ENGINE="$cpmd"
export RGPOT_CPMD_ENGINE="$cpmd"
export RGPOT_RPC_E2E_REQUIRED=1

exec cargo run -q -p potctl -- ci rpc-integ --server-bin "$potserv"
