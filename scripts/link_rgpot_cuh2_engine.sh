#!/usr/bin/env bash
# Link rgpot's tracked CuH2 C entry point into a narrow dlopen engine.
set -euo pipefail

if (($# != 4)); then
  echo "usage: $0 RGPOT_ROOT RGPOT_BUILD_DIR EXPORT_MAP OUTPUT" >&2
  exit 2
fi

rgpot=$1
build_dir=$2
export_map=$3
output=$4
capi_rel=CppCore/rgpot/fortran/rgpot_cuh2_capi.f90
kernels=$build_dir/CppCore/rgpot/fortran/librgpot_fortran_kernels.a
vesin_fortran=$build_dir/libvesin_fortran.a
vesin_internal=$build_dir/libvesin_internal.a
gfortran=${GFORTRAN:-gfortran}

git -C "$rgpot" ls-files --error-unmatch "$capi_rel" >/dev/null
if ! grep -F 'bind(c, name="rgpot_cuh2_force")' "$rgpot/$capi_rel" >/dev/null; then
  echo "$rgpot/$capi_rel does not define rgpot_cuh2_force" >&2
  exit 2
fi
for input in "$kernels" "$vesin_fortran" "$vesin_internal" "$export_map"; do
  if [[ ! -s $input ]]; then
    echo "missing CuH2 link input: $input" >&2
    exit 1
  fi
done
if ! command -v "$gfortran" >/dev/null; then
  echo "missing Fortran linker: $gfortran" >&2
  exit 1
fi
if ! nm --defined-only "$kernels" |
  awk '$NF == "rgpot_cuh2_force" { found = 1 } END { exit found ? 0 : 1 }'; then
  echo "$kernels does not contain rgpot_cuh2_force" >&2
  exit 1
fi

"$gfortran" \
  -shared \
  -Wl,--no-undefined \
  -Wl,-soname,librgpot_cuh2.so \
  -Wl,-u,rgpot_cuh2_force \
  -Wl,--version-script="$export_map" \
  -Wl,--start-group \
  "$kernels" \
  "$vesin_fortran" \
  "$vesin_internal" \
  -Wl,--end-group \
  -lstdc++ \
  -lm \
  -pthread \
  -o "$output"

mapfile -t actual_exports < <(
  nm -D --defined-only --format=posix "$output" | awk '{print $1}' | LC_ALL=C sort
)
expected_exports=(rgpot_cuh2_force rgpot_fortran_last_error)
if ((${#actual_exports[@]} != ${#expected_exports[@]})); then
  echo "$output exports an unexpected symbol set: ${actual_exports[*]}" >&2
  exit 1
fi
for index in "${!expected_exports[@]}"; do
  if [[ ${actual_exports[index]} != "${expected_exports[index]}" ]]; then
    echo "$output exports an unexpected symbol set: ${actual_exports[*]}" >&2
    exit 1
  fi
done

printf 'RGPOT_CUH2_ENGINE_OK %s\n' "$output"
