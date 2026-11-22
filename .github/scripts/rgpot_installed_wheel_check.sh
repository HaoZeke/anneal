#!/usr/bin/env bash
# Install each repaired wheel into a fresh venv and run the checks that
# py-wheels.yml's CIBW_TEST_COMMAND replaces. That includes
# tests/test_ase_plugin_wheel.py. The wheel is installed with its ase extra,
# and the test skips itself on an ASE release without the plugin API.
# The venv is started with LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, and
# LIBRARY_PATH unset, so the import has to stand on the wheel's own
# runpath.
#
# Interpreters are selected by wheel tag:
#   PY_310, PY_311, PY_312, PY_314T
# Usage: rgpot_installed_wheel_check.sh WHEELDIR
set -euo pipefail

wheel_dir=${1:-wheelhouse}
shopt -s nullglob
wheels=("${wheel_dir}"/*.whl)
if [ "${#wheels[@]}" -eq 0 ]; then
  echo "no wheels in ${wheel_dir}" >&2
  exit 1
fi

interpreter_for() {
  case "$1" in
    *cp314*t*) printf '%s\n' "${PY_314T:?PY_314T is unset}" ;;
    *cp310*) printf '%s\n' "${PY_310:?PY_310 is unset}" ;;
    *cp311*) printf '%s\n' "${PY_311:?PY_311 is unset}" ;;
    *cp312*) printf '%s\n' "${PY_312:?PY_312 is unset}" ;;
    *)
      echo "no interpreter for $1" >&2
      return 1
      ;;
  esac
}

root=$(pwd)
for whl in "${wheels[@]}"; do
  echo "== ${whl}"
  py=$(interpreter_for "$(basename "${whl}")")
  venv=$(mktemp -d)
  "${py}" -m venv "${venv}"
  "${venv}/bin/python" -m pip install -q -U pip
  "${venv}/bin/python" -m pip install -q pytest numpy "${whl}[ase]"
  env -u LD_LIBRARY_PATH -u DYLD_LIBRARY_PATH -u LIBRARY_PATH -u PYTHONPATH \
    "${venv}/bin/python" -c \
    "import rgpot; assert rgpot.has_expr and rgpot.has_dftd3 and rgpot.has_dftd4 and rgpot.has_xckernel, (rgpot.has_expr, rgpot.has_dftd3, rgpot.has_dftd4, rgpot.has_xckernel); print(rgpot.__file__, len(rgpot.XcKernel.catalog()))"
  env -u LD_LIBRARY_PATH -u DYLD_LIBRARY_PATH -u LIBRARY_PATH -u PYTHONPATH \
    "${venv}/bin/python" -m pytest -q "${root}/tests/test_rgpot_pip_core.py" \
    "${root}/tests/test_ase_plugin_wheel.py"
  rm -rf "${venv}"
done
