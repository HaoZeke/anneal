#!/bin/bash
# Reduced meson build of the Python extension: Fortran pots on, torch off.
# The build is installed into the site-packages of PY, so the tests import
# the installed rgpot and never the python/ source tree. Source this file.
# It exports PYTHONPATH, and appends the same variable to GITHUB_ENV when
# that file is set.
# PY selects the interpreter (default: python). ROOT is the repository
# root (default: GITHUB_WORKSPACE, else the working directory).
set -euo pipefail

PY="${PY:-python}"
ROOT="${ROOT:-${GITHUB_WORKSPACE:-$(pwd)}}"
cd "${ROOT}"

"${PY}" -m pip install -U pip
"${PY}" -m pip install 'nanobind==3.1.0' 'ase>=3.23' meson ninja
if ! "${PY}" -c 'import numpy' >/dev/null 2>&1; then
  "${PY}" -m pip install 'numpy'
fi

SITE="$("${PY}" -c 'import site; print(site.getsitepackages()[0])')"
EXE="$("${PY}" -c 'import sys; print(sys.executable)')"
BIN="$("${PY}" -c 'import pathlib, sys; print(pathlib.Path(sys.executable).parent)')"
cat > /tmp/rgpot-native.ini << EOF
[binaries]
python = '${EXE}'
EOF

BUILD=/tmp/rgpot-reduced-build
PREFIX=/tmp/rgpot-reduced-prefix
SANITIZE=()
if printf '%s' "${CFLAGS:-} ${CXXFLAGS:-} ${LDFLAGS:-}" | grep -q 'fsanitize=thread'; then
  # CFLAGS already instrument the objects. b_sanitize puts libtsan on the
  # link line. b_lundef would reject the uninstrumented Fortran runtime.
  SANITIZE+=(-Db_sanitize=thread -Db_lundef=false)
fi
"${BIN}/meson" setup "${BUILD}" \
  --native-file /tmp/rgpot-native.ini \
  --prefix "${PREFIX}" \
  -Dpython.platlibdir="${SITE}" \
  -Dpython.purelibdir="${SITE}" \
  -Dwith_python=true \
  -Dwith_fortran_pots=enabled \
  -Dwith_metatomic=false \
  -Dwith_rpc=false \
  -Dwith_tests=false \
  -Dwith_examples=false \
  -Dwith_dftd3=false \
  -Dwith_dftd4=false \
  -Dwith_expr=false \
  -Dwith_xckernel=false \
  -Dwith_cache=false \
  -Dwith_eigen=false \
  -Dwith_xtb=false \
  -Dwith_tblite=false \
  --buildtype=release \
  "${SANITIZE[@]}"
"${BIN}/meson" compile -C "${BUILD}"

"${BIN}/meson" install -C "${BUILD}" --skip-subprojects

export PYTHONPATH="${ROOT}/tests/python:${SITE}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "PYTHONPATH=${PYTHONPATH}" >> "${GITHUB_ENV}"
fi
