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

# CppCore/meson.build queries git, and ase pulls matplotlib -> pillow;
# free-threaded tags without a pillow wheel build pillow from source and
# require libjpeg-dev and zlib1g-dev.
if command -v dpkg >/dev/null 2>&1 && [ -f "${ROOT}/ci/gha/apt-install.sh" ]; then
  if ! dpkg -s git libjpeg-dev zlib1g-dev >/dev/null 2>&1; then
    bash "${ROOT}/ci/gha/apt-install.sh" git libjpeg-dev zlib1g-dev
  fi
fi

# Sanitizer flags from callers (such as py-tsan.yml) target rgpot itself.
# Leaving -fsanitize=thread in the environment while pip builds third-party
# test dependencies from sdist passes LDFLAGS=-fsanitize=thread to gfortran
# in SciPy and links GCC's libtsan.so.2 into a Clang-TSan interpreter.
env -u CFLAGS -u CXXFLAGS -u FFLAGS -u FCFLAGS -u LDFLAGS \
  "${PY}" -m pip install -U pip
if ! "${PY}" -c 'import numpy' >/dev/null 2>&1; then
  env -u CFLAGS -u CXXFLAGS -u FFLAGS -u FCFLAGS -u LDFLAGS \
    "${PY}" -m pip install 'numpy'
fi
PIP_SCIPY_ARGS=()
if "${PY}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) and hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled() else 1)'; then
  # Current SciPy, Pillow, Matplotlib, ContourPy, and KiwiSolver dropped
  # cp313t wheels; pin to the latest releases with cp313t wheels before ase,
  # and forbid source builds of scipy so pip never falls back to an sdist.
  PIP_SCIPY_ARGS=(--only-binary=scipy)
  env -u CFLAGS -u CXXFLAGS -u FFLAGS -u FCFLAGS -u LDFLAGS \
    "${PY}" -m pip install "${PIP_SCIPY_ARGS[@]}" \
      'scipy>=1.15,<1.18' \
      'pillow>=11,<12.3' \
      'matplotlib>=3.9,<3.11.2' \
      'contourpy>=1.3,<1.4' \
      'kiwisolver>=1.4,<1.5.1'
fi
env -u CFLAGS -u CXXFLAGS -u FFLAGS -u FCFLAGS -u LDFLAGS \
  "${PY}" -m pip install ${PIP_SCIPY_ARGS[@]+"${PIP_SCIPY_ARGS[@]}"} \
    'nanobind==3.1.0' 'ase>=3.23' meson ninja

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
