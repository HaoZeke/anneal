#!/usr/bin/env bash
# Copyright (c) anneal developers
# Conda prefix for the openconf head-to-head and the TRIC quench measurement.
# RDKit, xtb, tblite and geomeTRIC from conda-forge; openconf from PyPI.
set -euo pipefail

PREFIX=${OPENCONF_H2H_ENV:-/work/env}
MAMBA=${MAMBA_EXE:-/work/bin/micromamba}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/work/mamba}

if [[ ! -x $MAMBA ]]; then
  unpack="$(dirname "$MAMBA")/micromamba-unpack"
  mkdir -p "$unpack"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$unpack" bin/micromamba
  mv "$unpack/bin/micromamba" "$MAMBA"
  rm -rf "$unpack"
fi
if [[ ! -x $PREFIX/bin/python ]]; then
  "$MAMBA" create -y -p "$PREFIX" -c conda-forge \
    python=3.12 rdkit=2025.09.6 numpy scipy xtb=6.7.1 crest=3.0.2 tblite-python \
    geometric=1.1.1 asv pip cmake ninja gfortran meson pkg-config
fi
"$PREFIX/bin/pip" install -q openconf==0.0.19 maturin snakemake array-api-compat
