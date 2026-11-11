#!/bin/sh
# Copyright (c) anneal developers
# Branch coverage for the Python files.
# rustc does not instrument Python, so this uses coverage.py --branch.
# Usage: python_coverage.sh OUTDIR
set -eu
out=${1:?output directory}
cd "$(git rev-parse --show-toplevel)"
mkdir -p "$out"
export PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH=0
python3 -m coverage run --branch --source=paper/jctc/scripts,python/anneal,pytest \
  -m pytest \
  paper/jctc/scripts/test_script_behaviour.py \
  paper/jctc/scripts/test_redact_paths.py \
  paper/jctc/scripts/test_seal_outer.py
python3 -m coverage json -o "$out/coverage.json"
python3 -m coverage report -m > "$out/report.txt"
