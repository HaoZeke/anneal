#!/bin/sh
# Copyright (c) anneal developers
# Thirty water-hexamer starts. Four arms. One serial session.
set -eu
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export H2H_LOG_DIR="${H2H_LOG_DIR:-paper/jctc/logs/water-engrad}"
export H2H_SCRATCH_DIR="${H2H_SCRATCH_DIR:-paper/jctc/logs/scratch/water-engrad}"
export H2H_NUMBERS_DIR="$H2H_LOG_DIR"
export H2H_SEEDS="${H2H_SEEDS:-$(seq -s, 0 29)}"
export H2H_WALL_LABEL="${H2H_WALL_LABEL:-one session}"
unset H2H_LANDED_NUMBERS || true
unset H2H_LANDED_CREST_DIR || true
python3 paper/jctc/scripts/water_h2h.py
python3 paper/jctc/scripts/water_engrad_numbers.py
python3 paper/jctc/scripts/count_cost.py "$H2H_LOG_DIR" > "$H2H_LOG_DIR/COST.tsv"
python3 paper/jctc/scripts/water_significance.py "$H2H_LOG_DIR/NUMBERS.tsv"
