#!/usr/bin/env bash
# Copyright (c) anneal developers
# Python environment for the GAD rerun and the campaign runner: pinned
# wheels in WORK/venv and CondEvo at one commit in WORK/src/condevo.
# CondEvo's packaging installs no modules at that commit, so the checkout
# goes on PYTHONPATH instead of through pip.
set -euo pipefail

WORK=/work/gad-published
CONDEVO_URL=https://github.com/bhartl/CondEvo
CONDEVO_COMMIT=ae76bf0462de63fa6bf3bde97df7cc00237b8d1d
mkdir -p "$WORK/src"

if [[ ! -x $WORK/venv/bin/python ]]; then
  python3 -m venv "$WORK/venv"
fi
"$WORK/venv/bin/pip" install -q --disable-pip-version-check --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
"$WORK/venv/bin/pip" install -q --disable-pip-version-check ase==3.29.0 numpy==2.5.4 scipy==1.18.1 h5py==3.16.0 \
  PyYAML==6.0.3 tqdm==4.70.1 cma==3.3.0

if [[ ! -d $WORK/src/condevo/.git ]]; then
  git clone -q "$CONDEVO_URL" "$WORK/src/condevo"
fi
git -C "$WORK/src/condevo" fetch -q origin "$CONDEVO_COMMIT" 2>/dev/null || true
git -C "$WORK/src/condevo" checkout -q --detach "$CONDEVO_COMMIT"
test "$(git -C "$WORK/src/condevo" rev-parse HEAD)" = "$CONDEVO_COMMIT"
PYTHONPATH=$WORK/src/condevo "$WORK/venv/bin/python" -c 'import condevo.es, ase, h5py, torch'
echo "gad python=$WORK/venv/bin/python condevo=$CONDEVO_COMMIT"
