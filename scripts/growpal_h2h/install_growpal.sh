#!/usr/bin/env bash
# Install the published GrowPAL release and the example LJ10 seeds.
# PyPI 0.1.6 is the release whose documented LJ script matches the
# diversity-preserving search (selection_count = round(N * 1.8), five
# reoptimisations, energy window 5 N^(2/3)). GitHub main is an older
# tree that still calls the retired aegon ranking argument.
set -euo pipefail

WORK=/work/growpal-h2h
if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
SRC=$WORK/src
VENV=$WORK/venv
GROWPAL_VERSION=0.1.6
AEGON_VERSION=1.3.15
GIT_URL=https://github.com/isaacgcampos/GrowPAL.git

mkdir -p "$WORK" "$SRC"
if [[ ! -f $SRC/GrowPAL/examples/LJ/LJ010.xyz ]]; then
  git clone --depth 1 "$GIT_URL" "$SRC/GrowPAL"
fi
revision=$(git -C "$SRC/GrowPAL" rev-parse HEAD)

if [[ ! -x $VENV/bin/python ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "growpal==${GROWPAL_VERSION}" "aegon==${AEGON_VERSION}"

"$VENV/bin/python" - << PY
import importlib.metadata as meta
growpal = meta.version("growpal")
aegon = meta.version("aegon")
if growpal != "${GROWPAL_VERSION}":
    raise SystemExit(f"growpal {growpal}")
if aegon != "${AEGON_VERSION}":
    raise SystemExit(f"aegon {aegon}")
from growpal.libgrowpal import growpal_parallel
from growpal.libselect_clustering import select_by_clustering
from aegon.optimization.potentials.lj import opt_LJ_parallel, lj_energy
from aegon.population import sort_by_energy
print(f"imported growpal {growpal} aegon {aegon}")
PY

cat > "$WORK/growpal-source.txt" << EOF
growpal_version=$GROWPAL_VERSION
aegon_version=$AEGON_VERSION
growpal_source=https://pypi.org/project/growpal/${GROWPAL_VERSION}/
github_url=$GIT_URL
github_main=$revision
seeds=$SRC/GrowPAL/examples/LJ/LJ010.xyz
dtol=1.2
growatom=Mo
reopt_rounds=5
dedup_tols=0.99
dedup_tole=0.1
selection_count=round(n*1.8)
n_clusters=max(1, number of specials)
ecut=5*n^(2/3)
early_stop=0.01
documented_nproc=cpu_count-2
n_cores=1
n_jobs=1
optimizer=L-BFGS-B
gtol=1e-6
maxiter=1000
EOF
"$VENV/bin/pip" freeze > "$WORK/growpal-freeze.txt"
echo "growpal ${GROWPAL_VERSION} aegon ${AEGON_VERSION} github ${revision}"
