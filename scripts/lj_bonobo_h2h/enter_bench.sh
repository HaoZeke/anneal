#!/bin/bash
# Both arms, one UTS namespace, hostname bench, working directory under /work.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/lj-bonobo-h2h
PYAR_SRC=${PYAR_SRC:-/tmp/pyar-hist}
BONOBO_SRC=${BONOBO_SRC:-/tmp/modified_bo_framework_codes}
BINARY=${LEAN_BINARY:-"$ROOT/../tmp/anneal-target/release/examples/lj_lean_relax_budget"}

if [[ ! -x "$BINARY" ]]; then
  echo "missing lean binary: $BINARY" >&2
  exit 1
fi
if [[ ! -f "$PYAR_SRC/pyar/scripts/pyar-tabu" ]]; then
  echo "missing historical pyar-tabu under $PYAR_SRC" >&2
  exit 1
fi
if [[ ! -f "$BONOBO_SRC/bo.py" ]]; then
  echo "missing bonobo checkout under $BONOBO_SRC" >&2
  exit 1
fi

mkdir -p "$WORK/bin" "$WORK/upstream/bonobo" "$WORK/upstream/pyar-tree" \
  "$WORK/runs" "$WORK/home" "$WORK/tmp"
if [[ ! -x "$WORK/venv/bin/python" ]]; then
  python3 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install --quiet \
    'numpy==2.0.2' 'scipy==1.13.1' 'numba==0.60.0' 'networkx==3.2.1' \
    'ase==3.23.0' pandas scikit-learn hdbscan dscribe
fi

rm -rf "$WORK/upstream/pyar-tree/pyar"
mkdir -p "$WORK/upstream/pyar-tree"
cp -a "$PYAR_SRC/pyar" "$WORK/upstream/pyar-tree/pyar"
rm -rf "$WORK/upstream/pyar-tree/pyar/AIMNet2" \
  "$WORK/upstream/pyar-tree/pyar/mlatom" \
  "$WORK/upstream/pyar-tree/pyar/__pycache__"
cp -a "$BONOBO_SRC/bo.py" "$BONOBO_SRC/lbfgs_opt.py" "$BONOBO_SRC/lj.py" \
  "$BONOBO_SRC/initial_pop.py" "$BONOBO_SRC/check_geom.py" \
  "$BONOBO_SRC/make_valid_geom.py" "$BONOBO_SRC/similarity.py" \
  "$BONOBO_SRC/write_traj.py" "$BONOBO_SRC/read_xyz_files.py" \
  "$WORK/upstream/bonobo/"
cp -a "$HERE/bonobo_arm.py" "$HERE/run_campaign.py" "$HERE/pyar_tabu_seed.py" \
  "$HERE/faithful_rerun.sh" "$WORK/bin/"
chmod +x "$WORK/bin/faithful_rerun.sh"
cp -a "$BINARY" "$WORK/bin/lj_lean_relax_budget"
cat > "$WORK/bin/pyar-tabu" << EOF
#!/bin/bash
export PYAR_TABU_SCRIPT="$WORK/upstream/pyar-tree/pyar/scripts/pyar-tabu"
export PYTHONPATH="$WORK/upstream/pyar-tree\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$WORK/venv/bin/python" "$WORK/bin/pyar_tabu_seed.py" "\$@"
EOF
chmod +x "$WORK/bin/pyar-tabu" "$WORK/bin/lj_lean_relax_budget"

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:$WORK/venv/bin:\$PATH
export PYTHONPATH=$WORK/upstream/pyar-tree
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
cd $WORK
if [ "\${1:-}" = "--faithful" ]; then
  exec bash $WORK/bin/faithful_rerun.sh
fi
exec $WORK/venv/bin/python -u $WORK/bin/run_campaign.py \\
  --fixture $ROOT/bench/lj-bonobo-h2h/cambridge_lj.tsv \\
  --binary $WORK/bin/lj_lean_relax_budget \\
  --upstream $WORK/upstream/bonobo \\
  --runs $WORK/runs \\
  --summary $ROOT/bench/lj-bonobo-h2h/summary.tsv \\
  "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/lj-bonobo-h2h/bin/inside.sh "$@"' _ "$@"
