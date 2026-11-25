#!/bin/bash
# Copyright (c) anneal developers
# Timed Lennard-Jones arms, one UTS namespace, hostname bench, paths under /work.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/ase-gofee-h2h
BINARY=${H2H_BINARY:-/work/target/release/examples/h2h_gmin_proof}
EVAL=${H2H_EVAL:-/work/target/release/examples/asg_eval}
VENV=${H2H_VENV:-$WORK/venv}
PANEL=0
for arg in "$@"; do
  if [[ $arg == "--panel" ]]; then
    PANEL=1
  fi
done

mkdir -p "$WORK/runs" "$WORK/home" "$WORK/tmp" "$WORK/mpl" "$WORK/numba" "$WORK/cache" \
  /work/scratch /work/target

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing venv $VENV" >&2
  exit 1
fi

export CARGO_TARGET_DIR=/work/target
export RUSTFLAGS="--remap-path-prefix=${ROOT}=/src --remap-path-prefix=/work/target=/src --remap-path-prefix=/work/scratch=/build"
if [[ ! -x "$BINARY" ]]; then
  (cd "$ROOT" && cargo +1.88 build --release --example h2h_gmin_proof)
fi
if [[ $PANEL == 1 ]]; then
  if [[ ! -x "$EVAL" || "$ROOT/examples/asg_eval.rs" -nt "$EVAL" ]]; then
    (cd "$ROOT" && cargo +1.88 build --release --example asg_eval)
  fi
  if [[ ! -f /work/covalent-h2h/lib/librgpot_covalent.so ]]; then
    echo "missing librgpot_covalent.so" >&2
    exit 1
  fi
  BINARY=$EVAL
fi

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi

cat > "$WORK/inside.sh" << EOF
#!/bin/bash
set -euo pipefail
export HOME=$WORK/home TMPDIR=$WORK/tmp
export MPLBACKEND=Agg MPLCONFIGDIR=$WORK/mpl NUMBA_CACHE_DIR=$WORK/numba
export XDG_CACHE_HOME=$WORK/cache PYTHONUNBUFFERED=1
export PYTHONPATH=$ROOT/scripts/ase_gofee_h2h
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE OMP_PLACES=cores RAYON_NUM_THREADS=1 RAY_AUTH_MODE=disabled
export RGPOT_COVALENT_LIB=/work/covalent-h2h/lib/librgpot_covalent.so
export PATH=$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
OUTPUT=\${AGOX_GOFEE_OUTPUT:-$WORK/output}
mkdir -p "\$OUTPUT"
taskset -c 0 $VENV/bin/python $ROOT/scripts/session_metadata.py \\
  --output "\$OUTPUT/sessions.jsonl" \\
  --campaign agox-gofee --root $ROOT -- \\
  snakemake -s workflow/Snakefile agox_gofee_h2h --cores 1 --force
exec taskset -c 0 $VENV/bin/python -u $ROOT/scripts/ase_gofee_h2h/campaign.py \\
  --binary $BINARY --repo $ROOT --runs $WORK/runs --output-dir "\$OUTPUT" "\$@"
EOF
chmod +x "$WORK/inside.sh"

exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu "$0" "$@"' "$WORK/inside.sh" "$@"
