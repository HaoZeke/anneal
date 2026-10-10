#!/bin/bash
#SBATCH --job-name=lj75-qcg-0
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=qcg-%j.out
# One seed. Source is cloned onto the node scratch disk. The uv cache and
# the cargo target stay there. One log is copied back at the end.
set -euo pipefail
SCRATCH=/scratch/users/${USER}/${SLURM_JOB_ID}
mkdir -p "$SCRATCH"
df -PT "$SCRATCH" | awk 'NR==2 { if ($2 ~ /^nfs/) { print "scratch is nfs: " $0 > "/dev/stderr"; exit 1 } }'
export UV_CACHE_DIR=$SCRATCH/uv-cache
mkdir -p "$UV_CACHE_DIR"
git clone --depth 1 --branch main https://github.com/HaoZeke/anneal.git "$SCRATCH/src"
export LJ_ROOT=$SCRATCH/src
bash "$LJ_ROOT/scripts/elja_build_lj.sh"
BIN=$LJ_ROOT/target/release/examples/lj_cluster_search
test -x "$BIN"
uv run --script "$LJ_ROOT/scripts/elja_qcg_cell.py" cell \
  --n 75 --budget 4000000 --arm rec --seed 0 \
  --binary "$BIN" \
  --record "${LJ_RECORD:-$HOME/ljwork/qcg-lj75}"
