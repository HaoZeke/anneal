#!/bin/bash
#SBATCH --job-name=lj75-floor-exit
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=floor-exit-%j.out
# One exclusive job. A covering displacement, a minimum-mode climb,
# and a plain quench, from the LJ75 icosahedron.
# Work stays on the node scratch disk.
set -euo pipefail
SCRATCH=/scratch/users/${USER}/${SLURM_JOB_ID}
mkdir -p "$SCRATCH"
df -PT "$SCRATCH" | awk 'NR==2 { if ($2 ~ /^nfs/) { print "scratch is nfs" > "/dev/stderr"; exit 1 } }'
case "$SCRATCH" in
  /users/home/*) echo "work directory is the home filer" >&2; exit 1 ;;
esac
export UV_CACHE_DIR=$SCRATCH/uv-cache
mkdir -p "$UV_CACHE_DIR"
git clone --depth 1 --branch main https://github.com/HaoZeke/anneal.git "$SCRATCH/src"
export LJ_ROOT=$SCRATCH/src
bash "$LJ_ROOT/scripts/elja_build_lj.sh"
HOP=$LJ_ROOT/target/release/examples/ico75_hop
test -x "$HOP"
cd "$SCRATCH"
echo "pwd $(pwd)"
# Covering displacement, minimum-mode climb, plain quench.
# The log is one line per quenched candidate.
HOPS=${FLOOR_HOPS:-2000}
SEEDS=${FLOOR_SEEDS:-1}
"$HOP" search "$HOPS" "$SEEDS" > "$SCRATCH/climb.log"
RECORD=${LJ_RECORD:-$HOME/ljwork}
mkdir -p "$RECORD"
rsync -a --bwlimit=40000 "$SCRATCH/climb.log" "$RECORD/floor-exit-${SLURM_JOB_ID}.log"
