#!/bin/bash
#SBATCH --job-name=lj75-cover
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cover-%j.out
# One job. From the quenched LJ75 icosahedron, place each Plasencia
# covering point at the walk step and quench. The tally is how many
# leave that packing.
set -euo pipefail
SCRATCH=/scratch/users/${USER}/${SLURM_JOB_ID}
mkdir -p "$SCRATCH"
df -PT "$SCRATCH" | awk 'NR==2 { if ($2 ~ /^nfs/) { print "scratch is nfs: " $0 > "/dev/stderr"; exit 1 } }'
git clone --depth 1 --branch main https://github.com/HaoZeke/anneal.git "$SCRATCH/src"
export LJ_ROOT=$SCRATCH/src
bash "$LJ_ROOT/scripts/elja_build_lj.sh"
PROBE=$LJ_ROOT/target/release/examples/leave_packing_probe
test -x "$PROBE"
# 96 is the default cover. Step 0.7 is the step the LJ75 walk prints.
"$PROBE" 96 600 cover 0.7
