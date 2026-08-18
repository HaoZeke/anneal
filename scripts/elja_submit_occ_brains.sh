#!/usr/bin/env bash
# One exclusive long node per system, all 48 replicas in one wave.
# Elja has srun, not flux / qcg-pilot / snakemake.
set -euo pipefail
ROOT=${LJ_ROOT:-$HOME/anneal-occ-brains}
cd "$ROOT"
test -s SOURCE_COMMIT
test -x scripts/elja_lj38_occ_brains.sbatch
j38=$(sbatch --parsable scripts/elja_lj38_occ_brains.sbatch)
j75=$(sbatch --parsable scripts/elja_lj75_occ_brains.sbatch)
j98=$(sbatch --parsable scripts/elja_lj98_occ_brains.sbatch)
echo "SUBMIT_OK 38=$j38 75=$j75 98=$j98 SOURCE=$(cat SOURCE_COMMIT)"
