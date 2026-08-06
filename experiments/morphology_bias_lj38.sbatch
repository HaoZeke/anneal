#!/bin/bash
# Five-arm morphology-bias campaign at LJ38, 4e5 charged evaluations, 24 seeds
# an arm, on Elja.
#
# WHAT IS BEING TESTED
#
# Whether a well-tempered bias deposited on a morphology coordinate solves
# LJ38, against a no-bias control on the same driver, the same ledger and the
# same corrected relaxer. The manuscript asserts 5 of 5 on Q4, 8 of 8 on
# (Q4, Q6) and 5 of 7 on a learned SOAP projection against 1 of 8 for controls,
# from a table measured before the relaxer fix that took the local optimiser
# from 3389 evaluations per relaxation without converging to 386 converging.
#
# WHY ONE TASK PER ARM RATHER THAN ONE PER SEED
#
# The association carries MaxJobs 30 and MaxSubmit 100, and array tasks count
# against MaxSubmit, so 5 arms by 24 seeds as 120 tasks is refused. It does not
# need to be that shape: one seed at 4e5 takes 2 s on the control and 18 s on
# the SOAP arm, so a whole arm of 24 seeds is at most eight minutes and fits in
# one task.
#
#SBATCH --job-name=morphbias
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=any_cpu

set -u

BIN="$HOME/anneal-morphcv/target/release/examples/cluster_bench"
SEEDS=24
BUDGET=400000
N=38

# Arm 0 is the control: no mechanism flags at all, so the driver runs its
# default basin keying with no morphology coordinate anywhere.
ARMS=("" "q4" "q4q6" "soap" "coord")
NAMES=(control q4 q4q6 soap coord)

ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "arm ${NAME}, flags '${ARM}', LJ${N}, ${BUDGET} charged evaluations, ${SEEDS} seeds"
echo "binary $(ls -la "$BIN")"
echo "host $(hostname)"
date -Is

"$BIN" lj "$N" "$BUDGET" "$SEEDS" "$ARM"
rc=$?

date -Is
echo "exit ${rc}"
exit $rc
