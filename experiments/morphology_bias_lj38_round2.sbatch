#!/bin/bash
# Round two of the morphology-bias campaign at LJ38, 4e5 charged evaluations.
#
# TWO THINGS THIS ANSWERS
#
# The SOAP arm, rerun with the cold start removed. Round one reported 10 of 24
# against 14 of 24 for its own control, which measured a projector that held
# its coordinate at zero for the first 200 structures rather than measuring the
# idea. Same protocol, same seeds, same budget.
#
# And Q4 alone at 48 seeds against a 48-seed control. Round one gave 20 of 24
# against 14 of 24, a difference of +0.250 whose interval missed zero by 0.008
# at p=0.111. That is underpowered rather than absent, and 48 seeds an arm is
# the cheapest way to turn a near miss into a statement.
#
# Tasks 0 and 1 each run 24 seeds of an arm, tasks 2 and 3 the second 24 seeds
# of the same arms at a seed offset, so the two halves concatenate into 48
# without any seed being used twice.
#
#SBATCH --job-name=morphbias2
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --partition=any_cpu

set -u

BIN="$HOME/anneal-morphcv/target/release/examples/cluster_bench"
SEEDS=24
BUDGET=400000
N=38

# task:            0        1         2         3       4
ARMS=(""       "q4"     ""        "q4"    "soap")
NAMES=(control48a q4-48a control48b q4-48b soap-refit)
OFFSETS=(0        0       24        24      0)

ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
export SEED_OFFSET="${OFFSETS[$SLURM_ARRAY_TASK_ID]}"

echo "arm ${NAME}, flags '${ARM}', seed offset ${SEED_OFFSET}, LJ${N}, ${BUDGET} charged evaluations, ${SEEDS} seeds"
echo "binary $(ls -la "$BIN")"
echo "host $(hostname)"
date -Is

"$BIN" lj "$N" "$BUDGET" "$SEEDS" "$ARM"
rc=$?

date -Is
echo "exit ${rc}"
exit $rc
