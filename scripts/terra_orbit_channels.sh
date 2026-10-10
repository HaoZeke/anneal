#!/usr/bin/env bash
# Matched native-thread orbit campaign. `plan` is read-only; `submit` takes
# a source packet with bin/lj_cluster_search and its checksum manifests.
set -euo pipefail
set -o noclobber

plan() {
  local anneal_task=0 anneal_seed anneal_n anneal_budget anneal_arm
  local anneal_replicas anneal_bias anneal_gossip
  for ((anneal_seed = 0; anneal_seed < 72; anneal_seed++)); do
    for anneal_n in 38 75 98; do
      if ((anneal_n != 38 && anneal_seed >= 48)); then continue; fi
      anneal_budget=4000000
      if ((anneal_n == 38)); then anneal_budget=400000; fi
      for anneal_arm in single independent bias bias_ring; do
        anneal_replicas=4 anneal_bias=0 anneal_gossip=none
        case "$anneal_arm" in
          single) anneal_replicas=1 ;;
          bias) anneal_bias=1 ;;
          bias_ring) anneal_bias=1 anneal_gossip=ring ;;
        esac
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$anneal_task" "$anneal_n" "$anneal_budget" "$anneal_seed" \
          "$anneal_arm" "$anneal_replicas" "$anneal_bias" "$anneal_gossip"
        anneal_task=$((anneal_task + 1))
      done
    done
  done
}

submit() {
  local anneal_root anneal_script anneal_job
  anneal_root=$(cd "${1:?source packet}" && pwd)
  anneal_script=$(realpath "${BASH_SOURCE[0]}")
  cd "$anneal_root"
  sha256sum --quiet -c BINARY_SHA256SUMS
  sha256sum --quiet -c SOURCE_SHA256SUMS
  mkdir campaign
  mkdir campaign/38 campaign/75 campaign/98
  cp "$anneal_script" campaign/run.sh
  plan > campaign/plan.tsv
  cd campaign
  sha256sum run.sh plan.tsv ../BINARY_SHA256SUMS ../SOURCE_SHA256SUMS > CAMPAIGN_SHA256SUMS
  anneal_job=$(env SLURM_CONF=/etc/slurm-llnl/slurm.conf sbatch --parsable \
    --partition=cpu --cpus-per-task=4 --mem=4096 --time=01:00:00 \
    --array=0-671%6 --job-name=anneal-orbit-channels --export=ALL \
    --output="$anneal_root/campaign/slurm-%A_%a.log" \
    "$anneal_root/campaign/run.sh" run "$anneal_root")
  printf '%s\n' "$anneal_job" > SLURM_ARRAY_JOB_ID
  printf 'campaign=%s/campaign\njob=%s\n' "$anneal_root" "$anneal_job"
}

run() {
  local anneal_root anneal_task anneal_index anneal_n anneal_budget anneal_seed
  local anneal_arm anneal_replicas anneal_bias anneal_gossip anneal_stem anneal_status
  anneal_root=$(cd "${1:?source packet}" && pwd)
  anneal_task=${SLURM_ARRAY_TASK_ID:?Slurm array task required}
  [[ -n ${SLURM_JOB_ID:-} && $anneal_task =~ ^[0-9]+$ ]]
  ((anneal_task < 672))
  cd "$anneal_root"
  sha256sum --quiet -c BINARY_SHA256SUMS
  cd campaign
  sha256sum --quiet -c CAMPAIGN_SHA256SUMS
  IFS=$'\t' read -r anneal_index anneal_n anneal_budget anneal_seed anneal_arm \
    anneal_replicas anneal_bias anneal_gossip \
    < <(sed -n "$((anneal_task + 1))p" plan.tsv)
  [[ $anneal_index == "$anneal_task" ]]
  anneal_stem="$anneal_n/${anneal_arm}_$anneal_seed"
  printf 'task=%s host=%s slurm_job=%s cpus=%s\n' \
    "$anneal_task" "$(hostname)" "$SLURM_JOB_ID" "${SLURM_CPUS_PER_TASK:?}" \
    > "$anneal_stem.meta"
  printf 'n=%s budget=%s seed=%s arm=%s mechanisms=thompson,rscreen,orbit\n' \
    "$anneal_n" "$anneal_budget" "$anneal_seed" "$anneal_arm" >> "$anneal_stem.meta"
  if [[ $anneal_gossip == none ]]; then anneal_gossip=; fi
  TIMEFORMAT=$'wall_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
  set +e
  {
    time env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH="$anneal_root/bin" \
      RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      SEED_OFFSET="$anneal_seed" HISTORY_REPLICAS="$anneal_replicas" \
      HISTORY=none HISTORY_POLICY=accepted HISTORY_CHECKPOINT=1000 \
      SHARED_BIAS="$anneal_bias" SHARED_BIAS_WEIGHT=0.25 \
      GOSSIP="$anneal_gossip" GOSSIP_INTERVAL=20000 GOSSIP_WEIGHT=0.5 \
      GOSSIP_TOP=64 GOSSIP_ADAPTIVE=0 TWO_CHOICE_STALL=0 \
      "$anneal_root/bin/lj_cluster_search" "$anneal_n" "$anneal_budget" 1 \
      thompson,rscreen,orbit > "$anneal_stem.out" 2> "$anneal_stem.err"
  } 2> "$anneal_stem.timing"
  anneal_status=$?
  set -e
  printf '%s\n' "$anneal_status" > "$anneal_stem.exitcode"
  printf 'task=%s exitcode=%s result=%s\n' "$anneal_task" "$anneal_status" "$anneal_stem.out"
  exit "$anneal_status"
}

case "${1:-}" in
  plan) plan ;;
  submit) submit "${2:?source packet}" ;;
  run) run "${2:?source packet}" ;;
  *) printf 'usage: %s plan | submit <source-packet> | run <source-packet>\n' "$0" >&2; exit 2 ;;
esac
