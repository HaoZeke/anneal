#!/usr/bin/env bash
# Compute-node scratch on Elja is /scratch/users/$USER.
# IRHPC: https://irhpcwiki.hi.is/docs/elja/scratch_disk/
# Each compute node has its own /scratch disk. The job copies its inputs
# there, runs there, copies the needed files back, and deletes the directory.
# /users/home is the NFS filer (https://irhpcwiki.hi.is/docs/elja/Data_Management/).
# A directory named scratch under $HOME is still that filer.
set -euo pipefail

elja_on_home_filer() {
  local text=$1 resolved
  if resolved=$(readlink -f "$1" 2>/dev/null); then
    text=$resolved
  fi
  case "$text" in
    "$HOME"|"$HOME"/*|/users/home|/users/home/*) return 0 ;;
  esac
  return 1
}

# Fresh directory under /scratch/users/$USER, named with the Slurm job id.
# CARGO_TARGET_DIR and TMPDIR point at it. The caller copies inputs in.
elja_enter_scratch() {
  if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "elja_scratch: SLURM_JOB_ID is required; /scratch/users exists on the compute node" >&2
    exit 1
  fi
  local scratchlocation=/scratch/users
  if [[ ! -d $scratchlocation ]]; then
    echo "elja_scratch: $scratchlocation is absent on $(hostname)" >&2
    exit 1
  fi
  if elja_on_home_filer "$scratchlocation"; then
    echo "elja_scratch: $scratchlocation is on the home filer" >&2
    exit 1
  fi
  mkdir -p "$scratchlocation/$USER"
  local tdir fstype
  tdir=$(mktemp -d "$scratchlocation/$USER/${SLURM_JOB_ID}-XXXX")
  if elja_on_home_filer "$tdir"; then
    echo "elja_scratch: refusing $tdir on the home filer" >&2
    exit 1
  fi
  fstype=$(df -PT "$tdir" | awk 'NR==2 { print $2 }')
  case "$fstype" in
    nfs|nfs4)
      echo "elja_scratch: $tdir is $fstype, not the node disk" >&2
      exit 1
      ;;
  esac
  export ELJA_SCRATCH=$tdir
  export TMPDIR=$tdir
  export CARGO_TARGET_DIR=$tdir/target
  mkdir -p "$CARGO_TARGET_DIR"
}

# Copy one file onto the home filer at the documented NFS cap.
# rsync --bwlimit is the flag on the NFS page (40000).
elja_publish() {
  local src=$1 dest=$2
  mkdir -p "$(dirname "$dest")"
  rsync -a --bwlimit=40000 "$src" "$dest"
}

# Delete the directory this job created. Never a path on the home filer.
elja_leave_scratch() {
  local dir=${ELJA_SCRATCH:-}
  if [[ -z $dir ]]; then
    return 0
  fi
  case "$dir" in
    /scratch/users/"$USER"/"${SLURM_JOB_ID:-}"-*)
      if ! elja_on_home_filer "$dir"; then
        rm -rf "$dir"
      fi
      ;;
  esac
  unset ELJA_SCRATCH
}
