#!/bin/bash
# Copyright (c) anneal developers
# Both arms search GFN2 plus the same one-sided O-H wall.
# Seeds 15000-15005, serial, one thread, cap 280305.
set -u
BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
START_ROOT=${2:?start root}
OUT=${3:?output root}
SEED=${4:?seed}
XTB=${XTB_REAL:?xtb binary}
CREST=${CREST_BIN:?crest binary}
export H2H_OMIT_HOST=1
export H2H_OH_WALL=1
export XTB_REAL="$XTB"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export ANNEAL_SOURCE_COMMIT=${ANNEAL_SOURCE_COMMIT:-unset}
export ANNEAL_SOURCE_TREE=${ANNEAL_SOURCE_TREE:-unset}
MOLECULES=16
BUDGET=280305
case "$(rustc -V)" in
  "rustc 1.88.0 "*) ;;
  *)
    echo "rustc is not the Cargo.toml pin of 1.88.0" >&2
    rustc -V >&2
    exit 1
    ;;
esac
# Every timed command for both arms:
#   sudo unshare --uts sh -c 'hostname bench; exec <cmd>'
# sudo resets PATH, so the shell sources the wrapper env before exec.
timed() {
  sudo -n unshare --uts sh -c 'hostname bench; . /home/ubuntu/work/bench-env.sh; exec "$@"' timed-sh "$@"
}
stamp() {
  python3 - << 'PY'
import time
secs = int(time.time()) - 6 * 3600
print(time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(secs)) + "-0600")
PY
}
machine_raw() {
  local dest=$1
  local when=$2
  mkdir -p "$dest"
  cat /proc/sys/kernel/random/boot_id > "$dest/${when}_boot_id.raw"
  cp /proc/uptime "$dest/${when}_uptime.raw"
  awk '/^cpu MHz/ {print}' /proc/cpuinfo > "$dest/${when}_mhz.raw"
  lscpu > "$dest/${when}_lscpu.raw"
  rustc -V > "$dest/${when}_rustc.version.raw"
  cargo -V > "$dest/${when}_cargo.version.raw"
  "$CREST" --version > "$dest/${when}_crest.version.raw" 2>&1
  sha256sum "$CREST" > "$dest/${when}_crest.sha256.raw"
}
write_definition() {
  local dest=$1
  cat > "$dest/definition.txt" << 'EOF'
method: GFN2
added_term: one-sided O-H wall
onset_A: 1.15
stiffness_eV_per_A2: 200
form: stiffness * max(0, r - onset)^2
pairs: each water oxygen with the two hydrogens that follow it
units: eV and eV per angstrom
md_shake: all bonds
md_wall: zero while those bonds stay inside the onset
EOF
}
pins() {
  local dest=$1
  local role=$2
  mkdir -p "$dest"
  {
    echo "date: $(stamp)"
    echo "role: $role"
    echo "threads: 1"
    echo "budget: $BUDGET"
    echo "anneal_sha256: $(sha256sum "$BIN" | awk '{print $1}')"
    echo "xtb_sha256: $(sha256sum "$XTB" | awk '{print $1}')"
    echo "crest_sha256: $(sha256sum "$CREST" | awk '{print $1}')"
    echo "definition_sha256: $(sha256sum "$dest/definition.txt" | awk '{print $1}')"
    echo "source_commit: $ANNEAL_SOURCE_COMMIT"
    echo "source_tree: $ANNEAL_SOURCE_TREE"
  } > "$dest/pins.txt"
}
score_one() {
  local src=$1
  local dest=$2
  mkdir -p "$dest"
  cp "$src" "$dest/in.xyz"
  write_definition "$dest"
  pins "$dest" score
  machine_raw "$dest" start
  (
    cd "$dest"
    set +e
    timed "$XTB" in.xyz --gfn 2 --opt vtight --acc 0.01 --iterations 500 > opt.out 2> opt.err &
    echo $! > pid.txt
    wait $!
    echo $? > exit.code
  )
  machine_raw "$dest" end
}
case "$SEED" in
  15001|15002|15003|15004|15005) ;;
  *) echo "this handoff is for seeds 15001-15005" >&2; exit 1 ;;
esac
mkdir -p "$OUT"
seed="$SEED"
  base="$OUT/runs/$seed"
  mkdir -p "$base"
  cp "$START_ROOT/$seed/start.xyz" "$base/start.xyz"

  anneal="$base/anneal"
  mkdir -p "$anneal"
  cp "$base/start.xyz" "$anneal/start.xyz"
  write_definition "$anneal"
  pins "$anneal" anneal
  machine_raw "$anneal" start
  echo "ARM_START anneal $seed $(stamp)" >> "$OUT/schedule.txt"
  (
    cd "$anneal"
    set +e
    timed "$BIN" search "$MOLECULES" "$BUDGET" "$seed" . > stdout.txt 2> stderr.txt &
    echo $! > pid.txt
    wait $!
    echo $? > exit.code
  )
  machine_raw "$anneal" end
  echo "ARM_DONE anneal $seed $(stamp)" >> "$OUT/schedule.txt"
  if [[ -f $anneal/best.xyz ]]; then
    score_one "$anneal/best.xyz" "$anneal/score"
  fi

  crest="$base/crest"
  mkdir -p "$crest"
  cp "$base/start.xyz" "$crest/start.xyz"
  write_definition "$crest"
  pins "$crest" crest
  {
    echo "crest start.xyz --gfn2 -T 1 --legacy -xnam anneal"
    echo "optimizations: the anneal binary, GFN2 plus the wall"
    echo "dynamics: the xtb binary, SHAKE on all bonds"
  } > "$crest/command.txt"
  machine_raw "$crest" start
  echo "ARM_START crest $seed $(stamp)" >> "$OUT/schedule.txt"
  (
    cd "$crest"
    set +e
    timed "$CREST" start.xyz --gfn2 -T 1 --legacy -xnam "$BIN" > crest.out 2> crest.err &
    echo $! > pid.txt
    wait $!
    echo $? > exit.code
  )
  machine_raw "$crest" end
  echo "ARM_DONE crest $seed $(stamp)" >> "$OUT/schedule.txt"
  if [[ -f $crest/crest_best.xyz ]]; then
    score_one "$crest/crest_best.xyz" "$crest/score"
  fi

  {
    echo "seed: $seed"
    echo "budget: $BUDGET"
    echo "date: $(stamp)"
    echo "anneal_exit: $(cat "$anneal/exit.code" 2>/dev/null || echo missing)"
    echo "anneal_score_exit: $(cat "$anneal/score/exit.code" 2>/dev/null || echo missing)"
    echo "crest_exit: $(cat "$crest/exit.code" 2>/dev/null || echo missing)"
    echo "crest_score_exit: $(cat "$crest/score/exit.code" 2>/dev/null || echo missing)"
    echo "anneal_pid: $(cat "$anneal/pid.txt" 2>/dev/null || echo missing)"
    echo "crest_pid: $(cat "$crest/pid.txt" 2>/dev/null || echo missing)"
  } > "$base/seed.txt"
echo "SERIES_DONE $(stamp)" >> "$OUT/schedule.txt"
