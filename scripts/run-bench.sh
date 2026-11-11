#!/bin/bash
# Lennard-Jones rows from /work. One size and seed at a time.
set -u
WORK=/work
BIN=/work/lj_board_row
cd "$WORK"
mkdir -p "$WORK/logs" "$WORK/hw" "$WORK/driver"
SHA=$(sha256sum "$BIN" | awk '{print $1}')
printf '%s  %s\n' "$SHA" "$BIN" > "$WORK/hw/binary.sha256"
cat > "$WORK/hw/build-command.txt" <<'EOF'
CARGO_TARGET_DIR=/work/target RUSTFLAGS='--remap-path-prefix=/agent=/src --remap-path-prefix=/tmp=/build --remap-path-prefix=/home=/build' cargo +1.88.0 build --release --locked --example lj_board_row -j 4
EOF
printf '%s\n' "$SHA" > "$WORK/hw/binary.sha256.hex"

sudo unshare --uts sh -c 'hostname bench; cd /work; lscpu' > "$WORK/hw/lscpu.txt"
sudo unshare --uts sh -c 'hostname bench; cd /work; cat /proc/meminfo' > "$WORK/hw/meminfo.txt"
sudo unshare --uts sh -c 'hostname bench; cd /work; nproc' > "$WORK/hw/nproc.txt"
sudo unshare --uts sh -c 'hostname bench; cd /work; hostname; uname -n' > "$WORK/hw/uts.txt"
rustc +1.88.0 -vV > "$WORK/hw/rustc.txt"
cargo +1.88.0 -vV > "$WORK/hw/cargo.txt"
git -C /agent/repos/anneal rev-parse HEAD > "$WORK/hw/head.txt"
git -C /agent/repos/anneal rev-parse 'HEAD^{tree}' > "$WORK/hw/tree.txt"

forbid() {
  local file="$1"
  if grep -q 'cursor' "$file"; then
    echo "forbidden nodename in $file" >&2
    return 1
  fi
  if grep -q '/agent' "$file"; then
    echo "forbidden path /agent in $file" >&2
    return 1
  fi
  if grep -q '/home' "$file"; then
    echo "forbidden path /home in $file" >&2
    return 1
  fi
  if grep -q '/tmp' "$file"; then
    echo "forbidden path /tmp in $file" >&2
    return 1
  fi
  return 0
}

for file in "$WORK/hw/lscpu.txt" "$WORK/hw/meminfo.txt" "$WORK/hw/nproc.txt" "$WORK/hw/uts.txt" "$WORK/hw/rustc.txt" "$WORK/hw/cargo.txt" "$WORK/hw/binary.sha256" "$WORK/hw/head.txt" "$WORK/hw/tree.txt"; do
  forbid "$file"
done

done_ok() {
  local log="$1"
  [[ -f "$log" ]] || return 1
  [[ "$(head -n 1 "$log")" == "binary_sha256=$SHA" ]] || return 1
  grep -q '^hostname=bench$' "$log" || return 1
  grep -q '^boot_id=.\+' "$log" || return 1
  grep -q '/proc/sys/kernel/random/boot_id' "$log" || return 1
  grep -q 'process_exit=0' "$log" || return 1
  grep -q 'exit_code=0' "$log" || return 1
  grep -q 'energy_calls=' "$log" || return 1
  grep -q 'gradient_calls=' "$log" || return 1
  grep -q 'stop=hit' "$log" && return 0
  grep -q 'stop=ceiling' "$log" && return 0
  return 1
}

for n in 38 55 75 98 104; do
  for seed in 6400 6401 6402 6403 6404 6405; do
    log="$WORK/logs/lj${n}-seed${seed}.log"
    if done_ok "$log" && forbid "$log"; then
      echo "keep n=$n seed=$seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      continue
    fi
    wrapper="sudo unshare --uts sh -c 'hostname bench; cd /work; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export NUMBA_NUM_THREADS=1; export VECLIB_MAXIMUM_THREADS=1; export BLIS_NUM_THREADS=1; export OMP_PROC_BIND=TRUE; export OMP_PLACES=cores; export RAYON_NUM_THREADS=1; exec /work/lj_board_row $n $seed'"
    echo "launch n=$n seed=$seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\n' "$wrapper" >> "$WORK/hw/commands.txt"
    t0=$(date +%s.%N)
    {
      echo "binary_sha256=$SHA"
      echo "wrapper=$wrapper"
      echo "cwd=/work"
      echo "shell_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      sudo unshare --uts sh -c 'hostname bench; cd /work; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export NUMBA_NUM_THREADS=1; export VECLIB_MAXIMUM_THREADS=1; export BLIS_NUM_THREADS=1; export OMP_PROC_BIND=TRUE; export OMP_PLACES=cores; export RAYON_NUM_THREADS=1; printf "hostname=%s\n" "$(hostname)"; exec /work/lj_board_row "$1" "$2"' sh "$n" "$seed"
      echo "process_exit=$?"
    } > "$log" 2>&1 &
    pid=$!
    elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 5
      elapsed=$((elapsed + 5))
      if (( elapsed % 600 == 0 )); then
        last=$(grep '^progress ' "$log" | tail -n 1 || true)
        echo "progress $(date -u +%Y-%m-%dT%H:%M:%SZ) n=$n seed=$seed elapsed_s=$elapsed $last"
      fi
    done
    wait "$pid" || true
    t1=$(date +%s.%N)
    python3 -c 'import sys; t0=float(sys.argv[1]); t1=float(sys.argv[2]); print("process_wall_s={:.6f}".format(t1-t0))' "$t0" "$t1" >> "$log"
    if ! forbid "$log"; then
      echo "arm failed path check n=$n seed=$seed" >&2
      exit 1
    fi
    prc=$(grep '^process_exit=' "$log" | tail -n 1 | cut -d= -f2 || true)
    echo "returned n=$n seed=$seed process_exit=${prc:-missing} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ "${prc:-}" != "0" ]]; then
      echo "arm failed n=$n seed=$seed" >&2
      tail -n 20 "$log" >&2
      exit 1
    fi
    if ! done_ok "$log"; then
      echo "arm failed checks n=$n seed=$seed" >&2
      exit 1
    fi
  done
done

if ! forbid "$WORK/hw/commands.txt"; then
  echo "commands failed the path check" >&2
  exit 1
fi

echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 - <<'PY'
from pathlib import Path
root = Path("/work/logs")
for n in (38, 55, 75, 98, 104):
    hits = 0
    walls = []
    for seed in range(6400, 6406):
        text = (root / f"lj{n}-seed{seed}.log").read_text()
        fields = {}
        for line in text.splitlines():
            for token in line.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    fields[k] = v
        hit = int(fields["hit"])
        hits += hit
        walls.append(float(fields["wall_s"]))
        print(
            f"lj{n} seed {seed} best {fields['best']} hit {hit} "
            f"wall_s {fields['wall_s']} energy_calls {fields['energy_calls']} "
            f"gradient_calls {fields['gradient_calls']} stop {fields['stop']}"
        )
    ordered = sorted(walls)
    rank = 1.0 + (len(ordered) - 1) * 0.5
    lower = int(rank)
    weight = rank - lower
    median = ordered[lower - 1] + weight * (ordered[lower] - ordered[lower - 1])
    print(f"LJ{n} hits {hits}/6 median_wall_s {median:.6f}")
PY
