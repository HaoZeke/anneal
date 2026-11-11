#!/bin/bash
# TIP4P success rates for (H2O)10, 15 and 20.
# Paper preset, one budget, serial, one UTS namespace, hostname bench.
# Wales and Hodges, Chem. Phys. Lett. 286, 65 (1998), doi:10.1016/S0009-2614(98)00065-7.
# Kazachenko and Thakkar, Chem. Phys. Lett. 476, 120 (2009), doi:10.1016/j.cplett.2009.06.026,
# report the same minima through n = 21.
# Kazachenko and Thakkar, J. Chem. Phys. 138, 194302 (2013), doi:10.1063/1.4804399.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/tip4p-rates
BUDGET=500000
# 0.001 kcal/mol. The thermochemical calorie is 4.184 kJ.
TOL_KJ=0.004184
CSV="$ROOT/bench/tip4p-water-rates/rates.csv"

SIZE_ARG=
SEED_ARG=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --size) SIZE_ARG=$2; shift 2 ;;
    --seed) SEED_ARG=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$SIZE_ARG" && -z "$SEED_ARG" ]] || [[ -z "$SIZE_ARG" && -n "$SEED_ARG" ]]; then
  echo "--size and --seed are a pair" >&2
  exit 2
fi

mkdir -p "$WORK/bin" "$WORK/logs" "$WORK/home" "$WORK/tmp" /work/target /work/rustc-tmp
cargo_home=${CARGO_HOME:-/usr/local/cargo}
build_line="cd ${ROOT} && CARGO_TARGET_DIR=/work/target TMPDIR=/work/rustc-tmp RUSTFLAGS='--remap-path-prefix=${ROOT}=/src --remap-path-prefix=${HOME}=/user --remap-path-prefix=${cargo_home}=/cargo' cargo +1.88.0 build --release --locked --example h2h_seed -j 4"
printf '%s\n' "$build_line" > "$WORK/build-command.txt"
bash -c "$build_line"
cp -f /work/target/release/examples/h2h_seed "$WORK/bin/h2h_seed"
chmod +x "$WORK/bin/h2h_seed"

cat > "$WORK/bin/inside.sh" << EOF
#!/bin/bash
set -euo pipefail
export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export PATH=$WORK/bin:\$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export RAYON_NUM_THREADS=1
cd /work
exec $WORK/bin/h2h_seed "\$@"
EOF
chmod +x "$WORK/bin/inside.sh"

cat > "$WORK/bin/one.sh" << EOF
#!/bin/bash
set -euo pipefail
n=\$1
seed=\$2
budget=\$3
ref=\$4
tol=\$5
log=\$6
prefix=\$7
stop=\$(python3 -c 'import sys; print(repr(float(sys.argv[1]) + float(sys.argv[2])))' "\$ref" "\$tol")
cd /work
{
  printf 'hostname=%s\n' "\$(hostname)"
  printf 'cwd=%s\n' "\$(pwd)"
  printf 'boot_id=%s\n' "\$(cat /proc/sys/kernel/random/boot_id)"
  printf 'uptime=%s\n' "\$(cat /proc/uptime)"
  $WORK/bin/inside.sh tip4p "\$n" 0 "\$budget" "\$seed" recommended "\$stop" "\$prefix"
} > "\$log" 2>&1
EOF
chmod +x "$WORK/bin/one.sh"

cat > "$WORK/bin/campaign.sh" << EOF
#!/bin/bash
set -euo pipefail
cd /work
tol=$TOL_KJ
declare -A budget_of=(
  [10]=500000
  [15]=500000
  [20]=2000000
)
size_only='${SIZE_ARG}'
seed_only='${SEED_ARG}'
declare -A ref=(
  [10]=-391.0226842125
  [15]=-628.3722525077
  [20]=-872.9887544952
)
sizes=(10 15 20)
seeds=(0 1 2 3 4 5)
if [[ -n "\$size_only" ]]; then
  sizes=("\$size_only")
  seeds=("\$seed_only")
fi
rows=()
for n in "\${sizes[@]}"; do
  for seed in "\${seeds[@]}"; do
    log=$WORK/logs/n\${n}-seed\${seed}.log
    prefix=$WORK/logs/n\${n}-seed\${seed}
    budget=\${budget_of[\$n]}
    echo "start n=\$n seed=\$seed budget=\$budget \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    $WORK/bin/one.sh "\$n" "\$seed" "\$budget" "\${ref[\$n]}" "\$tol" "\$log" "\$prefix"
    if ! grep -q '^hostname=bench$' "\$log"; then
      echo "hostname was not bench" >&2
      exit 1
    fi
    if ! grep -q '^cwd=/work$' "\$log"; then
      echo "cwd was not /work" >&2
      exit 1
    fi
    calls=\$(sed -n 's/^calls: //p' "\$log" | tail -n 1)
    lowest=\$(sed -n 's/^lowest: //p' "\$log" | tail -n 1)
    hit_calls=
    hit=0
    if [[ -f "\${prefix}.trace.csv" ]]; then
      hit_calls=\$(python3 -c '
import sys
ref = float(sys.argv[1])
tol = float(sys.argv[2])
limit = ref + tol
first = ""
for line in open(sys.argv[3]):
    if line.startswith("calls"):
        continue
    got, energy = line.split(",")
    if float(energy) <= limit:
        first = got.strip()
        break
print(first)
' "\${ref[\$n]}" "\$tol" "\${prefix}.trace.csv")
    fi
    if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) + float(sys.argv[3]) else 1)' "\$lowest" "\${ref[\$n]}" "\$tol"; then
      hit=1
    fi
    if [[ "\$hit" == 1 && -z "\$hit_calls" ]]; then
      hit_calls=\$calls
    fi
    if [[ "\$hit" == 0 ]]; then
      hit_calls=
    fi
    echo "done n=\$n seed=\$seed hit=\$hit calls=\${hit_calls:-} best=\$lowest"
    rows+=("\$n,\$seed,\$hit,\${hit_calls:-},\$lowest,\${ref[\$n]},\$budget,recommended")
  done
done
if [[ -z "\$size_only" ]]; then
  mkdir -p "$(dirname "$CSV")"
  {
    echo "n,seed,hit,calls_to_first_hit,best_kj,reference_kj,budget,preset"
    printf '%s\n' "\${rows[@]}"
  } > "$CSV"
  echo "wrote $CSV"
fi
EOF
chmod +x "$WORK/bin/campaign.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  "hostname bench; exec sudo -u ubuntu $WORK/bin/campaign.sh"
