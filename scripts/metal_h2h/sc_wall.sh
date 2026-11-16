#!/usr/bin/env bash
# Copyright (c) anneal developers
# Where anneal's Sutton-Chen wall goes on sc126-55 and sc96-55.
#
# Builds rgpot's Sutton-Chen entry at the rgpot revision of Cargo.lock from
# cargo's checkout, alone, with the neighbour list and SIMD sources it
# needs. Then, in one UTS namespace with hostname bench, working directory
# /work, one job at a time on core 0: anneal (h2h_sc of this HEAD) on seeds
# 2000-2011 at 1000000 calls, and sc_wall_micro on the seed-2000 start of
# each system. The harness must be committed, because its HEAD is the
# harness_revision of every row.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-sc
OUT=$ROOT/bench/metal-cluster-h2h/sc-wall
LIBDIR=$WORK/rgpot-sc

harness=(scripts/metal_h2h)
if ! git -C "$ROOT" diff --quiet HEAD -- "${harness[@]}" ||
  [[ -n $(git -C "$ROOT" ls-files --others --exclude-standard -- "${harness[@]}") ]]; then
  echo "commit the harness before the run" >&2
  exit 1
fi

rgpot_rev=$(sed -n 's|^source = "git+https://github.com/OmniPotentRPC/rgpot?rev=\([0-9a-f]*\)#.*|\1|p' \
  "$ROOT/Cargo.lock" | sort -u)
[[ $(wc -w <<< "$rgpot_rev") == 1 ]] || { echo "one rgpot revision expected in Cargo.lock" >&2; exit 1; }
cargo_home=${CARGO_HOME:-$HOME/.cargo}
src=$(ls -d "$cargo_home"/git/checkouts/rgpot-*/"${rgpot_rev:0:7}" | head -1)
cpp=$src/CppCore

bash "$HERE/build_sc.sh"
mkdir -p "$LIBDIR" "$OUT" "$WORK/home" "$WORK/tmp"
g++ -O3 -std=c++20 -fPIC -shared -DNDEBUG -w -I"$cpp" -I"$src/include" -o "$LIBDIR/librgpot.so" \
  "$cpp/rgpot/SuttonChen/SuttonChenPot.cc" "$cpp"/rgpot/nlist/*.cc "$cpp"/rgpot/nlist/*.cpp \
  "$cpp/rgpot/simd/exact_fma_hw.cc" "$cpp/rgpot/simd/exp.cc" -Wl,--no-undefined
g++ -O3 -std=c++20 -DNDEBUG -w -I"$cpp" -I"$src/include" -o "$LIBDIR/sc_wall_micro" \
  "$HERE/sc_wall_micro.cc" "$LIBDIR/librgpot.so" -ldl -Wl,-rpath,"$LIBDIR"
printf '%s\n' "$rgpot_rev" > "$LIBDIR/rgpot-revision"

cat > "$WORK/bin/inside-sc-wall.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home TMPDIR=$WORK/tmp PATH=$WORK/bin:/usr/bin:/bin
export OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 RGPOT_SC_LIB=$LIBDIR/librgpot.so
cd /work
[ "\$(hostname)" = bench ] || { echo "hostname is not bench" >&2; exit 1; }
head=\$(cat $WORK/anneal-head)
rgpot=\$(cat $LIBDIR/rgpot-revision)
seeds=$OUT/seeds.csv
micro=$OUT/micro.csv
echo "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,pot_per_eval_s,outside_per_eval_s,code_revision,rgpot_revision,host,harness_revision" > \$seeds
echo "system,atoms,reps,entry_us,dlsym_alloc_us,kept_us,kept_stream_us,plain_us,abs_de,max_abs_df,rgpot_revision,host,harness_revision" > \$micro
for system in sc126-55 sc96-55; do
  for seed in \$(seq 2000 2011); do
    row=\$(H2H_BUDGET=1000000 taskset -c 0 $WORK/bin/h2h_sc run \$system \$seed 2>/dev/null | grep ",anneal,")
    echo "\$row,\$head,\$rgpot,bench,\$head" >> \$seeds
  done
done
for system in sc126-55 sc96-55; do
  case \$system in sc126-*) exps="12 6 144.41" ;; sc96-*) exps="9 6 39.432" ;; esac
  $WORK/bin/h2h_sc coords \$system 2000 $WORK/tmp/\$system-start.xyz
  for rep in 1 2 3; do
    row=\$(taskset -c 0 $LIBDIR/sc_wall_micro $LIBDIR/librgpot.so $WORK/tmp/\$system-start.xyz \$exps 20000)
    echo "\$system,\$row,\$rgpot,bench,\$head" >> \$micro
  done
done
EOF
chmod +x "$WORK/bin/inside-sc-wall.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/metal-sc/bin/inside-sc-wall.sh'
