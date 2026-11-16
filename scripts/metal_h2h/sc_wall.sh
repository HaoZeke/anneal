#!/usr/bin/env bash
# Copyright (c) anneal developers
# Where anneal's Sutton-Chen wall goes on sc126-55 and sc96-55.
#
# Builds rgpot's Sutton-Chen entry with build_rgpot_sc.sh. Then, in one UTS
# namespace with hostname bench, working directory /work, one job at a
# time on core 0: anneal (h2h_sc of this HEAD) on seeds
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

bash "$HERE/build_sc.sh"
bash "$HERE/build_rgpot_sc.sh"
mkdir -p "$OUT" "$WORK/home" "$WORK/tmp"

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
