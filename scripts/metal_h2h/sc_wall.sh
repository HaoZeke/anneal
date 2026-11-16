#!/usr/bin/env bash
# Copyright (c) anneal developers
# Where anneal's Sutton-Chen wall goes on sc126-55 and sc96-55.
#
# Builds rgpot's Sutton-Chen entry with build_rgpot_sc.sh, from RGPOT_SRC
# when that is set. Rows are appended to seeds.csv and micro.csv. Then, in one UTS
# namespace with hostname bench, working directory /work, one job at a
# time on core 0: anneal (h2h_sc of this HEAD) on seeds
# 2000-2011 at 1000000 calls, and sc_wall_micro on the seed-2000 start of
# each system. The harness must be committed, because its HEAD is the
# harness_revision of every row.
#
# SC_WALL_OUT names another directory for the rows. Its rows also take
# the final energy, hops, escapes, restarts and paths from h2h_sc's diag
# line, and the openmp flag of the build (RGPOT_OPENMP, passed on to
# build_rgpot_sc.sh), OMP_NUM_THREADS (SC_WALL_THREADS, default 1) and
# the cores the jobs are pinned to (SC_WALL_CPUS, default 0). The jobs
# stay one at a time. SC_WALL_SYSTEMS replaces the two systems; each is
# an sc-n12 system, 12-6, 10-8 or 9-6.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-sc
SYSTEMS=${SC_WALL_SYSTEMS:-sc126-55 sc96-55}
THREADS=${SC_WALL_THREADS:-1}
CPUS=${SC_WALL_CPUS:-0}
OPENMP=${RGPOT_OPENMP:-0}
if [[ -n ${SC_WALL_OUT:-} ]]; then
  OUT=$SC_WALL_OUT
  wide=1
else
  OUT=$ROOT/bench/metal-cluster-h2h/sc-wall
  wide=0
  [[ $THREADS == 1 && $CPUS == 0 && $OPENMP == 0 ]] ||
    { echo "the sc-wall rows have no column for OpenMP, threads or cores; set SC_WALL_OUT" >&2; exit 1; }
fi
if [[ -n ${RGPOT_SRC:-} ]]; then
  LIBDIR=$WORK/rgpot-sc-$(git -C "$RGPOT_SRC" rev-parse --short=8 HEAD)
else
  LIBDIR=$WORK/rgpot-sc
fi
[[ $OPENMP == 1 ]] && LIBDIR=$LIBDIR-omp

harness=(scripts/metal_h2h)
if ! git -C "$ROOT" diff --quiet HEAD -- "${harness[@]}" ||
  [[ -n $(git -C "$ROOT" ls-files --others --exclude-standard -- "${harness[@]}") ]]; then
  echo "commit the harness before the run" >&2
  exit 1
fi

bash "$HERE/build_sc.sh"
RGPOT_OPENMP=$OPENMP bash "$HERE/build_rgpot_sc.sh"
mkdir -p "$OUT" "$WORK/home" "$WORK/tmp"

if [[ $wide == 1 ]]; then
  seeds_header=system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,pot_per_eval_s,outside_per_eval_s,final_e,hops,stall_escapes,trail_escapes,restarts,paths,path_improvements,code_revision,rgpot_revision,openmp,omp_threads,cpus,host,harness_revision
  micro_header=system,atoms,reps,entry_us,dlsym_alloc_us,kept_us,kept_stream_us,plain_us,abs_de,max_abs_df,rgpot_revision,openmp,omp_threads,cpus,host,harness_revision
  seeds_tail=",\$head,\$rgpot,$OPENMP,$THREADS,$CPUS,bench,\$head"
  micro_tail=",\$rgpot,$OPENMP,$THREADS,$CPUS,bench,\$head"
else
  seeds_header=system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,pot_per_eval_s,outside_per_eval_s,code_revision,rgpot_revision,host,harness_revision
  micro_header=system,atoms,reps,entry_us,dlsym_alloc_us,kept_us,kept_stream_us,plain_us,abs_de,max_abs_df,rgpot_revision,host,harness_revision
  seeds_tail=",\$head,\$rgpot,bench,\$head"
  micro_tail=",\$rgpot,bench,\$head"
fi

cat > "$WORK/bin/inside-sc-wall.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home TMPDIR=$WORK/tmp PATH=$WORK/bin:/usr/bin:/bin
export OMP_NUM_THREADS=$THREADS RAYON_NUM_THREADS=1 RGPOT_SC_LIB=$LIBDIR/librgpot.so
cd /work
[ "\$(hostname)" = bench ] || { echo "hostname is not bench" >&2; exit 1; }
head=\$(cat $WORK/anneal-head)
rgpot=\$(cat $LIBDIR/rgpot-revision)
[ "\$(cat $LIBDIR/openmp 2>/dev/null || echo 0)" = $OPENMP ] || { echo "$LIBDIR is not built with openmp=$OPENMP" >&2; exit 1; }
seeds=$OUT/seeds.csv
micro=$OUT/micro.csv
[ -f \$seeds ] || echo "$seeds_header" > \$seeds
[ -f \$micro ] || echo "$micro_header" > \$micro
[ "\$(head -1 \$seeds)" = "$seeds_header" ] || { echo "\$seeds has another header" >&2; exit 1; }
[ "\$(head -1 \$micro)" = "$micro_header" ] || { echo "\$micro has another header" >&2; exit 1; }
diag=$WORK/tmp/sc-wall-diag
for system in $SYSTEMS; do
  for seed in \$(seq 2000 2011); do
    row=\$(H2H_BUDGET=1000000 taskset -c $CPUS $WORK/bin/h2h_sc run \$system \$seed 2>\$diag | grep ",anneal,")
    if [ $wide = 1 ]; then
      line=\$(grep '^diag ' \$diag)
      for key in final hops stall_escapes trail_escapes restarts paths path_improvements; do
        value=\$(printf '%s\n' "\$line" | tr ' ' '\n' | sed -n "s/^\$key=//p")
        [ -n "\$value" ] || { echo "no \$key in the diag line of \$system \$seed" >&2; exit 1; }
        row="\$row,\$value"
      done
    fi
    echo "\$row$seeds_tail" >> \$seeds
  done
done
for system in $SYSTEMS; do
  case \$system in sc126-*) exps="12 6 144.41" ;; sc108-*) exps="10 8 34.408" ;; sc96-*) exps="9 6 39.432" ;; *) exit 1 ;; esac
  $WORK/bin/h2h_sc coords \$system 2000 $WORK/tmp/\$system-start.xyz
  for rep in 1 2 3; do
    row=\$(taskset -c $CPUS $LIBDIR/sc_wall_micro $LIBDIR/librgpot.so $WORK/tmp/\$system-start.xyz \$exps 20000)
    echo "\$system,\$row$micro_tail" >> \$micro
  done
done
EOF
chmod +x "$WORK/bin/inside-sc-wall.sh"

if ! grep -Eq '(^|[[:space:]])bench([[:space:]]|$)' /etc/hosts; then
  echo '127.0.0.1 bench' | sudo -n tee -a /etc/hosts >/dev/null
fi
exec sudo -n unshare --uts sh -c \
  'hostname bench; exec sudo -u ubuntu /work/metal-sc/bin/inside-sc-wall.sh'
