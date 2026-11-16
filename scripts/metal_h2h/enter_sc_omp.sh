#!/usr/bin/env bash
# Copyright (c) anneal developers
# Anneal's Sutton-Chen wall with rgpot built with OpenMP. RGPOT_SRC is a
# clean rgpot checkout and RGPOT_REV the revision it must be at. sc_wall.sh
# runs the six 55- and 80-atom sc-n12 systems, seeds 2000-2011 at 1000000
# calls, against the -fopenmp build twice: with OMP_NUM_THREADS=1 on core
# 0, then with OMP_NUM_THREADS at the core count on every core. Both append
# to bench/metal-cluster-h2h/sc-omp. Then sc_omp_bits compares that build
# with the build without OpenMP on the seed-2000 start of each system and
# its 63 noisy copies, at both thread counts, into bits.csv there. One job
# at a time, each in a UTS namespace with hostname bench. The harness must
# be committed, because its HEAD is the harness_revision of every row.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=/work/metal-sc
OUT=$ROOT/bench/metal-cluster-h2h/sc-omp
SYSTEMS="sc126-55 sc108-55 sc96-55 sc126-80 sc108-80 sc96-80"
: "${RGPOT_SRC:?set RGPOT_SRC to an rgpot checkout}"
: "${RGPOT_REV:?set RGPOT_REV to the rgpot revision of RGPOT_SRC}"
[[ $(git -C "$RGPOT_SRC" rev-parse HEAD) == "$RGPOT_REV" ]] || { echo "$RGPOT_SRC is not at $RGPOT_REV" >&2; exit 1; }
SERIAL=$WORK/rgpot-sc-${RGPOT_REV:0:8}
OMP=$SERIAL-omp
CORES=$(nproc)

RGPOT_SRC=$RGPOT_SRC RGPOT_OPENMP=0 bash "$HERE/build_rgpot_sc.sh"
for pass in "1 0" "$CORES 0-$((CORES - 1))"; do
  read -r threads cpus <<< "$pass"
  RGPOT_SRC=$RGPOT_SRC RGPOT_OPENMP=1 SC_WALL_OUT=$OUT SC_WALL_SYSTEMS=$SYSTEMS SC_WALL_THREADS=$threads \
    SC_WALL_CPUS=$cpus bash "$HERE/sc_wall.sh"
done

header=system,atoms,geometries,energy_differ,force_differ,max_abs_de,max_abs_df,max_rel_de,reference_rgpot_revision,reference_openmp,rgpot_revision,openmp,omp_threads,cpus,host,harness_revision
cat > "$WORK/bin/inside-sc-omp.sh" << EOF
#!/bin/bash
set -eu
export HOME=$WORK/home TMPDIR=$WORK/tmp PATH=$WORK/bin:/usr/bin:/bin RAYON_NUM_THREADS=1
cd /work
[ "\$(hostname)" = bench ] || { echo "hostname is not bench" >&2; exit 1; }
head=\$(cat $WORK/anneal-head)
bits=$OUT/bits.csv
[ -f \$bits ] || echo "$header" > \$bits
[ "\$(head -1 \$bits)" = "$header" ] || { echo "\$bits has another header" >&2; exit 1; }
for threads_cpus in "1 0" "$CORES 0-$((CORES - 1))"; do
  set -- \$threads_cpus
  for system in $SYSTEMS; do
    case \$system in sc126-*) exps="12 6 144.41" ;; sc108-*) exps="10 8 34.408" ;; sc96-*) exps="9 6 39.432" ;; *) exit 1 ;; esac
    $WORK/bin/h2h_sc coords \$system 2000 $WORK/tmp/\$system-start.xyz
    row=\$(OMP_NUM_THREADS=\$1 taskset -c \$2 $OMP/sc_omp_bits $SERIAL/librgpot.so $OMP/librgpot.so \\
      $WORK/tmp/\$system-start.xyz \$exps)
    echo "\$system,\$row,\$(cat $SERIAL/rgpot-revision),\$(cat $SERIAL/openmp),\$(cat $OMP/rgpot-revision),\$(cat $OMP/openmp),\$1,\$2,bench,\$head" >> \$bits
  done
done
EOF
chmod +x "$WORK/bin/inside-sc-omp.sh"

exec sudo -n unshare --uts sh -c 'hostname bench; exec sudo -u ubuntu /work/metal-sc/bin/inside-sc-omp.sh'
