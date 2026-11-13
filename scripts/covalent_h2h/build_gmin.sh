#!/usr/bin/env bash
# Wales's public GMIN with the evaluator cap and a Stillinger-Weber timer.
# The public build brackets LJ and TIP4P only. SWTWO is timed the way the
# Morse bench times MORSEGH, so pot_s is the time inside the potential.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
GMIN=/work/gmin-h2h/src/wales/GMIN

# The shared script extracts, checks the revision and applies the cap.
# A timer patch removes the binary, and the second call rebuilds the
# objects it touched.
bash "$ROOT/scripts/gmin_h2h/build_gmin.sh"
if ! grep -q H2H_POT_ENTER "$GMIN/source/SW.f90"; then
  patch -p1 --forward --batch -d "$GMIN" < "$HERE/sw-timer.patch"
  rm -f /work/gmin-h2h/bin/GMIN
fi
bash "$ROOT/scripts/gmin_h2h/build_gmin.sh"
grep -q H2H_POT_ENTER "$GMIN/source/SW.f90"
