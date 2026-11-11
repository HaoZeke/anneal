#!/bin/bash
# Copyright (c) anneal developers
# Separate GFN2-xTB reference for (H2O)20. Not a timed arm.
# No published GFN2 (H2O)20 minimum is used: this run's best vtight
# energy, and the vtight energy of its starting geometry, are compared
# and the lower one is written to bench/water-h2o20-goat/reference.csv.
# Timed seeds must not start until that file exists.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
XTB=${XTB_BIN:-/work/xtb-gfn2}
CREST=${CREST_BIN:-/work/crestenv/.pixi/envs/default/bin/crest}
START=${H2O_REF_XYZ:?starting xyz}
WORK=${H2O_REF_WORK:-/work/h2o20-goat/reference}
DEST=$ROOT/bench/water-h2o20-goat
mkdir -p "$WORK" "$DEST"
cp "$START" "$WORK/start.xyz"
(
  cd "$WORK"
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/work/rgpot/.pixi/envs/xtbbld/lib}
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
  export OMP_DYNAMIC=FALSE
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export XTB_ARGV_LOG=$WORK/argv.log
  "$CREST" start.xyz --gfn2 -T "${H2O_REF_THREADS:-8}" --legacy -xnam "$XTB" -len "${H2O_REF_LEN:-1.0}" \
    > crest.out 2> crest.err
  echo $? > crest.exit
)
python3 - "$WORK" "$DEST" "$XTB" << 'PY'
import os, subprocess, sys
from pathlib import Path
work, dest, xtb = map(Path, sys.argv[1:])
hartree = 27.211386245988

def vtight(xyz: Path) -> float:
    folder = xyz.parent / ("score-" + xyz.stem)
    folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["LD_LIBRARY_PATH"] = os.environ.get(
        "LD_LIBRARY_PATH", "/work/rgpot/.pixi/envs/xtbbld/lib"
    )
    proc = subprocess.run(
        [str(xtb), str(xyz), "--gfn", "2", "--opt", "vtight", "--acc", "0.01", "--iterations", "500"],
        cwd=folder,
        env=env,
        capture_output=True,
        text=True,
    )
    (folder / "opt.out").write_text(proc.stdout)
    (folder / "opt.err").write_text(proc.stderr)
    energy = None
    for line in proc.stdout.splitlines():
        if "TOTAL ENERGY" in line and "Eh" in line:
            parts = line.split()
            energy = float(parts[parts.index("Eh") - 1])
    if energy is None:
        raise SystemExit(f"no energy for {xyz}")
    return energy

candidates = [("start", work / "start.xyz")]
best = work / "crest_best.xyz"
if best.is_file():
    candidates.append(("crest_best", best))
scored = [(name, vtight(path)) for name, path in candidates]
name, eh = min(scored, key=lambda item: item[1])
ev = eh * hartree
out = dest / "reference.csv"
out.write_text(
    "molecules,energy_Eh,energy_eV,hit_window_eV,picked,source\n"
    f"20,{eh:.12f},{ev:.6f},0.01,{name},"
    "\"long GFN2 CREST iMTD-GC, -len 1.0, no published GFN2 (H2O)20 minimum\"\n"
)
print(f"reference {name} {eh:.12f} Eh {ev:.6f} eV")
PY
