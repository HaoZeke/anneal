#!/bin/bash
# Copyright (c) anneal developers
# Anneal searches GFN2 plus the one-sided O-H wall.
# CREST --opt, --md and --sp all go to real xtb at /work/xtb-gfn2
# via -xnam, not to the anneal binary.
# Seeds 15000-15005, serial, one thread, cap 280305.
set -u
BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
START_ROOT=${2:?start root}
OUT=${3:?output root}
SEED=${4:?seed}
XTB=${XTB_BIN:-/work/xtb-gfn2}
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
REFERENCE=-81.396292669509
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
  sudo -n unshare --uts sh -c 'hostname bench; . /work/bench-env.sh; exec "$@"' timed-sh "$@"
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
anneal_search: GFN2 plus the one-sided O-H wall
crest_search: xtb 6.7.1 at /work/xtb-gfn2 via -xnam, no O-H wall
wall_onset_A: 1.15
wall_stiffness_eV_per_A2: 200
wall_form: stiffness * max(0, r - onset)^2
intact: every oxygen has exactly two hydrogens within 1.2 angstrom
score: xtb --gfn 2 --opt vtight --acc 0.01 --iterations 500
crest_rescore: five lowest ensemble members, best intact vtight energy
hit: E <= -81.396292669509 + 1e-6
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
last_energy() {
  python3 - "$1" << 'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
found = re.findall(r"TOTAL ENERGY\s+([+-]?\d+\.\d+)", text)
print(found[-1] if found else "nan")
PY
}
monomers_intact() {
  python3 - "$1" << 'PY'
import sys
lines = open(sys.argv[1], errors="replace").read().splitlines()
frames = []
i = 0
while i < len(lines):
    parts = lines[i].split()
    if not parts:
        i += 1
        continue
    try:
        count = int(parts[0])
    except ValueError:
        i += 1
        continue
    body = lines[i + 2:i + 2 + count]
    if len(body) < count:
        break
    symbols = []
    coords = []
    for line in body:
        item = line.split()
        symbols.append(item[0])
        coords.extend(float(value) for value in item[1:4])
    frames.append((symbols, coords))
    i += 2 + count
if not frames:
    print("broken")
    raise SystemExit
symbols, coords = frames[-1]
n_mol = len(coords) // 9
if len(symbols) != 3 * n_mol or len(coords) != 9 * n_mol:
    print("broken")
    raise SystemExit
limit2 = 1.2 * 1.2

def apart(a, b):
    return sum((coords[3 * a + k] - coords[3 * b + k]) ** 2 for k in range(3))

for molecule in range(n_mol):
    if (
        symbols[3 * molecule].upper() != "O"
        or symbols[3 * molecule + 1].upper() != "H"
        or symbols[3 * molecule + 2].upper() != "H"
    ):
        print("broken")
        raise SystemExit
    near = 0
    oxygen = 3 * molecule
    for other in range(n_mol):
        for hydrogen in (3 * other + 1, 3 * other + 2):
            if apart(oxygen, hydrogen) <= limit2:
                near += 1
    if near != 2:
        print("broken")
        raise SystemExit
print("intact")
PY
}
judge() {
  local dest=$1
  local energy intact hit
  if [[ ! -f $dest/opt.out || ! -f $dest/xtbopt.xyz ]]; then
    echo "nan" > "$dest/energy.txt"
    echo "broken" > "$dest/intact.txt"
    echo "no" > "$dest/hit.txt"
    return
  fi
  energy=$(last_energy "$dest/opt.out")
  intact=$(monomers_intact "$dest/xtbopt.xyz")
  hit=$(python3 - "$energy" "$intact" "$REFERENCE" << 'PY'
import sys
energy, intact, reference = sys.argv[1:]
value = float(energy)
print("hit" if intact == "intact" and value <= float(reference) + 1e-6 else "no")
PY
)
  printf '%s\n' "$energy" > "$dest/energy.txt"
  printf '%s\n' "$intact" > "$dest/intact.txt"
  printf '%s\n' "$hit" > "$dest/hit.txt"
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
  judge "$dest"
}
case "$SEED" in
  15000|15001|15002|15003|15004|15005) ;;
  *) echo "this handoff is for seeds 15000-15005" >&2; exit 1 ;;
esac
mkdir -p "$OUT"
seed="$SEED"
  base="$OUT/runs/$seed"
  mkdir -p "$base"
  cp "$START_ROOT/$seed/start.xyz" "$base/start.xyz"
  anneal="$base/anneal"
  crest="$base/crest"

set_start_xyz() {
  local xyz=$1
  grep -v '^export H2H_START_XYZ=' /work/bench-env.sh > /work/bench-env.sh.tmp
  if [[ -n $xyz ]]; then
    printf 'export H2H_START_XYZ=%s\n' "$xyz" >> /work/bench-env.sh.tmp
  fi
  mv /work/bench-env.sh.tmp /work/bench-env.sh
}

run_anneal() {
  mkdir -p "$anneal"
  cp "$base/start.xyz" "$anneal/start.xyz"
  write_definition "$anneal"
  pins "$anneal" anneal
  machine_raw "$anneal" start
  echo "ARM_START anneal $seed $(stamp)" >> "$OUT/schedule.txt"
  set_start_xyz "$anneal/start.xyz"
  (
    cd "$anneal"
    set +e
    timed "$BIN" "$MOLECULES" "$BUDGET" "$seed" . > stdout.txt 2> stderr.txt &
    echo $! > pid.txt
    wait $!
    echo $? > exit.code
  )
  set_start_xyz ""
  machine_raw "$anneal" end
  echo "ARM_DONE anneal $seed $(stamp)" >> "$OUT/schedule.txt"
  if [[ -f $anneal/best.xyz ]]; then
    score_one "$anneal/best.xyz" "$anneal/score"
  fi
}

crest_lowest_five() {
  local src=$1
  local dest=$2
  python3 - "$src" "$dest" << 'PY'
import os, sys
src, dest = sys.argv[1:]
lines = open(src, errors="replace").read().splitlines()
frames = []
i = 0
while i < len(lines):
    parts = lines[i].split()
    if not parts:
        i += 1
        continue
    try:
        count = int(parts[0])
    except ValueError:
        i += 1
        continue
    comment = lines[i + 1] if i + 1 < len(lines) else ""
    body = lines[i + 2:i + 2 + count]
    if len(body) < count:
        break
    energy = None
    for token in comment.split():
        try:
            energy = float(token)
            break
        except ValueError:
            continue
    frames.append((energy if energy is not None else 0.0, count, comment, body))
    i += 2 + count
frames.sort(key=lambda item: item[0])
os.makedirs(dest, exist_ok=True)
for index, (energy, count, comment, body) in enumerate(frames[:5], start=1):
    folder = os.path.join(dest, str(index))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "picked.xyz"), "w") as handle:
        handle.write(f"{count}\n{comment}\n")
        handle.write("\n".join(body) + "\n")
PY
}

run_crest() {
  mkdir -p "$crest"
  cp "$base/start.xyz" "$crest/start.xyz"
  write_definition "$crest"
  pins "$crest" crest
  {
    echo "crest start.xyz --gfn2 -T 1 --legacy -xnam /work/xtb-gfn2"
    echo "optimizations: real xtb at /work/xtb-gfn2"
    echo "dynamics: real xtb at /work/xtb-gfn2"
    echo "wall: not applied"
    echo "scoring: five lowest ensemble members, best intact vtight energy"
  } > "$crest/command.txt"
  machine_raw "$crest" start
  echo "ARM_START crest $seed $(stamp)" >> "$OUT/schedule.txt"
  set_start_xyz ""
  (
    cd "$crest"
    set +e
    timed "$CREST" start.xyz --gfn2 -T 1 --legacy -xnam "$XTB" > crest.out 2> crest.err &
    echo $! > pid.txt
    wait $!
    echo $? > exit.code
  )
  machine_raw "$crest" end
  echo "ARM_DONE crest $seed $(stamp)" >> "$OUT/schedule.txt"
  local ensemble=$crest/crest_conformers.xyz
  if [[ ! -f $ensemble && -f $crest/crest_best.xyz ]]; then
    ensemble=$crest/crest_best.xyz
  fi
  if [[ -f $ensemble ]]; then
    crest_lowest_five "$ensemble" "$crest/rescore"
    local member
    for member in "$crest"/rescore/[1-5]; do
      [[ -f $member/picked.xyz ]] || continue
      score_one "$member/picked.xyz" "$member/score"
    done
  fi
}

order_byte=$(od -An -N1 -tu1 /dev/urandom | tr -d ' ')
if (( order_byte % 2 == 0 )); then
  arm_order=(anneal crest)
else
  arm_order=(crest anneal)
fi
if [[ ${H2O_SKIP_ANNEAL:-0} == 1 ]]; then
  arm_order=(crest)
fi
echo "ARM_ORDER ${arm_order[*]} $(stamp)" >> "$OUT/schedule.txt"
for arm in "${arm_order[@]}"; do
  case "$arm" in
    anneal) run_anneal ;;
    crest) run_crest ;;
  esac
done

  {
    echo "seed: $seed"
    echo "budget: $BUDGET"
    echo "date: $(stamp)"
    echo "arm_order: ${arm_order[*]}"
    echo "anneal_exit: $(cat "$anneal/exit.code" 2>/dev/null || echo missing)"
    echo "anneal_score_exit: $(cat "$anneal/score/exit.code" 2>/dev/null || echo missing)"
    echo "anneal_energy: $(cat "$anneal/score/energy.txt" 2>/dev/null || echo missing)"
    echo "anneal_intact: $(cat "$anneal/score/intact.txt" 2>/dev/null || echo missing)"
    echo "anneal_hit: $(cat "$anneal/score/hit.txt" 2>/dev/null || echo missing)"
    echo "crest_exit: $(cat "$crest/exit.code" 2>/dev/null || echo missing)"
    echo "crest_pid: $(cat "$crest/pid.txt" 2>/dev/null || echo missing)"
    echo "anneal_pid: $(cat "$anneal/pid.txt" 2>/dev/null || echo missing)"
  } > "$base/seed.txt"
python3 - "$base" "$REFERENCE" << 'PY'
import glob, os, re, sys
base, reference = sys.argv[1:]
reference = float(reference)
crest = os.path.join(base, "crest")
lines = ["reference: %.12f" % reference]

def read(path):
    try:
        return open(path).read().strip()
    except FileNotFoundError:
        return "missing"

chosen = None
for member in sorted(glob.glob(os.path.join(crest, "rescore", "[1-5]"))):
    energy = read(os.path.join(member, "score", "energy.txt"))
    intact = read(os.path.join(member, "score", "intact.txt"))
    lines.append("crest_member %s %s %s" % (os.path.basename(member), energy, intact))
    if intact == "intact":
        try:
            value = float(energy)
        except ValueError:
            continue
        if chosen is None or value < chosen[0]:
            chosen = (value, os.path.basename(member), intact)
if chosen is None:
    lines.append("crest_energy: nan")
    lines.append("crest_intact: broken")
    lines.append("crest_hit: no")
    lines.append("crest_member_chosen: none")
else:
    value, member, intact = chosen
    lines.append("crest_energy: %.12f" % value)
    lines.append("crest_intact: %s" % intact)
    lines.append("crest_hit: %s" % ("hit" if value <= reference + 1e-6 else "no"))
    lines.append("crest_member_chosen: %s" % member)
text = open(os.path.join(crest, "crest.out"), errors="replace").read() if os.path.isfile(os.path.join(crest, "crest.out")) else ""
found = re.search(
    r"CREST runtime \(total\)\s+(\d+) d,\s+(\d+) h,\s+(\d+) min,\s+([0-9.]+) sec",
    text,
)
if found:
    day, hour, minute, second = found.groups()
    wall = int(day) * 86400 + int(hour) * 3600 + int(minute) * 60 + float(second)
    lines.append("crest_wall_s: %.6f" % wall)
else:
    lines.append("crest_wall_s: missing")
run = os.path.join(base, "anneal", "run.txt")
wall = "missing"
if os.path.isfile(run):
    for line in open(run):
        if line.startswith("wall_s:"):
            wall = line.split()[1]
lines.append("anneal_wall_s: %s" % wall)
lines.append("anneal_energy: %s" % read(os.path.join(base, "anneal", "score", "energy.txt")))
lines.append("anneal_intact: %s" % read(os.path.join(base, "anneal", "score", "intact.txt")))
lines.append("anneal_hit: %s" % read(os.path.join(base, "anneal", "score", "hit.txt")))
open(os.path.join(base, "summary.txt"), "w").write("\n".join(lines) + "\n")
PY
echo "SERIES_DONE $(stamp)" >> "$OUT/schedule.txt"
