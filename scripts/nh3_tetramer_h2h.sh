#!/bin/bash
# (NH3)4 head-to-head. Anneal is recommended_molecular at the catalog
# GFN2 molecular budget. CREST is native xtb. Both arms share one rigid
# repack. Scoring is the H2O16 vtight command. A hit is at or below the
# GFN2 relaxation of the published cyclic tetramer, plus 1e-6 Eh.
set -u
BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
START_ROOT=${2:?start root}
OUT=${3:?output root}
SEED=${4:?seed}
XTB=${XTB_BIN:-/work/xtb-gfn2}
CREST=${CREST_BIN:?crest binary}
export H2H_OMIT_HOST=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
MOLECULES=4
BUDGET=4000
REFERENCE=-17.729616186942
case "$(rustc -V)" in
  "rustc 1.88.0 "*) ;;
  *)
    echo "rustc is not the Cargo.toml pin of 1.88.0" >&2
    exit 1
    ;;
esac
# sudo drops the environment. BENCH_ENV is sourced inside the namespace and
# can point the develop pin at an engine that exports rgpot_xtb_reset.
BENCH_ENV=${BENCH_ENV:-/work/bench-env.sh}
timed() {
  sudo -n unshare --uts sh -c "hostname bench; . '$BENCH_ENV'; exec \"\$@\"" timed-sh "$@"
}
stamp() {
  python3 - << 'PY'
import time
secs = int(time.time()) - 6 * 3600
print(time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(secs)) + "-0600")
PY
}
base="$OUT/runs/$SEED"
anneal="$base/anneal"
crest="$base/crest"
mkdir -p "$base" "$anneal" "$crest"
cp "$START_ROOT/$SEED/start.xyz" "$base/start.xyz"
cp "$base/start.xyz" "$anneal/start.xyz"
cp "$base/start.xyz" "$crest/start.xyz"

order_byte=$(od -An -N1 -tu1 /dev/urandom | tr -d ' ')
if (( order_byte % 2 == 0 )); then
  arm_order=(anneal crest)
else
  arm_order=(crest anneal)
fi
echo "ARM_ORDER ${arm_order[*]} seed $SEED $(stamp)" >> "$OUT/schedule.txt"

run_anneal() {
  echo "ARM_START anneal $SEED $(stamp)" >> "$OUT/schedule.txt"
  (
    cd "$anneal"
    set +e
    H2H_START_XYZ="$base/start.xyz" \
      timed "$BIN" "$MOLECULES" "$BUDGET" "$SEED" . > stdout.txt 2> stderr.txt
    echo $? > exit.code
  )
  echo "ARM_DONE anneal $SEED $(stamp)" >> "$OUT/schedule.txt"
}

write_xtb_timer() {
  cat > "$OUT/xtb-timed" << EOF
#!/bin/sh
log="$crest/xtb-calls.log"
start=\$(date +%s.%N)
tmp=\$(mktemp)
"$XTB" "\$@" >"\$tmp" 2>&1
rc=\$?
end=\$(date +%s.%N)
cat "\$tmp"
python3 - "\$tmp" "\$log" "\$start" "\$end" "\$*" << 'PY'
import re, sys
path, log, start, end, args = sys.argv[1:]
text = open(path, errors="ignore").read()
stars = len(re.findall(r"(?m)^\\s*\\* total energy\\b", text))
colons = len(re.findall(r"(?m)^\\s*:: total energy\\b", text))
cycles = stars if stars else colons
kind = "other"
if "--md" in args or "--omd" in args:
    kind = "md"
elif "--opt" in args:
    kind = "opt"
seconds = float(end) - float(start)
with open(log, "a") as handle:
    handle.write(f"{seconds:.6f} {kind} {cycles}\\n")
PY
rm -f "\$tmp"
exit \$rc
EOF
  chmod 755 "$OUT/xtb-timed"
}

run_crest() {
  write_xtb_timer
  : > "$crest/xtb-calls.log"
  {
    echo "crest start.xyz --gfn2 -T 1 --legacy -xnam $OUT/xtb-timed"
    echo "optimizations: real xtb at $XTB"
    echo "dynamics: real xtb at $XTB"
    echo "wall: not applied"
  } > "$crest/command.txt"
  echo "ARM_START crest $SEED $(stamp)" >> "$OUT/schedule.txt"
  (
    cd "$crest"
    set +e
    timed "$CREST" start.xyz --gfn2 -T 1 --legacy -xnam "$OUT/xtb-timed" > crest.out 2> crest.err
    echo $? > exit.code
  )
  echo "ARM_DONE crest $SEED $(stamp)" >> "$OUT/schedule.txt"
}

score_one() {
  local src=$1 dest=$2
  mkdir -p "$dest"
  cp "$src" "$dest/in.xyz"
  (
    cd "$dest"
    set +e
    timed "$XTB" in.xyz --gfn 2 --opt vtight --acc 0.01 --iterations 500 > opt.out 2> opt.err
    echo $? > exit.code
  )
  python3 - "$dest" "$REFERENCE" << 'PY'
import math, os, re, sys
dest, reference = sys.argv[1:]
reference = float(reference)
text = open(os.path.join(dest, "opt.out"), errors="replace").read()
found = re.findall(r"TOTAL ENERGY\s+([+-]?\d+\.\d+)", text)
energy = found[-1] if found else "nan"
open(os.path.join(dest, "energy.txt"), "w").write(energy + "\n")
xyz = os.path.join(dest, "xtbopt.xyz")
intact = "broken"
if os.path.isfile(xyz):
    lines = open(xyz).read().splitlines()
    atoms = []
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
        for row in body:
            item = row.split()
            atoms.append((item[0], [float(v) for v in item[1:4]]))
        break
    nitrogens = [xyz for sym, xyz in atoms if sym == "N"]
    hydrogens = [xyz for sym, xyz in atoms if sym == "H"]
    def near(a, b):
        return math.dist(a, b) <= 1.2
    ok = bool(nitrogens) and all(sum(near(n, h) for h in hydrogens) == 3 for n in nitrogens)
    intact = "intact" if ok else "broken"
open(os.path.join(dest, "intact.txt"), "w").write(intact + "\n")
try:
    hit = "yes" if float(energy) <= reference + 1e-6 else "no"
except ValueError:
    hit = "no"
open(os.path.join(dest, "hit.txt"), "w").write(hit + "\n")
PY
}

crest_lowest_five() {
  python3 - "$1" "$2" << 'PY'
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

for arm in "${arm_order[@]}"; do
  case "$arm" in
    anneal) run_anneal ;;
    crest) run_crest ;;
  esac
done

if [[ -f $anneal/best.xyz ]]; then
  score_one "$anneal/best.xyz" "$anneal/score"
fi
ensemble=$crest/crest_conformers.xyz
if [[ ! -f $ensemble && -f $crest/crest_best.xyz ]]; then
  ensemble=$crest/crest_best.xyz
fi
if [[ -f $ensemble ]]; then
  crest_lowest_five "$ensemble" "$crest/rescore"
  for member in "$crest"/rescore/[1-5]; do
    [[ -f $member/picked.xyz ]] || continue
    score_one "$member/picked.xyz" "$member/score"
  done
fi

python3 - "$base" "$crest" "$REFERENCE" << 'PY'
import os, re, sys
base, crest, reference = sys.argv[1:]
reference = float(reference)

def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return "missing"

lines = [f"reference: {reference:.12f}"]

def add_member(folder, index):
    energy = read(os.path.join(folder, "energy.txt"))
    intact = read(os.path.join(folder, "intact.txt"))
    lines.append(f"crest_member {index} {energy} {intact}")
    try:
        value = float(energy)
    except ValueError:
        return None
    return value, intact, index

best = None
for index in range(1, 6):
    folder = os.path.join(crest, "rescore", str(index), "score")
    if not os.path.isdir(folder):
        continue
    row = add_member(folder, index)
    if row is None:
        continue
    if row[1] != "intact":
        continue
    if best is None or row[0] < best[0]:
        best = row
if best is None:
    lines.append("crest_energy: missing")
    lines.append("crest_intact: missing")
    lines.append("crest_hit: no")
else:
    lines.append(f"crest_energy: {best[0]:.12f}")
    lines.append(f"crest_intact: {best[1]}")
    lines.append("crest_hit: " + ("yes" if best[0] <= reference + 1e-6 else "no"))
    lines.append(f"crest_member_chosen: {best[2]}")
text = open(os.path.join(crest, "crest.out"), errors="replace").read() if os.path.isfile(os.path.join(crest, "crest.out")) else ""
steps = 0
rows = text.splitlines()
i = 0
while i < len(rows):
    if "MD time /ps" in rows[i]:
        t = float(rows[i].split(":")[1].split()[0])
        dt = None
        finished = False
        j = i + 1
        while j < len(rows) and "MD time /ps" not in rows[j]:
            if "dt /fs" in rows[j] and dt is None:
                dt = float(rows[j].split(":")[1].split()[0])
            if "finished" in rows[j].lower():
                finished = True
                break
            j += 1
        if finished and dt:
            steps += int(round(t * 1000.0 / dt))
        i = j
    else:
        i += 1
opt_cycles = 0
inside = 0.0
log = os.path.join(crest, "xtb-calls.log")
if os.path.isfile(log):
    for line in open(log):
        parts = line.split()
        if len(parts) != 3:
            continue
        seconds, kind, cycles = float(parts[0]), parts[1], int(parts[2])
        inside += seconds
        if kind == "opt":
            opt_cycles += cycles
calls = steps + opt_cycles
wall = None
match = re.search(r"CREST runtime \(total\)\s+(\d+) d,\s+(\d+) h,\s+(\d+) min,\s+([0-9.]+) sec", text)
if match:
    day, hour, minute, second = match.groups()
    wall = int(day) * 86400 + int(hour) * 3600 + int(minute) * 60 + float(second)
lines.append(f"crest_wall_s: {wall if wall is not None else 'missing'}")
lines.append(f"crest_md_steps: {steps}")
lines.append(f"crest_opt_cycles: {opt_cycles}")
lines.append(f"crest_calls: {calls}")
lines.append(f"crest_inside_s: {inside:.6f}")
outside = None if wall is None else max(0.0, wall - inside)
lines.append(f"crest_outside_s: {outside if outside is not None else 'missing'}")
if calls and wall is not None:
    lines.append(f"crest_inside_per_call_s: {inside / calls:.9f}")
    lines.append(f"crest_outside_per_call_s: {outside / calls:.9f}")

def field(name):
    path = os.path.join(base, "anneal", "run.txt")
    if not os.path.isfile(path):
        return "missing"
    for line in open(path):
        if line.startswith(name + ":"):
            return line.split(":", 1)[1].strip()
    return "missing"

lines.append("anneal_wall_s: " + field("wall_s"))
lines.append("anneal_calls: " + field("energy_calls"))
lines.append("anneal_pot_s: " + field("pot_s"))
lines.append("anneal_outside_s: " + field("outside_s"))
lines.append("anneal_pot_per_call_s: " + field("pot_per_call_s"))
lines.append("anneal_outside_per_call_s: " + field("outside_per_call_s"))
lines.append("anneal_energy: " + read(os.path.join(base, "anneal", "score", "energy.txt")))
lines.append("anneal_intact: " + read(os.path.join(base, "anneal", "score", "intact.txt")))
lines.append("anneal_hit: " + read(os.path.join(base, "anneal", "score", "hit.txt")))
open(os.path.join(base, "summary.txt"), "w").write("\n".join(lines) + "\n")
PY
