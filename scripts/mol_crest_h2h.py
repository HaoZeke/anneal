#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GFN2 CREST head-to-head.

Anneal uses Config::recommended_molecular at GFN2_HEXAMER_BUDGET (4000),
the molecular entry of the paper budgets. CREST is native xtb with no wall.
Both arms share one rigid repack. Scoring is xtb --gfn 2 --opt vtight.
A hit is within 1e-6 Eh of the best intact energy either arm finds for
that system, except (NH3)4, which keeps the published cyclic-tetramer gate.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path

BUDGET = 4000
HIT_WINDOW = 1e-6
NH3_REFERENCE = -17.729616186942
HEAVY_H = {
    "nh3-4": {"N": 3},
    "nh3-8": {"N": 3},
    "h2o-8": {"O": 2},
    "h2o-12": {"O": 2},
    "hf-6": {"F": 1},
    "meoh-4": {"C": 3, "O": 1},
    "nh3-2-h2o-4": {"N": 3, "O": 2},
}
# Anneal binary and first argument. nh3-4 reuses the tetramer example.
ANNEAL = {
    "nh3-4": ("gfn2_nh3_h2h", ["4"]),
    "nh3-8": ("gfn2_mol_h2h", ["nh3-8"]),
    "h2o-8": ("gfn2_mol_h2h", ["h2o-8"]),
    "h2o-12": ("gfn2_mol_h2h", ["h2o-12"]),
    "hf-6": ("gfn2_mol_h2h", ["hf-6"]),
    "meoh-4": ("gfn2_mol_h2h", ["meoh-4"]),
    "nh3-2-h2o-4": ("gfn2_mol_h2h", ["nh3-2-h2o-4"]),
}


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run(
    cmd: list[str],
    cwd: Path,
    extra: dict[str, str] | None = None,
    stdout: Path | None = None,
    stderr: Path | None = None,
) -> int:
    bench = os.environ.get("BENCH_ENV", "/work/bench-env-nh3.sh")
    exports = "".join(f"export {key}='{value}'; " for key, value in (extra or {}).items())
    inner = " ".join(f"'{part}'" for part in cmd)
    wrapped = [
        "sudo",
        "-n",
        "unshare",
        "--uts",
        "sh",
        "-c",
        f"hostname bench; . '{bench}'; {exports}cd '{cwd}'; exec {inner}",
    ]
    out = stdout.open("w") if stdout else subprocess.DEVNULL
    err = stderr.open("w") if stderr else subprocess.DEVNULL
    proc = subprocess.run(wrapped, stdout=out, stderr=err)
    if stdout:
        out.close()
    if stderr and stderr != stdout:
        err.close()
    return proc.returncode


def read_xyz(path: Path) -> list[tuple[str, list[float]]]:
    lines = path.read_text(errors="replace").splitlines()
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
        body = lines[i + 2 : i + 2 + count]
        for row in body:
            item = row.split()
            atoms.append((item[0], [float(v) for v in item[1:4]]))
        break
    return atoms


def intact_of(path: Path, expect: dict[str, int]) -> str:
    if not path.is_file():
        return "missing"
    atoms = read_xyz(path)
    hydrogens = [xyz for sym, xyz in atoms if sym == "H"]

    def near(a, b) -> bool:
        return math.dist(a, b) <= 1.2

    heavies = [(sym, xyz) for sym, xyz in atoms if sym in expect]
    if not heavies:
        return "broken"
    ok = all(sum(near(xyz, h) for h in hydrogens) == expect[sym] for sym, xyz in heavies)
    return "intact" if ok else "broken"


def score(xtb: str, src: Path, dest: Path, expect: dict[str, int]) -> tuple[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "in.xyz").write_text(src.read_text())
    rc = run(
        [xtb, "in.xyz", "--gfn", "2", "--opt", "vtight", "--acc", "0.01", "--iterations", "500"],
        dest,
        stdout=dest / "opt.out",
        stderr=dest / "opt.err",
    )
    (dest / "exit.code").write_text(str(rc) + "\n")
    text = (dest / "opt.out").read_text(errors="replace") if (dest / "opt.out").exists() else ""
    found = re.findall(r"TOTAL ENERGY\s+([+-]?\d+\.\d+)", text)
    energy = found[-1] if found else "nan"
    (dest / "energy.txt").write_text(energy + "\n")
    geometry = dest / "xtbopt.xyz"
    state = intact_of(geometry, expect)
    (dest / "intact.txt").write_text(state + "\n")
    return energy, state


def crest_calls(crest_dir: Path) -> tuple[int, int, float, float | None]:
    text = (crest_dir / "crest.out").read_text(errors="replace") if (crest_dir / "crest.out").is_file() else ""
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
    inside = 0.0
    opt_cycles = 0
    log = crest_dir / "xtb-calls.log"
    if log.is_file():
        for line in log.read_text().splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            seconds, kind, cycles = float(parts[0]), parts[1], int(parts[2])
            inside += seconds
            if kind == "opt":
                opt_cycles += cycles
    wall = None
    match = re.search(
        r"CREST runtime \(total\)\s+(\d+) d,\s+(\d+) h,\s+(\d+) min,\s+([0-9.]+) sec",
        text,
    )
    if match:
        day, hour, minute, second = match.groups()
        wall = int(day) * 86400 + int(hour) * 3600 + int(minute) * 60 + float(second)
    return steps + opt_cycles, steps, inside, wall


def lowest_frames(path: Path, dest: Path, count: int = 5) -> None:
    lines = path.read_text(errors="replace").splitlines()
    frames = []
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        try:
            n = int(parts[0])
        except ValueError:
            i += 1
            continue
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        body = lines[i + 2 : i + 2 + n]
        if len(body) < n:
            break
        energy = None
        for token in comment.split():
            try:
                energy = float(token)
                break
            except ValueError:
                continue
        frames.append((energy if energy is not None else 0.0, n, comment, body))
        i += 2 + n
    frames.sort(key=lambda item: item[0])
    for index, (energy, n, comment, body) in enumerate(frames[:count], start=1):
        folder = dest / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "picked.xyz").write_text(f"{n}\n{comment}\n" + "\n".join(body) + "\n")


def field(path: Path, name: str) -> str:
    if not path.is_file():
        return "missing"
    for line in path.read_text().splitlines():
        if line.startswith(name + ":"):
            return line.split(":", 1)[1].strip()
    return "missing"


def one_seed(
    system: str,
    seed: int,
    out: Path,
    binaries: Path,
    crest_too: bool,
    shared_start: Path | None,
) -> None:
    xtb = os.environ.get("XTB_BIN", "/work/xtb-gfn2")
    crest_bin = os.environ.get("CREST_BIN", "/opt/crestenv/bin/crest")
    expect = HEAVY_H[system]
    base = out / "runs" / str(seed)
    anneal = base / "anneal"
    crest = base / "crest"
    anneal.mkdir(parents=True, exist_ok=True)
    if crest_too:
        crest.mkdir(parents=True, exist_ok=True)
    binary_name, prefix = ANNEAL[system]
    binary = binaries / binary_name
    if shared_start is not None:
        text = shared_start.read_text()
        (base / "start.xyz").write_text(text)
        (anneal / "start.xyz").write_text(text)
    else:
        run(
            [str(binary), *prefix, str(BUDGET), str(seed), str(anneal)],
            anneal,
            {"H2H_START_ONLY": "1"},
        )
        (base / "start.xyz").write_text((anneal / "start.xyz").read_text())
    if crest_too:
        (crest / "start.xyz").write_text((base / "start.xyz").read_text())
    order = ["anneal", "crest"] if crest_too else ["anneal"]
    if crest_too and random.randrange(2):
        order = ["crest", "anneal"]
    (out / "schedule.txt").open("a").write(f"ARM_ORDER {' '.join(order)} {system} {seed} {stamp()}\n")
    for arm in order:
        (out / "schedule.txt").open("a").write(f"ARM_START {arm} {system} {seed} {stamp()}\n")
        if arm == "anneal":
            run(
                [str(binary), *prefix, str(BUDGET), str(seed), str(anneal)],
                anneal,
                {"H2H_START_XYZ": str(base / "start.xyz")},
                stdout=anneal / "stdout.txt",
                stderr=anneal / "stderr.txt",
            )
        else:
            timer = out / "xtb-timed"
            timer.write_text(
                "#!/bin/sh\n"
                f'log="{crest}/xtb-calls.log"\n'
                "start=$(date +%s.%N)\n"
                "tmp=$(mktemp)\n"
                f'"{xtb}" "$@" >"$tmp" 2>&1\n'
                "rc=$?\n"
                "end=$(date +%s.%N)\n"
                'cat "$tmp"\n'
                'python3 - "$tmp" "$log" "$start" "$end" "$*" << \'PY\'\n'
                "import re, sys\n"
                "path, log, start, end, args = sys.argv[1:]\n"
                "text = open(path, errors='ignore').read()\n"
                "stars = len(re.findall(r'(?m)^\\s*\\* total energy\\b', text))\n"
                "colons = len(re.findall(r'(?m)^\\s*:: total energy\\b', text))\n"
                "cycles = stars if stars else colons\n"
                "kind = 'other'\n"
                "if '--md' in args or '--omd' in args:\n"
                "    kind = 'md'\n"
                "elif '--opt' in args:\n"
                "    kind = 'opt'\n"
                "seconds = float(end) - float(start)\n"
                "open(log, 'a').write(f'{seconds:.6f} {kind} {cycles}\\n')\n"
                "PY\n"
                'rm -f "$tmp"\n'
                "exit $rc\n"
            )
            timer.chmod(0o755)
            (crest / "xtb-calls.log").write_text("")
            run(
                [crest_bin, "start.xyz", "--gfn2", "-T", "1", "--legacy", "-xnam", str(timer)],
                crest,
                stdout=crest / "crest.out",
                stderr=crest / "crest.err",
            )
        (out / "schedule.txt").open("a").write(f"ARM_DONE {arm} {system} {seed} {stamp()}\n")
    if (anneal / "best.xyz").is_file():
        score(xtb, anneal / "best.xyz", anneal / "score", expect)
    if crest_too:
        ensemble = crest / "crest_conformers.xyz"
        if not ensemble.is_file() and (crest / "crest_best.xyz").is_file():
            ensemble = crest / "crest_best.xyz"
        if ensemble.is_file():
            lowest_frames(ensemble, crest / "rescore")
            for member in sorted((crest / "rescore").glob("[1-5]")):
                picked = member / "picked.xyz"
                if picked.is_file():
                    score(xtb, picked, member / "score", expect)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--skip-crest", action="store_true")
    parser.add_argument("--start", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for system in args.systems:
        for seed in args.seeds:
            print(f"SEED {system} {seed} {stamp()}", flush=True)
            one_seed(
                system,
                seed,
                args.out / system,
                args.binaries,
                crest_too=not args.skip_crest,
                shared_start=args.start,
            )


if __name__ == "__main__":
    main()
