#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GFN2-xTB Si10 and C10 panel: anneal's general arm against CREST.

Both codes run real xtb 6.7.1 GFN2 at one electronic temperature,
``ETEMP`` K: anneal through rgpot's ``libxtb_engine.so``
(``examples/gfn2_covalent_h2h``), CREST 3.0.2 as

    crest start.xyz --gfn2 -T 1 --legacy -notopo -nocross -shake 0 \
        -tstep 2 -xnam <wrapper>

from the shared start after one xtb ``--opt`` of it. Each option is
there because a cluster search rearranges bonds. CREST stops when its
first optimisation changes the topology of a random cluster, so the start
is optimised before it. CREGEN removes every structure whose topology
differs from the input unless ``-notopo``. The genetic crossing then has
one structure and ends the run with an error, so ``-nocross``. The
default SHAKE holds every bond at its starting length during the
metadynamics, so ``-shake 0``, with the step at 2 fs in place of the 5
fs SHAKE allows.

The wrapper adds ``--etemp`` to every xtb process, times it, and logs the
energy calls it made: the ``max steps`` of an MD, and one
``* total energy`` line per optimisation cycle, or the
``:: total energy`` lines of any other run. CREST's calls are the sum of
that log.

Each seed's start is anneal's random start for that seed, written before
any search. The (seed, code) pairs of a system run in a shuffled order.
``--codes`` and ``--crest-seeds`` select the arms and CREST's seeds, and
``--summary`` writes one line per system and code, alone when no binaries
are given.
Anneal stops at the reference and its miss is the full budget. CREST runs
to its own end; its energy is the first frame of ``crest_best.xyz``. A
hit is an energy within ``HIT_TOL`` eV of the reference. This process
refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import socket
import statistics
import subprocess
import time
from pathlib import Path

SYSTEMS = ("si10", "c10")
CODES = ("anneal-general", "crest")
SEEDS = tuple(range(2000, 2012))
BUDGET = 100000
ETEMP = 3000.0
HIT_TOL = 0.001
# eV per Hartree in xtb 6.7.1.
AUTOEV = 27.21138505
# The lowest minimum any run of either code reached at these settings:
# anneal at 30000 calls on six seeds per system. The C10 minimum is the
# cumulenic ring, every bond 1.285 angstrom.
REFERENCE = {"si10": -460.68373809, "c10": -571.32273783}
SCHEDULE_SEED = 20261114

ROW_FIELDS = [
    "system",
    "code",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
]
SUMMARY_FIELDS = [
    "system",
    "code",
    "runs",
    "hits",
    "median_evals_to_hit",
    "median_wall_s",
]

WRAPPER = """#!/bin/sh
start=$(date +%s.%N)
tmp=$(mktemp)
"{xtb}" "$@" --etemp {etemp} >"$tmp" 2>&1
rc=$?
end=$(date +%s.%N)
cat "$tmp"
"{python}" - "$tmp" "{log}" "$start" "$end" "$*" << 'PY'
import re, sys
path, log, start, end, args = sys.argv[1:]
text = open(path, errors="ignore").read()
if "--md" in args or "--omd" in args or "--metadyn" in args:
    kind = "md"
    found = re.findall(r"(?m)^\\s*max steps\\s*:\\s*(\\d+)", text)
    calls = int(found[-1]) if found else 0
else:
    kind = "opt" if "--opt" in args else "other"
    stars = len(re.findall(r"(?m)^\\s*\\* total energy\\b", text))
    colons = len(re.findall(r"(?m)^\\s*:: total energy\\b", text))
    calls = stars if stars else colons
with open(log, "a") as handle:
    handle.write(f"{{float(end) - float(start):.6f}} {{kind}} {{calls}}\\n")
PY
rm -f "$tmp"
exit $rc
"""


def run_anneal(binary: Path, system: str, seed: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    target = f"{REFERENCE[system]:.8f}"
    proc = subprocess.run(
        [str(binary), system, str(seed), str(BUDGET), target, str(out / "best.xyz")],
        capture_output=True,
        text=True,
        check=True,
    )
    (out / "stdout.txt").write_text(proc.stdout)
    lines = proc.stdout.splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith("system,code"))
    row = dict(zip(lines[head].split(","), lines[head + 1].split(",")))
    return {
        "system": system,
        "code": "anneal-general",
        "seed": seed,
        "budget": BUDGET,
        "hit": row["hit"],
        "evals_to_hit": int(row["evals_to_hit"]),
        "charged": int(row["charged"]),
        "wall_s": float(row["wall_s"]),
        "best": float(row["best_eV"]),
        "pot_s": float(row["pot_s"]),
        "outside_s": float(row["outside_s"]),
    }


def first_frame_energy(path: Path) -> float:
    lines = path.read_text().splitlines()
    found = re.findall(r"[-+]?\d+\.\d+", lines[1])
    return float(found[0]) * AUTOEV


def run_crest(
    crest: Path, xtb: Path, python: str, system: str, seed: int, start: Path, out: Path
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log = out / "xtb-calls.log"
    log.write_text("")
    wrapper = out / "xtb-timed"
    wrapper.write_text(WRAPPER.format(xtb=xtb, etemp=ETEMP, python=python, log=log))
    wrapper.chmod(0o755)
    preopt = out / "preopt"
    preopt.mkdir(exist_ok=True)
    (preopt / "start.xyz").write_bytes(start.read_bytes())
    prepare = [str(wrapper), "start.xyz", "--gfn", "2", "--opt"]
    command = [
        str(crest),
        "start.xyz",
        "--gfn2",
        "-T",
        "1",
        "--legacy",
        "-notopo",
        "-nocross",
        "-shake",
        "0",
        "-tstep",
        "2",
        "-xnam",
        str(wrapper),
    ]
    (out / "command.txt").write_text(
        " ".join(prepare) + "\n" + " ".join(command) + "\n"
    )
    began = time.monotonic()
    with open(preopt / "opt.out", "w") as stdout:
        subprocess.run(
            prepare, check=False, cwd=preopt, stdout=stdout, stderr=subprocess.STDOUT
        )
    code = 1
    if (preopt / "xtbopt.xyz").is_file():
        (out / "start.xyz").write_bytes((preopt / "xtbopt.xyz").read_bytes())
        with (
            open(out / "crest.out", "w") as stdout,
            open(out / "crest.err", "w") as stderr,
        ):
            code = subprocess.run(
                command, check=False, cwd=out, stdout=stdout, stderr=stderr
            ).returncode
    wall = time.monotonic() - began
    (out / "exit.code").write_text(f"{code}\n")
    inside = 0.0
    calls = 0
    for line in log.read_text().splitlines():
        seconds, _, count = line.split()
        inside += float(seconds)
        calls += int(count)
    best_file = out / "crest_best.xyz"
    best = first_frame_energy(best_file) if best_file.is_file() else float("inf")
    hit = best <= REFERENCE[system] + HIT_TOL
    return {
        "system": system,
        "code": "crest",
        "seed": seed,
        "budget": calls,
        "hit": "true" if hit else "false",
        "evals_to_hit": calls,
        "charged": calls,
        "wall_s": wall,
        "best": best,
        "pot_s": inside,
        "outside_s": max(wall - inside, 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anneal", type=Path)
    ap.add_argument("--crest", type=Path)
    ap.add_argument("--xtb", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--crest-seeds", nargs="*", type=int)
    ap.add_argument("--codes", nargs="*", default=list(CODES))
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()
    if args.summary is not None and not any((args.anneal, args.crest, args.xtb)):
        summarise(args.rows, args.summary)
        return
    if None in (args.anneal, args.crest, args.xtb, args.out):
        raise SystemExit("--anneal, --crest, --xtb and --out run the panel")
    if socket.gethostname() != "bench":
        raise SystemExit("hostname is not bench")
    os.environ["RGPOT_XTB_ETEMP"] = f"{ETEMP}"
    os.environ["OMP_NUM_THREADS"] = "1"
    python = os.environ.get("PANEL_PYTHON", "/usr/bin/python3")
    rng = random.Random(SCHEDULE_SEED)
    write_header = not args.rows.exists()
    with open(args.rows, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        for system in args.systems:
            starts = args.out / system / "starts"
            starts.mkdir(parents=True, exist_ok=True)
            for seed in args.seeds:
                subprocess.run(
                    [
                        str(args.anneal),
                        system,
                        str(seed),
                        str(BUDGET),
                        "-",
                        str(starts / f"{seed}.xyz"),
                    ],
                    env=dict(os.environ, H2H_START_ONLY="1"),
                    capture_output=True,
                    check=True,
                )
            crest_seeds = set(
                args.seeds if args.crest_seeds is None else args.crest_seeds
            )
            pairs = [
                (seed, code)
                for seed in args.seeds
                for code in CODES
                if code in args.codes and (code != "crest" or seed in crest_seeds)
            ]
            rng.shuffle(pairs)
            for seed, code in pairs:
                where = args.out / system / code / str(seed)
                if code == "crest":
                    row = run_crest(
                        args.crest,
                        args.xtb,
                        python,
                        system,
                        seed,
                        starts / f"{seed}.xyz",
                        where,
                    )
                else:
                    row = run_anneal(args.anneal, system, seed, where)
                writer.writerow(row)
                handle.flush()
                print(",".join(str(row[k]) for k in ROW_FIELDS), flush=True)
    if args.summary is not None:
        summarise(args.rows, args.summary)


def summarise(rows: Path, summary: Path) -> None:
    """One line per system and code: hits, median calls and wall time."""
    with open(rows, newline="") as handle:
        table = list(csv.DictReader(handle))
    with open(summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for system in SYSTEMS:
            for code in CODES:
                runs = [
                    row
                    for row in table
                    if row["system"] == system and row["code"] == code
                ]
                if not runs:
                    continue
                writer.writerow(
                    {
                        "system": system,
                        "code": code,
                        "runs": len(runs),
                        "hits": sum(row["hit"] == "true" for row in runs),
                        "median_evals_to_hit": statistics.median(
                            int(row["evals_to_hit"]) for row in runs
                        ),
                        "median_wall_s": f"{statistics.median(float(row['wall_s']) for row in runs):.1f}",
                    }
                )


if __name__ == "__main__":
    main()
