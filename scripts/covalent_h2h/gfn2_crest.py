#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GFN2-xTB silicon and carbon panel: anneal's general arm against CREST.

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
``--codes``, ``--seeds`` and ``--crest-seeds`` select the arms and the
seeds, each system's defaults being ``SEEDS`` and ``CREST_SEEDS``; a
system without a reference is refused.
Rows are appended to ``--rows`` with the anneal revision, the host and the
binary hashes, and a file with another header is refused, never
rewritten. ``--history`` names earlier rows files, such as ``seeds.csv``,
read alongside it. ``--summary`` appends one line per system and code over
all of them, with Fisher's exact test on hits and a two-sided Mann-Whitney
on calls to the hit, with a miss ranked after every hit, both against
anneal; alone when no binaries are given, over ``--systems``, and with
``--anneal-revision`` over anneal's rows of that ``code_revision`` only.
``--false-hits`` names rows, by system, code, revision and seed, whose hit
is not the geometry's energy; the summary counts them as misses at their
budget.
Time per call skips a run that made no call.
Anneal stops at the reference and its miss is the full budget. CREST runs
to its own end; its energy is the first frame of ``crest_best.xyz``. A
hit is an energy within ``HIT_TOL`` eV of the reference. This process
refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import math
import os
import random
import re
import socket
import subprocess
import time
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

SYSTEMS = ("si10", "c10", "si13", "c20", "si20")
CODES = ("anneal-general", "crest")
TWELVE = tuple(range(2000, 2012))
# Si20 runs six seeds in both arms: a GFN2 call there costs 0.3-0.7 s.
SEEDS = {
    "si10": TWELVE,
    "c10": TWELVE,
    "si13": TWELVE,
    "c20": TWELVE,
    "si20": TWELVE[:6],
}
# CREST's seeds where its wall time does not allow twelve.
CREST_SEEDS = {**SEEDS, "si13": TWELVE[:6]}
BUDGET = 100000
ETEMP = 3000.0
HIT_TOL = 0.001
# eV per Hartree in xtb 6.7.1.
AUTOEV = 27.21138505
# The lowest minimum any run of either code reached at these settings,
# each reproduced by xtb from a fresh SCC. Si10 and C10: anneal at 30000
# calls on six seeds each; the C10 minimum is the cumulenic ring, every
# bond 1.285 angstrom. C20: the cage, every atom three-coordinated, from
# one anneal run without the call floor (seed 2006 at 1e5 calls); no
# other run of either code has reached it. Si13 and Si20 have none yet:
# the engine's reused wavefunction gives their clusters energies a fresh
# SCC does not reproduce.
REFERENCE = {
    "si10": -460.68373809,
    "c10": -571.32273783,
    "c20": -1148.85383633,
}
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
PROVENANCE_FIELDS = [
    "code_revision",
    "host",
    "date",
    "crest_sha256",
    "xtb_sha256",
    "engine_sha256",
]
SUMMARY_FIELDS = [
    "system",
    "code",
    "runs",
    "hits",
    "seeds",
    "median_evals_to_hit",
    "iqr_evals_to_hit",
    "median_best",
    "median_wall_s",
    "median_pot_s",
    "median_outside_s",
    "inside_ms_per_call",
    "outside_ms_per_call",
    "fisher_p",
    "mannwhitney_p",
    "code_revision",
    "host",
    "date",
]

WRAPPER = """#!/bin/sh
read -r start _ < /proc/uptime
tmp=$(mktemp)
"{xtb}" "$@" --etemp {etemp} >"$tmp" 2>&1
rc=$?
read -r end _ < /proc/uptime
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def append_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    if not new:
        with open(path, newline="") as handle:
            header = next(csv.reader(handle))
        if header != fields:
            raise SystemExit(
                f"{path} has another header; rows are appended, never rewritten"
            )
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anneal", type=Path)
    ap.add_argument("--crest", type=Path)
    ap.add_argument("--xtb", type=Path)
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--head-file", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rows", type=Path)
    ap.add_argument("--history", type=Path, nargs="*", default=[])
    ap.add_argument(
        "--systems", nargs="*", default=[s for s in SYSTEMS if s in REFERENCE]
    )
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--crest-seeds", nargs="*", type=int)
    ap.add_argument("--codes", nargs="*", default=list(CODES))
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--anneal-revision")
    ap.add_argument("--false-hits", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.rows is None:
        raise SystemExit("--rows is required")
    host = socket.gethostname()
    revision = args.head_file.read_text().strip() if args.head_file else ""
    if args.summary is not None and not any((args.anneal, args.crest, args.xtb)):
        summarise(
            [*args.history, args.rows],
            args.summary,
            revision,
            host,
            args.anneal_revision,
            args.systems,
            args.false_hits,
        )
        return
    if None in (
        args.anneal,
        args.crest,
        args.xtb,
        args.engine,
        args.head_file,
        args.out,
    ):
        raise SystemExit(
            "--anneal, --crest, --xtb, --engine, --head-file and --out run the panel"
        )
    for system in args.systems:
        if system not in REFERENCE:
            raise SystemExit(f"{system} has no reference")
    if host != "bench":
        raise SystemExit("hostname is not bench")
    os.environ["RGPOT_XTB_ETEMP"] = f"{ETEMP}"
    os.environ["OMP_NUM_THREADS"] = "1"
    python = os.environ.get("PANEL_PYTHON", "/usr/bin/python3")
    pins = {
        "code_revision": revision,
        "host": host,
        "crest_sha256": sha256(args.crest),
        "xtb_sha256": sha256(args.xtb),
        "engine_sha256": sha256(args.engine),
    }
    rng = random.Random(SCHEDULE_SEED)
    for system in args.systems:
        seeds = SEEDS[system] if args.seeds is None else args.seeds
        starts = args.out / system / "starts"
        starts.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
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
        if args.crest_seeds is not None:
            crest_seeds = set(args.crest_seeds)
        elif args.seeds is not None:
            crest_seeds = set(seeds)
        else:
            crest_seeds = set(CREST_SEEDS[system])
        pairs = [
            (seed, code)
            for seed in seeds
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
            row.update(pins, date=datetime.date.today().isoformat())
            append_rows(args.rows, ROW_FIELDS + PROVENANCE_FIELDS, [row])
            print(
                ",".join(str(row[k]) for k in ROW_FIELDS + PROVENANCE_FIELDS),
                flush=True,
            )
    if args.summary is not None:
        summarise([*args.history, args.rows], args.summary, revision, host)


def type7(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * p
    low = int(h)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def two_sided_mw(left: list[float], right: list[float]) -> float:
    if len(set(left) | set(right)) == 1:
        return 1.0
    return float(
        mannwhitneyu(
            left,
            right,
            alternative="two-sided",
            method="asymptotic",
            use_continuity=True,
        ).pvalue
    )


def median(runs: list[dict], key: str) -> float:
    return type7([float(row[key]) for row in runs], 0.5)


def per_call_ms(runs: list[dict], key: str) -> float:
    """Median time per call over the runs that made a call."""
    return type7(
        [
            float(row[key]) / int(row["charged"]) * 1e3
            for row in runs
            if int(row["charged"])
        ],
        0.5,
    )


def ranked_calls(runs: list[dict]) -> list[float]:
    """Calls to the hit, with a miss ranked after every hit."""
    return [
        float(row["evals_to_hit"]) if row["hit"] == "true" else math.inf for row in runs
    ]


def summarise(
    paths: list[Path],
    summary: Path,
    revision: str,
    host: str,
    anneal_revision: str | None = None,
    systems: list[str] | None = None,
    false_hits: Path | None = None,
) -> None:
    """Append one line per system and code over every rows file.

    With ``anneal_revision``, anneal's rows are only those of that
    ``code_revision``. A row named in ``false_hits`` by system, code,
    revision and seed counts as a miss at its budget.
    """
    table = []
    for path in paths:
        if path.is_file():
            with open(path, newline="") as handle:
                table.extend(csv.DictReader(handle))
    if anneal_revision is not None:
        table = [
            row
            for row in table
            if row["code"] != "anneal-general"
            or row.get("code_revision") == anneal_revision
        ]
    if false_hits is not None:
        with open(false_hits, newline="") as handle:
            named = {
                (row["system"], row["code"], row["code_revision"], row["seed"])
                for row in csv.DictReader(handle)
            }
        table = [
            dict(row, hit="false", evals_to_hit=row["budget"])
            if (row["system"], row["code"], row.get("code_revision", ""), row["seed"])
            in named
            else row
            for row in table
        ]
    date = datetime.date.today().isoformat()
    lines = []
    for system in systems or SYSTEMS:
        groups = {
            code: [
                row for row in table if row["system"] == system and row["code"] == code
            ]
            for code in CODES
        }
        anneal = groups["anneal-general"]
        for code in CODES:
            runs = groups[code]
            if not runs:
                continue
            evals = [float(row["evals_to_hit"]) for row in runs]
            hits = sum(row["hit"] == "true" for row in runs)
            seeds = sorted(int(row["seed"]) for row in runs)

            line = {
                "system": system,
                "code": code,
                "runs": len(runs),
                "hits": hits,
                "seeds": " ".join(str(seed) for seed in seeds),
                "median_evals_to_hit": f"{type7(evals, 0.5):.1f}",
                "iqr_evals_to_hit": f"{type7(evals, 0.75) - type7(evals, 0.25):.1f}",
                "median_best": f"{median(runs, 'best'):.8f}",
                "median_wall_s": f"{median(runs, 'wall_s'):.3f}",
                "median_pot_s": f"{median(runs, 'pot_s'):.3f}",
                "median_outside_s": f"{median(runs, 'outside_s'):.3f}",
                "inside_ms_per_call": f"{per_call_ms(runs, 'pot_s'):.4f}",
                "outside_ms_per_call": f"{per_call_ms(runs, 'outside_s'):.4f}",
                "fisher_p": "",
                "mannwhitney_p": "",
                "code_revision": revision,
                "host": host,
                "date": date,
            }
            if code != "anneal-general" and anneal:
                anneal_hits = sum(row["hit"] == "true" for row in anneal)
                odds = [
                    [hits, len(runs) - hits],
                    [anneal_hits, len(anneal) - anneal_hits],
                ]
                line["fisher_p"] = f"{fisher_exact(odds).pvalue:.6g}"
                mannwhitney = two_sided_mw(ranked_calls(runs), ranked_calls(anneal))
                line["mannwhitney_p"] = f"{mannwhitney:.6g}"
            lines.append(line)
    append_rows(summary, SUMMARY_FIELDS, lines)


def self_test() -> None:
    import tempfile

    if abs(type7([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) > 1e-12:
        raise SystemExit("type 7")
    if two_sided_mw([5.0, 5.0], [5.0, 5.0]) != 1.0:
        raise SystemExit("tied mann-whitney")
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "rows.csv"
        fields = ROW_FIELDS + PROVENANCE_FIELDS
        row = {key: "1" for key in fields}
        append_rows(path, fields, [row])
        append_rows(path, fields, [row])
        if len(path.read_text().splitlines()) != 3:
            raise SystemExit("append")
        path.write_text("system,code\n")
        try:
            append_rows(path, fields, [row])
        except SystemExit:
            pass
        else:
            raise SystemExit("header guard")
    print("ok")


if __name__ == "__main__":
    main()
