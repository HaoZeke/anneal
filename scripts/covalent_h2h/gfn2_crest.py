#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GFN2-xTB silicon and carbon panel: anneal's general arm against CREST.

Both codes run real xtb 6.7.1 GFN2 at one electronic temperature,
``ETEMP`` K: anneal through rgpot's ``libxtb_engine.so``
(``examples/gfn2_covalent_h2h``), CREST 3.0.2 as

    crest start.xyz --gfn2 -T 1 --legacy -notopo -noreftopo -nocross \
        -shake 0 -tstep 2 -xnam <wrapper>

from the shared start after one xtb ``--opt`` of it. Each option is
there because a cluster search rearranges bonds. CREST stops when its
first optimisation changes the topology of a random cluster, so the start
is optimised before it, and on Si20 seeds 2008, 2010 and 2012 CREST's
own first optimisation still changed the topology of that and stopped:
``-noreftopo`` takes the new topology and goes on, and where the topology
holds CREST runs as without it. CREGEN removes every structure whose topology
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
any search. ``--check-starts CSV`` appends instead whether xtb answers each
start from a fresh SCC, and runs nothing. The (seed, code) pairs of a
system run in a shuffled order.
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
``--superseded`` names rows, by system, code, revision and seed, that a
later run of the same seed replaces: the summary and the wall cap leave
them out, and the summary refuses a CREST seed with two rows left.
``--false-hits`` names rows, by system, code, revision and seed, whose hit
is not the geometry's energy; the summary counts them as misses at their
budget. ``--rescore SYSTEM ENERGY_EV`` appends to ``--rescored`` CREST's
hits on that system, over the same files, against another reference;
anneal stops at its own and is not rescored.
Time per call skips a run that made no call.
Anneal stops at the reference and its miss is the full budget. CREST runs
to its own end; its energy is the first frame of ``crest_best.xyz``. A
hit is an energy within ``HIT_TOL`` eV of the reference. This process
refuses to run unless the UTS hostname is bench.

On a system in ``WALL_CAP`` CREST runs first. Anneal then stops at the
budget, the reference, or ``WALL_CAP`` times the slowest CREST wall over
every seed it runs, whichever comes first, through ``H2H_WALL_S``; it is
refused until CREST has a row for each of those seeds. Those rows carry
the cap in ``wall_cap_s``, empty for CREST, so they go to their own file.
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
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

for _candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_candidate / "target_tol.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

SYSTEMS = ("si10", "c10", "si13", "c20", "si20")
CODES = ("anneal-general", "crest")
TWELVE = tuple(range(2000, 2012))
# Si13 runs six seeds in both arms: from a fresh SCC a call costs about
# 0.1 s. Si20 runs twelve under the wall cap, the first twelve from 2000
# whose start xtb answers from a fresh SCC (``gfn2/si20-starts.csv``, from
# ``--check-starts``): on the others CREST's first ``--opt`` stops at its
# first SCC, and on 2000-2011 that is seven of twelve.
SI20 = (2003, 2005, 2008, 2009, 2010, 2012, 2013, 2014, 2018, 2019, 2020, 2021)
SEEDS = {
    "si10": TWELVE,
    "c10": TWELVE,
    "si13": TWELVE[:6],
    "c20": TWELVE,
    "si20": SI20,
}
# CREST's seeds, anneal's unless its wall time does not allow them.
CREST_SEEDS = dict(SEEDS)
BUDGET = 100000
ETEMP = 3000.0
HIT_TOL = DEFAULT_HIT_TOL
# eV per Hartree in xtb 6.7.1.
AUTOEV = 27.21138505
# The lowest minimum any run of either code reached at these settings,
# each reproduced by xtb from a fresh SCC. Si10 and C10: anneal at 30000
# calls on six seeds each; the C10 minimum is the cumulenic ring, every
# bond 1.285 angstrom. C20: the cage, every atom three-coordinated, from
# one anneal run without the call floor (seed 2006 at 1e5 calls); no
# other run of either code has reached it. Si13: anneal from a fresh SCC
# on every call at 30000 calls, reached by seeds 3002-3005. Si20 is
# provisional: the twenty-membered ring two of eight capped anneal seeds
# reached (3003 and 3006 on a52726af), which xtb gives the same energy
# from a fresh start; it is a reference only if the panel reaches it again.
REFERENCE = {
    "si10": -460.68373809,
    "c10": -571.32273783,
    "si13": -599.34546694,
    "c20": -1148.85383633,
    "si20": -922.98227162,
}
# Anneal's wall cap over the slowest CREST wall, as the LJ75 ASE arms are
# capped at three anneal walls. Si20 at 1e5 calls from a fresh SCC is days
# a seed.
WALL_CAP = {"si20": 3.0}
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


def row_fields(system: str) -> list[str]:
    capped = ["wall_cap_s"] if system in WALL_CAP else []
    return ROW_FIELDS + capped + PROVENANCE_FIELDS


START_FIELDS = [
    "system",
    "seed",
    "converged",
    "xtb_fresh_Eh",
    "code_revision",
    "host",
    "date",
    "xtb_sha256",
]
CENSOR_FIELDS = [
    "system",
    "code",
    "seed",
    "outcome",
    "wall_observed_s",
    "xtb_log_records",
    "best_eV",
    "reason",
    "code_revision",
    "host",
    "date",
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


def run_anneal(
    binary: Path, system: str, seed: int, out: Path, wall_cap: float | None = None
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    target = f"{REFERENCE[system]:.8f}"
    env = dict(os.environ)
    env.pop("H2H_WALL_S", None)
    if wall_cap is not None:
        env["H2H_WALL_S"] = f"{wall_cap:.6f}"
    proc = subprocess.run(
        [str(binary), system, str(seed), str(BUDGET), target, str(out / "best.xyz")],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    (out / "stdout.txt").write_text(proc.stdout)
    (out / "stderr.txt").write_text(proc.stderr)
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


def write_start(binary: Path, system: str, seed: int, path: Path) -> None:
    subprocess.run(
        [str(binary), system, str(seed), str(BUDGET), "-", str(path)],
        env=dict(os.environ, H2H_START_ONLY="1"),
        capture_output=True,
        check=True,
    )


def fresh_energy(xtb: Path, start: Path, where: Path) -> float | None:
    """xtb's GFN2 energy of ``start`` in Hartree from a fresh SCC in an empty
    directory, or None when that SCC does not converge."""
    where.mkdir(parents=True, exist_ok=True)
    for stale in where.iterdir():
        if stale.is_file():
            stale.unlink()
    (where / "start.xyz").write_bytes(start.read_bytes())
    proc = subprocess.run(
        [str(xtb), "start.xyz", "--gfn", "2", "--sp", "--etemp", f"{ETEMP}"],
        cwd=where,
        capture_output=True,
        text=True,
        check=False,
    )
    found = re.findall(r"TOTAL ENERGY\s+([-+]?\d+\.\d+)\s+Eh", proc.stdout)
    return float(found[-1]) if proc.returncode == 0 and found else None


def row_key(row: dict) -> tuple[str, str, str, str]:
    return (row["system"], row["code"], row.get("code_revision", ""), str(row["seed"]))


def named_rows(path: Path | None) -> set[tuple[str, str, str, str]]:
    """The rows ``path`` names by system, code, code_revision and seed."""
    if path is None:
        return set()
    with open(path, newline="") as handle:
        return {row_key(row) for row in csv.DictReader(handle)}


def wall_cap(
    paths: list[Path],
    system: str,
    seeds: list[int],
    superseded: Path | None = None,
    censoring: Path | None = None,
) -> float:
    """Cap from completed CREST walls, with explicitly censored seeds absent."""
    left_out = named_rows(superseded)
    censored = set()
    if censoring is not None:
        with open(censoring, newline="") as handle:
            censored = {
                int(row["seed"])
                for row in csv.DictReader(handle)
                if row["system"] == system
                and row["code"] == "crest"
                and row["outcome"] == "censored"
            }
    walls = {}
    for path in paths:
        if path.is_file():
            with open(path, newline="") as handle:
                for row in csv.DictReader(handle):
                    if (
                        row["system"] == system
                        and row["code"] == "crest"
                        and row_key(row) not in left_out
                    ):
                        walls[int(row["seed"])] = float(row["wall_s"])
    missing = sorted(set(seeds) - set(walls) - censored)
    if missing:
        raise SystemExit(f"{system} anneal is capped by CREST, which has no row on {missing}")
    completed = [walls[seed] for seed in seeds if seed in walls]
    if not completed:
        raise SystemExit(f"{system} anneal has no completed CREST wall")
    return WALL_CAP[system] * max(completed)


def first_frame_energy(path: Path) -> float:
    lines = path.read_text().splitlines()
    if len(lines) < 2 or not lines[0].strip().isdigit() or int(lines[0]) <= 0:
        return float("inf")
    found = re.findall(r"[-+]?\d+\.\d+", lines[1])
    return float(found[0]) * AUTOEV if found else float("inf")


def output_exceeds(path: Path, limit: int) -> bool:
    """Whether a CREST output has crossed its corruption guard."""
    return path.is_file() and path.stat().st_size > limit


def run_crest(
    crest: Path, xtb: Path, python: str, system: str, seed: int, start: Path, out: Path
) -> tuple[dict | None, dict | None]:
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
        "-noreftopo",
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
    group = None
    if (preopt / "xtbopt.xyz").is_file():
        (out / "start.xyz").write_bytes((preopt / "xtbopt.xyz").read_bytes())
        with (
            open(out / "crest.out", "w") as stdout,
            open(out / "crest.err", "w") as stderr,
        ):
            crest_run = subprocess.Popen(
                command, cwd=out, stdout=stdout, stderr=stderr, start_new_session=True
            )
            group = crest_run.pid
            limit = int(os.environ.get("CREST_MAX_OUTPUT_BYTES", str(1 << 30)))
            garbage = out / "crest_rotamers_0.xyz"
            while crest_run.poll() is None:
                if output_exceeds(garbage, limit):
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    crest_run.wait()
                    wall = time.monotonic() - began
                    return None, {
                        "system": system,
                        "code": "crest",
                        "seed": seed,
                        "outcome": "censored",
                        "wall_observed_s": f"{wall:.6f}",
                        "xtb_log_records": len(log.read_text().splitlines()),
                        "best_eV": "",
                        "reason": (
                            f"crest_rotamers_0.xyz exceeded {limit} bytes while CREST "
                            "was still running"
                        ),
                    }
                try:
                    code = crest_run.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    continue
    wall = time.monotonic() - began
    if group is not None:
        # On Si20 seed 2005 a CREST process outlived the run and wrote 58 GB
        # of unformatted memory to crest_rotamers_0.xyz until the disk filled.
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    (out / "exit.code").write_text(f"{code}\n")
    inside = 0.0
    calls = 0
    for line in log.read_text().splitlines():
        seconds, _, count = line.split()
        inside += float(seconds)
        calls += int(count)
    best_file = out / "crest_best.xyz"
    best = first_frame_energy(best_file) if best_file.is_file() else float("inf")
    hit = energy_hits(best, REFERENCE[system], HIT_TOL)
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
    }, None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_header(path: Path, fields: list[str]) -> None:
    if path.is_file() and path.stat().st_size:
        with open(path, newline="") as handle:
            header = next(csv.reader(handle))
        if header != fields:
            raise SystemExit(
                f"{path} has another header; rows are appended, never rewritten"
            )


def append_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    check_header(path, fields)
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
    ap.add_argument("--superseded", type=Path)
    ap.add_argument("--rescore", nargs=2, metavar=("SYSTEM", "ENERGY_EV"))
    ap.add_argument("--rescored", type=Path)
    ap.add_argument("--check-starts", type=Path)
    ap.add_argument("--censoring", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.rows is None and args.check_starts is None:
        raise SystemExit("--rows is required")
    host = socket.gethostname()
    revision = args.head_file.read_text().strip() if args.head_file else ""
    if args.rescore is not None:
        if args.rescored is None:
            raise SystemExit("--rescore appends to --rescored")
        rescore(
            [*args.history, args.rows],
            args.rescored,
            args.rescore[0],
            float(args.rescore[1]),
            revision,
            host,
        )
        return
    if args.summary is not None and not any((args.anneal, args.crest, args.xtb)):
        summarise(
            [*args.history, args.rows],
            args.summary,
            revision,
            host,
            args.anneal_revision,
            args.systems,
            args.false_hits,
            args.superseded,
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
            write_start(args.anneal, system, seed, starts / f"{seed}.xyz")
        if args.check_starts is not None:
            rows = []
            for seed in seeds:
                energy = fresh_energy(
                    args.xtb, starts / f"{seed}.xyz", args.out / system / "fresh" / str(seed)
                )
                rows.append(
                    {
                        "system": system,
                        "seed": seed,
                        "converged": "false" if energy is None else "true",
                        "xtb_fresh_Eh": "" if energy is None else f"{energy:.12f}",
                        "code_revision": revision,
                        "host": host,
                        "date": datetime.date.today().isoformat(),
                        "xtb_sha256": pins["xtb_sha256"],
                    }
                )
            append_rows(args.check_starts, START_FIELDS, rows)
            continue
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
        if system in WALL_CAP:
            pairs.sort(key=lambda pair: pair[1] != "crest")
        cap = None
        fields = row_fields(system)
        check_header(args.rows, fields)
        for seed, code in pairs:
            where = args.out / system / code / str(seed)
            if code == "crest":
                row, censored = run_crest(
                    args.crest,
                    args.xtb,
                    python,
                    system,
                    seed,
                    starts / f"{seed}.xyz",
                    where,
                )
                if censored is not None:
                    if args.censoring is None:
                        raise SystemExit("a censored CREST run needs --censoring")
                    censored.update(
                        code_revision=revision,
                        host=host,
                        date=datetime.date.today().isoformat(),
                    )
                    append_rows(args.censoring, CENSOR_FIELDS, [censored])
                    print(",".join(str(censored[k]) for k in CENSOR_FIELDS), flush=True)
                    continue
                assert row is not None
            else:
                if system in WALL_CAP and cap is None:
                    cap = wall_cap(
                        [*args.history, args.rows],
                        system,
                        sorted(crest_seeds | set(seeds)),
                        args.superseded,
                        args.censoring,
                    )
                row = run_anneal(args.anneal, system, seed, where, cap)
            if system in WALL_CAP:
                row["wall_cap_s"] = f"{cap:.6f}" if code != "crest" else ""
            row.update(pins, date=datetime.date.today().isoformat())
            append_rows(args.rows, fields, [row])
            print(",".join(str(row[k]) for k in fields), flush=True)
    if args.summary is not None:
        summarise(
            [*args.history, args.rows],
            args.summary,
            revision,
            host,
            superseded=args.superseded,
        )


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
    superseded: Path | None = None,
) -> None:
    """Append one line per system and code over every rows file.

    With ``anneal_revision``, anneal's rows are only those of that
    ``code_revision``. A row named in ``false_hits`` by system, code,
    revision and seed counts as a miss at its budget. Rows of a seed outside
    ``SEEDS``, or ``CREST_SEEDS`` for CREST, and rows ``superseded`` names
    are left out. A CREST seed with two rows left is refused, as is an anneal
    one under ``anneal_revision``.
    """
    table = []
    for path in paths:
        if path.is_file():
            with open(path, newline="") as handle:
                table.extend(csv.DictReader(handle))
    left_out = named_rows(superseded)
    table = [
        row
        for row in table
        if int(row["seed"])
        in (CREST_SEEDS if row["code"] == "crest" else SEEDS)[row["system"]]
        and row_key(row) not in left_out
    ]
    if anneal_revision is not None:
        table = [
            row
            for row in table
            if row["code"] != "anneal-general"
            or row.get("code_revision") == anneal_revision
        ]
    seen = set()
    for row in table:
        run = (row["system"], row["code"], row["seed"])
        if run in seen and (row["code"] == "crest" or anneal_revision is not None):
            raise SystemExit(f"{run} has two rows; name the older in --superseded")
        seen.add(run)
    if false_hits is not None:
        named = named_rows(false_hits)
        table = [
            dict(row, hit="false", evals_to_hit=row["budget"])
            if row_key(row) in named
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


RESCORE_FIELDS = [
    "system",
    "code",
    "reference_eV",
    "runs",
    "hits",
    "seeds",
    "code_revision",
    "host",
    "date",
]


def rescore(
    paths: list[Path],
    rescored: Path,
    system: str,
    reference: float,
    revision: str,
    host: str,
) -> None:
    """Append CREST's hits on ``system`` against another reference.

    CREST runs to its own end, so its best energy is scored against any
    reference after the fact. Anneal stops at the reference it ran with,
    and its rows are not rescored.
    """
    runs = []
    for path in paths:
        if path.is_file():
            with open(path, newline="") as handle:
                runs.extend(
                    row
                    for row in csv.DictReader(handle)
                    if row["system"] == system and row["code"] == "crest"
                )
    if not runs:
        raise SystemExit(f"no CREST rows on {system}")
    line = {
        "system": system,
        "code": "crest",
        "reference_eV": f"{reference:.8f}",
        "runs": len(runs),
        "hits": sum(energy_hits(float(row["best"]), reference, HIT_TOL) for row in runs),
        "seeds": " ".join(str(seed) for seed in sorted(int(r["seed"]) for r in runs)),
        "code_revision": revision,
        "host": host,
        "date": datetime.date.today().isoformat(),
    }
    append_rows(rescored, RESCORE_FIELDS, [line])


def self_test() -> None:
    import tempfile

    if abs(type7([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) > 1e-12:
        raise SystemExit("type 7")
    if two_sided_mw([5.0, 5.0], [5.0, 5.0]) != 1.0:
        raise SystemExit("tied mann-whitney")
    with tempfile.TemporaryDirectory() as scratch:
        garbage = Path(scratch) / "garbage"
        garbage.write_bytes(b"12345")
        if output_exceeds(garbage, 5) or not output_exceeds(garbage, 4):
            raise SystemExit("CREST output guard")
        invalid_best = Path(scratch) / "invalid.xyz"
        invalid_best.write_text("0\n0.00000000\n")
        if first_frame_energy(invalid_best) != float("inf"):
            raise SystemExit("zero-atom CREST best")
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
        capped = Path(scratch) / "capped.csv"
        fields = row_fields("si20")
        if fields.index("wall_cap_s") != len(ROW_FIELDS) or "wall_cap_s" in row_fields("si13"):
            raise SystemExit("capped fields")
        for seed, wall in ((2000, "10.5"), (2001, "40.0"), (2002, "20.0")):
            append_rows(
                capped,
                fields,
                [dict(row, system="si20", code="crest", seed=str(seed), wall_s=wall)],
            )
        if wall_cap([capped], "si20", [2000, 2002]) != 60.0:
            raise SystemExit("wall cap")
        if wall_cap([capped], "si20", [2000, 2001]) != 120.0:
            raise SystemExit("wall cap over the slowest seed")
        try:
            wall_cap([capped], "si20", [2000, 2003])
        except SystemExit:
            pass
        else:
            raise SystemExit("wall cap without every CREST seed")
        censoring = Path(scratch) / "censoring.csv"
        append_rows(
            censoring,
            CENSOR_FIELDS,
            [
                {
                    key: (
                        "si20"
                        if key == "system"
                        else "crest"
                        if key == "code"
                        else "2003"
                        if key == "seed"
                        else "censored"
                        if key == "outcome"
                        else "1"
                    )
                    for key in CENSOR_FIELDS
                }
            ],
        )
        if wall_cap([capped], "si20", [2000, 2003], censoring=censoring) != 31.5:
            raise SystemExit("wall cap with a censored CREST seed")
        panel = Path(scratch) / "panel.csv"
        for seed in (2003, 2004):
            append_rows(
                panel,
                fields,
                [dict(row, system="si20", code="crest", seed=str(seed), hit="false")],
            )
        lines = Path(scratch) / "summary.csv"
        summarise([panel], lines, "", "", systems=["si20"])
        with open(lines, newline="") as handle:
            runs = {line["code"]: line["runs"] for line in csv.DictReader(handle)}
        if runs.get("crest") != "1":
            raise SystemExit(f"a seed outside the panel was summarised: {runs}")
        append_rows(
            panel,
            fields,
            [dict(row, system="si20", code="crest", seed="2003", code_revision="2")],
        )
        try:
            summarise([panel], lines, "", "", systems=["si20"])
        except SystemExit:
            pass
        else:
            raise SystemExit("two rows of one seed were summarised")
        superseded = Path(scratch) / "superseded.csv"
        superseded.write_text("system,code,code_revision,seed\nsi20,crest,1,2003\n")
        summarise([panel], lines, "", "", systems=["si20"], superseded=superseded)
        if wall_cap([panel], "si20", [2003], superseded) != 3.0 * float(row["wall_s"]):
            raise SystemExit("wall cap over a superseded row")
    print("ok")


if __name__ == "__main__":
    main()
