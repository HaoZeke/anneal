#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial covalent panel: three anneal arms against GMIN and ASE.

Systems are Stillinger-Weber Si10, Si13 and Si20 in sigma and epsilon,
and Tersoff C10 and C20 in Angstrom and eV. The anneal arms are
``Config::recommended``, the same with ``H2H_PRIORS=0``, and the general
arm on dimer scales (``examples/h2h_covalent``). With ``--before-binary``
the two prior-free arms also run on that build, coded ``@v2``. On silicon
every anneal arm runs again with ``H2H_GTOL=5e-3``, coded ``-sloppy``, to
quench as loosely as GMIN's data. GMIN runs twice, on its data and as
``gmin-tight`` with every quench to 1e-6: the ``SW`` keyword on silicon,
and on carbon a second build whose ``USERPOT`` is rgpot's Tersoff
(``build_gmin_tersoff.sh``) with the silicon data on the carbon dimer
scales. ASE basin hopping runs every system. Twelve seeds, the same list
on every code and the same start for each seed.
Within a system the (seed, code) pairs are shuffled. One budget per
system for every code. A hit is a recorded minimum within 0.001 of the
reference. A miss is the full budget. Medians are Hyndman-Fan type 7.
pot_s is time inside the potential. outside_s is the rest of the wall.
This process refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

SYSTEMS = ("sw10", "sw13", "sw20", "tc10", "tc20")
SEEDS = tuple(range(2000, 2012))
ASE_SEEDS = SEEDS
# The budgets of examples/h2h_covalent, which the panel refuses to differ
# from.
PAPER_BUDGETS = {"sw10": 100000, "sw13": 100000, "sw20": 400000, "tc10": 100000, "tc20": 400000}
RECOMMENDED = {"H2H_ARM": "recommended", "H2H_PRIORS": "1"}
NO_PRIORS = {"H2H_ARM": "recommended", "H2H_PRIORS": "0"}
GENERAL = {"H2H_ARM": "general", "H2H_PRIORS": "0"}
SLOPPY = {"H2H_GTOL": "5e-3"}
# code -> (binary, environment, code the binary prints). The before binary
# is the head that recorded the second panel; Config::recommended is the
# same in both, so only the two prior-free arms run on it.
ANNEAL = {
    "anneal": ("binary", RECOMMENDED, "anneal"),
    "anneal-nopriors": ("binary", NO_PRIORS, "anneal-nopriors"),
    "anneal-general": ("binary", GENERAL, "anneal-general"),
    "anneal-nopriors@v2": ("before_binary", NO_PRIORS, "anneal-nopriors"),
    "anneal-general@v2": ("before_binary", GENERAL, "anneal-general"),
    "anneal-sloppy": ("binary", RECOMMENDED | SLOPPY, "anneal-sloppy"),
    "anneal-nopriors-sloppy": ("binary", NO_PRIORS | SLOPPY, "anneal-nopriors-sloppy"),
    "anneal-general-sloppy": ("binary", GENERAL | SLOPPY, "anneal-general-sloppy"),
}
# GMIN's own data quenches to SLOPPYCONV 5e-3 RMS, 2.0e-2 eV/A on carbon;
# gmin-tight converges every quench to 1e-6 RMS, against anneal's 1e-6
# largest component.
GMIN = {"gmin": None, "gmin-tight": "1.0D-6"}
SCHEDULE_SEED = 20261113
HIT_TOL = 0.001

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
    "n",
    "hits",
    "budget",
    "reference",
    "best_min",
    "best_median",
    "evals_median",
    "evals_iqr",
    "charged_median",
    "wall_median_s",
    "pot_median_s",
    "outside_median_s",
    "pot_per_eval_s",
    "outside_per_eval_s",
    "seeds",
    "schedule_seed",
    "gmin_revision",
    "anneal_head",
]


def codes_for(name: str) -> tuple[str, ...]:
    base = tuple(code for code in ANNEAL if not code.endswith("-sloppy"))
    if name.startswith("sw"):
        sloppy = tuple(code for code in ANNEAL if code.endswith("-sloppy"))
        return base + sloppy + tuple(GMIN) + ("ase",)
    return base + tuple(GMIN) + ("ase",)


def seeds_for(code: str) -> tuple[int, ...]:
    return ASE_SEEDS if code == "ase" else SEEDS


def type7(sorted_values: list[float], p: float) -> float:
    n = len(sorted_values)
    if n < 1:
        raise ValueError("type 7 needs a sample")
    if n == 1:
        return sorted_values[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return type7(ordered, 0.5), type7(ordered, 0.75) - type7(ordered, 0.25)


def schedule(names, seeds, codes_of, rng_seed):
    rng = random.Random(rng_seed)
    order = []
    for name in names:
        jobs = [
            (name, seed, code)
            for code in codes_of(name)
            for seed in seeds
            if seed in seeds_for(code)
        ]
        rng.shuffle(jobs)
        order.extend(jobs)
    return order


def binary_lines(binary: Path, command: str, name: str) -> list[str]:
    proc = subprocess.run(
        [str(binary), command, name], check=True, capture_output=True, text=True
    )
    return proc.stdout.split()


def load_source(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("revision="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{path} has no revision")


def thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def csv_row(text: str, code: str) -> dict[str, str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("system,code,") and index + 1 < len(lines):
            row = next(csv.DictReader([line, lines[index + 1]]))
            if row["code"] == code:
                return row
    raise ValueError(f"no {code} row in {text[-400:]!r}")


def run_logged(command, env, log: Path, cwd=None) -> str:
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", *command],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=cwd,
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"{command[0]} exited {proc.returncode}\n{text[-400:]}")
    return text


def run_anneal(args, name, seed, code, budget, log) -> dict[str, str]:
    binary, settings, printed = ANNEAL[code]
    env = thread_env()
    env.update(settings)
    env["H2H_BUDGET"] = str(budget)
    text = run_logged([str(getattr(args, binary)), "run", name, str(seed)], env, log)
    row = csv_row(text, printed)
    row["code"] = code
    return row


def run_ase(args, name, seed, budget, log) -> dict[str, str]:
    env = thread_env()
    text = run_logged(
        [
            sys.executable,
            str(args.ase),
            "--binary",
            str(args.binary),
            "--library",
            str(args.library),
            "--system",
            name,
            "--seed",
            str(seed),
            "--budget",
            str(budget),
        ],
        env,
        log,
    )
    return csv_row(text, "ase")


def gmin_data(text: str, conv: str | None) -> str:
    if conv is None:
        return text
    lines = []
    for line in text.splitlines():
        if line.startswith(("SLOPPYCONV", "TIGHTCONV")):
            line = f"{line.split()[0]} {conv}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def run_gmin(args, name, seed, code, budget, target, radius, start, log) -> dict[str, str]:
    work = args.runs / name / f"{code}-{seed}"
    work.mkdir(parents=True, exist_ok=True)
    carbon = name.startswith("tc")
    (work / "data").write_text(
        gmin_data((args.carbon_data if carbon else args.data).read_text(), GMIN[code])
        + f"RADIUS {radius:.10f}\nRANSEED {seed}\n"
    )
    (work / "coords").write_bytes(start.read_bytes())
    row_path = work / "row.txt"
    row_path.unlink(missing_ok=True)
    env = thread_env()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "H2H_SYSTEM": name,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{target:.10f}",
            "H2H_TOL": f"{HIT_TOL}",
            "H2H_ROW": str(row_path),
        }
    )
    with log.open("w") as handle:
        subprocess.run(
            ["taskset", "-c", "0", str(args.gmin_tersoff if carbon else args.gmin)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=work,
        )
    if not row_path.is_file():
        raise RuntimeError(f"gmin {name} seed {seed} wrote no row\n{log.read_text()[-400:]}")
    fields = dict(part.split("=", 1) for part in row_path.read_text().split()[1:])
    wall = float(fields["wall_s"])
    pot = float(fields.get("pot_s") or 0.0)
    return {
        "system": name,
        "code": code,
        "seed": fields["seed"],
        "budget": fields["budget"],
        "hit": "true" if fields["hit"] in ("1", "true") else "false",
        "evals_to_hit": fields["evals_to_hit"],
        "charged": fields.get("charged", fields["evals_to_hit"]),
        "wall_s": f"{wall:.6f}",
        "best": fields["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{max(wall - pot, 0.0):.6f}",
    }


def normalise(row: dict[str, str]) -> dict[str, str]:
    out = {key: row[key] for key in ROW_FIELDS if key in row}
    out["hit"] = "true" if row["hit"] in ("1", "true", "True") else "false"
    if out["hit"] == "true":
        # The anneal ledger stops at the target. A hit recorded off the
        # improvement list printed the budget, and the charged count is
        # the call that reached it.
        out["evals_to_hit"] = str(min(int(float(out["evals_to_hit"])), int(float(out["charged"]))))
    if "outside_s" not in out:
        out["outside_s"] = f"{max(float(row['wall_s']) - float(row['pot_s']), 0.0):.6f}"
    return out


def summarise(rows, references, revision, heads):
    out = []
    for name in SYSTEMS:
        for code in codes_for(name):
            group = [row for row in rows if row["system"] == name and row["code"] == code]
            if not group:
                continue
            seed_text = " ".join(sorted({row["seed"] for row in group}, key=int))
            evals = median_iqr([float(row["evals_to_hit"]) for row in group])
            charged = [float(row["charged"]) for row in group]
            pot = [float(row["pot_s"]) for row in group]
            outside = [float(row["outside_s"]) for row in group]
            bests = [float(row["best"]) for row in group]
            total = sum(charged) or 1.0
            out.append(
                {
                    "system": name,
                    "code": code,
                    "n": str(len(group)),
                    "hits": str(sum(row["hit"] == "true" for row in group)),
                    "budget": group[0]["budget"],
                    "reference": f"{references[name]:.8f}",
                    "best_min": f"{min(bests):.8f}",
                    "best_median": f"{median_iqr(bests)[0]:.8f}",
                    "evals_median": f"{evals[0]:.1f}",
                    "evals_iqr": f"{evals[1]:.1f}",
                    "charged_median": f"{median_iqr(charged)[0]:.1f}",
                    "wall_median_s": f"{median_iqr([float(r['wall_s']) for r in group])[0]:.6f}",
                    "pot_median_s": f"{median_iqr(pot)[0]:.6f}",
                    "outside_median_s": f"{median_iqr(outside)[0]:.6f}",
                    "pot_per_eval_s": f"{sum(pot) / total:.9f}",
                    "outside_per_eval_s": f"{sum(outside) / total:.9f}",
                    "seeds": seed_text,
                    "schedule_seed": str(SCHEDULE_SEED),
                    "gmin_revision": revision,
                    "anneal_head": heads[ANNEAL.get(code, ("binary",))[0]],
                }
            )
    return out


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def note(progress: Path, line: str) -> None:
    print(line, flush=True)
    with progress.open("a") as handle:
        handle.write(line + "\n")


def self_test() -> None:
    if abs(type7([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7 median")
    order = schedule(SYSTEMS, SEEDS, codes_for, SCHEDULE_SEED)
    if len(order) != 12 * (3 * 11 + 2 * 8):
        raise SystemExit(f"schedule length {len(order)}")
    data = "SLOPPYCONV 5.0D-3\nTIGHTCONV 1.0D-3\nSW\n"
    if gmin_data(data, "1.0D-6") != "SLOPPYCONV 1.0D-6\nTIGHTCONV 1.0D-6\nSW\n":
        raise SystemExit("gmin-tight data")
    seen = []
    for name, _seed, _code in order:
        if not seen or seen[-1] != name:
            seen.append(name)
    if seen != list(SYSTEMS):
        raise SystemExit(f"system order {seen}")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--before-binary", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gmin-tersoff", type=Path, required=True)
    parser.add_argument("--carbon-data", type=Path, required=True)
    parser.add_argument("--ase", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--before-head-file", type=Path)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--system", action="append")
    parser.add_argument("--code", action="append")
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args(argv[1:])
    names = tuple(args.system) if args.system else SYSTEMS
    wanted = set(args.code) if args.code else None
    revision = load_source(args.source)
    head = args.head_file.read_text().strip()
    heads = {"binary": head, "before_binary": ""}
    if args.before_head_file:
        heads["before_binary"] = args.before_head_file.read_text().strip()
    progress = args.runs.parent / "progress.txt"
    progress.parent.mkdir(parents=True, exist_ok=True)

    def codes_of(name):
        return tuple(
            code
            for code in codes_for(name)
            if (wanted is None or code in wanted)
            and (args.before_binary or ANNEAL.get(code, ("",))[0] != "before_binary")
        )

    if args.summarise_only:
        rows = [normalise(row) for row in csv.DictReader(args.rows.read_text().splitlines())]
        references = {
            name: float(binary_lines(args.binary, "budget", name)[1]) for name in SYSTEMS
        }
        write_csv(args.rows, ROW_FIELDS, rows)
        write_csv(args.summary, SUMMARY_FIELDS, summarise(rows, references, revision, heads))
        return 0
    for name in names:
        budget = int(binary_lines(args.binary, "budget", name)[0])
        if budget != PAPER_BUDGETS[name]:
            raise SystemExit(f"{name} budget {budget} is not {PAPER_BUDGETS[name]}")
    order = schedule(names, SEEDS, codes_of, SCHEDULE_SEED)
    note(progress, f"head {head} schedule " + " ".join(f"{n}:{c}:{s}" for n, s, c in order))
    references = {}
    rows = []
    for name, seed, code in order:
        budget_text, reference_text, _count = binary_lines(args.binary, "budget", name)
        budget = int(budget_text)
        references[name] = float(reference_text)
        radius = float(binary_lines(args.binary, "scales", name)[2])
        start = args.starts / name / f"seed-{seed}.xyz"
        start.parent.mkdir(parents=True, exist_ok=True)
        if not start.is_file():
            subprocess.run([str(args.binary), "coords", name, str(seed), str(start)], check=True)
        saved = args.runs / name / f"{code}-{seed}.row"
        log = args.runs / name / f"{code}-{seed}.log"
        saved.parent.mkdir(parents=True, exist_ok=True)
        if saved.is_file() and saved.read_text().strip():
            row = next(csv.DictReader(saved.read_text().splitlines()))
            rows.append(row)
            note(progress, f"skip {name} code={code} seed={seed} hit={row['hit']}")
            continue
        note(progress, f"start {name} code={code} seed={seed} budget={budget}")
        if code in ANNEAL:
            raw = run_anneal(args, name, seed, code, budget, log)
        elif code == "ase":
            raw = run_ase(args, name, seed, budget, log)
        else:
            raw = run_gmin(args, name, seed, code, budget, references[name], radius, start, log)
        row = normalise(raw)
        write_csv(saved, ROW_FIELDS, [row])
        rows.append(row)
        note(
            progress,
            f"done {name} code={code} seed={seed} hit={row['hit']} evals={row['evals_to_hit']} "
            f"best={float(row['best']):.6f} pot_s={float(row['pot_s']):.3f} "
            f"outside_s={float(row['outside_s']):.3f}",
        )
        write_csv(args.rows, ROW_FIELDS, rows)
        write_csv(args.summary, SUMMARY_FIELDS, summarise(rows, references, revision, heads))
    write_csv(args.rows, ROW_FIELDS, rows)
    summary = summarise(rows, references, revision, heads)
    write_csv(args.summary, SUMMARY_FIELDS, summary)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
