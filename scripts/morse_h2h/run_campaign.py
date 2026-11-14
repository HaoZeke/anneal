#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GMIN against the recommended preset on Cambridge Morse clusters.

Systems are M38, M55 and M80 at rho = 6 and rho = 14.
Short range is the hard case, so rho = 14 is listed first. Six seeds,
the same list on both arms. Within a system the (seed, arm) pairs are
shuffled. A hit is a quenched energy within 0.001 of the Cambridge
table. A miss is the full budget. Medians are Hyndman-Fan type 7.
pot_s is time inside the potential. outside_s is the rest of the wall.
The per-call times divide those by the charged evaluations.

The search binary is examples/h2h_morse. GMIN is Wales's public build
with its MORSE keyword. This process refuses to run unless the UTS
hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

# (name, n, rho, budget, cambridge energy)
SYSTEMS = (
    ("m38r14", 38, 14.0, 400_000, -144.321054),
    ("m55r14", 55, 14.0, 1_000_000, -220.646208),
    ("m80r14", 80, 14.0, 4_000_000, -340.811371),
    ("m38r6", 38, 6.0, 400_000, -157.477108),
    ("m55r6", 55, 6.0, 1_000_000, -250.286609),
    ("m80r6", 80, 6.0, 4_000_000, -378.333471),
)
PAPER_BUDGETS = {name: budget for name, _n, _rho, budget, _ref in SYSTEMS}
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("anneal", "gmin")
SCHEDULE_SEED = 20261113
HIT_TOL = 0.001

SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "planned",
    "hits",
    "hits_over_n",
    "budget",
    "evals_median",
    "evals_iqr",
    "wall_median_s",
    "wall_iqr_s",
    "pot_median_s",
    "outside_median_s",
    "pot_per_eval_s",
    "outside_per_eval_s",
    "wall_ratio",
    "seeds",
    "schedule_seed",
    "gmin_url",
    "gmin_revision",
    "anneal_head",
]


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
    q1 = type7(ordered, 0.25)
    q3 = type7(ordered, 0.75)
    return type7(ordered, 0.5), q3 - q1


def schedule(names, seeds, arms, rng_seed):
    rng = random.Random(rng_seed)
    order = []
    for name in names:
        jobs = [(name, seed, arm) for seed in seeds for arm in arms]
        rng.shuffle(jobs)
        order.extend(jobs)
    return order


def spec(name: str):
    for row in SYSTEMS:
        if row[0] == name:
            return row
    raise SystemExit(f"unknown system {name}")


def load_source(path: Path) -> tuple[str, str]:
    fields = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    url = fields.get("url", "")
    revision = fields.get("revision", "")
    if not url or not revision:
        raise SystemExit(f"{path} has no url and revision")
    return url, revision


def binary_budget(binary: Path, name: str) -> tuple[int, float, float]:
    proc = subprocess.run(
        [str(binary), "budget", name],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        raise SystemExit(f"budget for {name} returned {proc.stdout!r}")
    return int(lines[0]), float(lines[1]), float(lines[2])


def write_start(binary: Path, name: str, seed: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return
    subprocess.run(
        [str(binary), "coords", name, str(seed), str(path)],
        check=True,
    )


def parse_anneal(text: str) -> dict[str, str]:
    data = [line for line in text.splitlines() if ",anneal," in line]
    if len(data) != 1:
        raise ValueError(f"expected one anneal row, saw {len(data)}")
    header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s"
    )
    return next(csv.DictReader([header, data[0]]))


def parse_gmin(text: str) -> dict[str, str]:
    fields = {}
    for part in text.split()[1:]:
        key, value = part.split("=", 1)
        fields[key] = value
    if "hit" not in fields or "evals_to_hit" not in fields:
        raise ValueError(f"no gmin row in {text!r}")
    wall = float(fields["wall_s"])
    pot = float(fields.get("pot_s") or 0.0)
    fields["outside_s"] = f"{max(wall - pot, 0.0):.6f}"
    fields.setdefault("charged", fields["evals_to_hit"])
    return fields


def _env(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(budget)
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def run_anneal(binary, name, seed, budget, coords, log, saved) -> dict[str, str]:
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), "run", name, str(seed), str(coords)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=_env(budget),
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal {name} seed {seed} exited {proc.returncode}\n{text[-400:]}")
    row = parse_anneal(text)
    saved.write_text(text)
    return row


def run_gmin(gmin, data, name, rho, seed, budget, target, coords, work, log, saved):
    work.mkdir(parents=True, exist_ok=True)
    (work / "data").write_text(data.read_text() + f"MORSE {rho:.1f}\nRANSEED {seed}\n")
    (work / "coords").write_bytes(coords.read_bytes())
    env = _env(budget)
    env.update(
        {
            "H2H_SYSTEM": name,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{target:.10f}",
            "H2H_TOL": f"{HIT_TOL}",
            "H2H_ROW": str(work / "row.txt"),
        }
    )
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(gmin)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=work,
        )
    row_path = work / "row.txt"
    if not row_path.is_file():
        raise RuntimeError(f"gmin {name} seed {seed} exited {proc.returncode} with no row\n{log.read_text()[-400:]}")
    text = row_path.read_text()
    row = parse_gmin(text)
    saved.write_text(text)
    return row


def as_result(name, arm, row):
    _name, n, rho, budget, _ref = spec(name)
    hit = row["hit"] in ("1", "true")
    charged = float(row.get("charged") or row["evals_to_hit"])
    wall = float(row["wall_s"])
    pot = float(row.get("pot_s") or 0.0)
    outside = float(row.get("outside_s") or max(wall - pot, 0.0))
    return {
        "system": name,
        "n": n,
        "rho": rho,
        "arm": arm,
        "seed": int(row["seed"]),
        "budget": int(row["budget"]),
        "hit": hit,
        "evals": float(row["evals_to_hit"]),
        "charged": charged,
        "wall_s": wall,
        "pot_s": pot,
        "outside_s": outside,
        "best": float(row["best"]),
    }


def summarise(rows, seeds, url, revision, head):
    by_name = {}
    for row in rows:
        by_name.setdefault(row["system"], {}).setdefault(row["arm"], []).append(row)
    seed_text = " ".join(str(seed) for seed in seeds)
    out = []
    for name, _n, _rho, _budget, _ref in SYSTEMS:
        arms = by_name.get(name)
        if not arms:
            continue
        walls = {
            arm: median_iqr([item["wall_s"] for item in group])[0]
            for arm, group in arms.items()
            if group
        }
        ratio = ""
        if "anneal" in walls and "gmin" in walls and walls["anneal"] > 0.0:
            ratio = f"{walls['gmin'] / walls['anneal']:.6f}"
        for arm in ARMS:
            group = arms.get(arm, [])
            if not group:
                continue
            evals_median, evals_iqr = median_iqr([item["evals"] for item in group])
            wall_median, wall_iqr = median_iqr([item["wall_s"] for item in group])
            pot_median, _ = median_iqr([item["pot_s"] for item in group])
            outside_median, _ = median_iqr([item["outside_s"] for item in group])
            charged = [item["charged"] for item in group if item["charged"] > 0]
            pot_per = ""
            outside_per = ""
            if charged:
                pot_per = f"{sum(item['pot_s'] for item in group) / sum(item['charged'] for item in group):.9f}"
                outside_per = f"{sum(item['outside_s'] for item in group) / sum(item['charged'] for item in group):.9f}"
            hits = sum(bool(item["hit"]) for item in group)
            out.append(
                {
                    "system": name,
                    "arm": arm,
                    "n": str(len(group)),
                    "planned": str(len(seeds)),
                    "hits": str(hits),
                    "hits_over_n": f"{hits}/{len(group)}",
                    "budget": str(group[0]["budget"]),
                    "evals_median": f"{evals_median:.6f}",
                    "evals_iqr": f"{evals_iqr:.6f}",
                    "wall_median_s": f"{wall_median:.6f}",
                    "wall_iqr_s": f"{wall_iqr:.6f}",
                    "pot_median_s": f"{pot_median:.6f}",
                    "outside_median_s": f"{outside_median:.6f}",
                    "pot_per_eval_s": pot_per,
                    "outside_per_eval_s": outside_per,
                    "wall_ratio": ratio,
                    "seeds": seed_text,
                    "schedule_seed": str(SCHEDULE_SEED),
                    "gmin_url": url,
                    "gmin_revision": revision,
                    "anneal_head": head,
                }
            )
    return out


def write_summary(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def job_paths(runs: Path, name: str, seed: int, arm: str):
    directory = runs / name
    return directory / f"{arm}-{seed}.row", directory / f"{arm}-{seed}.log"


def load_saved(path: Path, name: str, arm: str):
    if not path.is_file() or not path.read_text().strip():
        return None
    text = path.read_text()
    row = parse_anneal(text) if arm == "anneal" else parse_gmin(text)
    return as_result(name, arm, row)


def note(progress: Path, line: str) -> None:
    print(line, flush=True)
    with progress.open("a") as handle:
        handle.write(line + "\n")


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7 median")
    if any(n < 6 for _name, n, _rho, _budget, _ref in SYSTEMS):
        raise SystemExit("n < 6")
    if PAPER_BUDGETS["m38r14"] != 400_000 or PAPER_BUDGETS["m55r14"] != 1_000_000:
        raise SystemExit("paper budgets")
    if PAPER_BUDGETS["m80r14"] != 4_000_000:
        raise SystemExit("m80 budget")
    names = tuple(row[0] for row in SYSTEMS)
    order = schedule(names, SEEDS, ARMS, SCHEDULE_SEED)
    seen = []
    for name, _seed, _arm in order:
        if not seen or seen[-1] != name:
            seen.append(name)
    if seen != list(names):
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
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", choices=("anneal", "gmin", "both"), default="both")
    args = parser.parse_args(argv[1:])
    names = tuple(args.system) if args.system else tuple(row[0] for row in SYSTEMS)
    seeds = tuple(args.seed) if args.seed else SEEDS
    arms = ARMS if args.arm == "both" else (args.arm,)
    for name in names:
        if name not in PAPER_BUDGETS:
            raise SystemExit(f"no paper budget for {name}")
    url, revision = load_source(args.source)
    head = args.head_file.read_text().strip()
    progress = args.runs.parent / "progress.txt"
    progress.parent.mkdir(parents=True, exist_ok=True)
    order = schedule(names, seeds, arms, SCHEDULE_SEED)
    note(progress, "schedule " + " ".join(f"{name}:{arm}:{seed}" for name, seed, arm in order))
    rows = []
    for name, seed, arm in order:
        _name, _n, rho, budget, target = spec(name)
        binary_b, binary_ref, binary_rho = binary_budget(args.binary, name)
        if binary_b != budget or abs(binary_ref - target) > 1e-9 or abs(binary_rho - rho) > 1e-9:
            raise SystemExit(f"binary spec {binary_b} {binary_ref} {binary_rho} != {budget} {target} {rho}")
        coords = args.starts / name / f"seed-{seed}.xyz"
        write_start(args.binary, name, seed, coords)
        saved, log = job_paths(args.runs, name, seed, arm)
        saved.parent.mkdir(parents=True, exist_ok=True)
        existing = load_saved(saved, name, arm)
        if existing is not None:
            rows.append(existing)
            note(progress, f"skip {name} arm={arm} seed={seed} hit={int(existing['hit'])}")
            continue
        note(progress, f"start {name} arm={arm} seed={seed} budget={budget}")
        if arm == "anneal":
            raw = run_anneal(args.binary, name, seed, budget, coords, log, saved)
        else:
            raw = run_gmin(
                args.gmin,
                args.data,
                name,
                rho,
                seed,
                budget,
                target,
                coords,
                args.runs / name / f"gmin-{seed}",
                log,
                saved,
            )
        result = as_result(name, arm, raw)
        rows.append(result)
        note(
            progress,
            f"done {name} arm={arm} seed={seed} hit={int(result['hit'])} "
            f"evals={result['evals']:.0f} best={result['best']:.6f} "
            f"pot_s={result['pot_s']:.3f} outside_s={result['outside_s']:.3f}",
        )
        if args.summary:
            write_summary(args.summary, summarise(rows, seeds, url, revision, head))
    text_rows = summarise(rows, seeds, url, revision, head)
    if args.summary:
        write_summary(args.summary, text_rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(text_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
