#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Twelve-seed Lennard-Jones panel on the recommended preset.

Seeds are 2000 through 2011, one process, hostname bench. A row is
appended and never replaced. GMIN call counts are read from an existing
seed file when one is passed, and otherwise left for a later append.
Fisher is the two-sided exact test on the hit counts. Mann-Whitney is
the tie-corrected normal approximation on calls to the minimum, misses
stored as the budget.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

SIZES = (38, 55, 75, 76, 77, 98, 102, 103, 104, 150)
SEEDS = tuple(range(2000, 2012))
BUDGET = {
    38: 400_000,
    55: 1_000_000,
    75: 4_000_000,
    76: 4_000_000,
    77: 4_000_000,
    98: 4_000_000,
    102: 4_000_000,
    103: 4_000_000,
    104: 4_000_000,
    150: 4_000_000,
}
GMIN_REVISION = "aac1548a8aec13b53ad12a8d6904625081685d2e"
PANEL = "bench/lj-n12-500aaf65/summary.csv"

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
    "code_revision",
    "host",
]
SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "hits",
    "hits_over_n",
    "budget",
    "evals_median",
    "evals_iqr",
    "wall_median_s",
    "inside_us",
    "outside_us",
    "fisher_p",
    "mannwhitney_p",
    "code_revision",
    "host",
    "seeds",
    "previous_hits",
]


def type7(ordered: list[float], p: float) -> float:
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0], 0.0
    return type7(ordered, 0.5), type7(ordered, 0.75) - type7(ordered, 0.25)


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def fisher_exact_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b

    def probability(aa: int, bb: int, cc: int, dd: int) -> float:
        total = aa + bb + cc + dd
        return comb(aa + bb, aa) * comb(cc + dd, cc) / comb(total, aa + cc)

    observed = probability(a, b, c, d)
    total = 0.0
    for aa in range(a + b + 1):
        cc = a + c - aa
        bb = a + b - aa
        dd = c + d - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        mass = probability(aa, bb, cc, dd)
        if mass <= observed + 1e-15:
            total += mass
    return min(1.0, total)


def mann_whitney_u_two_sided(x: list[float], y: list[float]) -> float:
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = sorted(x + y)
    ranks: dict[float, float] = {}
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        rank = (i + 1 + j + 1) / 2.0
        ranks[pooled[i]] = rank
        i = j + 1
    r1 = sum(ranks[v] for v in x)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie = 0.0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        t = j - i + 1
        tie += t**3 - t
        i = j + 1
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if sigma2 <= 0.0:
        return 1.0
    z = (abs(u - mu) - 0.5) / math.sqrt(sigma2)
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2.0))))


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({key: row[key] for key in ROW_FIELDS})


def seen(rows: list[dict[str, str]], system: str, code: str, seed: str, revision: str) -> bool:
    return any(
        row["system"] == system
        and row["code"] == code
        and row["seed"] == seed
        and row["code_revision"] == revision
        for row in rows
    )


def previous_hits(root: Path) -> dict[int, int]:
    path = root / PANEL
    found: dict[int, int] = {}
    for row in csv.DictReader(path.open()):
        if row["arm"] != "anneal":
            continue
        n = int(row["system"].removeprefix("lj"))
        found[n] = int(row["hits"])
    return found


def parse_anneal(text: str, revision: str, host: str) -> dict[str, str]:
    data = [line for line in text.splitlines() if ",anneal," in line]
    if len(data) != 1:
        raise ValueError(f"expected one anneal row, saw {len(data)}")
    header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s"
    )
    row = next(csv.DictReader([header, data[0]]))
    hit = "1" if row["hit"] in ("1", "true") else "0"
    wall = float(row["wall_s"])
    pot = float(row["pot_s"]) if row.get("pot_s") else 0.0
    return {
        "system": row["system"],
        "code": "anneal",
        "seed": row["seed"],
        "budget": row["budget"],
        "hit": hit,
        "evals_to_hit": row["evals_to_hit"],
        "charged": row["charged"],
        "wall_s": f"{wall:.6f}",
        "best": row["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{max(wall - pot, 0.0):.6f}",
        "code_revision": revision,
        "host": host,
    }


REFERENCE = {
    38: -173.928427,
    55: -279.248470,
    75: -397.492331,
    76: -402.894866,
    77: -409.083517,
    98: -543.665361,
    102: -569.363652,
    103: -575.766131,
    104: -582.086642,
    150: -893.310258,
}


def parse_gmin(text: str, n: int, seed: int, revision: str, host: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "hit=" not in line:
            continue
        for part in line.split()[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value
    if "hit" not in fields or "evals_to_hit" not in fields:
        raise ValueError(f"no gmin row in {text[-400:]!r}")
    hit = "1" if fields["hit"] in ("1", "true") else "0"
    wall = float(fields.get("wall_s", "nan"))
    pot = float(fields["pot_s"]) if "pot_s" in fields else float("nan")
    outside = max(wall - pot, 0.0) if math.isfinite(pot) else float("nan")
    return {
        "system": f"lj{n}",
        "code": "gmin",
        "seed": str(seed),
        "budget": fields.get("budget", str(BUDGET[n])),
        "hit": hit,
        "evals_to_hit": fields["evals_to_hit"],
        "charged": fields.get("charged", fields["evals_to_hit"]),
        "wall_s": f"{wall:.6f}",
        "best": fields.get("best", ""),
        "pot_s": f"{pot:.6f}" if math.isfinite(pot) else "",
        "outside_s": f"{outside:.6f}" if math.isfinite(outside) else "",
        "code_revision": revision,
        "host": host,
    }


def run_gmin(
    gmin: Path,
    data: Path,
    binary: Path,
    n: int,
    seed: int,
    work: Path,
    log: Path,
) -> str:
    work.mkdir(parents=True, exist_ok=True)
    coords = work / "coords"
    subprocess.run(
        [str(binary), "coords", f"lj{n}", str(seed), str(coords)],
        check=True,
    )
    (work / "data").write_text(data.read_text() + f"RANSEED {seed}\n")
    env = os.environ.copy()
    env.update(
        {
            "H2H_BUDGET": str(BUDGET[n]),
            "H2H_SYSTEM": f"lj{n}",
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{REFERENCE[n]:.10f}",
            "H2H_TOL": "0.001",
            "H2H_ROW": str(work / "row.txt"),
            "OMP_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
        }
    )
    log.parent.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError(
            f"gmin LJ{n} seed {seed} exited {proc.returncode} with no row\n{log.read_text()[-400:]}"
        )
    return row_path.read_text()


def run_anneal(binary: Path, n: int, seed: int, log: Path) -> str:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(BUDGET[n])
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), "run", f"lj{n}", str(seed)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal LJ{n} seed {seed} exited {proc.returncode}\n{text[-400:]}")
    return text


def summarise(
    rows: list[dict[str, str]],
    revision: str,
    host: str,
    prior: dict[int, int],
) -> list[dict[str, str]]:
    anneal = [row for row in rows if row["code"] == "anneal" and row["code_revision"] == revision]
    gmin = [row for row in rows if row["code"] == "gmin"]
    by_a: dict[int, list[dict[str, str]]] = {}
    by_g: dict[int, list[dict[str, str]]] = {}
    for row in anneal:
        by_a.setdefault(int(row["system"].removeprefix("lj")), []).append(row)
    for row in gmin:
        by_g.setdefault(int(row["system"].removeprefix("lj")), []).append(row)
    seed_text = " ".join(str(seed) for seed in SEEDS)
    out = []
    for n in SIZES:
        group = by_a.get(n, [])
        if len(group) != len(SEEDS):
            continue
        evals = [float(row["evals_to_hit"]) for row in group]
        walls = [float(row["wall_s"]) for row in group]
        insides = []
        outsides = []
        for row in group:
            charged = max(int(row["charged"]), 1)
            insides.append(float(row["pot_s"]) / charged * 1e6)
            outsides.append(float(row["outside_s"]) / charged * 1e6)
        evals_median, evals_iqr = median_iqr(evals)
        wall_median, _wall_iqr = median_iqr(walls)
        hits = sum(row["hit"] == "1" for row in group)
        other = by_g.get(n, [])
        if len(other) == len(SEEDS):
            fisher = fisher_exact_two_sided(hits, len(group), sum(r["hit"] == "1" for r in other), len(other))
            mann = mann_whitney_u_two_sided(evals, [float(r["evals_to_hit"]) for r in other])
            mann_text = f"{mann:.6g}"
            fisher_text = f"{fisher:.6g}"
        else:
            fisher_text = ""
            mann_text = ""
        out.append(
            {
                "system": f"lj{n}",
                "arm": "anneal",
                "n": str(len(group)),
                "hits": str(hits),
                "hits_over_n": f"{hits}/{len(group)}",
                "budget": str(BUDGET[n]),
                "evals_median": f"{evals_median:.6f}",
                "evals_iqr": f"{evals_iqr:.6f}",
                "wall_median_s": f"{wall_median:.6f}",
                "inside_us": f"{median_iqr(insides)[0]:.4f}",
                "outside_us": f"{median_iqr(outsides)[0]:.4f}",
                "fisher_p": fisher_text,
                "mannwhitney_p": mann_text,
                "code_revision": revision,
                "host": host,
                "seeds": seed_text,
                "previous_hits": str(prior.get(n, "")),
            }
        )
    return out


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--gmin", type=Path)
    parser.add_argument("--gmin-data", type=Path)
    parser.add_argument("--size", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args(argv[1:])
    sizes = tuple(args.size) if args.size else SIZES
    seeds = tuple(args.seed) if args.seed else SEEDS
    revision = args.head_file.read_text().strip()
    host = socket.gethostname()
    prior = previous_hits(args.root)
    stored = load_rows(args.rows)
    for n in sizes:
        if n not in BUDGET:
            raise SystemExit(f"no paper budget for LJ{n}")
        for seed in seeds:
            key = (f"lj{n}", "anneal", str(seed), revision)
            if seen(stored, *key):
                print(f"skip lj{n} seed {seed}", flush=True)
                continue
            print(f"start lj{n} seed {seed} budget={BUDGET[n]}", flush=True)
            log = args.logs / f"lj{n}" / f"anneal-{seed}.log"
            text = run_anneal(args.binary, n, seed, log)
            row = parse_anneal(text, revision, host)
            append_row(args.rows, row)
            stored.append(row)
            print(
                f"done lj{n} seed {seed} hit={row['hit']} evals={row['evals_to_hit']} best={row['best']}",
                flush=True,
            )
            write_summary(args.summary, summarise(stored, revision, host, prior))
    if args.gmin is not None:
        if args.gmin_data is None:
            raise SystemExit("gmin data file is required with --gmin")
        for n in sizes:
            for seed in seeds:
                key = (f"lj{n}", "gmin", str(seed), GMIN_REVISION)
                if seen(stored, *key):
                    print(f"skip gmin lj{n} seed {seed}", flush=True)
                    continue
                print(f"start gmin lj{n} seed {seed} budget={BUDGET[n]}", flush=True)
                work = args.logs / f"lj{n}" / f"gmin-{seed}"
                log = args.logs / f"lj{n}" / f"gmin-{seed}.log"
                text = run_gmin(args.gmin, args.gmin_data, args.binary, n, seed, work, log)
                row = parse_gmin(text, n, seed, GMIN_REVISION, host)
                append_row(args.rows, row)
                stored.append(row)
                print(
                    f"done gmin lj{n} seed {seed} hit={row['hit']} evals={row['evals_to_hit']}",
                    flush=True,
                )
                write_summary(args.summary, summarise(stored, revision, host, prior))
    write_summary(args.summary, summarise(stored, revision, host, prior))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
