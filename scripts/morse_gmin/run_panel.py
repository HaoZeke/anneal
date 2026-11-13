#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Morse head-to-head of the recommended preset against GMIN.

Systems are M30, M38, M55 and M80 at rho 6 and 14, plus M80 at rho 3.
Twelve seeds, 2000-2011, one core, hostname bench. Both arms use the
paper budget. A hit is within 0.001 of the Cambridge table. A miss is
entered at the budget. Medians are Hyndman-Fan type 7. Fisher is the
two-sided exact test on hits. Mann-Whitney is the two-sided normal
approximation on the call counts, with the tie correction and the
continuity correction.

GMIN rows already stored for this source revision are copied. Anneal
is run on this tree.
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

# Cambridge Cluster Database, Doye and Wales. The bold entry at that rho.
# 30B and 30D' are the global minima at rho 6 and 14.
SYSTEMS = (
    ("m30r6", 30, 6.0, 400_000, -118.432844),
    ("m30r14", 30, 14.0, 400_000, -106.835790),
    ("m38r6", 38, 6.0, 400_000, -157.477108),
    ("m38r14", 38, 14.0, 400_000, -144.321054),
    ("m55r6", 55, 6.0, 1_000_000, -250.286609),
    ("m55r14", 55, 14.0, 1_000_000, -220.646208),
    ("m80r3", 80, 3.0, 4_000_000, -690.577890),
    ("m80r6", 80, 6.0, 4_000_000, -378.333471),
    ("m80r14", 80, 14.0, 4_000_000, -340.811371),
)
SEEDS = tuple(range(2000, 2012))
GMIN_REV = "aac1548a8aec13b53ad12a8d6904625081685d2e"
ARCHIVES = (
    "bench/morse-h2h/seeds.csv",
    "bench/morse14-h2h/seeds.csv",
    "bench/morse-panel/seeds.csv",
)
FIELDS = [
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
    "anneal_hits",
    "anneal_n",
    "gmin_hits",
    "gmin_n",
    "gmin_reused",
    "fisher_p",
    "mannwhitney_p_calls",
    "anneal_evals_median",
    "gmin_evals_median",
    "anneal_wall_median_s",
    "gmin_wall_median_s",
    "anneal_pot_median_s",
    "anneal_outside_median_s",
    "gmin_pot_median_s",
    "gmin_outside_median_s",
]


def type7(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * 0.5
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b
    total_n = a + b + c + d

    def prob(aa: int, bb: int, cc: int, dd: int) -> float:
        return math.comb(aa + bb, aa) * math.comb(cc + dd, cc) / math.comb(total_n, aa + cc)

    base = prob(a, b, c, d)
    total = 0.0
    for aa in range(a + b + 1):
        cc = a + c - aa
        bb = a + b - aa
        dd = c + d - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        p = prob(aa, bb, cc, dd)
        if p <= base * (1.0 + 1e-9):
            total += p
    return min(1.0, total)


def mann_whitney(left: list[float], right: list[float]) -> float:
    """Two-sided p, tie-corrected, with the continuity correction."""
    n1, n2 = len(left), len(right)
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = sorted(left + right)
    ranks: dict[float, float] = {}
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        ranks[pooled[i]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r1 = sum(ranks[value] for value in left)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie = 0
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
    if z < 0.0:
        z = 0.0
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2.0))))


def spec(name: str):
    for row in SYSTEMS:
        if row[0] == name:
            return row
    raise SystemExit(f"unknown system {name}")


def run_env(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "H2H_GENERAL",
        "H2H_BASIN_STRIDE",
        "H2H_BEST",
        "H2H_REFERENCE",
        "H2H_PRIORS",
        "H2H_REPLICAS",
        "H2H_TARGET",
        "H2H_TOL",
        "H2H_SEED",
        "H2H_SYSTEM",
        "H2H_ROW",
        "H2H_BUDGET_RESTART",
    ):
        env.pop(key, None)
    env["H2H_BUDGET"] = str(budget)
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["LJOS_SEAT"] = "anneal-paper"
    return env


def hit_bit(value: str) -> str:
    return "1" if value in ("1", "true", "True") else "0"


def anneal_line(text: str, name: str, revision: str) -> dict[str, str]:
    rows = [line for line in text.splitlines() if line.startswith(name + ",anneal,")]
    if len(rows) != 1:
        raise RuntimeError(f"expected one anneal row for {name}, saw {len(rows)}")
    cols = rows[0].split(",")
    budget = int(cols[3])
    hit = cols[4] == "true"
    evals = int(cols[5])
    charged = int(cols[6])
    if not hit:
        evals = budget
    elif evals >= budget:
        evals = min(charged, budget)
    return {
        "system": name,
        "code": "anneal",
        "seed": cols[2],
        "budget": cols[3],
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": cols[6],
        "wall_s": cols[7],
        "best": cols[8],
        "pot_s": cols[9],
        "outside_s": cols[10],
        "code_revision": revision,
        "host": "bench",
    }


def gmin_line(text: str, name: str, seed: int, budget: int, target: float) -> dict[str, str]:
    fields = {}
    for part in text.split()[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    if "best" not in fields:
        raise RuntimeError(f"no gmin row in {text!r}")
    best = float(fields["best"])
    hit = best <= target + 0.001
    charged = int(float(fields.get("charged") or fields.get("evals_to_hit") or budget))
    evals = int(float(fields.get("evals_to_hit") or charged))
    if not hit:
        evals = budget
    elif evals >= budget:
        evals = min(charged, budget)
    wall = float(fields["wall_s"])
    pot = float(fields.get("pot_s") or 0.0)
    outside = float(fields.get("outside_s") or max(wall - pot, 0.0))
    return {
        "system": name,
        "code": "gmin",
        "seed": str(seed),
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(charged),
        "wall_s": f"{wall:.6f}",
        "best": f"{best:.8f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "code_revision": GMIN_REV,
        "host": "bench",
    }


def load_archives(root: Path, svnrev: str) -> dict[tuple[str, int], dict[str, str]]:
    if svnrev != GMIN_REV:
        print(f"GMIN SVNREV {svnrev} is not {GMIN_REV}; stored rows are not reused", flush=True)
        return {}
    wanted = {row[0]: row for row in SYSTEMS}
    found: dict[tuple[str, int], dict[str, str]] = {}
    for relative in ARCHIVES:
        path = root / relative
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("code") != "gmin":
                    continue
                name = raw["system"]
                if name not in wanted:
                    continue
                if int(raw["budget"]) != wanted[name][3]:
                    continue
                seed = int(raw["seed"])
                if seed not in SEEDS:
                    continue
                found[(name, seed)] = {
                    "system": name,
                    "code": "gmin",
                    "seed": raw["seed"],
                    "budget": raw["budget"],
                    "hit": hit_bit(raw["hit"]),
                    "evals_to_hit": raw["evals_to_hit"],
                    "charged": raw["charged"],
                    "wall_s": raw["wall_s"],
                    "best": raw["best"],
                    "pot_s": raw["pot_s"],
                    "outside_s": raw["outside_s"],
                    "code_revision": GMIN_REV,
                    "host": "bench",
                }
    return found


def load_done(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    if not path.is_file():
        return {}
    done = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            done[(raw["system"], raw["code"], int(raw["seed"]))] = {key: raw.get(key, "") for key in FIELDS}
    return done


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row["system"], row["code"], int(row["seed"])))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def summarise(rows: list[dict[str, str]], reused: set[tuple[str, int]]) -> list[dict[str, str]]:
    out = []
    for name, _n, _rho, _budget, _ref in SYSTEMS:
        anneal = [row for row in rows if row["system"] == name and row["code"] == "anneal"]
        gmin = [row for row in rows if row["system"] == name and row["code"] == "gmin"]
        if len(anneal) != len(SEEDS) or len(gmin) != len(SEEDS):
            continue

        def col(group: list[dict[str, str]], key: str) -> list[float]:
            return [float(row[key]) for row in group]

        ah = sum(row["hit"] == "1" for row in anneal)
        gh = sum(row["hit"] == "1" for row in gmin)
        reused_seeds = " ".join(str(seed) for seed in SEEDS if (name, seed) in reused)
        out.append(
            {
                "system": name,
                "anneal_hits": ah,
                "anneal_n": len(SEEDS),
                "gmin_hits": gh,
                "gmin_n": len(SEEDS),
                "gmin_reused": reused_seeds,
                "fisher_p": f"{fisher(ah, len(SEEDS), gh, len(SEEDS)):.6g}",
                "mannwhitney_p_calls": f"{mann_whitney(col(anneal, 'evals_to_hit'), col(gmin, 'evals_to_hit')):.6g}",
                "anneal_evals_median": f"{type7(col(anneal, 'evals_to_hit')):.3f}",
                "gmin_evals_median": f"{type7(col(gmin, 'evals_to_hit')):.3f}",
                "anneal_wall_median_s": f"{type7(col(anneal, 'wall_s')):.6f}",
                "gmin_wall_median_s": f"{type7(col(gmin, 'wall_s')):.6f}",
                "anneal_pot_median_s": f"{type7(col(anneal, 'pot_s')):.6f}",
                "anneal_outside_median_s": f"{type7(col(anneal, 'outside_s')):.6f}",
                "gmin_pot_median_s": f"{type7(col(gmin, 'pot_s')):.6f}",
                "gmin_outside_median_s": f"{type7(col(gmin, 'outside_s')):.6f}",
            }
        )
    return out


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_anneal(binary: Path, name: str, seed: int, budget: int, revision: str) -> dict[str, str]:
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", name, str(seed)],
        check=False,
        capture_output=True,
        text=True,
        env=run_env(budget),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:] or proc.stdout[-500:])
    return anneal_line(proc.stdout, name, revision)


def run_gmin(
    binary: Path,
    gmin: Path,
    data: str,
    name: str,
    rho: float,
    seed: int,
    budget: int,
    target: float,
    work: Path,
) -> dict[str, str]:
    work.mkdir(parents=True, exist_ok=True)
    coords = work / "coords"
    subprocess.run(
        [str(binary), "coords", name, str(seed), str(coords)],
        check=True,
        env=run_env(budget),
    )
    (work / "data").write_text(data + f"MORSE {rho:.1f}\nRANSEED {seed}\n")
    env = run_env(budget)
    env.update(
        {
            "H2H_SYSTEM": name,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{target:.10f}",
            "H2H_TOL": "0.001",
            "H2H_ROW": str(work / "row.txt"),
        }
    )
    proc = subprocess.run(
        ["taskset", "-c", "0", str(gmin)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=work,
    )
    row_path = work / "row.txt"
    if not row_path.is_file():
        raise RuntimeError(f"gmin {name} {seed} exited {proc.returncode}\n{(proc.stdout or '')[-500:]}")
    return gmin_line(row_path.read_text(), name, seed, budget, target)


def git_head(root: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def svnrev(path: Path) -> str:
    return path.read_text().strip()


def self_test() -> None:
    if abs(type7([1.0, 2.0, 3.0, 4.0]) - 2.5) > 1e-12:
        raise SystemExit("type 7")
    exact = 2.0 / math.comb(24, 12)
    if abs(fisher(12, 12, 0, 12) - exact) > 1e-12:
        raise SystemExit(f"fisher {fisher(12, 12, 0, 12)} != {exact}")
    if not math.isnan(mann_whitney([], [1.0])):
        raise SystemExit("empty mann-whitney")
    names = [row[0] for row in SYSTEMS]
    if names != [
        "m30r6",
        "m30r14",
        "m38r6",
        "m38r14",
        "m55r6",
        "m55r14",
        "m80r3",
        "m80r6",
        "m80r14",
    ]:
        raise SystemExit(f"systems {names}")
    if spec("m30r6")[3] != 400_000 or spec("m80r3")[3] != 4_000_000:
        raise SystemExit("budgets")
    print("ok")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0
    if socket.gethostname() != "bench":
        print(f"uts hostname is {socket.gethostname()}, expected bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/work/morse-gmin"))
    parser.add_argument("--gmin", type=Path, default=Path("/work/morse-gmin/bin/GMIN"))
    parser.add_argument("--svnrev", type=Path, default=Path("/work/gmin-h2h/src/wales/GMIN/source/SVNREV"))
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", choices=("anneal", "gmin", "both"), default="both")
    args = parser.parse_args()
    root = args.root
    bench = root / "bench" / "morse-gmin"
    seeds_path = bench / "seeds.csv"
    summary_path = bench / "summary.csv"
    binary = root.parent / "tmp" / "anneal-target" / "release" / "examples" / "h2h_morse"
    if not binary.is_file():
        raise SystemExit(f"missing {binary}")
    if not args.gmin.is_file():
        raise SystemExit(f"missing {args.gmin}")
    revision = git_head(root)
    got = svnrev(args.svnrev)
    data = (root / "examples" / "h2h_gmin_morse.data").read_text()
    if not data.endswith("\n"):
        data += "\n"
    archives = load_archives(root, got)
    done = load_done(seeds_path)
    rows = list(done.values())
    systems = [row for row in SYSTEMS if not args.system or row[0] in args.system]
    seeds = tuple(args.seed) if args.seed else SEEDS
    reused: set[tuple[str, int]] = set()
    for name, _n, _rho, budget, _target in systems:
        for seed in seeds:
            key = (name, seed)
            if key in archives and (args.arm in ("gmin", "both")):
                reused.add(key)
                if (name, "gmin", seed) not in done:
                    rows.append(archives[key])
                    done[(name, "gmin", seed)] = archives[key]
                    print(f"reuse gmin {name} {seed} best={archives[key]['best']}", flush=True)
    write_rows(seeds_path, rows)
    for name, _n, rho, budget, target in systems:
        for seed in seeds:
            if args.arm in ("anneal", "both") and (name, "anneal", seed) not in done:
                print(f"anneal {name} {seed}", flush=True)
                row = run_anneal(binary, name, seed, budget, revision)
                rows.append(row)
                done[(name, "anneal", seed)] = row
                write_rows(seeds_path, rows)
                print(
                    f"done anneal {name} {seed} hit={row['hit']} best={row['best']} "
                    f"calls={row['evals_to_hit']} pot_s={row['pot_s']} outside_s={row['outside_s']}",
                    flush=True,
                )
            if args.arm in ("gmin", "both") and (name, "gmin", seed) not in done:
                print(f"gmin {name} {seed}", flush=True)
                row = run_gmin(
                    binary,
                    args.gmin,
                    data,
                    name,
                    rho,
                    seed,
                    budget,
                    target,
                    args.work / name / f"gmin-{seed}",
                )
                rows.append(row)
                done[(name, "gmin", seed)] = row
                write_rows(seeds_path, rows)
                print(
                    f"done gmin {name} {seed} hit={row['hit']} best={row['best']} "
                    f"calls={row['evals_to_hit']} wall_s={row['wall_s']}",
                    flush=True,
                )
    write_rows(seeds_path, rows)
    summary = summarise(rows, reused)
    write_summary(summary_path, summary)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
