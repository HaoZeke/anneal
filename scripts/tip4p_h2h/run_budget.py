#!/usr/bin/env python3
# Copyright (c) anneal developers
"""(H2O)21 head-to-head, anneal against GMIN, across a budget sweep.

Same serial arm as ``run_h2h.py`` -- one core, hostname bench, working
directory /work, both codes from the same molecular centres -- but for one
size at several charged-evaluation budgets. anneal runs ``recommended``, the
develop default preset, with no replica ladder. GMIN is the fixed step. A hit
is a quenched energy at or below the Cambridge header plus ``HIT_TOL``; a miss
is the full budget.

GMIN at a seed and budget is deterministic, so a prior ``--gmin-rows`` CSV
(for example the rows the 2M head-to-head and the 4M/8M sweep already wrote)
is reused when it carries that pair, and GMIN is only run for the pairs it is
missing. Medians are Hyndman-Fan type 7, and each summary row carries the
two-sided Fisher exact test on the hit counts and the two-sided Mann-Whitney
test on the twelve best energies, with the wall split inside and outside the
potential.
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

HIT_TOL = 0.001
SEEDS = tuple(range(2000, 2012))
BUDGETS = (2_000_000, 4_000_000, 8_000_000)
SIZE = 21
CAMBRIDGE = {21: -916.7074302940}


def type7(values: list[float], p: float = 0.5) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    if n == 1:
        return s[0]
    h = 1.0 + (n - 1) * p
    lo = int(h)
    frac = h - lo
    idx = lo - 1
    if frac == 0.0:
        return s[idx]
    return s[idx] * (1.0 - frac) + s[idx + 1] * frac


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c
    row2 = c + d

    def table_p(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    p0 = table_p(a)
    total = 0.0
    for x in range(max(0, col1 - row2), min(row1, col1) + 1):
        px = table_p(x)
        if px <= p0 * (1.0 + 1e-9):
            total += px
    return min(1.0, total)


def mann_whitney_u_two_sided(x: list[float], y: list[float]) -> float:
    """Two-sided Mann-Whitney U p with a tie-corrected normal approximation."""
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
    z = (u - mu) / math.sqrt(sigma2)
    return max(0.0, min(1.0, 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))))


def parse_kv(text: str, prefix: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith(prefix):
            fields = {}
            for part in line.split()[1:]:
                key, value = part.split("=", 1)
                fields[key] = value
            return fields
    raise ValueError(f"no {prefix} row")


def env_of(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(budget)
    for key in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "1"
    return env


def outside_of(row: dict[str, str]) -> float:
    if row.get("outside_s"):
        return float(row["outside_s"])
    return max(0.0, float(row["wall_s"]) - float(row["pot_s"]))


def run_anneal(binary: Path, n: int, seed: int, budget: int, work: Path) -> dict[str, str]:
    prefix = work / f"tip4p{n}-s{seed}-b{budget}"
    log = work / f"tip4p{n}-s{seed}-b{budget}-anneal.log"
    stop = repr(CAMBRIDGE[n] + HIT_TOL) if n in CAMBRIDGE else "-"
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), "tip4p", str(n), "0",
             str(budget), str(seed), "recommended", stop, str(prefix)],
            check=False, stdout=handle, stderr=subprocess.STDOUT, text=True,
            env=env_of(budget),
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal n={n} seed={seed} budget={budget} exited {proc.returncode}\n{text[-500:]}")
    row = parse_kv(text, "h2h_row ")
    row["start"] = str(Path(f"{prefix}.start.gmin"))
    return row


def run_gmin(gmin: Path, data: Path, n: int, seed: int, budget: int, start: Path, work: Path) -> dict[str, str]:
    run = work / f"gmin-tip4p{n}-s{seed}-b{budget}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "coords").write_bytes(start.read_bytes())
    (run / "data").write_text(data.read_text() + f"RANSEED {seed}\n")
    log = work / f"tip4p{n}-s{seed}-b{budget}-gmin.log"
    env = env_of(budget)
    env.update({"H2H_SYSTEM": f"tip4p{n}", "H2H_SEED": str(seed), "H2H_ROW": str(run / "row.txt")})
    if n in CAMBRIDGE:
        env["H2H_TARGET"] = f"{CAMBRIDGE[n]:.10f}"
        env["H2H_TOL"] = f"{HIT_TOL}"
    with log.open("w") as handle:
        subprocess.run(["taskset", "-c", "0", str(gmin)], check=False,
                       stdout=handle, stderr=subprocess.STDOUT, text=True, env=env, cwd=run)
    row_path = run / "row.txt"
    if not row_path.is_file():
        raise RuntimeError(f"gmin n={n} seed={seed} budget={budget}\n{log.read_text()[-800:]}")
    return parse_kv(row_path.read_text(), "h2h_row ")


def load_reuse(path: Path | None) -> dict[tuple[int, int], dict[str, str]]:
    """Map (budget, seed) to a stored GMIN row, from one or more CSVs."""
    out: dict[tuple[int, int], dict[str, str]] = {}
    if path is None:
        return out
    for piece in str(path).split(","):
        p = Path(piece)
        if not p.is_file():
            continue
        for row in csv.DictReader(p.open()):
            if row.get("code") != "gmin":
                continue
            out[(int(row["budget"]), int(row["seed"]))] = row
    return out


def summarise(rows: list[dict[str, str]], budget: int, n: int) -> dict[str, str]:
    anneal = [r for r in rows if r["code"] == "anneal" and int(r["budget"]) == budget]
    gmin = [r for r in rows if r["code"] == "gmin" and int(r["budget"]) == budget]
    ref = CAMBRIDGE[n]
    a_best = [float(r["best"]) for r in anneal]
    g_best = [float(r["best"]) for r in gmin]
    ah = sum(1 for b in a_best if b <= ref + HIT_TOL)
    gh = sum(1 for b in g_best if b <= ref + HIT_TOL)
    return {
        "system": f"tip4p{n}",
        "n": str(len(anneal)),
        "budget": str(budget),
        "reference_kj": f"{ref:.10f}",
        "anneal_hits": f"{ah}/{len(anneal)}",
        "gmin_hits": f"{gh}/{len(gmin)}",
        "anneal_best_kj": f"{min(a_best):.10f}",
        "gmin_best_kj": f"{min(g_best):.10f}",
        "anneal_best_median_kj": f"{type7(a_best):.6f}",
        "gmin_best_median_kj": f"{type7(g_best):.6f}",
        "anneal_evals_median": f"{type7([float(r['charged']) for r in anneal]):.1f}",
        "gmin_evals_median": f"{type7([float(r['charged']) for r in gmin]):.1f}",
        "anneal_wall_median_s": f"{type7([float(r['wall_s']) for r in anneal]):.6f}",
        "gmin_wall_median_s": f"{type7([float(r['wall_s']) for r in gmin]):.6f}",
        "anneal_pot_median_s": f"{type7([float(r['pot_s']) for r in anneal]):.6f}",
        "gmin_pot_median_s": f"{type7([float(r['pot_s']) for r in gmin]):.6f}",
        "anneal_outside_median_s": f"{type7([outside_of(r) for r in anneal]):.6f}",
        "gmin_outside_median_s": f"{type7([outside_of(r) for r in gmin]):.6f}",
        "fisher_p": f"{fisher_exact_two_sided(ah, len(anneal) - ah, gh, len(gmin) - gh):.6f}",
        "mwu_p": f"{mann_whitney_u_two_sided(a_best, g_best):.6f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gmin-rows", type=Path, help="comma-separated CSVs of prior GMIN rows to reuse")
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--budget", type=int, help="one budget instead of the full sweep")
    parser.add_argument("--seed", type=int, help="one seed instead of all twelve")
    args = parser.parse_args()
    if socket.gethostname() != "bench":
        print(f"hostname is {socket.gethostname()}, not bench", file=sys.stderr)
        return 2
    if Path.cwd() != Path("/work"):
        print(f"cwd is {Path.cwd()}, not /work", file=sys.stderr)
        return 2
    args.work.mkdir(parents=True, exist_ok=True)
    reuse = load_reuse(args.gmin_rows)
    n = args.size
    budgets = (args.budget,) if args.budget else BUDGETS
    seeds = (args.seed,) if args.seed is not None else SEEDS
    whole = args.budget is None and args.seed is None
    fields = ["system", "code", "budget", "seed", "hit", "evals_to_hit", "charged",
              "best", "wall_s", "pot_s", "outside_s", "basins"]
    rows: list[dict[str, str]] = []
    with args.rows.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for budget in budgets:
            for seed in seeds:
                print(f"anneal n={n} seed={seed} budget={budget}", flush=True)
                anneal = run_anneal(args.binary, n, seed, budget, args.work)
                start = Path(anneal.pop("start"))
                carried = reuse.get((budget, seed))
                if carried is not None:
                    gmin = carried
                    print(f"gmin n={n} seed={seed} budget={budget} (reused)", flush=True)
                else:
                    print(f"gmin n={n} seed={seed} budget={budget}", flush=True)
                    gmin = run_gmin(args.gmin, args.data, n, seed, budget, start, args.work)
                for code, row in (("anneal", anneal), ("gmin", gmin)):
                    out = {"system": f"tip4p{n}", "code": code, "budget": str(budget), "seed": str(seed),
                           "hit": row["hit"], "evals_to_hit": row["evals_to_hit"], "charged": row["charged"],
                           "best": row["best"], "wall_s": row["wall_s"], "pot_s": row["pot_s"],
                           "outside_s": f"{outside_of(row):.6f}", "basins": row.get("basins", "")}
                    rows.append(out)
                    writer.writerow(out)
                    handle.flush()
    if not whole:
        return 0
    summary = [summarise(rows, budget, n) for budget in budgets]
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
