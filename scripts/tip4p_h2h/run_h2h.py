#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial TIP4P head-to-head, anneal against GMIN, one core, twelve seeds.

Both arms open the same molecular centres: ``h2h_seed`` writes the start it
draws for a seed, and GMIN reads it as its ``coords``. Each run is capped at
2_000_000 charged evaluations, one entry to the potential on either side. A
hit is a quenched energy at or below the reference plus 0.001 kJ/mol.

The references are the Cambridge Cluster Database TIP4P headers, which the
database lists for n <= 21. (H2O)25 is not in the database, so its hit is
scored against the lowest energy either arm reached in the session and no
early stop is armed. Before the campaign both codes are checked to reproduce
the Cambridge header at the published minimum (``--verify``).

Medians are Hyndman-Fan type 7. ``pot_s`` is time inside the potential and
``outside_s`` is the rest of the wall. ``fisher_p`` is the two-sided Fisher
exact test on the hit counts; ``mwu_p`` is the two-sided Mann-Whitney U test
on the per-seed best energies.
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

BUDGET = 2_000_000
HIT_TOL = 0.001
SEEDS = tuple(range(2000, 2012))
SIZES = (8, 15, 20, 21, 25)
# Cambridge Cluster Database TIP4P headers (kJ/mol), listed for n <= 21.
CAMBRIDGE = {
    8: -305.5183221676,
    15: -628.3722525077,
    20: -872.9887544952,
    21: -916.7074302940,
}


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
        return (
            math.comb(row1, x)
            * math.comb(row2, col1 - x)
            / math.comb(n, col1)
        )

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
    for key in (
        "OMP_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def run_anneal(binary: Path, n: int, seed: int, work: Path) -> dict[str, str]:
    prefix = work / f"tip4p{n}-s{seed}"
    log = work / f"tip4p{n}-s{seed}-anneal.log"
    stop = repr(CAMBRIDGE[n] + HIT_TOL) if n in CAMBRIDGE else "-"
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), "tip4p", str(n), "0",
             str(BUDGET), str(seed), "recommended", stop, str(prefix)],
            check=False, stdout=handle, stderr=subprocess.STDOUT, text=True,
            env=env_of(BUDGET),
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal n={n} seed={seed} exited {proc.returncode}\n{text[-500:]}")
    return parse_kv(text, "h2h_row ")


def run_gmin(gmin: Path, data: Path, n: int, seed: int, start: Path, work: Path) -> dict[str, str]:
    run = work / f"gmin-tip4p{n}-s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "coords").write_bytes(start.read_bytes())
    (run / "data").write_text(data.read_text() + f"RANSEED {seed}\n")
    log = work / f"tip4p{n}-s{seed}-gmin.log"
    env = env_of(BUDGET)
    env.update({"H2H_SYSTEM": f"tip4p{n}", "H2H_SEED": str(seed),
                "H2H_ROW": str(run / "row.txt")})
    if n in CAMBRIDGE:
        env["H2H_TARGET"] = f"{CAMBRIDGE[n]:.10f}"
        env["H2H_TOL"] = f"{HIT_TOL}"
    with log.open("w") as handle:
        proc = subprocess.run(["taskset", "-c", "0", str(gmin)], check=False,
                              stdout=handle, stderr=subprocess.STDOUT, text=True,
                              env=env, cwd=run)
    row_path = run / "row.txt"
    if not row_path.is_file():
        raise RuntimeError(f"gmin n={n} seed={seed} exited {proc.returncode}\n{log.read_text()[-800:]}")
    return parse_kv(row_path.read_text(), "h2h_row ")


# The published structure sits at the TIP4P global minimum of both codes. A
# single point at it must land within this of the Cambridge header to confirm
# the shared parameters and energy units; GMIN carries a small constant
# Coulomb-prefactor offset that grows with the pair count.
VERIFY_TOL = 5.0e-3


def gmin_single_point(gmin: Path, verify_data: Path, block: Path, run: Path, n: int) -> float:
    (run / "coords").write_bytes(block.read_bytes())
    (run / "data").write_text(verify_data.read_text())
    log = run / f"verify-{n}.log"
    env = env_of(0)
    env.pop("H2H_BUDGET", None)
    env.update({"H2H_SYSTEM": f"tip4p{n}", "H2H_SEED": "0"})
    with log.open("w") as handle:
        subprocess.run(["taskset", "-c", "0", str(gmin)], check=True,
                       stdout=handle, stderr=subprocess.STDOUT, text=True,
                       env=env, cwd=run)
    for line in log.read_text().splitlines():
        if line.strip().startswith("Energy  ="):
            return float(line.split("=")[1].split()[0])
    raise RuntimeError(f"no CHECKD energy for n={n}\n{log.read_text()[-400:]}")


def verify(point: Path, gmin: Path, verify_data: Path, gm_dir: Path, out: Path, work: Path) -> None:
    """Both codes reproduce the Cambridge header at the published minimum.

    Each code takes a single point at the published centre/angle-axis block
    and must land within ``VERIFY_TOL`` of the header, confirming the shared
    TIP4P parameters and energy units.
    """
    rows = []
    run = work / "verify"
    run.mkdir(parents=True, exist_ok=True)
    for n, ref in CAMBRIDGE.items():
        block = gm_dir / f"tip4p{n}.gm.gmin"
        point_out = subprocess.run([str(point), str(n), str(block)], check=True,
                                   capture_output=True, text=True).stdout
        anneal_e = float(point_out.split("energy:")[1].split()[0])
        gmin_e = gmin_single_point(gmin, verify_data, block, run, n)
        ok = abs(anneal_e - ref) <= VERIFY_TOL and abs(gmin_e - ref) <= VERIFY_TOL
        rows.append({"n": n, "reference_kj": f"{ref:.10f}",
                     "anneal_kj": f"{anneal_e:.10f}", "gmin_kj": f"{gmin_e:.10f}",
                     "anneal_minus_ref": f"{anneal_e - ref:.2e}",
                     "gmin_minus_ref": f"{gmin_e - ref:.2e}",
                     "reproduced": int(ok)})
        print(f"verify n={n} ref={ref:.6f} anneal={anneal_e:.6f} gmin={gmin_e:.6f} ok={ok}", flush=True)
        if not ok:
            raise RuntimeError(f"n={n} not reproduced: anneal {anneal_e}, gmin {gmin_e}, ref {ref}")
    with out.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for n in SIZES:
        anneal = [r for r in rows if r["system"] == f"tip4p{n}" and r["code"] == "anneal"]
        gmin = [r for r in rows if r["system"] == f"tip4p{n}" and r["code"] == "gmin"]
        if len(anneal) != len(SEEDS) or len(gmin) != len(SEEDS):
            continue
        a_best = [float(r["best"]) for r in anneal]
        g_best = [float(r["best"]) for r in gmin]
        if n in CAMBRIDGE:
            ref = CAMBRIDGE[n]
            a_hit = [b <= ref + HIT_TOL for b in a_best]
            g_hit = [b <= ref + HIT_TOL for b in g_best]
            ref_s = f"{ref:.10f}"
        else:
            ref = min(a_best + g_best)
            a_hit = [b <= ref + HIT_TOL for b in a_best]
            g_hit = [b <= ref + HIT_TOL for b in g_best]
            ref_s = f"pooled:{ref:.10f}"
        ah, gh = sum(a_hit), sum(g_hit)

        def pot(r):
            return float(r["pot_s"])

        def outside(r):
            if r.get("outside_s"):
                return float(r["outside_s"])
            return max(0.0, float(r["wall_s"]) - float(r["pot_s"]))

        out.append({
            "system": f"tip4p{n}",
            "n": str(len(SEEDS)),
            "budget": str(BUDGET),
            "reference_kj": ref_s,
            "anneal_hits": f"{ah}/{len(SEEDS)}",
            "gmin_hits": f"{gh}/{len(SEEDS)}",
            "anneal_best_kj": f"{min(a_best):.10f}",
            "gmin_best_kj": f"{min(g_best):.10f}",
            "anneal_best_median_kj": f"{type7(a_best):.6f}",
            "gmin_best_median_kj": f"{type7(g_best):.6f}",
            "anneal_evals_median": f"{type7([float(r['evals_to_hit']) for r in anneal]):.1f}",
            "gmin_evals_median": f"{type7([float(r['evals_to_hit']) for r in gmin]):.1f}",
            "anneal_wall_median_s": f"{type7([float(r['wall_s']) for r in anneal]):.6f}",
            "gmin_wall_median_s": f"{type7([float(r['wall_s']) for r in gmin]):.6f}",
            "anneal_pot_median_s": f"{type7([pot(r) for r in anneal]):.6f}",
            "gmin_pot_median_s": f"{type7([pot(r) for r in gmin]):.6f}",
            "anneal_outside_median_s": f"{type7([outside(r) for r in anneal]):.6f}",
            "gmin_outside_median_s": f"{type7([outside(r) for r in gmin]):.6f}",
            "fisher_p": f"{fisher_exact_two_sided(ah, len(SEEDS) - ah, gh, len(SEEDS) - gh):.6f}",
            "mwu_p": f"{mann_whitney_u_two_sided(a_best, g_best):.6f}",
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--point", type=Path)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--verify-data", type=Path)
    parser.add_argument("--gm-dir", type=Path)
    parser.add_argument("--verify-out", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if socket.gethostname() != "bench":
        print(f"hostname is {socket.gethostname()}, not bench", file=sys.stderr)
        return 2
    if Path.cwd() != Path("/work"):
        print(f"cwd is {Path.cwd()}, not /work", file=sys.stderr)
        return 2
    args.work.mkdir(parents=True, exist_ok=True)
    whole = args.size is None and args.seed is None
    if whole and args.point and args.verify_data and args.gm_dir and args.verify_out:
        verify(args.point, args.gmin, args.verify_data, args.gm_dir, args.verify_out, args.work)
    sizes = (args.size,) if args.size else SIZES
    seeds = (args.seed,) if args.seed is not None else SEEDS
    rows: list[dict[str, str]] = []
    if args.rows.is_file() and whole:
        args.rows.unlink()
    for n in sizes:
        for seed in seeds:
            print(f"anneal n={n} seed={seed}", flush=True)
            anneal = run_anneal(args.binary, n, seed, args.work)
            start = args.work / f"tip4p{n}-s{seed}.start.gmin"
            print(f"gmin n={n} seed={seed}", flush=True)
            gmin = run_gmin(args.gmin, args.data, n, seed, start, args.work)
            rows.extend([anneal, gmin])
            with args.rows.open("a") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({*anneal, *gmin}))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(anneal)
                writer.writerow(gmin)
    if whole:
        summary = summarise(rows)
        with args.summary.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        for row in summary:
            print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
