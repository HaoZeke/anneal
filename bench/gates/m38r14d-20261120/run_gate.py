#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj102": 4_000_000,
    "lj103": 4_000_000,
    "lj104": 4_000_000,
    "lj150": 4_000_000,
    "au55": 1_000_000,
    "m55r6": 1_000_000,
    "m55r14": 1_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
    "sc126-38": 400_000,
    "sc126-55": 1_000_000,
    "sc126-80": 1_000_000,
    "sc108-38": 400_000,
    "sc108-55": 1_000_000,
    "sc108-80": 1_000_000,
    "sc96-38": 400_000,
    "sc96-55": 1_000_000,
    "sc96-80": 1_000_000,
    "tip4p8": 2_000_000,
    "tip4p12": 2_000_000,
    "tip4p16": 2_000_000,
    "tip4p20": 2_000_000,
    "tip4p21": 2_000_000,
}

REFERENCES = {
    "lj38": -173.928427,
    "lj55": -279.248470,
    "lj75": -397.492331,
    "lj98": -543.665361,
    "lj102": -569.363652,
    "lj103": -575.766131,
    "lj104": -582.086642,
    "lj150": -893.310258,
    "au55": -191.03085280,
    "m55r6": -250.286609,
    "m55r14": -220.646208,
    "m80r6": -378.333471,
    "m80r14": -340.811371,
    "sc126-38": -35419.9804,
    "sc126-55": -52884.6806,
    "sc126-80": -78414.6271,
    "sc108-38": -10117.2454,
    "sc108-55": -14814.5225,
    "sc108-80": -21819.9952,
    "sc96-38": -8917.7056,
    "sc96-55": -13217.8963,
    "sc96-80": -19560.8966,
    "tip4p8": -305.5183221676,
    "tip4p12": -492.9082357122,
    "tip4p16": -681.1925945898,
    "tip4p20": -872.9887544952,
    "tip4p21": -916.7074302940,
}

PRIORITY = (
    "lj75",
    "lj98",
    "lj102",
    "lj103",
    "lj104",
    "m55r14",
    "m80r6",
    "m80r14",
)
AA_CELLS = ("lj75", "m55r14", "tip4p20")


def machine_hash() -> str:
    machine = Path("/etc/machine-id").read_text().strip() if Path("/etc/machine-id").is_file() else socket.gethostname()
    return hashlib.sha256(machine.encode("utf-8")).hexdigest()[:16]


def env_one_thread() -> dict[str, str]:
    env = os.environ.copy()
    for key in THREAD_KEYS:
        env[key] = "1"
    return env


def system_binary(system: str) -> str:
    if system.startswith("lj"):
        return "h2h_gmin_proof"
    if system.startswith("m"):
        return "h2h_morse"
    if system.startswith("sc"):
        return "h2h_sc"
    if system.startswith("tip4p"):
        return "h2h_seed"
    return "h2h_metal"


def run_cmd(bin_dir: Path, system: str, seed: int, scratch: Path) -> list[str]:
    budget = PAPER_BUDGETS[system]
    if system.startswith("tip4p"):
        n = system.removeprefix("tip4p")
        stop = REFERENCES[system] + 0.001
        return [
            str(bin_dir / "h2h_seed"),
            "tip4p",
            n,
            "0",
            str(budget),
            str(seed),
            "recommended",
            f"{stop:.10f}",
            str(scratch / f"{system}-{seed}"),
        ]
    return [str(bin_dir / system_binary(system)), "run", system, str(seed)]


def parse_stdout(system: str, seed: int, text: str) -> dict[str, float | int]:
    budget = PAPER_BUDGETS[system]
    if system.startswith("tip4p"):
        row = {}
        for line in text.splitlines():
            if line.startswith("h2h_row "):
                row = dict(part.split("=", 1) for part in line.split()[1:])
        if not row:
            raise RuntimeError(f"{system} {seed}: no h2h_row")
        best = float(row["best"])
        charged = int(row["charged"])
        evals = int(row["evals_to_hit"])
        wall = float(row["wall_s"])
        outside = float(row["outside_s"])
        return {"best": best, "charged": charged, "evals_to_hit": evals, "wall_s": wall, "outside_s": outside}
    header = None
    data = None
    for line in text.splitlines():
        if line.startswith("system,"):
            header = line.split(",")
        elif f",{seed}," in line and line.startswith(system + ","):
            data = line.split(",")
    if header is None or data is None:
        raise RuntimeError(f"{system} {seed}: no csv row")
    row = dict(zip(header, data))
    if int(row["budget"]) != budget:
        raise RuntimeError(f"{system} {seed}: budget mismatch")
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    outside = float(row.get("outside_s", wall - pot))
    return {
        "best": float(row["best"]),
        "charged": int(row["charged"]),
        "evals_to_hit": int(row["evals_to_hit"]),
        "wall_s": wall,
        "outside_s": outside,
    }


def run_one(bin_dir: Path, arm: str, system: str, seed: int, rows: Path, session: Path) -> None:
    env = env_one_thread()
    env["LC_ALL"] = "C"
    aff = sorted(os.sched_getaffinity(0))
    with tempfile.TemporaryDirectory() as temp:
        cmd = ["taskset", "-c", "0", *run_cmd(bin_dir, system, seed, Path(temp))]
        start = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        elapsed = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{system} {arm} {seed} failed: {proc.returncode}")
    parsed = parse_stdout(system, seed, proc.stdout)
    best = float(parsed["best"])
    hit = 1 if best <= REFERENCES[system] + 0.001 else 0
    charged = int(parsed["charged"])
    outside = float(parsed["outside_s"])
    row = {
        "cell": system,
        "arm": arm,
        "seed": str(seed),
        "budget": str(PAPER_BUDGETS[system]),
        "reference": f"{REFERENCES[system]:.10f}",
        "best": f"{best:.10f}",
        "hit": str(hit),
        "charged": str(charged),
        "charged_rank": str(charged if hit == 1 else PAPER_BUDGETS[system]),
        "outside_s": f"{outside:.6f}",
        "outside_per_charged": f"{outside / max(charged, 1):.9f}",
        "wall_s": f"{float(parsed['wall_s']):.6f}",
    }
    fresh = not rows.is_file() or rows.stat().st_size == 0
    with rows.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell": system,
        "arm": arm,
        "seed": seed,
        "core": 0,
        "affinity": aff,
        "machine_hash": machine_hash(),
        "elapsed_s": round(elapsed, 6),
        "cmd": cmd[2:5],
    }
    with session.open("a") as handle:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")


def abba_pairs(a: str, b: str, seed: int) -> tuple[str, str]:
    return (a, b) if seed % 2 == 0 else (b, a)


def run_band(bin_a: Path, bin_b: Path, cells: list[str], seeds: list[int], rows: Path, session: Path, aa: bool) -> None:
    for cell in cells:
        for seed in seeds:
            left, right = abba_pairs("A", "B", seed)
            if aa:
                left, right = abba_pairs("A1", "A2", seed)
            for arm in (left, right):
                bind = bin_a if arm.startswith("A") else bin_b
                run_one(bind, arm, cell, seed, rows, session)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def median(values: list[float]) -> float:
    return statistics.median(values)


def summarize(rows: list[dict[str, str]], a_arm: str, b_arm: str) -> list[dict[str, str]]:
    out = []
    cells = sorted({r["cell"] for r in rows})
    for cell in cells:
        a = [r for r in rows if r["cell"] == cell and r["arm"] == a_arm]
        b = [r for r in rows if r["cell"] == cell and r["arm"] == b_arm]
        if not a or not b:
            continue
        a_hits = sum(int(r["hit"]) for r in a)
        b_hits = sum(int(r["hit"]) for r in b)
        fisher_p = float(fisher_exact([[b_hits, len(b) - b_hits], [a_hits, len(a) - a_hits]], alternative="two-sided")[1])
        a_rank = [float(r["charged_rank"]) for r in a]
        b_rank = [float(r["charged_rank"]) for r in b]
        if len(set(a_rank + b_rank)) == 1:
            mw_p = 1.0
        else:
            mw_p = float(mannwhitneyu(b_rank, a_rank, alternative="two-sided", method="asymptotic", use_continuity=True).pvalue)
        a_out = [float(r["outside_per_charged"]) for r in a]
        b_out = [float(r["outside_per_charged"]) for r in b]
        ratio = median(b_out) / median(a_out) if median(a_out) > 0 else math.inf
        fail = (b_hits < a_hits and fisher_p < 0.05) or (median(b_rank) > median(a_rank) and mw_p < 0.05) or ratio > 1.1
        out.append(
            {
                "cell": cell,
                "hits_a": f"{a_hits}/{len(a)}",
                "hits_b": f"{b_hits}/{len(b)}",
                "fisher_p": f"{fisher_p:.6g}",
                "mw_p_charged": f"{mw_p:.6g}",
                "outside_ratio_b_over_a": f"{ratio:.6f}",
                "status": "FAIL" if fail else "PASS",
            }
        )
    return out


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-a", type=Path, required=True)
    parser.add_argument("--bin-b", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--session-jsonl", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--mode", choices=("p1", "full"), default="p1")
    args = parser.parse_args()

    seeds12 = list(range(2000, 2012))
    seeds24 = list(range(2000, 2024))

    run_band(args.bin_a, args.bin_a, list(AA_CELLS), seeds12, args.rows, args.session_jsonl, aa=True)

    if args.mode == "p1":
        cells = list(PRIORITY)
    else:
        all_cells = list(PAPER_BUDGETS)
        rest = [c for c in all_cells if c not in PRIORITY]
        cells = list(PRIORITY) + rest

    for cell in cells:
        seeds = seeds24 if cell in {"lj38", "lj98"} else seeds12
        run_band(args.bin_a, args.bin_b, [cell], seeds, args.rows, args.session_jsonl, aa=False)

    rows = read_rows(args.rows)
    verdict = summarize(rows, "A", "B")
    write_table(args.verdict, verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
