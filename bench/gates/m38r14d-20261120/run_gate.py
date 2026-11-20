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

AA_CELLS = ("lj75", "m55r14", "tip4p20")
P1_AB_CELLS = ("lj75", "lj98", "lj102", "lj103", "lj104", "m55r14", "m80r6", "m80r14")

FULL_ORDER = (
    "lj75",
    "lj98",
    "lj102",
    "lj103",
    "lj104",
    "m55r14",
    "m80r6",
    "m80r14",
    "lj38",
    "lj55",
    "lj150",
    "au55",
    "m55r6",
    "sc126-38",
    "sc126-55",
    "sc126-80",
    "sc108-38",
    "sc108-55",
    "sc108-80",
    "sc96-38",
    "sc96-55",
    "sc96-80",
    "tip4p8",
    "tip4p12",
    "tip4p16",
    "tip4p20",
    "tip4p21",
)

ROW_FIELDS = [
    "cell",
    "band",
    "arm",
    "session_id",
    "seed",
    "budget",
    "reference",
    "best",
    "hit",
    "charged",
    "charged_rank",
    "outside_s",
    "outside_per_charged",
    "wall_s",
]


def machine_hash() -> str:
    machine = Path("/etc/machine-id").read_text().strip() if Path("/etc/machine-id").is_file() else socket.gethostname()
    return hashlib.sha256(machine.encode("utf-8")).hexdigest()[:16]


def env_one_thread() -> dict[str, str]:
    env = os.environ.copy()
    for key in THREAD_KEYS:
        env[key] = "1"
    env["LC_ALL"] = "C"
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
        return {
            "best": float(row["best"]),
            "charged": int(row["charged"]),
            "evals_to_hit": int(row["evals_to_hit"]),
            "wall_s": float(row["wall_s"]),
            "outside_s": float(row["outside_s"]),
        }
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
    return {
        "best": float(row["best"]),
        "charged": int(row["charged"]),
        "evals_to_hit": int(row["evals_to_hit"]),
        "wall_s": wall,
        "outside_s": float(row.get("outside_s", wall - pot)),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)


def append_session(path: Path, event: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")


def abba_pair(a: str, b: str, seed: int) -> tuple[str, str]:
    return (a, b) if seed % 2 == 0 else (b, a)


def seeds_for(cell: str, aa: bool) -> list[int]:
    if aa:
        return list(range(2000, 2012))
    if cell in {"lj38", "lj98"}:
        return list(range(2000, 2024))
    return list(range(2000, 2012))


def jobs(mode: str) -> list[tuple[str, str, str, int]]:
    out: list[tuple[str, str, str, int]] = []
    for cell in AA_CELLS:
        for seed in seeds_for(cell, aa=True):
            left, right = abba_pair("A1", "A2", seed)
            out.append(("AA", cell, left, seed))
            out.append(("AA", cell, right, seed))

    if mode == "p1":
        cells = list(P1_AB_CELLS)
    else:
        cells = list(FULL_ORDER)
    for cell in cells:
        for seed in seeds_for(cell, aa=False):
            left, right = abba_pair("A", "B", seed)
            out.append(("AB", cell, left, seed))
            out.append(("AB", cell, right, seed))
    return out


def done_set(rows: list[dict[str, str]]) -> set[tuple[str, str, str, str]]:
    return {(r["band"], r["cell"], r["arm"], r["seed"]) for r in rows}


def run_one(
    bin_a: Path,
    bin_b: Path,
    rows_path: Path,
    session_path: Path,
    session_id: str,
    band: str,
    cell: str,
    arm: str,
    seed: int,
) -> None:
    bin_dir = bin_a if arm.startswith("A") else bin_b
    env = env_one_thread()
    affinity = sorted(os.sched_getaffinity(0))
    with tempfile.TemporaryDirectory() as tmp:
        inner = run_cmd(bin_dir, cell, seed, Path(tmp))
        cmd = ["taskset", "-c", "0", *inner]
        start = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        elapsed = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{band} {cell} {arm} {seed}: exit {proc.returncode}")
    parsed = parse_stdout(cell, seed, proc.stdout)
    best = float(parsed["best"])
    hit = 1 if best <= REFERENCES[cell] + 0.001 else 0
    charged = int(parsed["charged"])
    outside = float(parsed["outside_s"])
    row = {
        "cell": cell,
        "band": band,
        "arm": arm,
        "session_id": session_id,
        "seed": str(seed),
        "budget": str(PAPER_BUDGETS[cell]),
        "reference": f"{REFERENCES[cell]:.10f}",
        "best": f"{best:.10f}",
        "hit": str(hit),
        "charged": str(charged),
        "charged_rank": str(charged if hit == 1 else PAPER_BUDGETS[cell]),
        "outside_s": f"{outside:.6f}",
        "outside_per_charged": f"{outside / max(charged, 1):.9f}",
        "wall_s": f"{float(parsed['wall_s']):.6f}",
    }
    append_row(rows_path, row)
    append_session(
        session_path,
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "band": band,
            "cell": cell,
            "arm": arm,
            "session_id": session_id,
            "seed": seed,
            "core": 0,
            "affinity": affinity,
            "machine_hash": machine_hash(),
            "elapsed_s": round(elapsed, 6),
            "binary": Path(inner[0]).name,
        },
    )


def fisher_two_sided(hits_left: int, n_left: int, hits_right: int, n_right: int) -> float:
    hits = hits_left + hits_right
    ways = math.comb(n_left + n_right, hits)

    def prob(k: int) -> float:
        return math.comb(n_left, k) * math.comb(n_right, hits - k) / ways

    observed = prob(hits_left)
    lo, hi = max(0, hits - n_right), min(n_left, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney_two_sided(left: list[float], right: list[float]) -> float:
    n1, n2 = len(left), len(right)
    if n1 == 0 or n2 == 0:
        raise ValueError("mann-whitney needs two groups")
    pooled = sorted(left + right)
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    n = n1 + n2
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def median(values: list[float]) -> float:
    return statistics.median(values)


def verdict(rows: list[dict[str, str]], band: str, left_arm: str, right_arm: str) -> list[dict[str, str]]:
    out = []
    cells = sorted({r["cell"] for r in rows if r["band"] == band})
    for cell in cells:
        left = [r for r in rows if r["band"] == band and r["cell"] == cell and r["arm"] == left_arm]
        right = [r for r in rows if r["band"] == band and r["cell"] == cell and r["arm"] == right_arm]
        if not left or not right:
            continue
        l_hits = sum(int(r["hit"]) for r in left)
        r_hits = sum(int(r["hit"]) for r in right)
        fisher_p = fisher_two_sided(r_hits, len(right), l_hits, len(left))
        l_rank = [float(r["charged_rank"]) for r in left]
        r_rank = [float(r["charged_rank"]) for r in right]
        mw_p = 1.0 if len(set(l_rank + r_rank)) == 1 else mann_whitney_two_sided(r_rank, l_rank)
        l_out = [float(r["outside_per_charged"]) for r in left]
        r_out = [float(r["outside_per_charged"]) for r in right]
        ratio = median(r_out) / median(l_out) if median(l_out) > 0 else math.inf
        fail = (r_hits < l_hits and fisher_p < 0.05) or (median(r_rank) > median(l_rank) and mw_p < 0.05) or ratio > 1.1
        out.append(
            {
                "band": band,
                "cell": cell,
                "hits_a": f"{l_hits}/{len(left)}",
                "hits_b": f"{r_hits}/{len(right)}",
                "fisher_p": f"{fisher_p:.6g}",
                "mw_p": f"{mw_p:.6g}",
                "outside_ratio": f"{ratio:.6f}",
                "status": "FAIL" if fail else "PASS",
            }
        )
    return out


def write_verdict(path: Path, rows: list[dict[str, str]]) -> None:
    merged = verdict(rows, "AA", "A1", "A2") + verdict(rows, "AB", "A", "B")
    if not merged:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)


def summarize_progress(rows: list[dict[str, str]], mode: str) -> dict[str, int]:
    required = jobs(mode)
    done = done_set(rows)
    total = len(required)
    complete = sum(1 for band, cell, arm, seed in required if (band, cell, arm, str(seed)) in done)
    missing = total - complete
    return {"total": total, "complete": complete, "missing": missing}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-a", type=Path, required=True)
    parser.add_argument("--bin-b", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--session-jsonl", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--mode", choices=("p1", "full"), default="p1")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--progress-json", type=Path)
    args = parser.parse_args()

    all_jobs = jobs(args.mode)
    rows = read_rows(args.rows)
    done = done_set(rows)

    for band, cell, arm, seed in all_jobs:
        key = (band, cell, arm, str(seed))
        if key in done:
            continue
        run_one(args.bin_a, args.bin_b, args.rows, args.session_jsonl, args.session_id, band, cell, arm, seed)
        done.add(key)

    rows = read_rows(args.rows)
    write_verdict(args.verdict, rows)
    if args.progress_json is not None:
        status = summarize_progress(rows, args.mode)
        args.progress_json.write_text(json.dumps(status, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
