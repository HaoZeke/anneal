# Copyright (c) anneal developers
"""Same-session baseline against the species-SOAP and acceptance fix.

The baseline is the rebased binary Lennard-Jones tip. The fix is its
child. Both arms use seeds 2000-2011, one process on core 0, hostname
bench. The binary Lennard-Jones panel is the seven Doye-Meyer clusters.
The Lennard-Jones panel is LJ38, LJ55, LJ75, LJ98 and LJ104 at the paper
budgets. A hit is a quenched energy at or below the reference plus 0.001.

Rows are appended and never rewritten. ``code_revision`` is the commit
the binary's tree is on. Fisher is the two-sided exact test on hits.
Mann-Whitney is two-sided, the normal approximation with the tie and
continuity corrections.

This process refuses to run unless the UTS hostname is bench and it is
pinned to core 0.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import socket
import subprocess
import time
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
HIT_TOL = 0.001
BLJ = (
    ("blj105n38", 38, 400_000, -177.2090846748),
    ("blj105n55", 55, 1_000_000, -283.1545313810),
    ("blj115n38", 38, 400_000, -187.3676867453),
    ("blj115n55", 55, 1_000_000, -289.2931853058),
    ("blj130n38", 38, 400_000, -191.1840192602),
    ("blj130n55", 55, 1_000_000, -306.7337643721),
    ("blj130n98", 98, 4_000_000, -590.7874132760),
)
LJ = (
    ("lj38", 38, 400_000, -173.928427),
    ("lj55", 55, 1_000_000, -279.248470),
    ("lj75", 75, 4_000_000, -397.492331),
    ("lj98", 98, 4_000_000, -543.665361),
    ("lj104", 104, 4_000_000, -582.086642),
)
SYSTEMS = BLJ + LJ

ROW_FIELDS = [
    "system",
    "arm",
    "code",
    "driver",
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
    "cpu",
    "session",
    "utc",
]


def type7(values: list[float], p: float) -> float:
    """Hyndman-Fan type 7."""
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided exact p, summing the tables no more likely than the observed."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """U of ``left`` and the two-sided p, normal approximation with the tie
    and continuity corrections. Every value tied gives p = 1."""
    n1, n2 = len(left), len(right)
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
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
        return u, 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return u, min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def reference_of(system: str) -> tuple[int, int, float]:
    for name, n, budget, reference in SYSTEMS:
        if name == system:
            return n, budget, reference
    raise SystemExit(f"unknown system {system}")


def is_blj(system: str) -> bool:
    return system.startswith("blj")


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip().replace(",", " ")
    return ""


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("H2H_"):
            del env[key]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[key] = "1"
    return env


def tree_head(tree: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def read_rows(path: Path) -> list[dict[str, str]]:
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
        handle.flush()
        os.fsync(handle.fileno())


def check_result(system: str, seed: int, budget: int, hit: bool, best: float, evals: int) -> None:
    _n, _budget, reference = reference_of(system)
    if hit != (best <= reference + HIT_TOL + 1e-9):
        raise RuntimeError(f"{system} seed {seed}: hit {hit} with best {best} against {reference}")
    if not hit and evals != budget:
        raise RuntimeError(f"{system} seed {seed}: a miss reports {evals} calls")
    if hit and not 0 < evals <= budget:
        raise RuntimeError(f"{system} seed {seed}: a hit reports {evals} calls")


def run_blj(binary: Path, tree: Path, revision: str, system: str, seed: int, log: Path) -> dict[str, str]:
    _n, budget, _reference = reference_of(system)
    command = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
    proc = subprocess.run(command, cwd=tree, env=single_thread_env(), capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    lines = [line for line in proc.stdout.splitlines() if line.startswith("system,") or f",anneal,{seed}," in line]
    headers = [line for line in lines if line.startswith("system,")]
    data = [line for line in lines if not line.startswith("system,")]
    if len(headers) != 1 or len(data) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one row\n{proc.stdout[-2000:]}")
    parsed = dict(zip(headers[0].split(","), data[0].split(",")))
    if parsed["system"] != system or int(parsed["seed"]) != seed or int(parsed["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: row {data[0]}")
    if parsed["head"] != revision:
        raise RuntimeError(f"{system} seed {seed}: binary reports {parsed['head']}, tree is {revision}")
    best = float(parsed["best"])
    hit = parsed["hit"].strip().lower() in ("1", "true")
    evals = int(parsed["evals_to_hit"])
    check_result(system, seed, budget, hit, best, evals)
    wall = float(parsed["wall_s"])
    pot = float(parsed["pot_s"])
    return {
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(int(parsed["charged"])),
        "wall_s": f"{wall:.6f}",
        "best": parsed["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{wall - pot:.6f}",
        "driver": "h2h_blj",
    }


def run_lj(binary: Path, tree: Path, system: str, seed: int, log: Path, scratch: Path) -> dict[str, str]:
    n, budget, reference = reference_of(system)
    stop = f"{reference + HIT_TOL:.12f}"
    prefix = scratch / system / str(seed)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset", "-c", "0", str(binary),
        "lj", str(n), "0", str(budget), str(seed), "recommended", stop, str(prefix),
    ]
    proc = subprocess.run(command, cwd=tree, env=single_thread_env(), capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    rows = [line for line in proc.stdout.splitlines() if line.startswith("h2h_row ")]
    if len(rows) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one h2h_row\n{proc.stdout[-2000:]}")
    fields = dict(part.split("=", 1) for part in rows[0].split()[1:] if "=" in part)
    if fields["system"] != system or int(fields["seed"]) != seed or int(fields["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: row {rows[0]}")
    best = float(fields["best"])
    hit = fields["hit"] == "1"
    evals = int(fields["evals_to_hit"])
    check_result(system, seed, budget, hit, best, evals)
    return {
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(int(fields["charged"])),
        "wall_s": f"{float(fields['wall_s']):.6f}",
        "best": f"{best:.10f}",
        "pot_s": f"{float(fields['pot_s']):.6f}",
        "outside_s": f"{float(fields['outside_s']):.6f}",
        "driver": "h2h_seed",
    }


def pp(value: float) -> str:
    if value >= 0.995:
        return "1"
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["arm"]), []).append(row)
    out = []
    for system, _n, budget, reference in SYSTEMS:
        base = sorted(grouped.get((system, "baseline"), []), key=lambda r: int(r["seed"]))
        fix = sorted(grouped.get((system, "fix"), []), key=lambda r: int(r["seed"]))
        if len(base) != 12 or len(fix) != 12:
            continue

        def col(group: list[dict[str, str]], key: str) -> list[float]:
            return [float(r[key]) for r in group]

        bh = sum(r["hit"] == "1" for r in base)
        fh = sum(r["hit"] == "1" for r in fix)
        p_hits = fisher(fh, 12, bh, 12)
        _u_calls, p_calls = mann_whitney(col(fix, "evals_to_hit"), col(base, "evals_to_hit"))
        _u_wall, p_wall = mann_whitney(col(fix, "wall_s"), col(base, "wall_s"))
        fix_calls = type7(col(fix, "evals_to_hit"), 0.5)
        base_calls = type7(col(base, "evals_to_hit"), 0.5)
        fix_wall = type7(col(fix, "wall_s"), 0.5)
        base_wall = type7(col(base, "wall_s"), 0.5)
        lose_hits = fh < bh and p_hits < 0.05
        lose_calls = fix_calls > base_calls and p_calls < 0.05
        lose_wall = fix_wall > base_wall and p_wall < 0.05
        out.append(
            {
                "system": system,
                "budget": str(budget),
                "reference": f"{reference:.10f}",
                "baseline_hits": f"{bh}/12",
                "fix_hits": f"{fh}/12",
                "fisher_p": pp(p_hits),
                "baseline_calls": f"{base_calls:.1f}",
                "fix_calls": f"{fix_calls:.1f}",
                "mw_p_calls": pp(p_calls),
                "baseline_wall": f"{base_wall:.3f}",
                "fix_wall": f"{fix_wall:.3f}",
                "mw_p_wall": pp(p_wall),
                "baseline_pot": f"{type7(col(base, 'pot_s'), 0.5):.3f}",
                "fix_pot": f"{type7(col(fix, 'pot_s'), 0.5):.3f}",
                "baseline_out": f"{type7(col(base, 'outside_s'), 0.5):.3f}",
                "fix_out": f"{type7(col(fix, 'outside_s'), 0.5):.3f}",
                "baseline_best": f"{min(col(base, 'best')):.8f}",
                "fix_best": f"{min(col(fix, 'best')):.8f}",
                "lose_hits": "yes" if lose_hits else "no",
                "lose_calls": "yes" if lose_calls else "no",
                "lose_wall": "yes" if lose_wall else "no",
            }
        )
    return out


def print_summary(rows: list[dict[str, str]]) -> None:
    table = summarise(rows)
    if not table:
        print("summary: no complete system yet", flush=True)
        return
    print(
        "system baseline_hits fix_hits fisher base_calls fix_calls p_calls "
        "base_in fix_in base_out fix_out base_wall fix_wall p_wall lose",
        flush=True,
    )
    for row in table:
        lose = ",".join(
            name
            for name, flag in (("hits", row["lose_hits"]), ("calls", row["lose_calls"]), ("wall", row["lose_wall"]))
            if flag == "yes"
        ) or "no"
        print(
            f"{row['system']} {row['baseline_hits']} {row['fix_hits']} {row['fisher_p']} "
            f"{row['baseline_calls']} {row['fix_calls']} {row['mw_p_calls']} "
            f"{row['baseline_pot']} {row['fix_pot']} {row['baseline_out']} {row['fix_out']} "
            f"{row['baseline_wall']} {row['fix_wall']} {row['mw_p_wall']} {lose}",
            flush=True,
        )


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    table = summarise(rows)
    if not table:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)


def run(args: argparse.Namespace) -> int:
    if socket.gethostname() != "bench":
        print(f"uts hostname is {socket.gethostname()}, expected bench", flush=True)
        return 2
    if os.sched_getaffinity(0) != {0}:
        print(f"pinned to {sorted(os.sched_getaffinity(0))}, expected core 0", flush=True)
        return 2
    arms = {
        "baseline": (args.baseline_bin, args.baseline_tree, tree_head(args.baseline_tree)),
        "fix": (args.fix_bin, args.fix_tree, tree_head(args.fix_tree)),
    }
    if arms["baseline"][2] != args.baseline_rev or arms["fix"][2] != args.fix_rev:
        raise SystemExit(
            f"trees are {arms['baseline'][2]} and {arms['fix'][2]}, "
            f"expected {args.baseline_rev} and {args.fix_rev}"
        )
    if arms["baseline"][2] == arms["fix"][2]:
        raise SystemExit("baseline and fix are the same commit")
    stored = read_rows(args.rows)
    sessions = {row["session"] for row in stored}
    if len(sessions) > 1:
        raise SystemExit(f"rows already hold sessions {sorted(sessions)}")
    session = sessions.pop() if sessions else time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done = {(row["system"], row["arm"], row["seed"], row["code_revision"]) for row in stored}
    host = socket.gethostname()
    cpu = cpu_model()
    systems = [name for name, *_rest in SYSTEMS if not args.system or name in args.system]
    seeds = tuple(args.seed) if args.seed else SEEDS
    print(f"session {session} host {host} baseline {arms['baseline'][2]} fix {arms['fix'][2]}", flush=True)
    for system in systems:
        for seed in seeds:
            for arm in ("baseline", "fix"):
                binary_root, tree, revision = arms[arm]
                key = (system, arm, str(seed), revision)
                if key in done:
                    print(f"skip {system} {arm} {seed}", flush=True)
                    continue
                binary = binary_root / ("h2h_blj" if is_blj(system) else "h2h_seed")
                print(f"start {system} {arm} {seed}", flush=True)
                log = args.logs / system / f"{arm}-{seed}.log"
                if is_blj(system):
                    result = run_blj(binary, tree, revision, system, seed, log)
                else:
                    scratch = args.scratch / arm
                    result = run_lj(binary, tree, system, seed, log, scratch)
                row = {
                    "system": system,
                    "arm": arm,
                    "code": "anneal",
                    "seed": str(seed),
                    "code_revision": revision,
                    "host": host,
                    "cpu": cpu,
                    "session": session,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    **result,
                }
                append_row(args.rows, row)
                done.add(key)
                print(
                    f"done {system} {arm} {seed} hit={row['hit']} calls={row['evals_to_hit']} "
                    f"wall_s={row['wall_s']} pot_s={row['pot_s']} outside_s={row['outside_s']} best={row['best']}",
                    flush=True,
                )
        stored_now = read_rows(args.rows)
        print_summary(stored_now)
        write_summary(args.summary, stored_now)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--baseline-bin", type=Path, required=True)
    parser.add_argument("--fix-bin", type=Path, required=True)
    parser.add_argument("--baseline-tree", type=Path, required=True)
    parser.add_argument("--fix-tree", type=Path, required=True)
    parser.add_argument("--baseline-rev", required=True)
    parser.add_argument("--fix-rev", required=True)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv[1:])
    if args.summarize:
        rows = read_rows(args.rows)
        print_summary(rows)
        write_summary(args.summary, rows)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
