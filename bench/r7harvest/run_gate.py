# Copyright (c) anneal developers
"""Serial gate of two builds on the CSM panels.

Seeds 2000-2011. Panels, in order, are the binary Lennard-Jones set
(`h2h_blj`), the Morse panel (`h2h_morse`), and the rigid TIP4P sizes
8, 12, 16, 20 and 21 (`h2h_seed`, the water panel's anneal arm). Each
(system, seed) round runs one build then the other. The build that
runs first flips with the sum of the panel, system and seed indices.

One process, core 0, one thread in every pool the child inherits.
Rows are appended after every run. A finished (panel, system, seed,
arm) row is kept. The summary is the tie-corrected Mann-Whitney
normal approximation with the continuity correction, per system and
pooled per panel, on wall time, time inside the potential, and time
outside it.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
BUDGET = 2_000_000
HIT_TOL = 0.001
CAMBRIDGE = {
    8: -305.5183221676,
    12: -492.9082357122,
    16: -681.1925945898,
    20: -872.9887544952,
    21: -916.7074302940,
}
BLJ = (
    "blj105n38",
    "blj105n55",
    "blj115n38",
    "blj115n55",
    "blj130n38",
    "blj130n55",
    "blj130n98",
)
MORSE = (
    "m55r6",
    "m55r14",
    "m70r3",
    "m70r6",
    "m70r10",
    "m70r14",
    "m80r3",
    "m80r6",
    "m80r10",
    "m80r14",
)
RIGID = tuple(CAMBRIDGE)

FIELDS = [
    "panel",
    "system",
    "seed",
    "arm",
    "position",
    "hit",
    "evals_to_hit",
    "charged",
    "calls",
    "best_bits",
    "outcome_bits",
    "wall_s",
    "pot_s",
    "outside_s",
]
SUMMARY_FIELDS = [
    "panel",
    "system",
    "n",
    "identical",
    "a_wall_median_s",
    "b_wall_median_s",
    "mw_p_wall",
    "mw_p_wall_loss",
    "a_pot_median_s",
    "b_pot_median_s",
    "mw_p_pot",
    "mw_p_pot_loss",
    "a_outside_median_s",
    "b_outside_median_s",
    "mw_p_outside",
    "mw_p_outside_loss",
]
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "OPENMP_NUM_THREADS": "1",
}


def child_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("H2H_")}
    env.update(THREAD_ENV)
    return env


def type7(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * 0.5
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mannwhitney(x: list[float], y: list[float], alternative: str) -> tuple[float, float]:
    """U for `x` and the tie-corrected normal p, with continuity correction.

    `greater` asks whether `x` is stochastically larger than `y`.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    ranks = _ranks(x + y)
    u1 = sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    if n < 2:
        return u1, 1.0
    tie = 0
    i = 0
    ordered = sorted(x + y)
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        t = j - i + 1
        tie += t * t * t - t
        i = j + 1
    var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if var <= 0.0:
        return u1, 1.0
    sigma = math.sqrt(var)
    mu = n1 * n2 / 2.0
    if alternative == "two-sided":
        z = (abs(u1 - mu) - 0.5) / sigma
        p = min(1.0, 2.0 * _sf(max(z, 0.0)))
    elif alternative == "greater":
        z = (u1 - 0.5 - mu) / sigma
        p = min(1.0, _sf(z))
    elif alternative == "less":
        z = (u1 + 0.5 - mu) / sigma
        p = min(1.0, 0.5 * math.erfc(-z / math.sqrt(2.0)))
    else:
        raise ValueError(alternative)
    return u1, p


def jobs() -> list[dict[str, object]]:
    panels = (
        ("blj", [(name, name) for name in BLJ]),
        ("morse", [(name, name) for name in MORSE]),
        ("rigid", [(f"tip4p{n}", n) for n in RIGID]),
    )
    planned = []
    for panel_index, (panel, systems) in enumerate(panels):
        for system_index, (system, key) in enumerate(systems):
            for seed_index, seed in enumerate(SEEDS):
                order = ("A", "B") if (panel_index + system_index + seed_index) % 2 == 0 else ("B", "A")
                for position, arm in enumerate(order, start=1):
                    planned.append(
                        {
                            "panel": panel,
                            "system": system,
                            "key": key,
                            "seed": seed,
                            "arm": arm,
                            "position": position,
                        }
                    )
    return planned


def parse_bits(text: str, name: str) -> str:
    prefix = name + " "
    found = [line.split()[1] for line in text.splitlines() if line.startswith(prefix)]
    if len(found) != 1:
        raise RuntimeError(f"expected one {name}, saw {found!r}")
    int(found[0])
    return found[0]


def parse_csv_row(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("system,code,")]
    if len(lines) != 1:
        raise RuntimeError("missing the result header")
    data = [line for line in text.splitlines() if line.startswith(tuple(BLJ + MORSE))]
    if len(data) != 1:
        raise RuntimeError(f"expected one result row, saw {data!r}")
    row = next(csv.DictReader([lines[0], data[0]]))
    hit = "1" if row["hit"] in ("1", "true") else "0"
    return {
        "hit": hit,
        "evals_to_hit": row["evals_to_hit"],
        "charged": row["charged"],
        "calls": row["charged"],
        "wall_s": row["wall_s"],
        "pot_s": row["pot_s"],
        "outside_s": row["outside_s"],
        "best_bits": parse_bits(text, "best_bits"),
        "outcome_bits": parse_bits(text, "outcome_bits"),
    }


def parse_h2h_row(text: str) -> dict[str, str]:
    line = next(line for line in text.splitlines() if line.startswith("h2h_row "))
    fields = {}
    for part in line.split()[1:]:
        key, value = part.split("=", 1)
        fields[key] = value
    calls = next(
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith("calls:")
    )
    return {
        "hit": "1" if fields["hit"] in ("1", "true") else "0",
        "evals_to_hit": fields["evals_to_hit"],
        "charged": fields["charged"],
        "calls": calls,
        "wall_s": fields["wall_s"],
        "pot_s": fields["pot_s"],
        "outside_s": fields["outside_s"],
        "best_bits": parse_bits(text, "best_bits"),
        "outcome_bits": parse_bits(text, "outcome_bits"),
    }


def example_binary(root: Path, panel: str) -> Path:
    name = {"blj": "h2h_blj", "morse": "h2h_morse", "rigid": "h2h_seed"}[panel]
    path = root / name
    if not path.is_file():
        raise RuntimeError(f"missing binary for {panel}")
    return path


def run_one(binary: Path, job: dict[str, object], scratch: Path) -> dict[str, str]:
    panel = str(job["panel"])
    seed = int(job["seed"])
    work = None
    if panel == "rigid":
        n = int(job["key"])
        work = Path(tempfile.mkdtemp(prefix="gate-", dir=scratch))
        prefix = work / "seed"
        command = [
            "taskset",
            "-c",
            "0",
            str(binary),
            "tip4p",
            str(n),
            "0",
            str(BUDGET),
            str(seed),
            "recommended",
            repr(CAMBRIDGE[n] + HIT_TOL),
            str(prefix),
        ]
        parser = parse_h2h_row
    else:
        command = ["taskset", "-c", "0", str(binary), "run", str(job["key"]), str(seed)]
        parser = parse_csv_row
    proc = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=child_env(),
    )
    if work is not None:
        shutil.rmtree(work, ignore_errors=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{job['panel']} {job['system']} {seed} {job['arm']} exited {proc.returncode}\n"
            f"{proc.stderr[-800:]}\n{proc.stdout[-800:]}"
        )
    parsed = parser(proc.stdout)
    return {
        "panel": panel,
        "system": str(job["system"]),
        "seed": str(seed),
        "arm": str(job["arm"]),
        "position": str(job["position"]),
        **parsed,
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def identical(left: dict[str, str], right: dict[str, str]) -> bool:
    keys = ("hit", "evals_to_hit", "charged", "calls", "best_bits", "outcome_bits")
    return all(left[key] == right[key] for key in keys)


def metric_row(panel: str, system: str, group: list[tuple[dict[str, str], dict[str, str]]]) -> dict[str, str]:
    a = [left for left, _right in group]
    b = [right for _left, right in group]
    same = all(identical(left, right) for left, right in group)

    def col(rows: list[dict[str, str]], key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    def cells(key: str) -> tuple[str, str, str, str]:
        left = col(a, key)
        right = col(b, key)
        _u, two = mannwhitney(right, left, "two-sided")
        _u, loss = mannwhitney(right, left, "greater")
        return (
            f"{type7(left):.6f}",
            f"{type7(right):.6f}",
            f"{two:.6g}",
            f"{loss:.6g}",
        )

    wall = cells("wall_s")
    pot = cells("pot_s")
    outside = cells("outside_s")
    return {
        "panel": panel,
        "system": system,
        "n": str(len(group)),
        "identical": "1" if same else "0",
        "a_wall_median_s": wall[0],
        "b_wall_median_s": wall[1],
        "mw_p_wall": wall[2],
        "mw_p_wall_loss": wall[3],
        "a_pot_median_s": pot[0],
        "b_pot_median_s": pot[1],
        "mw_p_pot": pot[2],
        "mw_p_pot_loss": pot[3],
        "a_outside_median_s": outside[0],
        "b_outside_median_s": outside[1],
        "mw_p_outside": outside[2],
        "mw_p_outside_loss": outside[3],
    }


def pairs_of(rows: list[dict[str, str]], panel: str, system: str | None) -> list[tuple[dict[str, str], dict[str, str]]]:
    chosen = [row for row in rows if row["panel"] == panel and (system is None or row["system"] == system)]
    by_seed: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in chosen:
        by_seed.setdefault((row["system"], row["seed"]), {})[row["arm"]] = row
    out = []
    for key in sorted(by_seed):
        arms = by_seed[key]
        if "A" in arms and "B" in arms:
            out.append((arms["A"], arms["B"]))
    return out


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    panels = []
    for row in rows:
        if row["panel"] not in panels:
            panels.append(row["panel"])
    for panel in panels:
        systems = []
        for row in rows:
            if row["panel"] == panel and row["system"] not in systems:
                systems.append(row["system"])
        for system in systems:
            group = pairs_of(rows, panel, system)
            if len(group) == len(SEEDS):
                out.append(metric_row(panel, system, group))
        pooled = pairs_of(rows, panel, None)
        if pooled and all(len(pairs_of(rows, panel, system)) == len(SEEDS) for system in systems):
            out.append(metric_row(panel, "all", pooled))
    return out


def mismatches(rows: list[dict[str, str]]) -> list[str]:
    lines = []
    for panel in ("blj", "morse", "rigid"):
        for left, right in pairs_of(rows, panel, None):
            if identical(left, right):
                continue
            for key in ("hit", "evals_to_hit", "charged", "calls", "best_bits", "outcome_bits"):
                if left[key] != right[key]:
                    lines.append(
                        f"{left['panel']} {left['system']} seed {left['seed']} {key} {left[key]} {right[key]}"
                    )
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-a", type=Path, required=True)
    parser.add_argument("--examples-b", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv[1:])
    rows = load_rows(args.rows)
    done = {(row["panel"], row["system"], row["seed"], row["arm"]) for row in rows}
    if not args.summarize_only:
        args.scratch.mkdir(parents=True, exist_ok=True)
        roots = {"A": args.examples_a, "B": args.examples_b}
        for job in jobs():
            key = (str(job["panel"]), str(job["system"]), str(job["seed"]), str(job["arm"]))
            if key in done:
                continue
            print(
                f"run {job['panel']} {job['system']} {job['seed']} {job['arm']} position {job['position']}",
                file=sys.stderr,
                flush=True,
            )
            binary = example_binary(roots[str(job["arm"])], str(job["panel"]))
            row = run_one(binary, job, args.scratch)
            rows.append(row)
            done.add(key)
            write_rows(args.rows, rows)
            print(
                f"done {row['panel']} {row['system']} {row['seed']} {row['arm']} "
                f"hit={row['hit']} calls={row['calls']} wall={row['wall_s']} "
                f"pot={row['pot_s']} outside={row['outside_s']}",
                file=sys.stderr,
                flush=True,
            )
    expected = {
        (str(job["panel"]), str(job["system"]), str(job["seed"]), str(job["arm"])) for job in jobs()
    }
    have = {(row["panel"], row["system"], row["seed"], row["arm"]) for row in rows}
    if have != expected:
        print(f"incomplete {len(have)}/{len(expected)}", file=sys.stderr)
        return 1
    summary = summarise(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    found = mismatches(rows)
    for line in found:
        print(f"mismatch {line}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
