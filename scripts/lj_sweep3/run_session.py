#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-session Lennard-Jones panel, develop against this tree.

One process, hostname bench. Seeds 2000-2011 on LJ38, 55, 75, 76, 77,
98, 102, 103, 104 and 150. Seeds 2000-2023 on LJ76, 77, 102, 103 and
150. Rows are appended with the code revision and the host. A repeated
seed keeps the earliest row. GMIN rows already stored are copied in
and not rerun. Fisher is the two-sided exact test on the hit counts.
Mann-Whitney is the tie-corrected normal approximation on calls to
the minimum.
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
MOVED = (76, 77, 102, 103, 150)
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
    "panel",
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
        ranks[pooled[i]] = (i + 1 + j + 1) / 2.0
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
        writer.writerow({key: row.get(key, "") for key in ROW_FIELDS})


def seen(rows: list[dict[str, str]], system: str, code: str, seed: str, revision: str) -> bool:
    return any(
        row["system"] == system
        and row["code"] == code
        and row["seed"] == seed
        and row["code_revision"] == revision
        for row in rows
    )


def reuse_gmin(stored: list[dict[str, str]], source: Path, dest: Path) -> list[dict[str, str]]:
    if not source.is_file():
        return stored
    for row in csv.DictReader(source.open()):
        if row.get("code") != "gmin":
            continue
        if seen(stored, row["system"], "gmin", row["seed"], row["code_revision"]):
            continue
        append_row(dest, row)
        stored.append(row)
    return stored


def parse_csv_row(text: str, code: str, revision: str, host: str) -> dict[str, str]:
    data = [line for line in text.splitlines() if f",{code}," in line]
    if len(data) != 1:
        raise ValueError(f"expected one {code} row, saw {len(data)} in {text[-300:]!r}")
    parts = next(csv.reader([data[0]]))
    if len(parts) < 9:
        raise ValueError(f"short row {data[0]!r}")
    hit = "1" if parts[4] in ("1", "true") else "0"
    wall = float(parts[7])
    # The proof row puts pot_s last, after head, tree, cpu and pins.
    # Morse and metal put pot_s and outside_s immediately after best.
    if len(parts) >= 14:
        pot = float(parts[13])
        outside = max(wall - pot, 0.0)
    else:
        pot = float(parts[9]) if len(parts) > 9 and parts[9] else 0.0
        outside = float(parts[10]) if len(parts) > 10 and parts[10] else max(wall - pot, 0.0)
    return {
        "system": parts[0],
        "code": code,
        "seed": parts[2],
        "budget": parts[3],
        "hit": hit,
        "evals_to_hit": parts[5],
        "charged": parts[6],
        "wall_s": f"{wall:.6f}",
        "best": parts[8],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "code_revision": revision,
        "host": host,
    }


def run_cmd(args: list[str], log: Path, env: dict[str, str]) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", *args],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"{args} exited {proc.returncode}\n{text[-400:]}")
    return text


def base_env(budget: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["LJOS_SEAT"] = "anneal-paper"
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    if budget is not None:
        env["H2H_BUDGET"] = str(budget)
    return env


def arm_row(group: list[dict[str, str]], system: str, panel: str, arm: str, revision: str, host: str, fisher: str, mann: str) -> dict[str, str]:
    evals = [float(row["evals_to_hit"]) for row in group]
    walls = [float(row["wall_s"]) for row in group]
    insides, outsides = [], []
    for row in group:
        charged = max(int(row["charged"]), 1)
        insides.append(float(row["pot_s"]) / charged * 1e6)
        outsides.append(float(row["outside_s"]) / charged * 1e6)
    evals_median, evals_iqr = median_iqr(evals)
    wall_median, _ = median_iqr(walls)
    hits = sum(row["hit"] == "1" for row in group)
    return {
        "system": system,
        "panel": panel,
        "arm": arm,
        "n": str(len(group)),
        "hits": str(hits),
        "hits_over_n": f"{hits}/{len(group)}",
        "budget": group[0]["budget"],
        "evals_median": f"{evals_median:.6f}",
        "evals_iqr": f"{evals_iqr:.6f}",
        "wall_median_s": f"{wall_median:.6f}",
        "inside_us": f"{median_iqr(insides)[0]:.4f}",
        "outside_us": f"{median_iqr(outsides)[0]:.4f}",
        "fisher_p": fisher,
        "mannwhitney_p": mann,
        "code_revision": revision,
        "host": host,
    }


def compare(
    rows: list[dict[str, str]],
    system: str,
    seeds: tuple[int, ...],
    head: str,
    base: str,
    host: str,
    panel: str,
) -> list[dict[str, str]]:
    def take(revision: str) -> list[dict[str, str]]:
        wanted = {str(seed) for seed in seeds}
        chosen: dict[str, dict[str, str]] = {}
        for row in rows:
            if (
                row["system"] == system
                and row["code"] == "anneal"
                and row["code_revision"] == revision
                and row["seed"] in wanted
                and row["seed"] not in chosen
            ):
                chosen[row["seed"]] = row
        return [chosen[str(seed)] for seed in seeds if str(seed) in chosen]

    left = take(head)
    right = take(base)
    if len(left) != len(seeds) or len(right) != len(seeds):
        return []
    hits_h = sum(row["hit"] == "1" for row in left)
    hits_b = sum(row["hit"] == "1" for row in right)
    fisher = f"{fisher_exact_two_sided(hits_h, len(left), hits_b, len(right)):.6g}"
    mann = f"{mann_whitney_u_two_sided([float(r['evals_to_hit']) for r in left], [float(r['evals_to_hit']) for r in right]):.6g}"
    return [
        arm_row(left, system, panel, "head", head, host, fisher, mann),
        arm_row(right, system, panel, "develop", base, host, fisher, mann),
    ]


def write_summary(path: Path, rows: list[dict[str, str]], head: str, base: str, host: str) -> None:
    out: list[dict[str, str]] = []
    n12 = tuple(range(2000, 2012))
    n24 = tuple(range(2000, 2024))
    for n in SIZES:
        out.extend(compare(rows, f"lj{n}", n12, head, base, host, "12"))
    for n in MOVED:
        out.extend(compare(rows, f"lj{n}", n24, head, base, host, "24"))
    for system in ("au55", "m80r6"):
        out.extend(compare(rows, system, n12, head, base, host, "12"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)


def record_run(
    stored: list[dict[str, str]],
    dest: Path,
    text: str,
    code: str,
    revision: str,
    host: str,
    summary: Path,
    head: str,
    base: str,
) -> None:
    row = parse_csv_row(text, code, revision, host)
    if seen(stored, row["system"], row["code"], row["seed"], revision):
        return
    append_row(dest, row)
    stored.append(row)
    print(
        f"done {row['system']} {row['code']} seed {row['seed']} rev {revision[:12]} hit={row['hit']} evals={row['evals_to_hit']}",
        flush=True,
    )
    write_summary(summary, stored, head, base, host)


def main(argv: list[str]) -> int:
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-bin", type=Path, required=True)
    parser.add_argument("--develop-bin", type=Path, required=True)
    parser.add_argument("--head-rev", type=Path, required=True)
    parser.add_argument("--develop-rev", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--gmin-rows", type=Path, required=True)
    parser.add_argument("--head-examples", type=Path, required=True)
    parser.add_argument("--develop-examples", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    args = parser.parse_args(argv[1:])
    head = args.head_rev.read_text().strip()
    base = args.develop_rev.read_text().strip()
    host = socket.gethostname()
    stored = load_rows(args.rows)
    stored = reuse_gmin(stored, args.gmin_rows, args.rows)
    develop_examples = args.develop_examples
    head_examples = args.head_examples
    arms = (
        ("develop", args.develop_bin, base, develop_examples),
        ("head", args.head_bin, head, head_examples),
    )

    def one(kind: str, revision: str, system: str, seed: int, budget: int | None, cmd: list[str]) -> None:
        if seen(stored, system, "anneal", str(seed), revision):
            print(f"skip {system} {kind} seed {seed}", flush=True)
            return
        print(f"start {system} {kind} seed {seed}", flush=True)
        log = args.logs / system / f"{kind}-{seed}.log"
        text = run_cmd(cmd, log, base_env(budget))
        record_run(stored, args.rows, text, "anneal", revision, host, args.summary, head, base)

    for n in SIZES:
        last = 2024 if n in MOVED else 2012
        for seed in range(2000, last):
            for kind, binary, revision, _examples in arms:
                one(kind, revision, f"lj{n}", seed, BUDGET[n], [str(binary), "run", f"lj{n}", str(seed)])
    for seed in range(2000, 2012):
        for kind, _binary, revision, examples in arms:
            one(kind, revision, "au55", seed, None, [str(examples / "h2h_metal"), "run", "au55", str(seed)])
    for seed in range(2000, 2012):
        for kind, _binary, revision, examples in arms:
            one(
                kind,
                revision,
                "m80r6",
                seed,
                4_000_000,
                [str(examples / "h2h_morse"), "run", "m80r6", str(seed)],
            )
    pin_dir = args.pins
    pin_dir.mkdir(parents=True, exist_ok=True)
    pins_script = Path(__file__).resolve().parents[1] / "covalent_h2h" / "pins.sh"
    for kind, _binary, revision, examples in arms:
        dest = pin_dir / f"pins-{kind}-{revision[:12]}.txt"
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"skip pins {kind}", flush=True)
            continue
        print(f"start pins {kind}", flush=True)
        env = base_env()
        with dest.open("w") as handle:
            proc = subprocess.run(
                ["taskset", "-c", "0", "bash", str(pins_script), str(examples)],
                check=False,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"pins {kind} exited {proc.returncode}")
    write_summary(args.summary, stored, head, base, host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
