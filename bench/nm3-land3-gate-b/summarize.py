#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Score the nm3-land3 part B rows and write the verdict.

A gate system is FIX when B has fewer hits than A, when B is slower to hit
at a two-sided Mann-Whitney p below 0.05, or when the type-7 median of
outside-potential time per charged call has a B/A ratio above 1.1.
Misses rank strictly after every hit. The blj105n38 seed 2101 rows are
reported on their own and are not scored.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from collections import Counter
from pathlib import Path

ALPHA = 0.05
RATIO_LIMIT = 1.1
SC126 = "sc126-55"
REPRO_SEED = "2101"

SUMMARY_FIELDS = [
    "system",
    "budget",
    "n",
    "hits_a",
    "hits_b",
    "fisher_p",
    "calls_median_a",
    "calls_median_b",
    "mw_u",
    "mw_p_calls",
    "calls_direction",
    "wall_median_a",
    "wall_median_b",
    "outside_per_call_median_a",
    "outside_per_call_median_b",
    "outside_per_call_ratio",
    "loss_hits",
    "loss_calls",
    "loss_outside",
    "verdict",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return float("nan")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def tie_groups(ordered: list[float]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[index]:
            end += 1
        groups.append((index, end))
        index = end + 1
    return groups


def fisher_two_sided(hits_a: int, miss_a: int, hits_b: int, miss_b: int) -> float:
    table_n = hits_a + miss_a + hits_b + miss_b
    if table_n == 0:
        return 1.0
    row1 = hits_a + miss_a
    col1 = hits_a + hits_b
    row2 = hits_b + miss_b

    def mass(value: int) -> float:
        return math.comb(row1, value) * math.comb(row2, col1 - value) / math.comb(table_n, col1)

    observed = mass(hits_a)
    total = 0.0
    for value in range(max(0, col1 - row2), min(row1, col1) + 1):
        point = mass(value)
        if point <= observed * (1.0 + 1e-12):
            total += point
    return min(1.0, total)


def mann_whitney(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    """Two-sided Mann-Whitney U with tie and continuity corrections.

    Returns U for x, p, mean rank of x, mean rank of y.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    ordered = sorted(x + y)
    n = len(ordered)
    ranks: dict[float, float] = {}
    tie = 0.0
    for start, end in tie_groups(ordered):
        ranks[ordered[start]] = (start + 1 + end + 1) / 2.0
        size = end - start + 1
        tie += size**3 - size
    r1 = sum(ranks[value] for value in x)
    r2 = sum(ranks[value] for value in y)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    if n < 2:
        return u1, 1.0, r1 / n1, r2 / n2
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if sigma2 <= 0.0:
        return u1, 1.0, r1 / n1, r2 / n2
    z = max(0.0, abs(u1 - n1 * n2 / 2.0) - 0.5) / math.sqrt(sigma2)
    return u1, min(1.0, math.erfc(z / math.sqrt(2.0))), r1 / n1, r2 / n2


def call_score(row: dict[str, str]) -> float:
    calls = float(row["calls"])
    budget = float(row["budget"])
    if row["hit"] == "1":
        return calls
    return budget + 1.0


def outside_per_call(row: dict[str, str]) -> float:
    return float(row["outside_per_call_s"])


def ratio(b: float, a: float) -> float:
    if a > 0.0 and b == b and a == a:
        return b / a
    if a == 0.0 and b == 0.0:
        return 1.0
    return float("inf")


def fmt(value: float, digits: str = ".6g") -> str:
    if value != value:
        return "nan"
    if math.isinf(value):
        return "inf"
    return format(value, digits)


def gate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["cohort"] == "gate" and row["status"] == "ok"]


def repro_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["cohort"] == "reproduction" and row["status"] == "ok"]


def systems_in_order(rows: list[dict[str, str]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row["system"] not in seen:
            seen.append(row["system"])
    return seen


def by_arm(rows: list[dict[str, str]], system: str) -> dict[str, list[dict[str, str]]]:
    grouped = {"A": [], "B": []}
    for row in rows:
        if row["system"] == system:
            grouped[row["arm"]].append(row)
    for arm in grouped:
        grouped[arm].sort(key=lambda row: int(row["seed"]))
    return grouped


def verify_abba(rows: list[dict[str, str]]) -> dict[tuple[int, str], int]:
    overall: Counter[tuple[int, str]] = Counter()
    per_system: dict[str, Counter[tuple[int, str]]] = {}
    for row in rows:
        key = (int(row["slot"]), row["arm"])
        overall[key] += 1
        per_system.setdefault(row["system"], Counter())[key] += 1
    expected = {(1, "A"), (1, "B"), (2, "A"), (2, "B")}
    for system, counts in per_system.items():
        if set(counts) != expected or len(set(counts.values())) != 1:
            raise SystemExit(f"unbalanced slots for {system}: {dict(counts)}")
    if set(overall) != expected or len(set(overall.values())) != 1:
        raise SystemExit(f"unbalanced slots overall: {dict(overall)}")
    return dict(overall)


def verify_objects(repo: Path, rows: list[dict[str, str]]) -> tuple[int, list[str]]:
    objects: set[str] = set()
    for row in rows:
        for field in ("library", "library_tree", "base_commit", "base_tree"):
            value = row[field].strip()
            if value:
                objects.add(value.split(":")[-1])
        for item in row["harness_blobs"].split(";"):
            if ":" in item:
                objects.add(item.rsplit(":", 1)[1].strip())
    missing = []
    for sha in sorted(objects):
        proc = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", sha],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            missing.append(sha)
    return len(objects), missing


def score_system(system: str, rows: list[dict[str, str]]) -> dict[str, str]:
    grouped = by_arm(rows, system)
    hits = {arm: sum(int(row["hit"]) for row in grouped[arm]) for arm in ("A", "B")}
    n = {arm: len(grouped[arm]) for arm in ("A", "B")}
    scores = {arm: [call_score(row) for row in grouped[arm]] for arm in ("A", "B")}
    calls = {arm: [float(row["calls"]) for row in grouped[arm]] for arm in ("A", "B")}
    walls = {arm: [float(row["wall_s"]) for row in grouped[arm]] for arm in ("A", "B")}
    outside = {arm: [outside_per_call(row) for row in grouped[arm]] for arm in ("A", "B")}
    fisher = fisher_two_sided(hits["A"], n["A"] - hits["A"], hits["B"], n["B"] - hits["B"])
    u_stat, p_calls, rank_a, rank_b = mann_whitney(scores["A"], scores["B"])
    med_out_a = type7(outside["A"])
    med_out_b = type7(outside["B"])
    out_ratio = ratio(med_out_b, med_out_a)
    loss_hits = hits["B"] < hits["A"]
    loss_calls = rank_b > rank_a and p_calls < ALPHA
    loss_outside = out_ratio > RATIO_LIMIT
    if rank_b > rank_a:
        direction = "B more calls"
    elif rank_b < rank_a:
        direction = "B fewer calls"
    else:
        direction = "equal"
    verdict = "FIX" if loss_hits or loss_calls or loss_outside else "PASS"
    budget = grouped["A"][0]["budget"]
    return {
        "system": system,
        "budget": budget,
        "n": str(n["A"]),
        "hits_a": str(hits["A"]),
        "hits_b": str(hits["B"]),
        "fisher_p": fmt(fisher, ".6g"),
        "calls_median_a": fmt(type7(calls["A"])),
        "calls_median_b": fmt(type7(calls["B"])),
        "mw_u": fmt(u_stat),
        "mw_p_calls": fmt(p_calls, ".6g"),
        "calls_direction": direction,
        "wall_median_a": fmt(type7(walls["A"]), ".6f"),
        "wall_median_b": fmt(type7(walls["B"]), ".6f"),
        "outside_per_call_median_a": f"{med_out_a:.6e}",
        "outside_per_call_median_b": f"{med_out_b:.6e}",
        "outside_per_call_ratio": fmt(out_ratio, ".4f"),
        "loss_hits": str(int(loss_hits)),
        "loss_calls": str(int(loss_calls)),
        "loss_outside": str(int(loss_outside)),
        "verdict": verdict,
    }


def failing_reasons(row: dict[str, str]) -> list[str]:
    reasons = []
    if row["loss_hits"] == "1":
        reasons.append(f"fewer hits ({row['hits_b']} vs {row['hits_a']})")
    if row["loss_calls"] == "1":
        reasons.append(f"calls loss p={row['mw_p_calls']}")
    if row["loss_outside"] == "1":
        reasons.append(f"outside/call B/A {row['outside_per_call_ratio']}")
    return reasons


def hit_table(system: str, rows: list[dict[str, str]]) -> list[str]:
    grouped = by_arm(rows, system)
    seeds = [row["seed"] for row in grouped["A"]]
    lines = [
        f"### {system}",
        "",
        "| seed | hit A | hit B | calls A | calls B |",
        "| --- | --- | --- | --- | --- |",
    ]
    b_by_seed = {row["seed"]: row for row in grouped["B"]}
    for row in grouped["A"]:
        other = b_by_seed[row["seed"]]
        lines.append(
            f"| {row['seed']} | {row['hit']} | {other['hit']} | {row['calls']} | {other['calls']} |"
        )
    lines.append("")
    if seeds != [other["seed"] for other in grouped["B"]]:
        raise SystemExit(f"{system} seed lists differ")
    return lines


def sc126_table(rows: list[dict[str, str]]) -> list[str]:
    grouped = by_arm(rows, SC126)
    b_by_seed = {row["seed"]: row for row in grouped["B"]}
    lines = [
        "## sc126-55 outside time per call",
        "",
        "| seed | A (s) | B (s) | B/A |",
        "| --- | --- | --- | --- |",
    ]
    for row in grouped["A"]:
        other = b_by_seed[row["seed"]]
        a = outside_per_call(row)
        b = outside_per_call(other)
        lines.append(f"| {row['seed']} | {a:.6e} | {b:.6e} | {fmt(ratio(b, a), '.4f')} |")
    lines.append("")
    return lines


def render(rows: list[dict[str, str]], scored: list[dict[str, str]], balance: dict, objects: int, missing: list[str]) -> str:
    gate = gate_rows(rows)
    repro = repro_rows(rows)
    failing = [row for row in scored if row["verdict"] == "FIX"]
    verdict = "FIX" if failing else "PASS"
    session = sorted({row["session_id"] for row in gate})
    machine = sorted({row["machine_id"] for row in gate})
    base = gate[0]
    lines = [
        "# nm3-land3 no-loss gate, part B",
        "",
        f"**Verdict: {verdict}**",
        "",
        "FIX if B has fewer hits than A on any system, if calls to hit are worse for B at p < 0.05, or if the median outside-potential time per call has B/A above 1.1. Otherwise PASS.",
        "",
        "## Failing cells",
        "",
    ]
    if not failing:
        lines.append("None.")
    else:
        for row in failing:
            lines.append(f"- {row['system']}: " + "; ".join(failing_reasons(row)))
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Base commit `{base['base_commit']}` tree `{base['base_tree']}`",
            f"- Arm A library `{next(r['library'] for r in gate if r['arm']=='A')}` tree `{next(r['library_tree'] for r in gate if r['arm']=='A')}`",
            f"- Arm B library `{next(r['library'] for r in gate if r['arm']=='B')}` tree `{next(r['library_tree'] for r in gate if r['arm']=='B')}`",
            f"- Sessions: {', '.join(session)}",
            f"- Machine id sha256: `{machine[0]}`",
            f"- Core {base['core']}; ABBA slot counts: "
            + ", ".join(f"slot {slot} {arm}={count}" for (slot, arm), count in sorted(balance.items())),
            f"- git cat-file -e resolved {objects - len(missing)}/{objects} recorded ids",
            "- Hit is the driver's full-precision comparison against the reference plus 1e-3. Sutton-Chen best is the eight-decimal printed value, TIP4P best is the ten-decimal printed value, and the binary Lennard-Jones best is the round-trip value together with its bits.",
            "",
            "## Systems",
            "",
            "| system | n | budget | hits A | hits B | Fisher p | calls A | calls B | MW p | direction | wall A | wall B | out/call A | out/call B | B/A | verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in scored:
        lines.append(
            f"| {row['system']} | {row['n']} | {row['budget']} | {row['hits_a']}/{row['n']} | {row['hits_b']}/{row['n']} | {row['fisher_p']} | {row['calls_median_a']} | {row['calls_median_b']} | {row['mw_p_calls']} | {row['calls_direction']} | {row['wall_median_a']} | {row['wall_median_b']} | {row['outside_per_call_median_a']} | {row['outside_per_call_median_b']} | {row['outside_per_call_ratio']} | {row['verdict']} |"
        )
    lines.extend(["", "## Per-seed hits", ""])
    for system in systems_in_order(gate):
        lines.extend(hit_table(system, gate))
    lines.extend(sc126_table(gate))
    lines.extend(
        [
            "## blj105n38 seed 2101 reproduction",
            "",
            "These two rows are not in the gate counts and have no verdict.",
            "",
            "| arm | slot | hit | calls | charged | wall_s | pot_s | outside_s | outside_per_call_s | best | best_bits |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(repro, key=lambda item: (int(item["slot"]), item["arm"])):
        lines.append(
            f"| {row['arm']} | {row['slot']} | {row['hit']} | {row['calls']} | {row['charged']} | {row['wall_s']} | {row['pot_s']} | {row['outside_s']} | {row['outside_per_call_s']} | {row['best']} | {row['best_bits']} |"
        )
    if missing:
        lines.extend(["", "## Unresolved ids", ""])
        for sha in missing:
            lines.append(f"- {sha}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = read_csv(args.rows)
    gate = gate_rows(rows)
    if not gate:
        raise SystemExit("no gate rows")
    if len({row["session_id"] for row in rows}) != 1:
        raise SystemExit("more than one session")
    if any(row["core"] != "0" for row in rows):
        raise SystemExit("a row is not on core 0")
    balance = verify_abba(gate)
    objects, missing = verify_objects(args.repo, rows)
    if missing:
        raise SystemExit(f"unresolved provenance: {missing}")
    scored = [score_system(system, gate) for system in systems_in_order(gate)]
    write_csv(args.summary, SUMMARY_FIELDS, scored)
    text = render(rows, scored, balance, objects, missing)
    args.readme.write_text(text, encoding="utf-8")
    failing = [row["system"] for row in scored if row["verdict"] == "FIX"]
    print("PASS" if not failing else "FIX " + ",".join(failing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
