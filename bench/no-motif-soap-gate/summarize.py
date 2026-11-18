#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize the no-motif SOAP A/B gate rows."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

ALPHA = 0.05

# Declared gate rules over summary_rows:
# 1. LOSE if any system has a statistically significant regression (p < ALPHA)
#    under any criterion in GATE_LOSS_CRITERIA.
# 2. INCOMPLETE if no criterion triggers and at least one declared system has
#    status != "ok" (or no summary rows exist).
# 3. PASS if all declared systems have status == "ok" and no criterion triggers.
GATE_LOSS_CRITERIA = (
    ("lose_hit_sig", "fisher_p", "Fisher", "hits"),
    ("lose_call_sig", "mannwhitney_p_calls", "Mann-Whitney", "calls to hit"),
    ("lose_outside_sig", "mannwhitney_p_outside_s", "Mann-Whitney", "time outside the potential"),
    ("lose_wall_sig", "mannwhitney_p_wall_s", "Mann-Whitney", "wall time"),
)

VERDICT_PASS = "PASS"
VERDICT_LOSE = "LOSE"
VERDICT_INCOMPLETE = "INCOMPLETE"

SUMMARY_FIELDS = [
    "panel",
    "system",
    "n_expected",
    "n_a",
    "n_b",
    "hits_a",
    "hits_b",
    "hits_a_over_n",
    "hits_b_over_n",
    "fisher_p",
    "median_calls_a",
    "median_calls_b",
    "mannwhitney_u_calls",
    "mannwhitney_p_calls",
    "median_pot_s_a",
    "median_pot_s_b",
    "median_outside_s_a",
    "median_outside_s_b",
    "mannwhitney_u_outside_s",
    "mannwhitney_p_outside_s",
    "median_wall_s_a",
    "median_wall_s_b",
    "mannwhitney_u_wall_s",
    "mannwhitney_p_wall_s",
    "bit_identical",
    "bit_identical_over_n",
    "hit_drop",
    "call_increase",
    "outside_increase",
    "wall_increase",
    "lose_hit_sig",
    "lose_call_sig",
    "lose_outside_sig",
    "lose_wall_sig",
    "status",
]

IDENTITY_FIELDS = [
    "panel",
    "system",
    "seed",
    "hit_a",
    "hit_b",
    "evals_a",
    "evals_b",
    "charged_a",
    "charged_b",
    "best_a",
    "best_b",
    "bit_identical",
]

DELTA_FIELDS = [
    "panel",
    "system",
    "hit_drop",
    "call_increase",
    "outside_increase",
    "wall_increase",
    "fisher_p",
    "mannwhitney_p_calls",
    "mannwhitney_p_outside_s",
    "mannwhitney_p_wall_s",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total_hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, total_hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, total_hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, total_hits - n_b), min(n_a, total_hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    n1, n2 = len(left), len(right)
    if n1 < 1 or n2 < 1:
        raise ValueError("mann-whitney needs two non-empty samples")
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


def bit_identical(a: dict[str, str], b: dict[str, str]) -> bool:
    return (
        a["hit"] == b["hit"]
        and a["evals_to_hit"] == b["evals_to_hit"]
        and a["charged"] == b["charged"]
        and a["best"] == b["best"]
    )


def format_float(value: float) -> str:
    return f"{value:.6g}"


def unsupported_reason(
    panel: str, system: str, unsupported_map: dict[tuple[str, str], str]
) -> str | None:
    return unsupported_map.get((panel, system)) or unsupported_map.get((panel, "all"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--declared", type=Path)
    parser.add_argument("--superseded", type=Path)
    parser.add_argument("--unsupported", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--deltas", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = read_csv(args.rows)
    declared_rows = read_csv(args.declared) if args.declared else []
    superseded_rows = read_csv(args.superseded) if args.superseded else []
    declared_map = {(row["panel"], row["system"]): int(row["n"]) for row in declared_rows}
    declared_reason_map = {(row["panel"], row["system"]): row.get("reason", "") for row in declared_rows}
    superseded_cutoff: dict[tuple[str, str], str] = {}
    for row in superseded_rows:
        superseded_cutoff[(row["panel"], row["system"])] = row["restart_session"]
    unsupported = read_csv(args.unsupported)
    unsupported_map = {(row["panel"], row["system"]): row.get("reason", "") for row in unsupported}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("panel", ""), row.get("system", ""))
        cutoff = superseded_cutoff.get(key)
        if cutoff and row.get("session", "") < cutoff:
            continue
        grouped.setdefault((row["panel"], row["system"]), []).append(row)
    expected_keys = set(grouped) | set(declared_map)

    summary_rows: list[dict[str, str]] = []
    identity_rows: list[dict[str, str]] = []
    delta_rows: list[dict[str, str]] = []
    partial_notes: dict[tuple[str, str], list[str]] = {}
    max_partial_outside: tuple[str, str, int, float, float] | None = None

    for panel, system in sorted(expected_keys):
        items = grouped.get((panel, system), [])
        expected_n = declared_map.get((panel, system), 0)
        by_arm = {"A": {}, "B": {}}
        for row in items:
            arm = row["arm"]
            if arm not in by_arm:
                continue
            by_arm[arm][int(row["seed"])] = row
        seeds_a = sorted(by_arm["A"])
        seeds_b = sorted(by_arm["B"])
        common = sorted(set(seeds_a) & set(seeds_b))
        a_rows = [by_arm["A"][seed] for seed in seeds_a]
        b_rows = [by_arm["B"][seed] for seed in seeds_b]

        hits_a = sum(row["hit"] == "1" for row in a_rows)
        hits_b = sum(row["hit"] == "1" for row in b_rows)
        calls_a = [float(row["evals_to_hit"]) for row in a_rows]
        calls_b = [float(row["evals_to_hit"]) for row in b_rows]
        pot_a = [float(row["pot_s"]) for row in a_rows]
        pot_b = [float(row["pot_s"]) for row in b_rows]
        out_a = [float(row["outside_s"]) for row in a_rows]
        out_b = [float(row["outside_s"]) for row in b_rows]
        wall_a = [float(row["wall_s"]) for row in a_rows]
        wall_b = [float(row["wall_s"]) for row in b_rows]

        same = 0
        for seed in common:
            a = by_arm["A"][seed]
            b = by_arm["B"][seed]
            identical = bit_identical(a, b)
            same += int(identical)
            identity_rows.append(
                {
                    "panel": panel,
                    "system": system,
                    "seed": str(seed),
                    "hit_a": a["hit"],
                    "hit_b": b["hit"],
                    "evals_a": a["evals_to_hit"],
                    "evals_b": b["evals_to_hit"],
                    "charged_a": a["charged"],
                    "charged_b": b["charged"],
                    "best_a": a["best"],
                    "best_b": b["best"],
                    "bit_identical": "1" if identical else "0",
                }
            )

        status = "ok"
        if not a_rows or not b_rows or seeds_a != seeds_b or (
            expected_n and (len(a_rows) != expected_n or len(b_rows) != expected_n)
        ):
            status = f"incomplete:A={len(a_rows)}:B={len(b_rows)}:expect={expected_n}"

        fisher_p = fisher(hits_b, len(b_rows), hits_a, len(a_rows)) if a_rows and b_rows else 1.0
        u_calls, p_calls = mann_whitney(calls_b, calls_a) if a_rows and b_rows else (0.0, 1.0)
        u_outside, p_outside = mann_whitney(out_b, out_a) if a_rows and b_rows else (0.0, 1.0)
        u_wall, p_wall = mann_whitney(wall_b, wall_a) if a_rows and b_rows else (0.0, 1.0)
        med_a = statistics.median(calls_a) if calls_a else float("nan")
        med_b = statistics.median(calls_b) if calls_b else float("nan")
        med_out_a = statistics.median(out_a) if out_a else float("nan")
        med_out_b = statistics.median(out_b) if out_b else float("nan")
        med_wall_a = statistics.median(wall_a) if wall_a else float("nan")
        med_wall_b = statistics.median(wall_b) if wall_b else float("nan")
        hit_drop = hits_b < hits_a
        call_increase = med_b > med_a
        outside_increase = med_out_b > med_out_a
        wall_increase = med_wall_b > med_wall_a
        lose_hit_sig = hit_drop and fisher_p < ALPHA
        lose_call_sig = call_increase and p_calls < ALPHA
        lose_outside_sig = outside_increase and p_outside < ALPHA
        lose_wall_sig = wall_increase and p_wall < ALPHA
        if status != "ok" and (a_rows or b_rows):
            notes: list[str] = []
            for seed in sorted(set(by_arm["A"]) | set(by_arm["B"])):
                row_a = by_arm["A"].get(seed)
                row_b = by_arm["B"].get(seed)
                out_a_s = row_a["outside_s"] if row_a else "NA"
                out_b_s = row_b["outside_s"] if row_b else "NA"
                wall_a_s = row_a["wall_s"] if row_a else "NA"
                wall_b_s = row_b["wall_s"] if row_b else "NA"
                notes.append(
                    f"seed {seed}: outside_s A={out_a_s} B={out_b_s}; "
                    f"wall_s A={wall_a_s} B={wall_b_s}"
                )
                if row_a and row_b and unsupported_reason(panel, system, unsupported_map) is None:
                    out_a_val = float(row_a["outside_s"])
                    out_b_val = float(row_b["outside_s"])
                    if max_partial_outside is None or out_b_val > max_partial_outside[4]:
                        max_partial_outside = (panel, system, seed, out_a_val, out_b_val)
            partial_notes[(panel, system)] = notes

        summary_rows.append(
            {
                "panel": panel,
                "system": system,
                "n_expected": str(expected_n),
                "n_a": str(len(a_rows)),
                "n_b": str(len(b_rows)),
                "hits_a": str(hits_a),
                "hits_b": str(hits_b),
                "hits_a_over_n": f"{hits_a}/{len(a_rows)}" if a_rows else "",
                "hits_b_over_n": f"{hits_b}/{len(b_rows)}" if b_rows else "",
                "fisher_p": format_float(fisher_p),
                "median_calls_a": format_float(med_a) if a_rows else "",
                "median_calls_b": format_float(med_b) if b_rows else "",
                "mannwhitney_u_calls": format_float(u_calls),
                "mannwhitney_p_calls": format_float(p_calls),
                "median_pot_s_a": format_float(statistics.median(pot_a)) if pot_a else "",
                "median_pot_s_b": format_float(statistics.median(pot_b)) if pot_b else "",
                "median_outside_s_a": format_float(med_out_a) if out_a else "",
                "median_outside_s_b": format_float(med_out_b) if out_b else "",
                "mannwhitney_u_outside_s": format_float(u_outside),
                "mannwhitney_p_outside_s": format_float(p_outside),
                "median_wall_s_a": format_float(med_wall_a) if wall_a else "",
                "median_wall_s_b": format_float(med_wall_b) if wall_b else "",
                "mannwhitney_u_wall_s": format_float(u_wall),
                "mannwhitney_p_wall_s": format_float(p_wall),
                "bit_identical": str(same),
                "bit_identical_over_n": f"{same}/{len(common)}" if common else "",
                "hit_drop": "1" if hit_drop else "0",
                "call_increase": "1" if call_increase else "0",
                "outside_increase": "1" if outside_increase else "0",
                "wall_increase": "1" if wall_increase else "0",
                "lose_hit_sig": "1" if lose_hit_sig else "0",
                "lose_call_sig": "1" if lose_call_sig else "0",
                "lose_outside_sig": "1" if lose_outside_sig else "0",
                "lose_wall_sig": "1" if lose_wall_sig else "0",
                "status": status,
            }
        )

        if hit_drop or call_increase or outside_increase or wall_increase:
            delta_rows.append(
                {
                    "panel": panel,
                    "system": system,
                    "hit_drop": "1" if hit_drop else "0",
                    "call_increase": "1" if call_increase else "0",
                    "outside_increase": "1" if outside_increase else "0",
                    "wall_increase": "1" if wall_increase else "0",
                    "fisher_p": format_float(fisher_p),
                    "mannwhitney_p_calls": format_float(p_calls),
                    "mannwhitney_p_outside_s": format_float(p_outside),
                    "mannwhitney_p_wall_s": format_float(p_wall),
                }
            )

    write_csv(args.summary, SUMMARY_FIELDS, summary_rows)
    write_csv(args.identity, IDENTITY_FIELDS, identity_rows)
    write_csv(args.deltas, DELTA_FIELDS, delta_rows)

    triggered: list[tuple[str, str, str, str, list[dict[str, str]]]] = []
    for flag, p_field, test_name, label in GATE_LOSS_CRITERIA:
        matches = [row for row in summary_rows if row[flag] == "1"]
        if matches:
            triggered.append((flag, p_field, test_name, label, matches))

    completed = [row for row in summary_rows if row["status"] == "ok"]
    not_run = [row for row in summary_rows if row["status"] != "ok"]

    if triggered:
        parts: list[str] = []
        for _flag, p_field, test_name, label, matches in triggered:
            sys_strs: list[str] = []
            for row in matches:
                n_str = row["n_a"] if row["n_a"] == row["n_b"] else f"A={row['n_a']},B={row['n_b']}"
                sys_strs.append(f"{row['panel']}/{row['system']} n={n_str} {test_name} p={row[p_field]}")
            parts.append(f"{label} ({', '.join(sys_strs)})")
        lines = [f"verdict: {VERDICT_LOSE}: B loses on {'; '.join(parts)}"]
    elif not summary_rows or not_run:
        lines = [
            f"verdict: {VERDICT_INCOMPLETE}: no gate loss rule triggered (alpha={ALPHA}), "
            f"{len(not_run)} of {len(summary_rows)} declared systems incomplete"
        ]
    else:
        lines = [
            f"verdict: {VERDICT_PASS}: all {len(completed)} declared systems completed "
            f"with no gate loss at alpha={ALPHA}"
        ]

    if triggered:
        lines.append(f"gate loss triggers (alpha={ALPHA}):")
        for flag, p_field, test_name, label, matches in triggered:
            for row in matches:
                lines.append(
                    f"- {row['panel']}/{row['system']}: {label} ({flag}=1, "
                    f"{test_name} p={row[p_field]}, n_a={row['n_a']}, n_b={row['n_b']})"
                )

    if partial_notes:
        lines.append("partial runs noted:")
        for (panel, system), notes in sorted(partial_notes.items()):
            for note in notes:
                lines.append(f"- {panel}/{system} {note}")

    lines.append("completed systems:")
    for row in completed:
        lines.append(f"- {row['panel']}/{row['system']}: n={row['n_a']} and n={row['n_b']}")

    never_run_unsupported: list[tuple[dict[str, str], str]] = []
    skipped_for_cost: list[tuple[dict[str, str], str]] = []
    outside_losers = [
        f"{row['panel']}/{row['system']}"
        for row in summary_rows
        if row["lose_outside_sig"] == "1" or row["lose_wall_sig"] == "1"
    ]
    for row in not_run:
        key = (row["panel"], row["system"])
        u_reason = unsupported_reason(row["panel"], row["system"], unsupported_map)
        if u_reason is not None:
            never_run_unsupported.append((row, u_reason))
            continue
        d_reason = declared_reason_map.get(key, "")
        if key in partial_notes:
            cost_note = f"stopped for outside-potential cost ({'; '.join(partial_notes[key])})"
        else:
            evidence: list[str] = []
            if outside_losers:
                evidence.append(f"gate loss on {', '.join(outside_losers)}")
            if max_partial_outside is not None:
                p_panel, p_sys, p_seed, p_a, p_b = max_partial_outside
                evidence.append(f"{p_panel}/{p_sys} seed {p_seed} outside_s B={p_b:.6f} vs A={p_a:.6f}")
            if evidence:
                cost_note = f"skipped for outside-potential cost ({'; '.join(evidence)})"
            else:
                cost_note = "not yet run"
        reason = f"{d_reason}; {cost_note}" if d_reason else cost_note
        skipped_for_cost.append((row, reason))

    if never_run_unsupported:
        lines.append("not run (never run; unsupported):")
        for row, reason in never_run_unsupported:
            lines.append(
                f"- {row['panel']}/{row['system']}: A={row['n_a']} B={row['n_b']} declared={row['n_expected']} — {reason}"
            )
    if skipped_for_cost:
        lines.append("not run (skipped or stopped for outside-potential cost):")
        for row, reason in skipped_for_cost:
            lines.append(
                f"- {row['panel']}/{row['system']}: A={row['n_a']} B={row['n_b']} declared={row['n_expected']} — {reason}"
            )

    if delta_rows:
        lines.append("systems with any hit drop, call increase, or time increase:")
        lines.extend(
            "- "
            + f"{row['panel']}/{row['system']} hit_drop={row['hit_drop']} "
            + f"call_increase={row['call_increase']} outside_increase={row['outside_increase']} "
            + f"wall_increase={row['wall_increase']}"
            for row in delta_rows
        )
    if unsupported:
        lines.append("unsupported systems:")
        lines.extend(f"- {row['panel']}/{row['system']}: {row['reason']}" for row in unsupported)
    if superseded_rows:
        lines.append("superseded systems:")
        lines.extend(
            f"- {row['panel']}/{row['system']}: restart_session={row['restart_session']} ({row['reason']})"
            for row in superseded_rows
        )
    args.verdict.parent.mkdir(parents=True, exist_ok=True)
    args.verdict.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
