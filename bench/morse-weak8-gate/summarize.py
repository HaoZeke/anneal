#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarise the weak8 paired gate and write the verdict from the rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

from protocol import (
    ARMS,
    ARM_LABEL,
    CELL_ORDER,
    CELLS,
    FRESH_LEDGER_RULE,
    FLOAT_FIELDS,
    INT_FIELDS,
    MASK64,
    ProtocolError,
    aa_ratio_band,
    fisher_two_sided,
    mann_whitney_two_sided,
    median,
    median_ci,
    slot_arm_counts,
)

BOOTSTRAP_SEED = 20261119


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_sessions(path):
    with open(path, newline="") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write(text)


def load_rows(path):
    rows = read_csv(path)
    for row in rows:
        for key in INT_FIELDS:
            row[key] = int(row[key])
        for key in FLOAT_FIELDS:
            row[key] = float(row[key])
    return rows


def group_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["cell"], row["arm"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["seed"])
    return groups


def arm_numbers(rows, seed):
    hit_calls = [row["evals_to_hit"] for row in rows if row["hit"]]
    return {
        "n": len(rows),
        "hits": sum(row["hit"] for row in rows),
        "hit_calls": median_ci(hit_calls, seed),
        "total_calls": sum(row["charged"] for row in rows),
        "wall_median_s": median([row["wall_s"] for row in rows]),
    }


def cell_numbers(cell, groups):
    by_arm = {arm: groups[(cell, arm)] for arm in ARMS}
    seed = (BOOTSTRAP_SEED * 1_000_003 + zlib.crc32(cell.encode())) & MASK64
    arms = {
        arm: arm_numbers(by_arm[arm], seed + index * 100_003)
        for index, arm in enumerate(ARMS)
    }
    calls = {
        arm: [row["evals_to_hit"] for row in by_arm[arm] if row["hit"]]
        for arm in ARMS
    }
    n = arms["A"]["n"]
    fisher_ba = fisher_two_sided(arms["A"]["hits"], n, arms["B"]["hits"], n)
    fisher_apa = fisher_two_sided(arms["A"]["hits"], n, arms["AP"]["hits"], n)
    mw_ba = mann_whitney_two_sided(calls["A"], calls["B"])
    mw_apa = mann_whitney_two_sided(calls["A"], calls["AP"])
    band = aa_ratio_band(calls["A"], calls["AP"], seed + 700_021)
    median_a = arms["A"]["hit_calls"]["median"] if arms["A"]["hit_calls"] else None
    median_b = arms["B"]["hit_calls"]["median"] if arms["B"]["hit_calls"] else None
    ratio_ba = (
        median_b / median_a
        if median_a is not None and median_b is not None and median_a > 0
        else None
    )
    by_seed_b = {row["seed"]: row for row in by_arm["B"]}
    paired = []
    for left in by_arm["A"]:
        right = by_seed_b[left["seed"]]
        energy = left["best_bits"] == right["best_bits"]
        charged = left["charged"] == right["charged"]
        to_hit = left["evals_to_hit"] == right["evals_to_hit"]
        paired.append(
            {
                "seed": left["seed"],
                "energy_bits_identical": energy,
                "charged_calls_identical": charged,
                "calls_to_hit_identical": to_hit,
                "identical": energy and charged and to_hit,
                "a_best_bits": left["best_bits"],
                "b_best_bits": right["best_bits"],
                "a_charged": left["charged"],
                "b_charged": right["charged"],
                "a_evals_to_hit": left["evals_to_hit"],
                "b_evals_to_hit": right["evals_to_hit"],
                "a_ledger": left["ledger"],
                "b_ledger": right["ledger"],
                "a_replicas": left["replicas"],
                "b_replicas": right["replicas"],
            }
        )
    identical = [item["identical"] for item in paired]
    return {
        "display": CELLS[cell][0],
        "arms": arms,
        "fisher_p_b_vs_a": fisher_ba,
        "fisher_p_ap_vs_a": fisher_apa,
        "mann_whitney_p_hit_calls_b_vs_a": mw_ba,
        "mann_whitney_p_hit_calls_ap_vs_a": mw_apa,
        "aa_bootstrap_95_band_hit_call_median_ratio_ap_over_a": band,
        "hit_call_median_ratio_b_over_a": ratio_ba,
        "ab_final_energy_and_call_counts_identical": {
            "count": sum(identical),
            "n": len(identical),
            "all": all(identical),
        },
        "per_seed": paired,
    }


def fresh_ledger_cells(rows):
    """Cells whose every arm and seed ran on a fresh single-replica ledger."""
    found = []
    for cell in CELL_ORDER:
        group = [row for row in rows if row["cell"] == cell]
        if group and all(
            row["ledger"] == "fresh" and row["replicas"] == 1 for row in group
        ):
            found.append(cell)
    return found


def data_checks(rows, sessions, groups, cells, rounds):
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    starts = [record for record in sessions if record["record"] == "start"]
    ends = {record["session"] for record in sessions if record["record"] == "end"}
    aborts = [record for record in sessions if record["record"] == "abort"]
    latest = starts[-1]["session"] if starts else ""
    check(
        "latest session ended without abort",
        bool(starts)
        and latest in ends
        and all(record["session"] != latest for record in aborts),
        f"{len(starts)} start, {len(ends)} end, {len(aborts)} abort",
    )
    expected = len(cells) * rounds * len(ARMS)
    check("complete schedule", len(rows) == expected, f"{len(rows)} of {expected} rows")
    check(
        "one row per sequence",
        [row["sequence"] for row in rows] == list(range(len(rows))),
    )
    check(
        "same run count in every cell and arm",
        all(len(groups.get((cell, arm), [])) == rounds for cell in cells for arm in ARMS),
    )
    check(
        "same seeds and starts in every arm",
        all(
            {(row["seed"], row["start_sha256"]) for row in groups.get((cell, "A"), [])}
            == {(row["seed"], row["start_sha256"]) for row in groups.get((cell, arm), [])}
            for cell in cells
            for arm in ("AP", "B")
        ),
    )
    counts = slot_arm_counts(rows)
    balanced = len(set(counts.values())) == 1 and all(counts.values())
    check(
        "balanced slot and arm counts",
        balanced,
        ", ".join(
            f"slot {slot} {ARM_LABEL[arm]}={counts[(slot, arm)]}"
            for slot in range(3)
            for arm in ARMS
        ),
    )
    check(
        "one pinned core and one thread",
        all(
            row["core_pinned"] == row["core_start"] == row["core_end"] == starts[0]["core"]
            and row["cpus_allowed_list"] == str(starts[0]["core"])
            and row["threads"] == 1
            and row["mode"] == "recommended"
            for row in rows
        )
        if starts
        else False,
    )
    check(
        "arm source identities and binaries are stable",
        all(
            len(
                {
                    (row["source_revision"], row["source_tree"], row["binary_sha256"])
                    for row in rows
                    if row["arm"] == arm
                }
            )
            == 1
            for arm in ARMS
        ),
    )
    check(
        "B records the base revision and the patched tree",
        all(
            row["source_revision"] == starts[-1]["base_revision"]
            and row["source_tree"] != starts[-1]["base_tree"]
            for row in rows
            if row["arm"] == "B"
        )
        if starts
        else False,
    )
    check(
        "A and A' are separate release builds of the base tree",
        (
            rows
            and {row["source_revision"] for row in rows if row["arm"] in ("A", "AP")}
            == {starts[-1]["base_revision"]}
            and {row["source_tree"] for row in rows if row["arm"] in ("A", "AP")}
            == {starts[-1]["base_tree"]}
            and len({row["binary_sha256"] for row in rows if row["arm"] == "A"}) == 1
            and len({row["binary_sha256"] for row in rows if row["arm"] == "AP"}) == 1
        )
        if starts
        else False,
    )
    fresh = fresh_ledger_cells(rows)
    check(
        "every cell ran on a fresh ledger",
        fresh == list(cells),
        " ".join(CELLS[cell][0] for cell in fresh) if fresh else "none",
    )
    return checks


def compute_summary(rows, sessions):
    starts = [record for record in sessions if record["record"] == "start"]
    if not starts:
        raise ProtocolError("no start session")
    cells = starts[-1]["cells"]
    rounds = starts[-1]["rounds"]
    groups = group_rows(rows)
    missing = [
        (cell, arm) for cell in cells for arm in ARMS if (cell, arm) not in groups
    ]
    if missing:
        raise ProtocolError("missing cell-arm groups")
    per_cell = {cell: cell_numbers(cell, groups) for cell in cells}
    checks = data_checks(rows, sessions, groups, cells, rounds)
    loss_hits = []
    loss_calls = []
    wins = []
    marginal_losses = []
    any_hit_drop = []
    any_call_increase = []
    bonferroni = 0.05 / len(cells)
    for cell, stats in per_cell.items():
        a = stats["arms"]["A"]
        b = stats["arms"]["B"]
        fisher = stats["fisher_p_b_vs_a"]
        mw = stats["mann_whitney_p_hit_calls_b_vs_a"]
        ratio = stats["hit_call_median_ratio_b_over_a"]
        band = stats["aa_bootstrap_95_band_hit_call_median_ratio_ap_over_a"]
        if b["hits"] < a["hits"]:
            any_hit_drop.append({"cell": cell, "a_hits": a["hits"], "b_hits": b["hits"], "p": fisher})
        if b["hits"] < a["hits"] and fisher < 0.05:
            loss_hits.append({"cell": cell, "p": fisher})
        call_increased = False
        if ratio is not None and ratio > 1.0:
            call_increased = True
        if b["total_calls"] > a["total_calls"]:
            call_increased = True
        if call_increased:
            any_call_increase.append(
                {
                    "cell": cell,
                    "ratio": ratio,
                    "p": mw,
                    "a_total_calls": a["total_calls"],
                    "b_total_calls": b["total_calls"],
                }
            )
        if (
            ratio is not None
            and band is not None
            and ratio > 1.0
            and ratio > band[1]
            and mw is not None
            and mw < 0.05
        ):
            loss_calls.append({"cell": cell, "p": mw, "ratio": ratio, "aa_band": band})
        if b["hits"] > a["hits"] and fisher < bonferroni:
            wins.append(
                {
                    "cell": cell,
                    "metric": "hits",
                    "p": fisher,
                    "bonferroni_p": min(1.0, fisher * len(cells)),
                }
            )
        if ratio is not None and ratio < 1.0 and mw is not None and mw < bonferroni:
            wins.append(
                {
                    "cell": cell,
                    "metric": "hit calls",
                    "p": mw,
                    "bonferroni_p": min(1.0, mw * len(cells)),
                }
            )
        if b["hits"] < a["hits"] and 0.05 <= fisher < 0.1:
            marginal_losses.append({"cell": cell, "metric": "hits", "p": fisher})
        if ratio is not None and ratio > 1.0 and mw is not None and 0.05 <= mw < 0.1:
            marginal_losses.append(
                {
                    "cell": cell,
                    "metric": "hit calls",
                    "p": mw,
                    "outside_aa_band": band is not None and ratio > band[1],
                }
            )
    comparable = all(
        item["pass"]
        for item in checks
        if item["name"]
        in {
            "latest session ended without abort",
            "complete schedule",
            "one row per sequence",
            "same run count in every cell and arm",
            "same seeds and starts in every arm",
            "one pinned core and one thread",
            "arm source identities and binaries are stable",
        }
    )
    verdict = "PASS" if comparable and not loss_hits and not loss_calls else "FAIL"
    fresh = fresh_ledger_cells(rows)
    contradictions = []
    for cell in fresh:
        for seed in per_cell[cell]["per_seed"]:
            if not seed["identical"]:
                contradictions.append({"cell": cell, "display": CELLS[cell][0], **seed})
    resumed_rows = sum(start.get("resumed_rows", 0) for start in starts)
    counts = slot_arm_counts(rows)
    return {
        "verdict": verdict,
        "rule": (
            "PASS only when the paired runs are complete and B has no cell with "
            "fewer hits at two-sided Fisher p<0.05 and no cell with a higher "
            "hit-call median at two-sided Mann-Whitney p<0.05 above the A/A' "
            "bootstrap band"
        ),
        "fresh_ledger_rule": FRESH_LEDGER_RULE,
        "fresh_ledger_cells": fresh,
        "fresh_ledger_contradictions": contradictions,
        "fresh_ledger_claim": "CONTRADICTED" if contradictions else "HOLDS",
        "cells": per_cell,
        "slot_arm_counts": {
            f"{slot}:{arm}": counts[(slot, arm)] for slot in range(3) for arm in ARMS
        },
        "gate_losses": {"hits": loss_hits, "hit_calls": loss_calls},
        "wins_bonferroni_8_cells": wins,
        "marginal_losses_0.05_to_0.1": marginal_losses,
        "any_hit_drop": any_hit_drop,
        "any_call_increase": any_call_increase,
        "bonferroni_threshold": bonferroni,
        "checks": checks,
        "resume": {
            "resumed": resumed_rows > 0,
            "resumed_rows": resumed_rows,
            "session_count": len(starts),
        },
    }


def fmt_p(value):
    if value is None:
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def fmt_int(value):
    return f"{value:.0f}"


def fmt_calls(summary):
    if summary is None:
        return "none"
    lo, hi = summary["bootstrap_95_ci"]
    return f"{fmt_int(summary['median'])} [{fmt_int(lo)}, {fmt_int(hi)}] (n={summary['n']})"


def md_table(header, body):
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in body)
    return lines


def json_or_none(value):
    return json.dumps(value, sort_keys=True) if value else "none"


def render_report(summary, sessions, hosts):
    starts = [record for record in sessions if record["record"] == "start"]
    last = starts[-1]
    host = next(row for row in hosts if row["host_id"] == last["host_id"])
    lines = [
        "# Morse weak8 paired gate",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        summary["rule"] + ".",
        "",
        f"Fresh-ledger claim: **{summary['fresh_ledger_claim']}**.",
        "",
        summary["fresh_ledger_rule"],
        "",
        "- A and A' are separate release builds of the base tree. B is a release build of the patched tree, recorded as the base revision plus that tree.",
        "- All arms use Config::recommended, the paper budgets, and the same 12 seeds and starts per cell.",
        "- Arm order is a balanced flip: rotations of A, A', B, then those orders reversed, repeated across the twelve seeds.",
        "- Runs are serial, one thread, pinned to one core, with no warm-up, retry, replacement, quick mode, or reduced budget.",
        "- Calls-to-hit statistics include hitting runs only. Confidence intervals are 95% percentile bootstrap intervals with 20000 fixed-stream draws.",
        "- The calls gate compares B/A's hit-call median ratio with the 95% bootstrap band of A'/A.",
        f"- Resume: {'yes' if summary['resume']['resumed'] else 'no'}"
        + (
            f" ({summary['resume']['resumed_rows']} rows)"
            if summary["resume"]["resumed"]
            else ""
        )
        + ".",
        "",
        "## Provenance",
        "",
    ]
    provenance = [
        ("base revision", last["base_revision"]),
        ("base tree", last["base_tree"]),
        ("harness revision", last["harness_revision"]),
        ("harness tree", last["harness_tree"]),
        ("host id", last["host_id"]),
        ("cpu", host["cpu_model"]),
        ("logical cpus", host["logical_cpus"]),
        ("memory kB", host["mem_total_kb"]),
        ("kernel", host["kernel_release"]),
        ("os", host["os_release"]),
        ("arch", host["arch"]),
        ("rustc", last["rustc"]),
        ("cargo", last["cargo"]),
        ("python", last["python"]),
        ("taskset", last["taskset"]),
        ("pinned core", str(last["core"])),
        (
            "run environment",
            " ".join(f"{key}={value}" for key, value in sorted(last["run_environment"].items())),
        ),
    ]
    lines += md_table(["field", "value"], [[key, f"`{value}`"] for key, value in provenance])
    lines += ["", "Arm builds:", ""]
    build_rows = []
    for arm in ARMS:
        identity = last["arms"][arm]
        build_rows.append(
            [
                ARM_LABEL[arm],
                identity["source_revision"],
                identity["source_tree"],
                identity["binary_sha256"],
                "empty" if not identity["git_status_porcelain_before_build"] else "nonempty",
                "empty" if not identity["git_status_porcelain_after_build"] else "nonempty",
            ]
        )
    lines += md_table(
        [
            "arm",
            "source revision",
            "source tree",
            "binary sha256",
            "porcelain before",
            "porcelain after",
        ],
        build_rows,
    )
    lines += ["", "## Order balance", ""]
    counts = summary["slot_arm_counts"]
    lines += md_table(
        ["slot", "A", "A'", "B"],
        [
            [
                str(slot),
                str(counts[f"{slot}:A"]),
                str(counts[f"{slot}:AP"]),
                str(counts[f"{slot}:B"]),
            ]
            for slot in range(3)
        ],
    )
    lines += ["", "## Per-cell results", ""]
    body = []
    for _cell, stats in summary["cells"].items():
        arms = stats["arms"]
        body.append(
            [
                stats["display"],
                f"{arms['A']['hits']}/{arms['A']['n']}",
                f"{arms['AP']['hits']}/{arms['AP']['n']}",
                f"{arms['B']['hits']}/{arms['B']['n']}",
                fmt_p(stats["fisher_p_b_vs_a"]),
                fmt_p(stats["fisher_p_ap_vs_a"]),
                fmt_calls(arms["A"]["hit_calls"]),
                fmt_calls(arms["AP"]["hit_calls"]),
                fmt_calls(arms["B"]["hit_calls"]),
                fmt_p(stats["mann_whitney_p_hit_calls_b_vs_a"]),
                fmt_p(stats["mann_whitney_p_hit_calls_ap_vs_a"]),
            ]
        )
    lines += md_table(
        [
            "cell",
            "A hits",
            "A' hits",
            "B hits",
            "Fisher B/A",
            "Fisher A'/A",
            "A hit calls median [boot CI]",
            "A' hit calls median [boot CI]",
            "B hit calls median [boot CI]",
            "MW B/A",
            "MW A'/A",
        ],
        body,
    )
    lines += ["", "Totals, wall medians, A/A' bands, and bit identity:", ""]
    body = []
    for _cell, stats in summary["cells"].items():
        arms = stats["arms"]
        band = stats["aa_bootstrap_95_band_hit_call_median_ratio_ap_over_a"]
        identity = stats["ab_final_energy_and_call_counts_identical"]
        body.append(
            [
                stats["display"],
                f"{arms['A']['total_calls']} / {arms['AP']['total_calls']} / {arms['B']['total_calls']}",
                f"{arms['A']['wall_median_s']:.2f} / {arms['AP']['wall_median_s']:.2f} / {arms['B']['wall_median_s']:.2f}",
                f"{stats['hit_call_median_ratio_b_over_a']:.3f}"
                if stats["hit_call_median_ratio_b_over_a"] is not None
                else "n/a",
                f"[{band[0]:.3f}, {band[1]:.3f}]" if band else "n/a",
                f"{identity['count']}/{identity['n']}",
            ]
        )
    lines += md_table(
        [
            "cell",
            "total charged calls A / A' / B",
            "wall median s A / A' / B",
            "hit-call median B/A",
            "A/A' bootstrap band",
            "A/B energy bits and call counts identical",
        ],
        body,
    )
    lines += [
        "",
        "## Gate decision",
        "",
        "Significant hit losses: " + json_or_none(summary["gate_losses"]["hits"]) + ".",
        "",
        "Significant hit-call losses outside the A/A' band: "
        + json_or_none(summary["gate_losses"]["hit_calls"])
        + ".",
        "",
        "Bonferroni wins over eight cells: " + json_or_none(summary["wins_bonferroni_8_cells"]) + ".",
        "",
        "Marginal losses (0.05 <= p < 0.1): "
        + json_or_none(summary["marginal_losses_0.05_to_0.1"])
        + ".",
        "",
        "Any hit drop, including non-significant: " + json_or_none(summary["any_hit_drop"]) + ".",
        "",
        "Any call increase, including non-significant: "
        + json_or_none(summary["any_call_increase"])
        + ".",
        "",
        "## Fresh-ledger bit identity",
        "",
        "Per seed, A against B, for every cell that ran on a fresh ledger. "
        "A seed is identical when the final-energy bits, the charged call count, "
        "and the calls to hit all match.",
        "",
    ]
    fresh_body = []
    for cell in summary["fresh_ledger_cells"]:
        stats = summary["cells"][cell]
        for seed in stats["per_seed"]:
            fresh_body.append(
                [
                    stats["display"],
                    str(seed["seed"]),
                    "yes" if seed["energy_bits_identical"] else "NO",
                    "yes" if seed["charged_calls_identical"] else "NO",
                    "yes" if seed["calls_to_hit_identical"] else "NO",
                    "yes" if seed["identical"] else "NO",
                ]
            )
    if fresh_body:
        lines += md_table(
            [
                "cell",
                "seed",
                "energy bits",
                "charged calls",
                "calls to hit",
                "identical",
            ],
            fresh_body,
        )
    else:
        lines.append("No cell ran on a fresh ledger.")
    lines += [
        "",
        "Fresh-ledger seeds that contradict the unchanged-trajectory claim: "
        + json_or_none(summary["fresh_ledger_contradictions"])
        + ".",
        "",
        "## Checks",
        "",
    ]
    for item in summary["checks"]:
        suffix = f" ({item['detail']})" if item["detail"] else ""
        lines.append(f"- {'PASS' if item['pass'] else 'FAIL'}: {item['name']}{suffix}")
    return "\n".join(lines) + "\n"


def cmd_report(args):
    rows = load_rows(args.rows)
    if not rows:
        raise ProtocolError("no rows")
    sessions = load_sessions(args.sessions)
    hosts = read_csv(args.hosts)
    summary = compute_summary(rows, sessions)
    write_text(args.summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_text(args.out, render_report(summary, sessions, hosts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        cmd_report(args)
    except ProtocolError as err:
        print(f"summarize: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
