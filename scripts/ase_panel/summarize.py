#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Reprint the ASE panel from its committed rows.

Compares hits by the two-sided Fisher exact test. Does not start a search.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from paired_stats import fisher_exact  # noqa: E402
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402
from machine_provenance import csv_fields  # noqa: E402
import revision_reach  # noqa: E402

SEEDS = tuple(range(2000, 2012))
ARMS = ("anneal", "recommended", "ase_bh", "ase_mh")
ASE_ARMS = ("ase_bh", "ase_mh")
LABELS = {
    "anneal": "anneal",
    "recommended": "recommended",
    "ase_bh": "basin hopping",
    "ase_mh": "minima hopping",
}
SYSTEMS = {
    "lj38": (38, 400_000, -173.928427, "LJ38"),
    "lj55": (55, 1_000_000, -279.248470, "LJ55"),
    "lj98": (98, 4_000_000, -543.665361, "LJ98"),
    "lj104": (104, 4_000_000, -582.086642, "LJ104"),
    "lj150": (150, 4_000_000, -893.310258, "LJ150"),
    "au55": (55, 1_000_000, -191.03085280, "Au55"),
}
ASE_VERSION = "3.29.0"
HIT_TOL = DEFAULT_HIT_TOL
HEADER = (
    "arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,"
    "median_outside_s,median_over_anneal,median_anneal_over,fisher_p,"
    "wall_stops,wall_cap_s,fisher_p_bh,fisher_p_mh"
)
ROW_FIELDS = (
    "system", "arm", "seed", "budget", "hit", "evals_to_hit", "charged", "best",
    "wall_s", "inside_s", "outside_s", "code_revision", "host", "stop", "cap_s",
)
RERUN_ROW_FIELDS = (
    *ROW_FIELDS,
    "harness_blob", "ase_version", "session",
)
CURRENT_ROW_FIELDS = (*ROW_FIELDS, "machine_id", "cpu_model")
ROWS_FILE = "rows-61fa67cf.csv"


def type7(ordered: list[float], probability: float) -> float:
    count = len(ordered)
    rank = 1.0 + (count - 1) * probability
    lower = math.floor(rank)
    weight = rank - lower
    left = ordered[int(lower) - 1]
    if weight == 0.0 or int(lower) >= count:
        return float(left)
    return float(left * (1.0 - weight) + ordered[int(lower)] * weight)


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def reference_text(value: float) -> str:
    text = f"{value:.8f}"
    return text[:-2] if text.endswith("00") else text


def load(system: str) -> dict[str, list[dict[str, str]]]:
    directory = ROOT / "bench" / f"ase-{system}"
    rerun_path = directory / ROWS_FILE
    is_rerun = rerun_path.exists()
    if is_rerun:
        if b"\r" in rerun_path.read_bytes():
            raise SystemExit(f"{rerun_path} contains CR")
        with rerun_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) not in (
                RERUN_ROW_FIELDS,
                (*RERUN_ROW_FIELDS, "machine_id", "cpu_model"),
            ):
                raise SystemExit(f"{rerun_path}: unexpected columns {reader.fieldnames}")
            rows = list(reader)
    else:
        paths = [directory / "rows.csv"]
        if system == "au55":
            paths.append(directory / "rows-for-cluster.csv")
        rows = []
        for path in paths:
            if b"\r" in path.read_bytes():
                raise SystemExit(f"{path} contains CR")
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) not in (ROW_FIELDS, CURRENT_ROW_FIELDS):
                    raise SystemExit(f"{path}: unexpected columns {reader.fieldnames}")
                for row in reader:
                    if system != "au55" or (
                        (path.name == "rows.csv" and row["arm"] == "recommended")
                        or (path.name == "rows-for-cluster.csv" and row["arm"] != "recommended")
                    ):
                        rows.append(row)
    _, budget, reference, _ = SYSTEMS[system]
    grouped = {arm: [] for arm in ARMS}
    by_arm: dict[str, list[tuple[int, dict[str, str]]]] = {arm: [] for arm in ARMS}
    for row in rows:
        valid_host = (
            bool(re.fullmatch(r"[0-9a-f]{64}", row["host"]))
            if is_rerun
            else (row["host"] == "bench" or bool(re.fullmatch(r"[0-9a-f]{64}", row["host"])))
        )
        if (
            row["system"] != system
            or row["arm"] not in grouped
            or row["budget"] != str(budget)
            or not valid_host
        ):
            raise SystemExit(f"unexpected row {row}")
        if not revision_reach.is_full_sha(row["code_revision"]):
            raise SystemExit(f"invalid revision {row}")
        if is_rerun:
            if not revision_reach.is_full_sha(row["harness_blob"]):
                raise SystemExit(f"invalid harness blob {row}")
            if row["ase_version"] != ASE_VERSION or not re.fullmatch(r"[0-9a-f]+", row["session"]):
                raise SystemExit(f"invalid ase version or session {row}")
        charged = int(row["charged"])
        if not 0 < charged <= budget:
            raise SystemExit(f"invalid charged count {row}")
        hit = row["hit"] == "1"
        if row["hit"] not in {"0", "1"} or hit != bool(row["evals_to_hit"]):
            raise SystemExit(f"hit/calls mismatch {row}")
        if hit:
            evals_to_hit = int(row["evals_to_hit"])
            if row["stop"] != "hit" or not 0 < evals_to_hit <= charged:
                raise SystemExit(f"invalid hit row {row}")
            if not energy_hits(float(row["best"]), reference, HIT_TOL + 5.0e-9):
                raise SystemExit(f"hit above threshold {row}")
        elif row["stop"] == "budget":
            if charged != budget:
                raise SystemExit(f"invalid budget stop {row}")
        elif row["stop"] == "wall":
            if charged >= budget or not row["cap_s"]:
                raise SystemExit(f"invalid wall stop {row}")
        else:
            raise SystemExit(f"invalid miss stop {row}")
        by_arm[row["arm"]].append((len(by_arm[row["arm"]]), row))

    def choose_campaign(arm: str) -> list[dict[str, str]]:
        # Append-only rows can hold multiple campaigns. Choose the latest full
        # 12-seed campaign of this arm, grouped by code revision and cap.
        groups: dict[tuple[str, str], tuple[int, dict[int, dict[str, str]]]] = {}
        for index, row in by_arm[arm]:
            key = (row["code_revision"], row["cap_s"])
            last, seeds = groups.get(key, (-1, {}))
            seeds[int(row["seed"])] = row
            groups[key] = (index, seeds)
        full = [
            (last, seeds)
            for last, seeds in groups.values()
            if tuple(sorted(seeds)) == SEEDS
        ]
        if full:
            _last, seeds = max(full, key=lambda item: item[0])
            return [seeds[seed] for seed in SEEDS]
        # Older tables (for example LJ55 ASE-BH) can be a complete campaign
        # split across multiple revisions. Fall back to the latest row of each
        # seed in append order.
        latest_by_seed: dict[int, dict[str, str]] = {}
        for _index, row in by_arm[arm]:
            latest_by_seed[int(row["seed"])] = row
        if tuple(sorted(latest_by_seed)) != SEEDS:
            raise SystemExit(f"{system} {arm}: no complete 2000-2011 campaign")
        seeds = latest_by_seed
        return [seeds[seed] for seed in SEEDS]

    for arm in ARMS:
        grouped[arm] = choose_campaign(arm)
    for arm in ARMS:
        seeds = tuple(sorted(int(row["seed"]) for row in grouped[arm]))
        if seeds != SEEDS:
            raise SystemExit(f"{arm} seeds {seeds}")
        grouped[arm].sort(key=lambda row: int(row["seed"]))
    return grouped


def fisher(hits: int, base: int) -> float:
    return fisher_exact(hits, 12, base, 12)


def comparison(left: list[dict[str, str]], right: list[dict[str, str]]) -> str:
    left_hits = sum(row["hit"] == "1" for row in left)
    right_hits = sum(row["hit"] == "1" for row in right)
    return f"{fisher(left_hits, right_hits):.6g}"


def joined(rows: list[dict[str, str]], key: str) -> str:
    return ", ".join(dict.fromkeys(row[key] for row in rows))


def render(system: str) -> tuple[str, str]:
    _, budget, reference, title = SYSTEMS[system]
    grouped = load(system)
    flat = [row for arm in ARMS for row in grouped[arm]]
    hosts = list(dict.fromkeys(row["host"] for row in flat))
    if len(hosts) != 1:
        raise SystemExit(f"{system} host is not unique: {hosts}")
    host = hosts[0]
    anneal = grouped["anneal"]
    lines = [HEADER]
    reports = []
    base_wall = median([float(row["wall_s"]) for row in anneal])
    for arm in ARMS:
        group = grouped[arm]
        hits = sum(row["hit"] == "1" for row in group)
        best = median([float(row["best"]) for row in group])
        wall = median([float(row["wall_s"]) for row in group])
        inside = median([float(row["inside_s"]) for row in group])
        outside = median([float(row["outside_s"]) for row in group])
        wall_stops = sum(row.get("stop") == "wall" for row in group)
        caps = [row["cap_s"] for row in group if row.get("cap_s")]
        cap = caps[0] if caps else ""
        if arm == "anneal":
            over = back = 1.0
            fisher_p = "1"
            cap = ""
        else:
            over = wall / base_wall
            back = base_wall / wall
            fisher_p = comparison(group, anneal)
        versus = {other: comparison(group, grouped[other]) for other in ASE_ARMS}
        lines.append(
            f"{arm},{hits},12,{best:.8f},{best - reference:.8f},{wall:.6f},{inside:.6f},{outside:.6f},"
            f"{over:.6f},{back:.6f},{fisher_p},{wall_stops},{cap},"
            f"{versus['ase_bh']},{versus['ase_mh']}"
        )
        wall_charged = [int(row["charged"]) for row in group if row["stop"] == "wall"]
        low_charged = min(wall_charged) if wall_charged else 0
        high_charged = max(wall_charged) if wall_charged else 0
        reports.append(
            (arm, hits, best, wall, inside, outside, over, back, fisher_p,
             wall_stops, cap, low_charged, high_charged, versus)
        )

    revisions = list(dict.fromkeys(row["code_revision"] for row in flat))
    cap_details = []
    for arm in ARMS:
        arm_caps = sorted({row["cap_s"] for row in grouped[arm] if row["cap_s"]})
        if arm_caps:
            cap_details.append(f"{LABELS[arm]} {', '.join(arm_caps)} s")
    report = [
        "Copyright (c) anneal developers",
        "",
        f"{title}, seeds 2000-2011, maximum budget {budget:,}, host {host}, one core.",
        f"The reference minimum is {reference:.8f}; a hit is at most {HIT_TOL:g} above it.",
        "Anneal uses the cluster preset. Recommended uses Config::recommended.",
        "ASE basin hopping and minima hopping use",
        "the ASE 3.29.0 constructor defaults.",
        f"Rows are in `{ROWS_FILE}` and carry code_revision " + ", ".join(revisions) + ".",
        "Rows carry harness_blob " + joined(flat, "harness_blob") + ".",
        "Rows carry ase_version " + joined(flat, "ase_version") + ".",
        "Rows carry session " + joined(flat, "session") + ".",
    ]
    if cap_details:
        report.extend([
            f"The recorded wall caps are {'; '.join(cap_details)}. A wall-stopped row did not receive",
            "the full call budget, so its call count is right-censored at `charged`.",
        ])
    if system == "lj38":
        report.extend([
            "",
            "| arm | stop | hits | median best | "
            f"gap to {reference_text(reference)} | median wall s | "
            "median inside s | median outside s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
    else:
        report.extend([
            "",
            "| arm | hits | Fisher p vs anneal | median best | "
            f"gap to {reference_text(reference)} | median wall s | over anneal | anneal over | "
            "median inside s | median outside s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
    def signature(arm: str) -> tuple[int, int, int]:
        group = grouped[arm]
        return (
            sum(row.get("stop") == "hit" for row in group),
            sum(row.get("stop") == "budget" for row in group),
            sum(row.get("stop") == "wall" for row in group),
        )

    stop_notes = []
    for (
        arm, hits, best, wall, inside, outside, over, back, fisher_p,
        wall_stops, cap, low_charged, high_charged, versus,
    ) in reports:
        if system == "lj38":
            stop = "hit or budget" if arm in ("anneal", "recommended") else f"wall cap on {wall_stops}"
            report.append(
                f"| {LABELS[arm]} | {stop} | {hits}/12 | {best:.8f} | {best - reference:.8f} | "
                f"{wall:.6f} | {inside:.6f} | {outside:.6f} |"
            )
        else:
            report.append(
                f"| {LABELS[arm]} | {hits}/12 | {fisher_p} | {best:.8f} | {best - reference:.8f} | "
                f"{wall:.6f} | {over:.6f} | {back:.6f} | {inside:.6f} | {outside:.6f} |"
            )
        if wall_stops:
            stop_notes.append(
                f"{LABELS[arm].capitalize()} stopped at the wall cap on {wall_stops}/12 seeds; "
                f"those rows charged {low_charged:,}-{high_charged:,} calls."
            )
        # LJ38 keeps the wall-stop blank from the earlier report. Other systems
        # blank a Fisher cell when the hit, budget, and wall counts differ.
        # The summary CSV keeps the computed p.
        if system == "lj38":
            if wall_stops == 12:
                report[-1] = report[-1].replace(f"| {fisher_p} |", "| — |", 1)
        elif signature(arm) != signature("anneal"):
            report[-1] = report[-1].replace(f"| {fisher_p} |", "| — |", 1)
    recommended = next(item for item in reports if item[0] == "recommended")
    recommended_hits = recommended[1]
    recommended_versus = recommended[-1]
    mixed_wall = any(signature(item[0]) != signature("anneal") for item in reports) or any(
        signature(other) != signature("recommended") for other in ASE_ARMS
    )
    if system == "lj38":
        report.extend(["", *stop_notes, "",
            "Basin hopping and minima hopping stopped at the wall on every seed.",
            "Recommended did not. The stops differ, and there is no paired test.",
            "Medians are type 7. Each wall is that arm's own median. The Rust and",
            "NumPy random generators do not make a shared starting geometry from the",
            "same numeric seed, so the report does not treat seed labels as pairs.",
        ])
    elif mixed_wall:
        report.extend(["", *stop_notes, "",
            "The Fisher cells are blank where the stops differ, and there is no paired test.",
            "",
            "| comparison | hits | Fisher p | median wall ratio | inverse ratio |",
            "| --- | --- | --- | --- | --- |",
        ])
    else:
        report.extend(["", *stop_notes, "",
            "Fisher below compares recommended with each ASE arm.",
            "",
            "| comparison | hits | Fisher p | median wall ratio | inverse ratio |",
            "| --- | --- | --- | --- | --- |",
        ])
    for other in (() if system == "lj38" else ASE_ARMS):
        other_report = next(item for item in reports if item[0] == other)
        other_hits = other_report[1]
        fisher_text = recommended_versus[other]
        if system == "lj38":
            if other_report[9] == 12 and recommended[9] != 12:
                fisher_text = "—"
        elif signature(other) != signature("recommended"):
            fisher_text = "—"
        wall_ratio = recommended[3] / other_report[3]
        inverse_ratio = other_report[3] / recommended[3]
        report.append(
            f"| recommended against {LABELS[other]} | {recommended_hits}/12 vs {other_hits}/12 | "
            f"{fisher_text} | {wall_ratio:.6f} | {inverse_ratio:.6f} |"
        )
    if system == "lj38":
        fisher_line = ""
    elif mixed_wall:
        fisher_line = "The Fisher cells above are blank where the arms do not share a stop."
    else:
        fisher_line = "Fisher is the two-sided exact test on hits under the recorded stopping rule."
    if system != "lj38":
        report.extend(["",
            fisher_line,
            "Medians are type 7. Wall ratios are ratios of arm medians. The Rust and",
            "NumPy random generators do not make a shared starting geometry from the",
            "same numeric seed, so the report does not treat seed labels as pairs.",
        ])
    if system == "au55":
        report.extend([
            "",
            "## Historical rows",
            "",
            "The earlier `rows.csv` and `rows-for-cluster.csv` tables are retained;",
            "the anneal, basin-hopping and minima-hopping rows of `rows.csv` at 33bed75f are kept",
            "and superseded by `rows-for-cluster.csv`; `superseded.csv` lists each of them.",
        ])
    report.append("")
    return "\n".join(lines) + "\n", "\n".join(report)


AGGREGATE_HEADER = (
    "system,arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,"
    "median_outside_s,fisher_p_bh,fisher_p_mh,wall_cap_s,code_revision,host"
)
CURRENT_AGGREGATE_HEADER = AGGREGATE_HEADER + ",machine_id,cpu_model"


def aggregate_rows(system: str, summary: str) -> list[str]:
    fields = (
        "arm", "hits", "n", "median_best", "gap_to_reference", "median_wall_s",
        "median_inside_s", "median_outside_s", "fisher_p_bh", "fisher_p_mh", "wall_cap_s",
    )
    loaded = load(system)
    revisions = {arm: {row["code_revision"] for row in rows} for arm, rows in loaded.items()}
    hosts = {arm: ";".join(sorted({row["host"] for row in rows})) for arm, rows in loaded.items()}
    provenance = {
        arm: (
            ";".join(sorted({row.get("machine_id", "") for row in rows if row.get("machine_id")})),
            ";".join(sorted({row.get("cpu_model", "") for row in rows if row.get("cpu_model")})),
        )
        for arm, rows in loaded.items()
    }
    out = []
    for row in csv.DictReader(summary.splitlines()):
        found = ";".join(sorted(revisions[row["arm"]]))
        values = [system, *(row[field] for field in fields), found, hosts[row["arm"]]]
        if all(provenance[row["arm"]]):
            values.extend(provenance[row["arm"]])
        out.append(csv_fields(values))
    return out


def aggregate_header(rows: list[str]) -> str:
    """Choose the aggregate schema from CSV-decoded row widths."""
    widths = {len(next(csv.reader([row]))) for row in rows}
    if len(widths) != 1:
        raise SystemExit("cannot mix legacy and machine-provenance aggregate rows")
    current_width = len(next(csv.reader([CURRENT_AGGREGATE_HEADER])))
    legacy_width = len(next(csv.reader([AGGREGATE_HEADER])))
    if widths == {current_width}:
        return CURRENT_AGGREGATE_HEADER
    if widths == {legacy_width}:
        return AGGREGATE_HEADER
    raise SystemExit(f"unexpected aggregate row width {next(iter(widths))}")


def main() -> None:
    args = sys.argv[1:]
    flags = [arg for arg in args if arg.startswith("--") and arg != "--all"]
    selection = [arg for arg in args if arg not in flags]
    if len(selection) != 1 or any(flag != "--check" for flag in flags) or len(set(flags)) != len(flags):
        raise SystemExit("usage: summarize.py SYSTEM|--all [--check]")
    selected = list(SYSTEMS) if selection[0] == "--all" else [selection[0]]
    if any(system not in SYSTEMS for system in selected):
        raise SystemExit(f"unknown system {selection[0]}")
    outputs = {}
    aggregate = []
    for system in selected:
        summary, report = render(system)
        directory = ROOT / "bench" / f"ase-{system}"
        outputs[directory / "summary.csv"] = summary
        outputs[directory / "README.md"] = report
        aggregate.extend(aggregate_rows(system, summary))
    if selected == list(SYSTEMS):
        aggregate.insert(0, aggregate_header(aggregate))
        outputs[ROOT / "bench" / "ase-recommended" / "summary.csv"] = (
            "\n".join(aggregate) + "\n"
        )
    if "--check" in flags:
        stale = [str(path.relative_to(ROOT)) for path, text in outputs.items() if path.read_text() != text]
        if stale:
            raise SystemExit("stale generated report: " + ", ".join(stale))
    else:
        for path, text in outputs.items():
            path.write_text(text)


if __name__ == "__main__":
    main()
