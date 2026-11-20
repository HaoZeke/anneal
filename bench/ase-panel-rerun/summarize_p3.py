#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize one ASE panel rerun from its rows.

Fisher is the two-sided exact test on hits. Mann-Whitney is not estimated
for an arm with wall-censored misses. Does not start a search.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
    "lj150": (150, 4_000_000, -893.310258, "LJ150"),
}
HIT_TOL = 1.0e-3
REV = "61fa67cf380e46876b569da45e7cc091647a775b"
BASE_FIELDS = (
    "system", "arm", "seed", "budget", "hit", "evals_to_hit", "charged", "best",
    "wall_s", "inside_s", "outside_s", "code_revision", "host", "stop", "cap_s",
)
EXTRA_FIELDS = ("harness_blob", "ase_version", "session")
HEADER = (
    "system,arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,"
    "median_outside_s,fisher_p_bh,mann_whitney_U_bh,mann_whitney_p_bh,"
    "fisher_p_mh,mann_whitney_U_mh,mann_whitney_p_mh,wall_cap_s,code_revision,host"
)


def energy_hits(energy: float, target: float, tol: float = HIT_TOL) -> bool:
    if not (math.isfinite(energy) and math.isfinite(target) and math.isfinite(tol)):
        return False
    return tol >= 0.0 and float(energy) <= float(target) + float(tol)


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


def calls(row: dict[str, str]) -> float:
    if row["evals_to_hit"]:
        return float(row["evals_to_hit"])
    return float(row["budget"])


def wall_censored(rows: list[dict[str, str]]) -> bool:
    return any(row["stop"] == "wall" and row["hit"] == "0" for row in rows)


def fisher(hits: int, base: int) -> float:
    row = 12
    col = hits + base
    total = 24

    def probability(value: int) -> float:
        return math.comb(row, value) * math.comb(row, col - value) / math.comb(total, col)

    observed = probability(hits)
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(max(0, col - row), min(row, col) + 1)
            if probability(value) <= observed * (1.0 + 1.0e-12)
        ),
    )


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    n_left, n_right = len(left), len(right)
    pooled = sorted([(value, 0) for value in left] + [(value, 1) for value in right])
    ranks = [0.0] * len(pooled)
    tie_term = 0
    index = 0
    while index < len(pooled):
        end = index + 1
        while end < len(pooled) and pooled[end][0] == pooled[index][0]:
            end += 1
        average = 0.5 * (index + 1 + end)
        ranks[index:end] = [average] * (end - index)
        width = end - index
        tie_term += width**3 - width
        index = end
    rank_left = sum(rank for rank, (_, arm) in zip(ranks, pooled) if arm == 0)
    statistic = rank_left - n_left * (n_left + 1) / 2.0
    total = n_left + n_right
    variance = n_left * n_right / 12.0 * (
        total + 1 - tie_term / (total * (total - 1))
    )
    if variance <= 0.0:
        return statistic, 1.0
    z = max(0.0, (abs(statistic - n_left * n_right / 2.0) - 0.5) / math.sqrt(variance))
    return statistic, min(1.0, math.erfc(z / math.sqrt(2.0)))


def comparison(
    left: list[dict[str, str]], right: list[dict[str, str]]
) -> tuple[str, str, str]:
    left_hits = sum(row["hit"] == "1" for row in left)
    right_hits = sum(row["hit"] == "1" for row in right)
    fisher_text = f"{fisher(left_hits, right_hits):.6g}"
    if wall_censored(left) or wall_censored(right):
        return fisher_text, "", ""
    statistic, probability = mann_whitney(
        [calls(row) for row in left], [calls(row) for row in right]
    )
    return fisher_text, f"{statistic:.6g}", f"{probability:.6g}"


def load(path: Path, system: str) -> dict[str, list[dict[str, str]]]:
    _, budget, reference, _ = SYSTEMS[system]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields[: len(BASE_FIELDS)] != BASE_FIELDS:
            raise SystemExit(f"{path}: unexpected columns {reader.fieldnames}")
        if fields[len(BASE_FIELDS):] != EXTRA_FIELDS and fields != BASE_FIELDS:
            raise SystemExit(f"{path}: unexpected extra columns {fields[len(BASE_FIELDS):]}")
        rows = list(reader)
    extra = fields[len(BASE_FIELDS):] == EXTRA_FIELDS
    grouped: dict[str, list[dict[str, str]]] = {arm: [] for arm in ARMS}
    for row in rows:
        if row["system"] != system or row["arm"] not in grouped or row["budget"] != str(budget):
            raise SystemExit(f"unexpected row {row}")
        if not row["host"]:
            raise SystemExit(f"missing host {row}")
        charged = int(row["charged"])
        if not 0 < charged <= budget:
            raise SystemExit(f"invalid charged count {row}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["code_revision"]):
            raise SystemExit(f"invalid revision {row}")
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
        if extra:
            if not re.fullmatch(r"[0-9a-f]{64}", row["host"]):
                raise SystemExit(f"host is not a hashed machine id {row}")
            if not re.fullmatch(r"[0-9a-f]{40}", row["harness_blob"]):
                raise SystemExit(f"invalid harness blob {row}")
            if row["ase_version"] != "3.29.0":
                raise SystemExit(f"unexpected ASE version {row}")
            if not re.fullmatch(r"[0-9a-f]{64}", row["session"]):
                raise SystemExit(f"invalid session {row}")
            if row["code_revision"] != REV:
                raise SystemExit(f"unexpected revision {row}")
        grouped[row["arm"]].append(row)
    for arm in ARMS:
        seeds = [int(row["seed"]) for row in grouped[arm]]
        if len(seeds) != len(set(seeds)) or tuple(sorted(seeds)) != SEEDS:
            raise SystemExit(f"{system} {arm}: seeds {sorted(seeds)}")
        grouped[arm].sort(key=lambda row: int(row["seed"]))
    hosts = {row["host"] for arm in ARMS for row in grouped[arm]}
    if len(hosts) != 1:
        raise SystemExit(f"host is not unique: {sorted(hosts)}")
    if extra:
        blobs = {row["harness_blob"] for arm in ARMS for row in grouped[arm]}
        sessions = {row["session"] for arm in ARMS for row in grouped[arm]}
        if len(blobs) != 1 or len(sessions) != 1:
            raise SystemExit("harness blob or session is not unique")
    return grouped


def check_cap(grouped: dict[str, list[dict[str, str]]]) -> str:
    anneal_caps = {row["cap_s"] for row in grouped["anneal"]}
    if anneal_caps != {""}:
        raise SystemExit(f"anneal cap must be empty, found {sorted(anneal_caps)}")
    caps = {row["cap_s"] for arm in ("recommended",) + ASE_ARMS for row in grouped[arm]}
    if len(caps) != 1 or "" in caps:
        raise SystemExit(f"the wall cap is not unique: {sorted(caps)}")
    cap = next(iter(caps))
    slowest = max(float(row["wall_s"]) for row in grouped["anneal"])
    if abs(float(cap) - 3.0 * slowest) > 1.0e-6:
        raise SystemExit("the wall cap is not three times the slowest anneal wall")
    return cap


def build(grouped: dict[str, list[dict[str, str]]], system: str) -> tuple[str, str]:
    _, budget, reference, title = SYSTEMS[system]
    check_cap(grouped)
    host = next(iter({row["host"] for arm in ARMS for row in grouped[arm]}))
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
        wall_stops = sum(row["stop"] == "wall" for row in group)
        caps = [row["cap_s"] for row in group if row["cap_s"]]
        cap = caps[0] if caps else ""
        if arm == "anneal":
            over = back = 1.0
            fisher_p = "1"
            u_text, mw_text = "72", "1"
            cap = ""
        else:
            over = wall / base_wall
            back = base_wall / wall
            fisher_p, u_text, mw_text = comparison(group, anneal)
        versus = {other: comparison(group, grouped[other]) for other in ASE_ARMS}
        revision = ";".join(sorted({row["code_revision"] for row in group}))
        lines.append(
            ",".join([
                system,
                arm,
                str(hits),
                "12",
                f"{best:.8f}",
                f"{best - reference:.8f}",
                f"{wall:.6f}",
                f"{inside:.6f}",
                f"{outside:.6f}",
                versus["ase_bh"][0],
                versus["ase_bh"][1],
                versus["ase_bh"][2],
                versus["ase_mh"][0],
                versus["ase_mh"][1],
                versus["ase_mh"][2],
                cap,
                revision,
                host,
            ])
        )
        reports.append(
            (arm, hits, best, wall, inside, outside, over, back, fisher_p, u_text, mw_text,
             wall_stops, cap, min(int(row["charged"]) for row in group),
             max(int(row["charged"]) for row in group), versus)
        )
    revisions = list(dict.fromkeys(row["code_revision"] for arm in ARMS for row in grouped[arm]))
    cap_details = []
    for arm in ARMS:
        arm_caps = sorted({row["cap_s"] for row in grouped[arm] if row["cap_s"]})
        if arm_caps:
            cap_details.append(f"{LABELS[arm]} {', '.join(arm_caps)} s")
    blobs = sorted({row["harness_blob"] for arm in ARMS for row in grouped[arm] if row.get("harness_blob")})
    versions = sorted({row["ase_version"] for arm in ARMS for row in grouped[arm] if row.get("ase_version")})
    sessions = sorted({row["session"] for arm in ARMS for row in grouped[arm] if row.get("session")})
    report = [
        "Copyright (c) anneal developers",
        "",
        f"{title}, seeds 2000-2011, maximum budget {budget:,}, host {host}, one core.",
        f"The reference minimum is {reference:.8f}; a hit is at most {HIT_TOL:g} above it.",
        "Anneal uses the cluster preset. Recommended uses Config::recommended.",
        "ASE basin hopping and minima hopping use",
        "the ASE 3.29 constructor defaults.",
        "Rows carry code_revision " + ", ".join(revisions) + ".",
    ]
    if blobs or versions or sessions:
        report.append(
            "Each row also records harness_blob "
            + ", ".join(blobs)
            + ", ase_version "
            + ", ".join(versions)
            + ", and session "
            + ", ".join(sessions)
            + "."
        )
    if cap_details:
        report.extend([
            f"The recorded wall caps are {'; '.join(cap_details)}. A wall-stopped row did not receive",
            "the full call budget, so its call count is right-censored at `charged`.",
        ])
    report.extend([
        "",
        "| arm | hits | Fisher p vs anneal | Mann-Whitney U vs anneal | Mann-Whitney p vs anneal | median best | "
        f"gap to {reference_text(reference)} | median wall s | over anneal | anneal over | "
        "median inside s | median outside s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    stop_notes = []
    for (
        arm, hits, best, wall, inside, outside, over, back, fisher_p, u_text, mw_text,
        wall_stops, cap, low_charged, high_charged, versus,
    ) in reports:
        report.append(
            f"| {LABELS[arm]} | {hits}/12 | {fisher_p} | {u_text or 'not estimated'} | "
            f"{mw_text or 'not estimated'} | {best:.8f} | {best - reference:.8f} | "
            f"{wall:.6f} | {over:.6f} | {back:.6f} | {inside:.6f} | {outside:.6f} |"
        )
        if wall_stops:
            stop_notes.append(
                f"{LABELS[arm].capitalize()} stopped at the wall cap on {wall_stops}/12 seeds; "
                f"those rows charged {low_charged:,}-{high_charged:,} calls."
            )
    recommended = next(item for item in reports if item[0] == "recommended")
    recommended_hits = recommended[1]
    recommended_versus = recommended[-1]
    report.extend(["", *stop_notes, "",
        "Fisher and Mann-Whitney below compare recommended with each ASE arm.",
        "",
        "| comparison | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | "
        "median wall ratio | inverse ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for other in ASE_ARMS:
        other_report = next(item for item in reports if item[0] == other)
        other_hits = other_report[1]
        fisher_text, u_text, mw_text = recommended_versus[other]
        wall_ratio = recommended[3] / other_report[3]
        inverse_ratio = other_report[3] / recommended[3]
        report.append(
            f"| recommended against {LABELS[other]} | {recommended_hits}/12 vs {other_hits}/12 | "
            f"{fisher_text} | {u_text or 'not estimated'} | {mw_text or 'not estimated'} | "
            f"{wall_ratio:.6f} | {inverse_ratio:.6f} |"
        )
    report.extend([
        "",
        "Fisher is the two-sided exact test on hits under the recorded stopping rule.",
        "Mann-Whitney is two-sided with tie and continuity corrections. It is not",
        "estimated for an arm with wall-censored misses; imputing those misses at the",
        "larger paper budget would manufacture observations that were never run.",
        "Medians are type 7. Wall ratios are ratios of arm medians. The Rust and",
        "NumPy random generators do not make a shared starting geometry from the",
        "same numeric seed, so the report does not treat seed labels as pairs.",
        "",
    ])
    return "\n".join(lines) + "\n", "\n".join(report)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in SYSTEMS:
        raise SystemExit("usage: summarize_p3.py lj150")
    system = sys.argv[1]
    directory = ROOT / "bench" / "ase-panel-rerun" / system
    grouped = load(directory / "rows.csv", system)
    summary, report = build(grouped, system)
    (directory / "summary.csv").write_text(summary)
    (directory / "README.md").write_text(report)


if __name__ == "__main__":
    main()
