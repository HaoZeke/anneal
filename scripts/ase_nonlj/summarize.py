#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Generate the non-LJ (Morse) ASE panel report from its committed rows.

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

SEEDS = tuple(range(2000, 2012))
SYSTEMS = ("m38r6", "m38r14", "m55r6", "m55r14")
ARMS = ("for_cluster", "recommended", "ase_bh", "ase_mh")
ASE_ARMS = ("ase_bh", "ase_mh")
LABEL = {
    "for_cluster": "for_cluster",
    "recommended": "recommended",
    "ase_bh": "basin hopping",
    "ase_mh": "minima hopping",
}
SPEC = {
    "m38r6": (38, 6.0, "400000", -157.477108, "38E"),
    "m38r14": (38, 14.0, "400000", -144.321054, "38D"),
    "m55r6": (55, 6.0, "1000000", -250.286609, "55B"),
    "m55r14": (55, 14.0, "1000000", -220.646208, "55C"),
}
ROW_FIELDS = (
    "system", "arm", "seed", "budget", "hit", "evals_to_hit", "charged", "best",
    "wall_s", "inside_s", "outside_s", "code_revision", "host", "stop", "cap_s",
)
HIT_TOL = DEFAULT_HIT_TOL
ASE_VERSION = "3.29.0"
REVISION = "9a30f3190b3bb7040c675a904f8388eb68224151"


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


def g6(value: float) -> str:
    return f"{value:.6g}"


def join_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def seed_list(rows: list[dict[str, str]]) -> str:
    return join_list([row["seed"] for row in rows])


def fisher(hits: int, base: int) -> float:
    return fisher_exact(hits, 12, base, 12)


def wall_ratio(left: list[dict[str, str]], right: list[dict[str, str]]) -> tuple[float, float]:
    left_wall = median([float(row["wall_s"]) for row in left])
    right_wall = median([float(row["wall_s"]) for row in right])
    return left_wall / right_wall, right_wall / left_wall


def against(group: list[dict[str, str]], other: list[dict[str, str]]) -> str:
    hits = sum(row["hit"] == "1" for row in group)
    base = sum(row["hit"] == "1" for row in other)
    return g6(fisher(hits, base))


def load_rows() -> dict[str, dict[str, list[dict[str, str]]]]:
    path = ROOT / "bench" / "ase-nonlj" / "rows.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROW_FIELDS:
            raise SystemExit(f"{path}: unexpected columns {reader.fieldnames}")
        rows = list(reader)
    grouped = {system: {arm: [] for arm in ARMS} for system in SYSTEMS}
    seen = set()
    for row in rows:
        system = row["system"]
        arm = row["arm"]
        if system not in SPEC or arm not in ARMS:
            raise SystemExit(f"unexpected row {row}")
        if row["budget"] != SPEC[system][2] or row["host"] != "bench":
            raise SystemExit(f"unexpected budget or host {row}")
        key = (row["code_revision"], system, arm, row["seed"])
        if key in seen:
            raise SystemExit(f"duplicate row {key}")
        seen.add(key)
        budget = int(row["budget"])
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
            if not energy_hits(float(row["best"]), SPEC[system][3], HIT_TOL + 5.0e-9):
                raise SystemExit(f"hit above threshold {row}")
        elif row["stop"] == "budget":
            if charged != budget:
                raise SystemExit(f"invalid budget stop {row}")
        elif row["stop"] == "wall":
            if charged >= budget or not row["cap_s"]:
                raise SystemExit(f"invalid wall stop {row}")
        else:
            raise SystemExit(f"invalid miss stop {row}")
        if row["code_revision"] == REVISION:
            grouped[system][arm].append(row)
    for system in SYSTEMS:
        for arm in ARMS:
            seeds = tuple(sorted(int(row["seed"]) for row in grouped[system][arm]))
            if seeds != SEEDS:
                raise SystemExit(f"{system} {arm} seeds {seeds}")
            grouped[system][arm].sort(key=lambda row: int(row["seed"]))
    return grouped


def load_identity() -> list[dict[str, str]]:
    path = ROOT / "bench" / "ase-nonlj" / "identity.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_sys = {row["system"]: row for row in rows}
    if tuple(by_sys) != SYSTEMS:
        raise SystemExit(f"identity systems {tuple(by_sys)}")
    for row in rows:
        if float(row["abs_energy_diff"]) > 1.0e-8 or float(row["max_force_diff"]) > 1.0e-8:
            raise SystemExit(f"identity check exceeded 1e-8: {row}")
    return rows


def group_revision(group: list[dict[str, str]]) -> str:
    found = {row["code_revision"] for row in group}
    if len(found) != 1:
        raise SystemExit(f"a summary row needs one code revision, found {sorted(found)}")
    return found.pop()


def main() -> None:
    flags = sys.argv[1:]
    if any(flag != "--check" for flag in flags) or len(set(flags)) != len(flags):
        raise SystemExit("usage: summarize.py [--check]")
    grouped = load_rows()
    identity = load_identity()
    revisions = sorted({row["code_revision"] for s in SYSTEMS for a in ARMS for row in grouped[s][a]})
    summary_header = (
        "system,arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,median_outside_s,"
        "fisher_p_bh,fisher_p_mh,"
        "wall_over_bh,wall_over_mh,wall_stops,wall_cap_s,code_revision,host"
    )
    summary_path = ROOT / "bench" / "ase-nonlj" / "summary.csv"
    summary_lines = [summary_header]
    if summary_path.exists():
        existing = summary_path.read_text().splitlines()
        if existing and existing[0] != summary_header:
            raise SystemExit(f"{summary_path}: unexpected columns")
        summary_lines.extend(
            line
            for line in existing[1:]
            if dict(zip(summary_header.split(","), line.split(",")))["code_revision"] != REVISION
        )
    readme = [
        "Copyright (c) anneal developers",
        "",
        "# Non-LJ ASE head-to-head (Cambridge Morse clusters)",
        "",
        "Four Cambridge Morse systems across two sizes and two ranges: M38 and M55 at rho = 6 and rho = 14.",
        "Seeds 2000-2011, host bench, serial one core (`taskset -c 0`).",
        f"code_revision {', '.join(revisions)}.",
        "Potential: untruncated Morse (`epsilon = 1`, `r0 = 1`, `rho0 = rho`).",
        "Anneal evaluates `PairPotential::morse` (`rgpot_pair::cambridge_morse`).",
        "ASE evaluates `ase.calculators.morse.MorsePotential` with `rcut1 = 1e6`, `rcut2 = 2e6`.",
        "Anneal `for_cluster` uses `Config::with_scales` at `PairKind::length_scale()` plus `for_cluster` Bayes screening.",
        "Anneal `recommended` uses `Config::recommended_with_scales` at `PairKind::length_scale()`.",
        "ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults (`fmax` 0.1 and 0.05).",
        "Hits reach `reference + 0.001`. Misses are censored at the paper budget (400,000 at 38 atoms, 1,000,000 at 55).",
        "The wall cap for `recommended`, `ase_bh` and `ase_mh` is three times the slowest `for_cluster` wall per system.",
        "",
        "## Cross-code potential agreement (seed 2000 start)",
        "",
        "| system | N | rho | E anneal | E ASE | |dE| | max |dF| |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in identity:
        n, rho, _, _, _ = SPEC[row["system"]]
        readme.append(
            f"| {row['system']} | {n} | {rho:g} | {float(row['energy_anneal']):.10f} | "
            f"{float(row['energy_ase']):.10f} | {float(row['abs_energy_diff']):.3e} | "
            f"{float(row['max_force_diff']):.3e} |"
        )

    readme.extend(
        [
            "",
            "## Summary",
            "",
            "| system | arm | hits | median best | gap | median wall s | median inside s | median outside s | "
            "stops (hit/budget/wall) | cap s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    vs_lines = [
        "",
        "## Fisher against ASE arms",
        "",
        "| system | comparison | hits | Fisher p | wall ratio | inverse ratio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for system in SYSTEMS:
        n, rho, budget, reference, label = SPEC[system]
        base = grouped[system]["for_cluster"]
        caps = sorted({row["cap_s"] for row in grouped[system]["recommended"] if row["cap_s"]})
        if len(caps) != 1:
            raise SystemExit(f"{system} cap not unique: {caps}")
        cap = caps[0]
        for arm in ARMS:
            group = grouped[system][arm]
            hits = sum(row["hit"] == "1" for row in group)
            best = median([float(row["best"]) for row in group])
            wall = median([float(row["wall_s"]) for row in group])
            inside = median([float(row["inside_s"]) for row in group])
            outside = median([float(row["outside_s"]) for row in group])
            hit_stops = sum(row["stop"] == "hit" for row in group)
            budget_stops = sum(row["stop"] == "budget" for row in group)
            wall_stops = sum(row["stop"] == "wall" for row in group)
            f_bh = against(group, grouped[system]["ase_bh"])
            f_mh = against(group, grouped[system]["ase_mh"])
            w_bh, inv_bh = wall_ratio(group, grouped[system]["ase_bh"])
            w_mh, inv_mh = wall_ratio(group, grouped[system]["ase_mh"])
            arm_cap = "" if arm == "for_cluster" else cap
            summary_lines.append(
                f"{system},{arm},{hits},12,{best:.8f},{best - reference:.8f},"
                f"{wall:.6f},{inside:.6f},{outside:.6f},"
                f"{f_bh},{f_mh},"
                f"{w_bh:.6f},{w_mh:.6f},{wall_stops},{arm_cap},{group_revision(group)},bench"
            )
            readme.append(
                f"| {system} ({label}, {reference:.6f}) | {LABEL[arm]} | {hits}/12 | {best:.8f} | "
                f"{best - reference:.8f} | {wall:.6f} | {inside:.6f} | {outside:.6f} | "
                f"{hit_stops}/{budget_stops}/{wall_stops} | {arm_cap} |"
            )
            if arm in ("for_cluster", "recommended"):
                for other_arm, (fp, wr, ir) in (
                    ("ase_bh", (f_bh, w_bh, inv_bh)),
                    ("ase_mh", (f_mh, w_mh, inv_mh)),
                ):
                    other_hits = sum(row["hit"] == "1" for row in grouped[system][other_arm])
                    vs_lines.append(
                        f"| {system} | {LABEL[arm]} vs {LABEL[other_arm]} | {hits}/12 vs {other_hits}/12 | "
                        f"{fp} | {wr:.6f} | {ir:.6f} |"
                    )

    readme.extend(vs_lines)
    readme.extend(
        [
            "",
            "Medians are Hyndman-Fan type 7. Wall ratios are ratios of arm medians.",
            "The Rust and NumPy random generators do not make a shared starting",
            "geometry from the same numeric seed, so seed labels are not treated as pairs.",
            "Fisher is two-sided exact.",
            "Inside is time in the potential. Outside is the rest of the wall.",
            "",
        ]
    )
    out = ROOT / "bench" / "ase-nonlj"
    outputs = {
        summary_path: "\n".join(summary_lines) + "\n",
        out / "README.md": "\n".join(readme),
    }
    if "--check" in flags:
        stale = [
            str(path.relative_to(ROOT))
            for path, text in outputs.items()
            if path.read_text() != text
        ]
        if stale:
            raise SystemExit("stale generated report: " + ", ".join(stale))
    else:
        for path, text in outputs.items():
            path.write_text(text)


if __name__ == "__main__":
    main()
