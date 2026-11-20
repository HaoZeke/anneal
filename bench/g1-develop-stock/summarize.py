#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Pair the measured rows with stored comparison rows and summarise them."""

from __future__ import annotations

import ast
import csv
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

SEEDS = tuple(range(2000, 2012))
# code_revision on the GMIN arm of bench/gmin-panel/rows.csv.
GMIN_REVISION = "aac1548a8aec13b53ad12a8d6904625081685d2e"
SYSTEMS = (
    ("m30r6", 30, 400_000),
    ("m30r14", 30, 400_000),
    ("m38r6", 38, 400_000),
    ("m38r14", 38, 400_000),
    ("m70r3", 70, 1_000_000),
    ("m70r6", 70, 1_000_000),
    ("m70r10", 70, 1_000_000),
    ("m70r14", 70, 1_000_000),
    ("m80r3", 80, 4_000_000),
    ("m80r10", 80, 4_000_000),
)
G0_CELLS = (
    "lj38",
    "lj55",
    "lj75",
    "lj98",
    "lj102",
    "lj103",
    "lj104",
    "lj150",
    "au55",
    "m55r6",
    "m55r14",
    "m80r6",
    "m80r14",
)
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
    "config",
    "source",
]
SUMMARY_FIELDS = [
    "system",
    "n",
    "anneal_hits",
    "gmin_hits",
    "gmin_revision",
    "fisher_p",
    "anneal_calls_median",
    "gmin_calls_median",
    "mannwhitney_u",
    "mannwhitney_p",
    "anneal_outside_s_per_call_median",
]
PANEL_FIELDS = [
    "cell",
    "seed",
    "start",
    "hit",
    "calls_to_hit",
    "charged",
    "best",
    "best_bits",
    "wall_s",
    "pot_s",
    "outside_s",
    "hop_first_best",
    "preset_hash",
    "revision",
    "core",
    "affinity",
    "machine_id_sha256",
    "session",
    "sequence",
    "source",
]


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def script_systems(path: Path) -> dict[str, tuple[int, int]]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "SYSTEMS" for target in node.targets):
                rows = ast.literal_eval(node.value)
                return {row[0]: (int(row[1]), int(row[3])) for row in rows}
    raise SystemExit(f"{path}: SYSTEMS is missing")


def check_specs() -> None:
    expected = {name: (n, budget) for name, n, budget in SYSTEMS}
    gmin = script_systems(ROOT / "scripts/morse_gmin/run_panel.py")
    panel = script_systems(ROOT / "scripts/morse_panel/run_panel.py")
    for name, pair in expected.items():
        source = gmin if name in ("m30r6", "m30r14", "m38r6", "m38r14") else panel
        if source.get(name) != pair:
            raise SystemExit(f"{name}: campaign specification differs from source script")


def hashed_host(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode()).hexdigest()


def g1_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    measured = load(HERE / "anneal-rows.csv")
    specs = {name: (n, budget) for name, n, budget in SYSTEMS}
    by_key = {(row["system"], int(row["seed"])): row for row in measured}
    if len(by_key) != len(SYSTEMS) * len(SEEDS):
        raise SystemExit("added panel does not contain exactly 120 unique rows")
    primary = {
        (row["system"], int(row["seed"])): row
        for row in load(ROOT / "bench/morse-gmin/seeds.csv")
        if row["code"] == "gmin"
    }
    fallback = {
        (row["system"], int(row["seed"])): row
        for row in load(ROOT / "bench/morse-panel/seeds.csv")
        if row["code"] == "gmin"
    }
    paired: list[dict[str, str]] = []
    for name, _n, budget in SYSTEMS:
        for seed in SEEDS:
            row = by_key[(name, seed)]
            if int(row["budget"]) != budget:
                raise SystemExit(f"{name} {seed}: measured budget differs")
            if row["config"] != "recommended":
                raise SystemExit(f"{name} {seed}: configuration differs")
            paired.append({**{key: row.get(key, "") for key in ROW_FIELDS}, "source": "g1"})
            old = primary.get((name, seed))
            source = "bench/morse-gmin/seeds.csv"
            if old is None:
                old = fallback.get((name, seed))
                source = "bench/morse-panel/seeds.csv"
            if old is None:
                paired.append(
                    {
                        **{key: "" for key in ROW_FIELDS},
                        "system": name,
                        "code": "gmin",
                        "seed": str(seed),
                        "budget": str(budget),
                    }
                )
                continue
            if int(old["budget"]) != specs[name][1]:
                raise SystemExit(f"{name} {seed}: stored budget differs")
            clean = {key: old.get(key, "") for key in ROW_FIELDS}
            clean["hit"] = "1" if old["hit"] in ("1", "true", "True") else "0"
            clean["host"] = hashed_host(old.get("host", ""))
            clean["config"] = ""
            clean["source"] = source
            paired.append(clean)
    return measured, paired


def g0_rows() -> list[dict[str, str]]:
    raw = load(ROOT / "bench/g0-lj-stock/rows.csv")
    manifest = load(ROOT / "bench/g0-lj-stock/manifest.csv")
    specs: dict[str, tuple[int, int]] = {}
    manifest_rows: dict[str, int] = {}
    for row in manifest:
        if not row["cell"].startswith("gmin-"):
            continue
        name = row["cell"][5:]
        if name not in G0_CELLS:
            continue
        specs[name] = (int(row["n"]), int(row["budget"]))
        manifest_rows[name] = manifest_rows.get(name, 0) + 1
    out = []
    for row in raw:
        if not row["cell"].startswith("gmin-"):
            continue
        name = row["cell"][5:]
        if name not in G0_CELLS:
            continue
        n, budget = specs[name]
        hit = row["hit"]
        out.append(
            {
                "system": name,
                "code": "anneal",
                "seed": row["seed"],
                "budget": str(budget),
                "hit": hit,
                "evals_to_hit": row["calls_to_hit"] if hit == "1" else str(budget),
                "charged": row["charged"],
                "wall_s": row["wall_s"],
                "best": row["best"],
                "pot_s": row["pot_s"],
                "outside_s": row["outside_s"],
                "code_revision": row["revision"],
                "host": row["machine_id_sha256"],
                "config": "recommended",
                "source": "g0",
            }
        )
    if len(out) != len(G0_CELLS) * len(SEEDS):
        raise SystemExit("reused panel does not contain exactly 156 rows")
    for name in G0_CELLS:
        if manifest_rows.get(name, 0) != 12:
            raise SystemExit(f"{name}: manifest does not contain 12 rows")
    return out


def panel_rows(measured: list[dict[str, str]]) -> list[dict[str, str]]:
    old = load(ROOT / "bench/g0-lj-stock/rows.csv")
    out = [
        {**row, "source": "g0"}
        for row in old
        if row["cell"].startswith("gmin-") and row["cell"][5:] in G0_CELLS
    ]
    for sequence, row in enumerate(measured, start=1):
        out.append(
            {
                "cell": f"gmin-{row['system']}",
                "seed": row["seed"],
                "start": "",
                "hit": row["hit"],
                "calls_to_hit": row["evals_to_hit"] if row["hit"] == "1" else "",
                "charged": row["charged"],
                "best": row["best"],
                "best_bits": "",
                "wall_s": row["wall_s"],
                "pot_s": row["pot_s"],
                "outside_s": row["outside_s"],
                "hop_first_best": "",
                "preset_hash": "",
                "revision": row["code_revision"],
                "core": "0",
                "affinity": "0",
                "machine_id_sha256": row["host"],
                "session": "",
                "sequence": str(sequence),
                "source": "g1",
            }
        )
    return out


def panel_references() -> dict[str, float]:
    """Reference energy each anneal comparison row was scored against."""
    refs: dict[str, float] = {}
    for row in load(ROOT / "bench/g0-lj-stock/manifest.csv"):
        if not row["cell"].startswith("gmin-"):
            continue
        name = row["cell"][5:]
        value = float(row["reference"])
        previous = refs.get(name)
        if previous is not None and previous != value:
            raise SystemExit(f"{name}: reference differs across seeds")
        refs[name] = value
    return refs


def gmin_arm(
    rows: list[dict[str, str]],
    references: dict[str, float],
) -> dict[str, list[dict[str, str]]]:
    """Latest GMIN row of each seed, with the hit re-derived from best energy.

    A row counts only when its code is ``gmin`` and its code revision is
    the GMIN revision. The hit uses the same rule as the anneal arm:
    best energy at or below that system's reference plus the shared
    tolerance. A miss is entered at the budget. A system with no such
    row is omitted.
    """
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("code") != "gmin" or row.get("code_revision") != GMIN_REVISION:
            continue
        latest[(row["system"], row["seed"])] = row
    grouped: dict[str, list[dict[str, str]]] = {}
    for (name, _seed), row in latest.items():
        grouped.setdefault(name, []).append(row)
    out: dict[str, list[dict[str, str]]] = {}
    for name, group in grouped.items():
        by_seed = {row["seed"]: row for row in group}
        if [str(seed) for seed in SEEDS] != sorted(by_seed, key=int) or len(by_seed) != len(SEEDS):
            raise SystemExit(f"{name}: GMIN rows do not cover seeds 2000-2011")
        if name not in references:
            raise SystemExit(f"{name}: anneal reference is missing")
        reference = references[name]
        arm: list[dict[str, str]] = []
        for seed in SEEDS:
            row = by_seed[str(seed)]
            hit = energy_hits(float(row["best"]), reference, DEFAULT_HIT_TOL)
            arm.append(
                {
                    "seed": str(seed),
                    "hit": "1" if hit else "0",
                    "evals_to_hit": row["evals_to_hit"] if hit else row["budget"],
                    "revision": row["code_revision"],
                }
            )
        out[name] = arm
    return out


def type7(values: list[float]) -> float:
    values = sorted(values)
    position = 1.0 + (len(values) - 1) * 0.5
    lower = int(position)
    fraction = position - lower
    left, right = values[lower - 1], values[min(lower, len(values) - 1)]
    if math.isinf(left) or math.isinf(right):
        return math.inf
    return left * (1.0 - fraction) + right * fraction


def fisher(hits_a: int, hits_b: int) -> float:
    hits = hits_a + hits_b
    ways = math.comb(24, hits)

    def probability(k: int) -> float:
        return math.comb(12, k) * math.comb(12, hits - k) / ways

    observed = probability(hits_a)
    lo, hi = max(0, hits - 12), min(12, hits)
    return min(
        1.0,
        sum(
            probability(k)
            for k in range(lo, hi + 1)
            if probability(k) <= observed * (1.0 + 1e-9)
        ),
    )


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    pooled = sorted(left + right)
    ranks: dict[float, float] = {}
    tie_sum = 0
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[index]:
            end += 1
        ranks[pooled[index]] = (index + end + 2) / 2.0
        size = end - index + 1
        tie_sum += size**3 - size
        index = end + 1
    u = sum(ranks[value] for value in left) - 78.0
    variance = 12.0 * ((25.0) - tie_sum / (24.0 * 23.0))
    if variance <= 0.0:
        return u, 1.0
    deviation = max(abs(u - 72.0) - 0.5, 0.0)
    p_value = math.erfc(deviation / math.sqrt(variance) / math.sqrt(2.0))
    return u, min(1.0, p_value)


def fmt(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.6g}"


def summary(
    anneal_rows: list[dict[str, str]],
    comparisons: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    specs = {name: (n, budget) for name, n, budget in SYSTEMS}
    g0_manifest = load(ROOT / "bench/g0-lj-stock/manifest.csv")
    for name in G0_CELLS:
        sample = next(row for row in g0_manifest if row["cell"] == f"gmin-{name}")
        specs[name] = (int(sample["n"]), int(sample["budget"]))
    out = []
    for name in (*G0_CELLS, *(row[0] for row in SYSTEMS)):
        n, budget = specs[name]
        arm = sorted(
            (row for row in anneal_rows if row["system"] == name),
            key=lambda row: int(row["seed"]),
        )
        other = sorted(comparisons.get(name, []), key=lambda row: int(row["seed"]))
        if len(arm) != 12:
            raise SystemExit(f"{name}: n differs from 12")
        if any(int(row["budget"]) != budget for row in arm):
            raise SystemExit(f"{name}: budget differs from source script")
        arm_hit = [row["hit"] == "1" for row in arm]
        arm_calls = [
            float(row["evals_to_hit"]) if hit else math.inf
            for row, hit in zip(arm, arm_hit, strict=True)
        ]
        outside = [float(row["outside_s"]) / float(row["charged"]) for row in arm]
        ah = sum(arm_hit)
        record = {
            "system": name,
            "n": "12",
            "anneal_hits": str(ah),
            "gmin_hits": "",
            "gmin_revision": "",
            "fisher_p": "",
            "anneal_calls_median": fmt(type7(arm_calls)),
            "gmin_calls_median": "",
            "mannwhitney_u": "",
            "mannwhitney_p": "",
            "anneal_outside_s_per_call_median": f"{type7(outside):.12g}",
        }
        if not other:
            out.append(record)
            continue
        if len(other) != 12:
            raise SystemExit(f"{name}: GMIN n differs from 12")
        revisions = {row["revision"] for row in other}
        if revisions != {GMIN_REVISION}:
            raise SystemExit(f"{name}: GMIN revision differs")
        other_hit = [row["hit"] == "1" for row in other]
        other_calls = [
            float(row["evals_to_hit"]) if hit else math.inf
            for row, hit in zip(other, other_hit, strict=True)
        ]
        gh = sum(other_hit)
        u, mw_p = mann_whitney(arm_calls, other_calls)
        record["gmin_hits"] = str(gh)
        record["gmin_revision"] = GMIN_REVISION
        record["fisher_p"] = f"{fisher(ah, gh):.6g}"
        record["gmin_calls_median"] = fmt(type7(other_calls))
        record["mannwhitney_u"] = f"{u:.6g}"
        record["mannwhitney_p"] = f"{mw_p:.6g}"
        out.append(record)
    return out


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    check_specs()
    measured, paired = g1_rows()
    g0 = g0_rows()
    combined = [*g0, *({**row, "source": "g1"} for row in measured)]
    comparisons = gmin_arm(load(ROOT / "bench/gmin-panel/rows.csv"), panel_references())
    write(HERE / "rows.csv", ROW_FIELDS, paired)
    write(HERE / "panel-rows.csv", PANEL_FIELDS, panel_rows(measured))
    write(HERE / "summary.csv", SUMMARY_FIELDS, summary(combined, comparisons))


if __name__ == "__main__":
    main()
