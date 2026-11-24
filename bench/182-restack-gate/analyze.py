#!/usr/bin/env python3
"""Registered summary for the restack gate. Rules live in PROTOCOL.md."""

from __future__ import annotations

import csv
import math
import random
import statistics
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(2000, 2012))
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20_261_123
LIMIT = 1.10
ENERGY_NOISE = 1e-9
CLUSTER_ARM = {"D": "A", "D2": "A2", "G": "B", "P": "P"}
DEFAULT = ("lj38", "lj75", "lj98", "m55r6")
PROBE = (
    "lj55",
    "m38r14",
    "m38r6",
    "sc126-38",
    "sc126-55",
    "sc108-38",
    "sc96-38",
    "sc96-55",
)
TIP4P = ("tip4p9", "tip4p10", "tip4p14", "tip4p16", "tip4p17")
ORDER = (
    "m38r14",
    "sc126-38",
    "sc126-55",
    "sc108-38",
    "sc96-38",
    "sc96-55",
    "lj38",
    "lj55",
    "m38r6",
    "m55r6",
    "lj98",
    "lj75",
    *TIP4P,
)
# Previous gate, tree d62db742, published B evaluation at the named seed.
ANCHORS = {
    "lj38": (2000, 603),
    "lj75": (2001, 3329914),
    "lj98": (2000, 365919),
    "m55r6": (2000, 1156),
    "lj55": (2000, 1487),
    "m38r14": (2000, 1500),
    "m38r6": (2000, 1615),
    "sc126-38": (2000, 1164),
    "sc126-55": (2000, 2911),
    "sc108-38": (2000, 1678),
    "sc96-38": (2000, 1289),
    "sc96-55": (2000, 2685),
}
FIELDS = (
    "cell",
    "category",
    "arm",
    "seed",
    "budget",
    "hit",
    "evals",
    "charged",
    "calls_raw",
    "best",
    "best_bits",
    "outside_s",
    "per_call_s",
    "rss_kb",
    "rss_nonzero",
    "packing_offered",
    "packing_entered",
    "code_tree",
    "binary_sha256",
    "b_reps",
    "b_agree",
)
SUMMARY_FIELDS = (
    "cell",
    "category",
    "n",
    "status",
    "performance",
    "hits_a",
    "hits_b",
    "hits_p",
    "hits_a2",
    "no_hit_loss",
    "fisher_b_a",
    "fisher_b_p",
    "mwu_b_a",
    "mwu_b_p",
    "outside_b_a",
    "outside_b_a_lo",
    "outside_b_a_hi",
    "outside_a2_a",
    "outside_a2_a_lo",
    "outside_a2_a_hi",
    "outside_limit",
    "outside_strict",
    "outside_adjusted",
    "outside_real",
    "per_call_b_a",
    "per_call_b_a_lo",
    "per_call_b_a_hi",
    "per_call_strict",
    "early_stop_artifact",
    "rss_b_a",
    "rss_b_a_lo",
    "rss_b_a_hi",
    "rss_a2_a",
    "rss_a2_a_lo",
    "rss_a2_a_hi",
    "rss_limit",
    "rss_pass",
    "rss_real",
    "parity_b_p",
    "identity_b_a",
    "identity_note",
    "anchor_seed",
    "anchor_evals",
    "anchor_match",
    "first_b_p_difference",
    "first_b_a_difference",
)


def category(cell: str) -> str:
    if cell in DEFAULT:
        return "default"
    if cell in PROBE:
        return "probe"
    return "tip4p"


def type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def bootstrap(values: list[float], statistic) -> tuple[float, float, float]:
    generator = random.Random(BOOTSTRAP_SEED)
    estimate = statistic(values)
    size = len(values)
    samples = [
        statistic([values[generator.randrange(size)] for _ in range(size)])
        for _ in range(BOOTSTRAPS)
    ]
    return estimate, type7(samples, 0.025), type7(samples, 0.975)


def fmt(value: float) -> str:
    return f"{value:.8g}"


def bits(text: str) -> str:
    return struct.pack(">d", float(text)).hex()


def fisher_two_sided(hits_left: int, hits_right: int, n: int) -> float:
    total = hits_left + hits_right
    denominator = math.comb(2 * n, total)

    def probability(left: int) -> float:
        right = total - left
        if not 0 <= left <= n or not 0 <= right <= n:
            return 0.0
        return math.comb(n, left) * math.comb(n, right) / denominator

    observed = probability(hits_left)
    return min(
        1.0,
        sum(
            probability(left)
            for left in range(n + 1)
            if probability(left) <= observed + 1.0e-15
        ),
    )


def mann_whitney_two_sided(left: list[float], right: list[float]) -> float:
    pooled = sorted(left + right)
    ranks: dict[float, float] = {}
    tie_term = 0
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[index]:
            end += 1
        ranks[pooled[index]] = (index + end + 2) / 2
        width = end - index + 1
        tie_term += width**3 - width
        index = end + 1
    n_left, n_right = len(left), len(right)
    u_left = sum(ranks[value] for value in left) - n_left * (n_left + 1) / 2
    mean = n_left * n_right / 2
    count = n_left + n_right
    variance = n_left * n_right / 12 * (
        count + 1 - tie_term / (count * (count - 1))
    )
    if variance <= 0:
        return 1.0
    z_score = (abs(u_left - mean) - 0.5) / math.sqrt(variance)
    return min(1.0, math.erfc(z_score / math.sqrt(2)))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cluster_rows(path: Path) -> list[dict[str, str]]:
    output = []
    for row in read_csv(path):
        arm = CLUSTER_ARM[row["arm"]]
        charged = int(row["charged"])
        outside = float(row["outside_s"])
        per_call = float(row["outside_per_call_s"])
        rss = int(row["rss_anon_self_peak_kb"])
        output.append(
            {
                "cell": row["system"],
                "category": category(row["system"]),
                "arm": arm,
                "seed": row["seed"],
                "budget": row["budget"],
                "hit": row["hit"],
                "evals": row["evals_to_hit"],
                "charged": str(charged),
                "calls_raw": row["evals_to_hit"],
                "best": row["best"],
                "best_bits": bits(row["best"]),
                "outside_s": f"{outside:.8g}",
                "per_call_s": f"{per_call:.8g}",
                "rss_kb": str(rss),
                "rss_nonzero": "1" if rss > 0 else "0",
                "packing_offered": row["packing_offered"],
                "packing_entered": str(int(int(row["packing_probe_calls"]) > 0)),
                "code_tree": row["code_tree"],
                "binary_sha256": row["binary_sha256"],
                "b_reps": "1",
                "b_agree": "1",
            }
        )
    return output


def tip_rows(path: Path) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in read_csv(path):
        grouped.setdefault((row["cell"], row["arm"], row["seed"]), []).append(row)
    output = []
    for (cell, arm, seed), rows in sorted(grouped.items()):
        agree = 1
        if arm == "B":
            keys = ("hit", "calls", "charged", "energy_bits")
            agree = int(all(
                all(row[key] == rows[0][key] for key in keys) for row in rows
            ))
        outside = statistics.fmean(float(row["outside_s"]) for row in rows)
        per_call = statistics.fmean(float(row["per_call_s"]) for row in rows)
        rss = statistics.fmean(float(row["rss_anon_end_kb"]) for row in rows)
        hit = rows[0]["hit"]
        calls = rows[0]["calls"]
        output.append(
            {
                "cell": cell,
                "category": "tip4p",
                "arm": arm,
                "seed": seed,
                "budget": rows[0]["budget"],
                "hit": hit,
                "evals": calls,
                "charged": rows[0]["charged"],
                "calls_raw": calls,
                "best": rows[0]["energy"],
                "best_bits": rows[0]["energy_bits"],
                "outside_s": f"{outside:.8g}",
                "per_call_s": f"{per_call:.8g}",
                "rss_kb": f"{rss:.8g}",
                "rss_nonzero": "1" if rss > 0 else "0",
                "packing_offered": "0",
                "packing_entered": "0",
                "code_tree": rows[0]["tree"],
                "binary_sha256": rows[0]["binary_sha256"],
                "b_reps": str(len(rows)),
                "b_agree": str(agree),
            }
        )
    return output


def index_cell(rows: list[dict[str, str]], cell: str) -> dict[tuple[str, int], dict[str, str]]:
    return {
        (row["arm"], int(row["seed"])): row
        for row in rows
        if row["cell"] == cell
    }


def scored(row: dict[str, str]) -> float:
    if row["hit"] == "1":
        return float(row["calls_raw"])
    return float(row["budget"]) + 1.0


def same_record(left: dict[str, str], right: dict[str, str], fields: tuple[str, ...]) -> bool:
    return all(left[field] == right[field] for field in fields)


def first_difference(
    indexed: dict[tuple[str, int], dict[str, str]],
    left_arm: str,
    right_arm: str,
) -> str:
    fields = ("hit", "evals", "charged", "calls_raw", "best_bits")
    for seed in SEEDS:
        left = indexed.get((left_arm, seed))
        right = indexed.get((right_arm, seed))
        if left is None or right is None:
            continue
        for field in fields:
            if left[field] != right[field]:
                return f"seed={seed} field={field} {left_arm}={left[field]} {right_arm}={right[field]}"
    return ""


def energy_agreement(left: dict[str, str], right: dict[str, str]) -> tuple[bool, str]:
    delta = abs(float(left["best"]) - float(right["best"]))
    bits_same = left["best_bits"] == right["best_bits"]
    if delta <= ENERGY_NOISE:
        if bits_same:
            return True, ""
        return True, f"energy noise {delta:.3g}"
    return False, f"energy delta {delta:.3g}"


def ratio_interval(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    values = [left / right for left, right in pairs]
    if any(value <= 0 for value in values):
        raise RuntimeError("nonpositive ratio")
    return bootstrap(values, geomean)


def blank(cell: str, status: str) -> dict[str, str]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update({"cell": cell, "category": category(cell), "n": "0", "status": status})
    return row


def summarize_cell(cell: str, rows: list[dict[str, str]]) -> dict[str, str]:
    indexed = index_cell(rows, cell)
    arms = ("A", "B", "A2") if cell in TIP4P else ("A", "B", "P", "A2")
    expected = {(arm, seed) for arm in arms for seed in SEEDS}
    if not set(indexed) & {(arm, seed) for arm, seed in expected}:
        return blank(cell, "NOT RUN")
    if set(indexed) != expected:
        return blank(cell, "PARTIAL")
    if cell in TIP4P and any(
        indexed[("B", seed)]["b_reps"] != "2" for seed in SEEDS
    ):
        return blank(cell, "PARTIAL")
    if any(row["rss_nonzero"] != "1" for row in indexed.values()):
        return blank(cell, "INVALID")
    kind = category(cell)
    hits = {
        arm: sum(int(indexed[(arm, seed)]["hit"]) for seed in SEEDS) for arm in arms
    }
    scores = {arm: [scored(indexed[(arm, seed)]) for seed in SEEDS] for arm in arms}
    no_hit_loss = hits["B"] >= hits["A"] and (kind == "tip4p" or hits["B"] >= hits["P"])
    outside_ba = ratio_interval(
        [
            (
                float(indexed[("B", seed)]["outside_s"]),
                float(indexed[("A", seed)]["outside_s"]),
            )
            for seed in SEEDS
        ]
    )
    outside_a2 = ratio_interval(
        [
            (
                float(indexed[("A2", seed)]["outside_s"]),
                float(indexed[("A", seed)]["outside_s"]),
            )
            for seed in SEEDS
        ]
    )
    per_ba = ratio_interval(
        [
            (
                float(indexed[("B", seed)]["per_call_s"]),
                float(indexed[("A", seed)]["per_call_s"]),
            )
            for seed in SEEDS
        ]
    )
    rss_ba = ratio_interval(
        [
            (float(indexed[("B", seed)]["rss_kb"]), float(indexed[("A", seed)]["rss_kb"]))
            for seed in SEEDS
        ]
    )
    rss_a2 = ratio_interval(
        [
            (
                float(indexed[("A2", seed)]["rss_kb"]),
                float(indexed[("A", seed)]["rss_kb"]),
            )
            for seed in SEEDS
        ]
    )
    outside_limit = max(LIMIT, outside_a2[2])
    rss_limit = max(LIMIT, rss_a2[2])
    outside_strict = outside_ba[2] <= LIMIT
    outside_adjusted = outside_ba[2] <= outside_limit
    outside_real = outside_ba[1] > 1 and outside_ba[1] > outside_a2[2]
    per_strict = per_ba[2] <= LIMIT
    calls_b = geomean([float(indexed[("B", seed)]["calls_raw"]) for seed in SEEDS])
    calls_a = geomean([float(indexed[("A", seed)]["calls_raw"]) for seed in SEEDS])
    early = (not per_strict) and outside_strict and calls_b < calls_a
    rss_pass = not (rss_ba[1] > rss_limit)
    rss_real = rss_ba[1] > 1 and rss_ba[1] > rss_a2[2]
    identity_fields = ("hit", "evals", "charged", "calls_raw")
    notes = []
    if kind == "tip4p":
        identity = True
        for seed in SEEDS:
            left = indexed[("B", seed)]
            right = indexed[("A", seed)]
            if left["b_agree"] != "1":
                identity = False
                notes.append(f"seed={seed} B repetitions disagree")
            if not same_record(left, right, identity_fields):
                identity = False
            agreed, note = energy_agreement(left, right)
            if not agreed:
                identity = False
            if note:
                notes.append(f"seed={seed} {note}")
        parity = ""
    else:
        parity_ok = all(
            same_record(
                indexed[("B", seed)],
                indexed[("P", seed)],
                ("hit", "evals", "charged", "calls_raw", "best_bits"),
            )
            for seed in SEEDS
        )
        parity = "1" if parity_ok else "0"
        identity = ""
    performance = no_hit_loss and outside_strict and rss_pass
    if kind == "tip4p":
        performance = performance and identity
    anchor = ANCHORS.get(cell)
    anchor_match = ""
    anchor_seed = ""
    anchor_evals = ""
    if anchor is not None:
        anchor_seed, anchor_evals = anchor
        observed = indexed[("B", int(anchor_seed))]["evals"]
        anchor_match = "1" if int(observed) == int(anchor_evals) else "0"
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "cell": cell,
            "category": kind,
            "n": str(len(SEEDS)),
            "status": "COMPLETE",
            "performance": "PASS" if performance else "FAIL",
            "hits_a": str(hits["A"]),
            "hits_b": str(hits["B"]),
            "hits_p": "" if kind == "tip4p" else str(hits["P"]),
            "hits_a2": str(hits["A2"]),
            "no_hit_loss": str(int(no_hit_loss)),
            "fisher_b_a": fmt(fisher_two_sided(hits["B"], hits["A"], len(SEEDS))),
            "fisher_b_p": ""
            if kind == "tip4p"
            else fmt(fisher_two_sided(hits["B"], hits["P"], len(SEEDS))),
            "mwu_b_a": fmt(mann_whitney_two_sided(scores["B"], scores["A"])),
            "mwu_b_p": ""
            if kind == "tip4p"
            else fmt(mann_whitney_two_sided(scores["B"], scores["P"])),
            "outside_b_a": fmt(outside_ba[0]),
            "outside_b_a_lo": fmt(outside_ba[1]),
            "outside_b_a_hi": fmt(outside_ba[2]),
            "outside_a2_a": fmt(outside_a2[0]),
            "outside_a2_a_lo": fmt(outside_a2[1]),
            "outside_a2_a_hi": fmt(outside_a2[2]),
            "outside_limit": fmt(outside_limit),
            "outside_strict": str(int(outside_strict)),
            "outside_adjusted": str(int(outside_adjusted)),
            "outside_real": str(int(outside_real)),
            "per_call_b_a": fmt(per_ba[0]),
            "per_call_b_a_lo": fmt(per_ba[1]),
            "per_call_b_a_hi": fmt(per_ba[2]),
            "per_call_strict": str(int(per_strict)),
            "early_stop_artifact": str(int(early)),
            "rss_b_a": fmt(rss_ba[0]),
            "rss_b_a_lo": fmt(rss_ba[1]),
            "rss_b_a_hi": fmt(rss_ba[2]),
            "rss_a2_a": fmt(rss_a2[0]),
            "rss_a2_a_lo": fmt(rss_a2[1]),
            "rss_a2_a_hi": fmt(rss_a2[2]),
            "rss_limit": fmt(rss_limit),
            "rss_pass": str(int(rss_pass)),
            "rss_real": str(int(rss_real)),
            "parity_b_p": parity,
            "identity_b_a": "" if identity == "" else str(int(identity)),
            "identity_note": "; ".join(notes),
            "anchor_seed": str(anchor_seed),
            "anchor_evals": str(anchor_evals),
            "anchor_match": anchor_match,
            "first_b_p_difference": ""
            if kind == "tip4p"
            else first_difference(indexed, "B", "P"),
            "first_b_a_difference": first_difference(indexed, "B", "A"),
        }
    )
    return row


def write_table(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary: list[dict[str, str]], fingerprint: list[str]) -> None:
    perf_fail = [row for row in summary if row["performance"] == "FAIL"]
    parity_fail = [row for row in summary if row["parity_b_p"] == "0"]
    identity_fail = [row for row in summary if row["identity_b_a"] == "0"]
    incomplete = [row for row in summary if row["status"] != "COMPLETE"]
    if incomplete:
        overall = "PARTIAL"
    elif perf_fail or parity_fail or identity_fail:
        overall = "FAIL"
    else:
        overall = "PASS"
    lines = [
        "# Restack gate",
        "",
        f"Overall: {overall}.",
        "",
        "Performance uses hit loss, strict total outside time, and the",
        "one-sided RssAnon bar. Parity is B against P on cluster cells.",
        "TIP4P identity is B against A. A strict time failure stays a",
        "failure when the adjusted limit would pass. Per-call time is not",
        "the time bar.",
        "",
        "| Cell | Hits A/B/P/A2 | Outside B/A | Rss B/A | Perf | Parity | Id |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        if row["status"] != "COMPLETE":
            lines.append(
                f"| {row['cell']} | - | - | - | {row['status']} | - | - |"
            )
            continue
        hits_p = row["hits_p"] or "-"
        parity = row["parity_b_p"] or "-"
        identity = row["identity_b_a"] or "-"
        lines.append(
            f"| {row['cell']} | {row['hits_a']}/{row['hits_b']}/{hits_p}/"
            f"{row['hits_a2']} | {row['outside_b_a']} "
            f"[{row['outside_b_a_lo']}, {row['outside_b_a_hi']}] | "
            f"{row['rss_b_a']} [{row['rss_b_a_lo']}, {row['rss_b_a_hi']}] | "
            f"{row['performance']} | {parity} | {identity} |"
        )
    lines += ["", "## Gates", ""]
    for row in summary:
        if row["status"] != "COMPLETE":
            lines.append(f"- {row['cell']}: {row['status']}")
            continue
        lines.append(
            f"- {row['cell']}: hits {row['no_hit_loss']} "
            f"Fisher B/A {row['fisher_b_a']} MWU B/A {row['mwu_b_a']}; "
            f"time strict {row['outside_strict']} adjusted {row['outside_adjusted']} "
            f"real {row['outside_real']} limit {row['outside_limit']}; "
            f"per-call {row['per_call_b_a']} "
            f"[{row['per_call_b_a_lo']}, {row['per_call_b_a_hi']}] "
            f"artifact {row['early_stop_artifact']}; "
            f"rss pass {row['rss_pass']} real {row['rss_real']} "
            f"limit {row['rss_limit']}"
        )
    lines += ["", "## Parity and identity", ""]
    for row in summary:
        if row["status"] != "COMPLETE":
            continue
        extra = row["identity_note"]
        if row["first_b_p_difference"]:
            lines.append(f"- {row['cell']} B/P: {row['first_b_p_difference']}")
        if row["first_b_a_difference"]:
            lines.append(f"- {row['cell']} B/A: {row['first_b_a_difference']}")
        if row["anchor_seed"]:
            lines.append(
                f"- {row['cell']} anchor seed {row['anchor_seed']} "
                f"evals {row['anchor_evals']} match {row['anchor_match']}"
            )
        if extra:
            lines.append(f"- {row['cell']} note: {extra}")
    if fingerprint:
        lines += ["", "## Heated lattice exchange", ""]
        lines.extend(fingerprint)
    lines.append("")
    (HERE / "SUMMARY.md").write_text("\n".join(lines), newline="\n")


def main() -> int:
    cluster_path = HERE / "cluster-rows.csv"
    tip_path = HERE / "tip4p-rows.csv"
    combined = cluster_rows(cluster_path) + tip_rows(tip_path)
    combined.sort(key=lambda row: (ORDER.index(row["cell"]), int(row["seed"]), row["arm"]))
    write_table(HERE / "rows.csv", FIELDS, combined)
    with (HERE / "rows.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in combined:
            handle.write(__import__("json").dumps(row, separators=(",", ":")) + "\n")
    summary = [summarize_cell(cell, combined) for cell in ORDER]
    write_table(HERE / "summary.csv", SUMMARY_FIELDS, summary)
    fingerprint_path = HERE / "fingerprint.txt"
    fingerprint = []
    if fingerprint_path.is_file():
        fingerprint = [
            line
            for line in fingerprint_path.read_text().splitlines()
            if line.strip()
        ]
    write_summary(summary, fingerprint)
    print("cells", len(summary), "rows", len(combined))
    return 0


if __name__ == "__main__":
    sys.exit(main())
