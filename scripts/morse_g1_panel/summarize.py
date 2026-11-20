#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize the fresh panel and audit the stored G1 comparison rows."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

from run_panel import CELLS, GMIN_REVISION, SEEDS

SUMMARY_FIELDS = (
    "system",
    "rho",
    "budget",
    "reference_energy",
    "anneal_hits",
    "gmin_hits",
    "fisher_p",
    "anneal_calls_median",
    "gmin_calls_median",
    "mwu_u_calls",
    "mwu_p_calls",
    "anneal_wall_median_s",
    "gmin_wall_median_s",
)
CHECK_FIELDS = (
    "system",
    "seed",
    "calls_equal",
    "energy_equal",
    "fresh_calls",
    "stored_calls",
    "fresh_best_bits",
    "stored_best_bits",
    "status",
)
PROVENANCE_FIELDS = (
    "source_table",
    "system",
    "seed",
    "presented_label",
    "code",
    "revision",
    "host",
    "source",
    "classification",
    "eligible_gmin_comparison",
)
COMPARISON_FIELDS = (
    "system",
    "fresh_gmin_hits",
    "stored_gmin_hits",
    "fresh_calls_median",
    "stored_calls_median",
    "fresh_wall_median_s",
    "stored_wall_median_s",
    "stored_code",
    "stored_revision",
    "stored_host",
    "fresh_input",
    "stored_input",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def type7(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    position = (len(ordered) - 1) * 0.5
    lower = int(position)
    fraction = position - lower
    if fraction == 0.0:
        return ordered[lower]
    return ordered[lower] * (1.0 - fraction) + ordered[lower + 1] * fraction


def fisher(a_hits: int, a_n: int, b_hits: int, b_n: int) -> float:
    total_hits = a_hits + b_hits
    denominator = math.comb(a_n + b_n, total_hits)

    def probability(value: int) -> float:
        return (
            math.comb(a_n, value)
            * math.comb(b_n, total_hits - value)
            / denominator
        )

    observed = probability(a_hits)
    low = max(0, total_hits - b_n)
    high = min(a_n, total_hits)
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(low, high + 1)
            if probability(value) <= observed * (1.0 + 1e-12)
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
        count = end - index + 1
        ranks[pooled[index]] = (index + 1 + end + 1) / 2.0
        tie_sum += count**3 - count
        index = end + 1
    n_left = len(left)
    n_right = len(right)
    rank_left = sum(ranks[value] for value in left)
    u_left = rank_left - n_left * (n_left + 1) / 2.0
    u = min(u_left, n_left * n_right - u_left)
    mean = n_left * n_right / 2.0
    total = n_left + n_right
    variance = n_left * n_right / 12.0 * (
        total + 1 - tie_sum / (total * (total - 1))
    )
    if variance <= 0:
        return u, 1.0
    z_value = max(0.0, (abs(u - mean) - 0.5) / math.sqrt(variance))
    return u, math.erfc(z_value / math.sqrt(2.0))


def groups(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    result: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[(row["system"], row["arm"])].append(row)
    for values in result.values():
        values.sort(key=lambda row: int(row["seed"]))
    return result


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = groups(rows)
    output = []
    for system, _n, rho, budget, reference in CELLS:
        anneal = grouped[(system, "anneal")]
        gmin = grouped[(system, "gmin")]
        expected = list(SEEDS)
        if [int(row["seed"]) for row in anneal] != expected:
            raise RuntimeError(f"{system}: incomplete anneal rows")
        if [int(row["seed"]) for row in gmin] != expected:
            raise RuntimeError(f"{system}: incomplete GMIN rows")
        a_hits = sum(row["hit"] == "1" for row in anneal)
        g_hits = sum(row["hit"] == "1" for row in gmin)
        a_ranked = [
            float(row["calls"]) if row["hit"] == "1" else math.inf
            for row in anneal
        ]
        g_ranked = [
            float(row["calls"]) if row["hit"] == "1" else math.inf
            for row in gmin
        ]
        u_value, p_value = mann_whitney(a_ranked, g_ranked)
        output.append(
            {
                "system": system,
                "rho": f"{rho:g}",
                "budget": str(budget),
                "reference_energy": f"{reference:.6f}",
                "anneal_hits": str(a_hits),
                "gmin_hits": str(g_hits),
                "fisher_p": f"{fisher(a_hits, 12, g_hits, 12):.9g}",
                "anneal_calls_median": f"{type7([float(row['calls']) for row in anneal]):.2f}",
                "gmin_calls_median": f"{type7([float(row['calls']) for row in gmin]):.2f}",
                "mwu_u_calls": f"{u_value:.1f}",
                "mwu_p_calls": f"{p_value:.9g}",
                "anneal_wall_median_s": f"{type7([float(row['wall_s']) for row in anneal]):.9f}",
                "gmin_wall_median_s": f"{type7([float(row['wall_s']) for row in gmin]):.9f}",
            }
        )
    return output


def best_bits(value: str) -> str:
    import struct

    return struct.pack(">d", float(value)).hex()


def check_anneal(
    fresh: list[dict[str, str]], stored: list[dict[str, str]]
) -> list[dict[str, str]]:
    current = {
        (row["system"], row["seed"]): row
        for row in fresh
        if row["arm"] == "anneal"
    }
    prior = {
        (row["system"], row["seed"]): row
        for row in stored
        if row.get("code") == "anneal"
    }
    output = []
    for system, _n, _rho, _budget, _reference in CELLS:
        for seed in SEEDS:
            key = (system, str(seed))
            if key not in prior:
                output.append(
                    {
                        "system": system,
                        "seed": str(seed),
                        "calls_equal": "",
                        "energy_equal": "",
                        "fresh_calls": current[key]["calls"],
                        "stored_calls": "",
                        "fresh_best_bits": current[key]["best_bits"],
                        "stored_best_bits": "",
                        "status": "not present in G1 rows",
                    }
                )
                continue
            new = current[key]
            old = prior[key]
            old_bits = best_bits(old["best"])
            calls_equal = int(new["calls"]) == int(old["evals_to_hit"])
            energy_equal = new["best_bits"] == old_bits
            output.append(
                {
                    "system": system,
                    "seed": str(seed),
                    "calls_equal": "1" if calls_equal else "0",
                    "energy_equal": "1" if energy_equal else "0",
                    "fresh_calls": new["calls"],
                    "stored_calls": old["evals_to_hit"],
                    "fresh_best_bits": new["best_bits"],
                    "stored_best_bits": old_bits,
                    "status": "match" if calls_equal and energy_equal else "mismatch",
                }
            )
    return output


def provenance(
    stored: list[dict[str, str]], panel: list[dict[str, str]]
) -> list[dict[str, str]]:
    output = []
    for row in stored:
        if row.get("code") != "gmin":
            continue
        revision = row.get("code_revision") or "none recorded"
        host = row.get("host") or "none recorded"
        eligible = revision != "none recorded"
        output.append(
            {
                "source_table": "rows.csv",
                "system": row["system"],
                "seed": row["seed"],
                "presented_label": "gmin",
                "code": row.get("code") or "none recorded",
                "revision": revision,
                "host": host,
                "source": row.get("source") or "none recorded",
                "classification": (
                    "recorded GMIN" if eligible else "insufficient provenance"
                ),
                "eligible_gmin_comparison": "1" if eligible else "0",
            }
        )
    for row in panel:
        label = row.get("cell", "")
        if not label.startswith("gmin-m"):
            continue
        output.append(
            {
                "source_table": "panel-rows.csv",
                "system": label.removeprefix("gmin-"),
                "seed": row["seed"],
                "presented_label": label,
                "code": "none recorded",
                "revision": row.get("revision") or "none recorded",
                "host": row.get("machine_id_sha256") or "none recorded",
                "source": row.get("source") or "none recorded",
                "classification": "anneal row under gmin label",
                "eligible_gmin_comparison": "0",
            }
        )
    return output


def compare(
    fresh: list[dict[str, str]], stored: list[dict[str, str]]
) -> list[dict[str, str]]:
    current = groups(fresh)
    prior: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stored:
        if row.get("code") == "gmin" and row.get("code_revision"):
            prior[row["system"]].append(row)
    output = []
    for system, values in sorted(prior.items()):
        if (system, "gmin") not in current:
            continue
        values.sort(key=lambda row: int(row["seed"]))
        new = current[(system, "gmin")]
        revisions = " ".join(sorted({row["code_revision"] for row in values}))
        hosts = " ".join(sorted({row["host"] or "none recorded" for row in values}))
        output.append(
            {
                "system": system,
                "fresh_gmin_hits": str(sum(row["hit"] == "1" for row in new)),
                "stored_gmin_hits": str(sum(row["hit"] == "1" for row in values)),
                "fresh_calls_median": f"{type7([float(row['calls']) for row in new]):.2f}",
                "stored_calls_median": f"{type7([float(row['evals_to_hit']) for row in values]):.2f}",
                "fresh_wall_median_s": f"{type7([float(row['wall_s']) for row in new]):.9f}",
                "stored_wall_median_s": f"{type7([float(row['wall_s']) for row in values]):.9f}",
                "stored_code": "gmin",
                "stored_revision": revisions,
                "stored_host": hosts,
                "fresh_input": (
                    "example Morse keywords; explicit native Morse default "
                    f"radius; revision {GMIN_REVISION}"
                ),
                "stored_input": (
                    "example Morse keywords; no explicit radius; recorded "
                    f"revision {revisions}"
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--g1-rows", type=Path, required=True)
    parser.add_argument("--g1-panel-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh = read_csv(args.rows)
    stored = read_csv(args.g1_rows)
    panel = read_csv(args.g1_panel_rows)
    write_csv(args.output / "summary.csv", SUMMARY_FIELDS, summarize(fresh))
    write_csv(
        args.output / "anneal-check.csv",
        CHECK_FIELDS,
        check_anneal(fresh, stored),
    )
    write_csv(
        args.output / "g1-stored-provenance.csv",
        PROVENANCE_FIELDS,
        provenance(stored, panel),
    )
    write_csv(
        args.output / "gmin-comparison.csv",
        COMPARISON_FIELDS,
        compare(fresh, stored),
    )


if __name__ == "__main__":
    main()
