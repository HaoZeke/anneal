#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Derive plain CSV tables from the committed G3 trace."""

from __future__ import annotations

import argparse
import csv
import io
from collections import Counter, defaultdict
from pathlib import Path

import zstandard

FUNNELS = ("icosahedral", "decahedral", "fcc", "other")
PREFIXES = {
    "accepted": "accepted_minima",
    "calls": "force_calls",
    "hops": "observed_hop_steps",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open("rb") as compressed:
        reader = zstandard.ZstdDecompressor().stream_reader(compressed)
        with reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
            return list(csv.DictReader(text))


def write_csv(
    path: Path, fields: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def occupancy_fields(keys: list[str]) -> list[str]:
    fields = keys + [
        "runs",
        "accepted_minima_total",
        "force_calls_total",
        "force_calls_before_first",
        "observed_hop_steps_total",
        "observed_hop_steps_before_first",
    ]
    for funnel in FUNNELS:
        for prefix in PREFIXES.values():
            fields.append(f"{prefix}_{funnel}")
    return fields


def occupancy(
    row: dict[str, str], minima: list[dict[str, str]]
) -> dict[str, object]:
    accepted: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    hops: Counter[str] = Counter()
    charged = int(row["charged"])
    for minimum in minima:
        accepted[minimum["funnel"]] += 1
    call_values = [int(minimum["call"]) for minimum in minima]
    hop_values = [int(minimum["hop"]) for minimum in minima]
    calls_before = call_values[0] if minima else charged
    hops_before = hop_values[0] if minima else 0
    for index, minimum in enumerate(minima):
        next_call = (
            call_values[index + 1]
            if index + 1 < len(minima)
            else max(charged, call_values[index])
        )
        calls[minimum["funnel"]] += next_call - call_values[index]
        if index + 1 < len(minima):
            hops[minimum["funnel"]] += (
                hop_values[index + 1] - hop_values[index]
            )
    result: dict[str, object] = {
        "method": row["method"],
        "system": row["system"],
        "seed": row["seed"],
        "runs": 1,
        "accepted_minima_total": len(minima),
        "force_calls_total": charged,
        "force_calls_before_first": calls_before,
        "observed_hop_steps_total": hop_values[-1] if minima else 0,
        "observed_hop_steps_before_first": hops_before,
    }
    for funnel in FUNNELS:
        result[f"accepted_minima_{funnel}"] = accepted[funnel]
        result[f"force_calls_{funnel}"] = calls[funnel]
        result[f"observed_hop_steps_{funnel}"] = hops[funnel]
    return result


def sum_occupancy(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    fields = occupancy_fields([])
    totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = str(row["method"]), str(row["system"])
        for field in fields:
            totals[key][field] += int(row[field])
    result = []
    for (method, system), values in sorted(totals.items()):
        result.append(
            {"method": method, "system": system, **dict(values)}
        )
    return result


def transitions(
    minima_by_run: dict[tuple[str, str, str], list[dict[str, str]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for (method, system, seed), minima in minima_by_run.items():
        for previous, current in zip(minima, minima[1:]):
            source = previous["funnel"]
            target = current["funnel"]
            if source != target:
                run_counts[
                    (
                        method,
                        system,
                        seed,
                        source,
                        target,
                        current["move"],
                    )
                ] += 1
    run_rows = [
        {
            "method": key[0],
            "system": key[1],
            "seed": key[2],
            "from_funnel": key[3],
            "to_funnel": key[4],
            "move": key[5],
            "count": count,
        }
        for key, count in sorted(run_counts.items())
    ]
    crossed_runs = {key[:3] for key in run_counts}
    for method, system, seed in sorted(minima_by_run):
        if (method, system, seed) not in crossed_runs:
            run_rows.append(
                {
                    "method": method,
                    "system": system,
                    "seed": seed,
                    "from_funnel": "",
                    "to_funnel": "",
                    "move": "",
                    "count": 0,
                }
            )
    run_rows.sort(
        key=lambda row: (
            str(row["method"]),
            str(row["system"]),
            int(str(row["seed"])),
            str(row["from_funnel"]),
            str(row["to_funnel"]),
            str(row["move"]),
        )
    )
    summary_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for key, count in run_counts.items():
        summary_counts[(key[0], key[1], key[3], key[4], key[5])] += count
    summary_rows = [
        {
            "method": key[0],
            "system": key[1],
            "from_funnel": key[2],
            "to_funnel": key[3],
            "move": key[4],
            "count": count,
        }
        for key, count in sorted(summary_counts.items())
    ]
    crossed_panels = {key[:2] for key in summary_counts}
    for method, system, _seed in sorted(minima_by_run):
        if (method, system) not in crossed_panels:
            summary_rows.append(
                {
                    "method": method,
                    "system": system,
                    "from_funnel": "",
                    "to_funnel": "",
                    "move": "",
                    "count": 0,
                }
            )
            crossed_panels.add((method, system))
    summary_rows.sort(
        key=lambda row: (
            str(row["method"]),
            str(row["system"]),
            str(row["from_funnel"]),
            str(row["to_funnel"]),
            str(row["move"]),
        )
    )
    return run_rows, summary_rows


def misses(
    rows: list[dict[str, str]], classified: list[dict[str, str]]
) -> list[dict[str, object]]:
    row_by_run = {
        (row["method"], row["system"], row["seed"]): row for row in rows
    }
    result = []
    for miss in classified:
        key = miss["method"], miss["system"], miss["seed"]
        row = row_by_run[key]
        if row["hit"] == "1":
            raise ValueError(f"classified hit present for {key}")
        result.append(
            {
                "method": miss["method"],
                "system": miss["system"],
                "seed": miss["seed"],
                "final_energy": row["best"],
                "point_group": miss["point_group"],
                "cna_class": miss["cna_class"],
                "miss_class": miss["miss_class"],
                "structure": miss["structure"],
                "con_path": f"misses/{miss['con']}",
            }
        )
    return sorted(
        result,
        key=lambda item: (
            str(item["method"]),
            str(item["system"]),
            int(str(item["seed"])),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def main() -> int:
    directory = parse_args().directory
    rows = read_csv(directory / "rows.csv")
    classified = read_csv(directory / "misses.csv")
    trace = read_trace(directory / "minima.csv.zst")
    minima_by_run: dict[
        tuple[str, str, str], list[dict[str, str]]
    ] = defaultdict(list)
    for minimum in trace:
        key = minimum["method"], minimum["system"], minimum["seed"]
        minima_by_run[key].append(minimum)
    occupancy_rows = []
    for row in rows:
        key = row["method"], row["system"], row["seed"]
        minima = minima_by_run[key]
        minima.sort(key=lambda item: int(item["index"]))
        occupancy_rows.append(occupancy(row, minima))
    occupancy_rows.sort(
        key=lambda item: (
            str(item["method"]),
            str(item["system"]),
            int(str(item["seed"])),
        )
    )
    occupancy_summary = sum_occupancy(occupancy_rows)
    transition_rows, transition_summary = transitions(minima_by_run)
    occupancy_run_fields = occupancy_fields(["method", "system", "seed"])
    occupancy_summary_fields = occupancy_fields(["method", "system"])
    transition_run_fields = [
        "method",
        "system",
        "seed",
        "from_funnel",
        "to_funnel",
        "move",
        "count",
    ]
    transition_summary_fields = [
        "method",
        "system",
        "from_funnel",
        "to_funnel",
        "move",
        "count",
    ]
    miss_fields = [
        "method",
        "system",
        "seed",
        "final_energy",
        "point_group",
        "cna_class",
        "miss_class",
        "structure",
        "con_path",
    ]
    write_csv(
        directory / "funnel_occupancy_runs.csv",
        occupancy_run_fields,
        occupancy_rows,
    )
    write_csv(
        directory / "funnel_occupancy_summary.csv",
        occupancy_summary_fields,
        occupancy_summary,
    )
    write_csv(
        directory / "funnel_transitions_runs.csv",
        transition_run_fields,
        transition_rows,
    )
    write_csv(
        directory / "funnel_transitions_summary.csv",
        transition_summary_fields,
        transition_summary,
    )
    write_csv(
        directory / "funnel_misses.csv",
        miss_fields,
        misses(rows, classified),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
