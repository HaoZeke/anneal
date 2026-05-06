"""Summarize a long-form CUTEst benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Coverage:
    problem_count: int
    cell_count: int
    kind_count: int


@dataclass(frozen=True)
class DriverSummary:
    driver: str
    cells: int
    ok: int
    timeout: int
    error: int
    solved: int
    solved_rate: float
    median_fevals: float
    median_best: float
    best_cells: int
    best_share: float


@dataclass(frozen=True)
class Summary:
    coverage: Coverage
    by_driver: dict[str, DriverSummary]


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int_or_zero(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best_cells(rows: list[dict]) -> dict[str, int]:
    cells: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        status = row.get("status", "ok")
        best_val = _float_or_nan(row.get("best_val"))
        if status != "ok" or not math.isfinite(best_val):
            continue
        cells[(str(row["problem"]), str(row["seed"]))].append((str(row["driver"]), best_val))

    wins: dict[str, int] = defaultdict(int)
    for entries in cells.values():
        if not entries:
            continue
        best = min(value for _driver, value in entries)
        winners = [driver for driver, value in entries if value == best]
        for driver in winners:
            wins[driver] += 1
    return dict(wins)


def summarize_rows(rows: list[dict]) -> Summary:
    problems = {str(row["problem"]) for row in rows}
    kinds = {str(row.get("kind", "")) for row in rows if row.get("kind")}
    wins = _best_cells(rows)

    by_driver_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_driver_rows[str(row["driver"])].append(row)

    summaries: dict[str, DriverSummary] = {}
    for driver, sub in sorted(by_driver_rows.items()):
        ok_rows = [
            row for row in sub
            if row.get("status", "ok") == "ok" and math.isfinite(_float_or_nan(row.get("best_val")))
        ]
        timeout = sum(1 for row in sub if row.get("status") == "timeout")
        error = sum(1 for row in sub if str(row.get("status", "")).startswith("err:"))
        solved = sum(_int_or_zero(row.get("solved")) for row in sub)
        fevals = [_float_or_nan(row.get("fevals")) for row in ok_rows]
        best_vals = [_float_or_nan(row.get("best_val")) for row in ok_rows]
        best_cells = wins.get(driver, 0)
        summaries[driver] = DriverSummary(
            driver=driver,
            cells=len(sub),
            ok=len(ok_rows),
            timeout=timeout,
            error=error,
            solved=solved,
            solved_rate=solved / len(sub) if sub else 0.0,
            median_fevals=float(np.median(fevals)) if fevals else float("nan"),
            median_best=float(np.median(best_vals)) if best_vals else float("nan"),
            best_cells=best_cells,
            best_share=best_cells / len(problems) if problems else 0.0,
        )

    return Summary(
        coverage=Coverage(
            problem_count=len(problems),
            cell_count=len(rows),
            kind_count=len(kinds),
        ),
        by_driver=summaries,
    )


def write_summary(path: Path, summary: Summary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "driver",
                "cells",
                "ok",
                "timeout",
                "error",
                "solved",
                "solved_rate",
                "median_fevals",
                "median_best",
                "best_cells",
                "best_share",
            ],
        )
        writer.writeheader()
        for driver in sorted(summary.by_driver):
            item = summary.by_driver[driver]
            writer.writerow(
                {
                    "driver": item.driver,
                    "cells": item.cells,
                    "ok": item.ok,
                    "timeout": item.timeout,
                    "error": item.error,
                    "solved": item.solved,
                    "solved_rate": f"{item.solved_rate:.6g}",
                    "median_fevals": f"{item.median_fevals:.6g}",
                    "median_best": f"{item.median_best:.6g}",
                    "best_cells": item.best_cells,
                    "best_share": f"{item.best_share:.6g}",
                }
            )


def summarize_csv(csv_path: Path | str, out_path: Path | str | None = None) -> Summary:
    rows = load_rows(Path(csv_path))
    summary = summarize_rows(rows)
    if out_path is not None:
        write_summary(Path(out_path), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out", default="data/cutest_summary.csv")
    parser.add_argument("--min-problems", type=int, default=1)
    parser.add_argument("--min-ok-rate", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = summarize_csv(args.csv_path, args.out)
    print(
        f"Problems: {summary.coverage.problem_count}; "
        f"cells: {summary.coverage.cell_count}; "
        f"kinds: {summary.coverage.kind_count}"
    )
    total_ok = sum(item.ok for item in summary.by_driver.values())
    total_cells = sum(item.cells for item in summary.by_driver.values())
    ok_rate = total_ok / total_cells if total_cells else 0.0
    for driver, item in summary.by_driver.items():
        print(
            f"  {driver:<16} ok {item.ok}/{item.cells}, "
            f"solved {item.solved}, timeout {item.timeout}, error {item.error}, "
            f"best-cells {item.best_cells}"
        )
    if summary.coverage.problem_count < args.min_problems:
        print(
            f"Expected at least {args.min_problems} problem(s); "
            f"found {summary.coverage.problem_count}",
            file=sys.stderr,
        )
        return 1
    if ok_rate < args.min_ok_rate:
        print(f"Expected ok-rate >= {args.min_ok_rate:g}; found {ok_rate:g}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
