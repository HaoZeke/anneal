# Copyright (c) anneal developers
"""Gate table for the develop re-session of the native-unit ASE panel.

The table is computed from the re-session rows. Misses are ranked strictly
worse than every hit. The run is budget only: there is no wall-clock cap.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mapping import (  # noqa: E402
    ASE_ARMS,
    BASE_SHA,
    SYSTEMS,
    fisher_exact_two_sided,
    mann_whitney_exact,
    median,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench" / "ase-native"
SHORT = BASE_SHA[:8]
NEW_ROWS = DATA / f"rows-{SHORT}.csv"
OLD_ROWS = DATA / "rows.csv"
OUT = DATA / f"gate-{SHORT}.md"


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def scored(row: dict[str, str]) -> float:
    """Evaluations to the hit. A miss sorts after every hit."""
    if int(row["hit"]) == 1:
        return float(row["evals_to_hit"])
    return float(row["budget"]) + 1.0


def fmt(value: float) -> str:
    if value != value:
        return ""
    return f"{value:.6g}"


def cell_rows(rows: list[dict[str, str]], system: str, arm: str) -> list[dict[str, str]]:
    return [row for row in rows if row["system"] == system and row["arm"] == arm]


def hits(rows: list[dict[str, str]]) -> int:
    return sum(int(row["hit"]) for row in rows)


def render(new_rows: list[dict[str, str]], old_rows: list[dict[str, str]]) -> str:
    preprune_blobs = next(row["blobs"] for row in old_rows if row["system"] == "lj38")
    preprune = [row for row in old_rows if row["blobs"] == preprune_blobs]
    lines = [
        f"Native-unit ASE gate on {SHORT}",
        "",
        "This comparison is budget only. There is no wall-clock cap.",
        "A hit is a finite best energy at or below the reference plus 0.001.",
        "That hit column is the accept count: a run accepts the reference or it does not.",
        "Mann-Whitney uses evaluations to the hit. A miss is ranked strictly worse than every hit,",
        "and misses tie with each other.",
        "Fisher is the two-sided exact test of anneal against that ASE arm.",
        "",
        "| system | arm | hits | n | Fisher p | MW p evals | median wall s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in SYSTEMS:
        anneal = cell_rows(new_rows, system, "anneal")
        anneal_scores = [scored(row) for row in anneal]
        anneal_hits = hits(anneal)
        lines.append(
            f"| {system} | anneal | {anneal_hits} | {len(anneal)} |  |  | {fmt(median([float(row['wall_s']) for row in anneal]))} |"
        )
        for arm in ASE_ARMS:
            other = cell_rows(new_rows, system, arm)
            fisher = fisher_exact_two_sided(
                anneal_hits,
                len(anneal) - anneal_hits,
                hits(other),
                len(other) - hits(other),
            )
            mw, _u = mann_whitney_exact(anneal_scores, [scored(row) for row in other])
            lines.append(
                f"| {system} | {arm} | {hits(other)} | {len(other)} | {fmt(fisher)} | {fmt(mw)} | "
                f"{fmt(median([float(row['wall_s']) for row in other]))} |"
            )
    lines.extend(
        [
            "",
            "Diff versus 5eb79cdc",
            "",
            "The earlier rows file is unchanged. This re-session supersedes it,",
            "including the rows recorded before the minima-comparison prune.",
            "A cell changed when its hit count, which is also its accept count, differs.",
            "",
            "| system | arm | earlier hits | new hits | changed | pre-prune seeds | pre-prune hits | new on those seeds | pre-prune changed |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for system in SYSTEMS:
        for arm in ASE_ARMS:
            earlier = cell_rows(old_rows, system, arm)
            current = cell_rows(new_rows, system, arm)
            old_hits = hits(earlier)
            new_hits = hits(current)
            prior = cell_rows(preprune, system, arm)
            prior_seeds = {row["seed"] for row in prior}
            current_on_prior = [row for row in current if row["seed"] in prior_seeds]
            prior_hits = hits(prior)
            now_hits = hits(current_on_prior)
            lines.append(
                f"| {system} | {arm} | {old_hits}/{len(earlier)} | {new_hits}/{len(current)} | "
                f"{'yes' if old_hits != new_hits else 'no'} | {len(prior)} | {prior_hits} | {now_hits} | "
                f"{'yes' if prior and prior_hits != now_hits else 'no' if prior else 'none'} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not NEW_ROWS.is_file():
        raise SystemExit(f"missing {NEW_ROWS.name}")
    text = render(load(NEW_ROWS), load(OLD_ROWS))
    OUT.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
