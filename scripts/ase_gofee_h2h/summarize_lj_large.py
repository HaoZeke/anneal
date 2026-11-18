#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append LJ98-LJ150 seed rows and rebuild the summary from them.

Existing seed lines are left as written. Wall-capped misses do not enter a
rank test: Mann-Whitney is then not estimated. Fisher, the tie-corrected
Mann-Whitney and the medians are the ASE panel's functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ase_panel.summarize import fisher, mann_whitney, median  # noqa: E402

SEED_COLUMNS = [
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "calls_to_hit",
    "charged",
    "wall_s",
    "inside_s",
    "outside_s",
    "best",
    "stop",
    "code_revision",
    "host",
]
SUMMARY_COLUMNS = [
    "system",
    "arm",
    "n",
    "hits",
    "calls_to_hit_median",
    "best",
    "inside_median_s",
    "outside_median_s",
    "wall_median_s",
    "budget",
    "code_revision",
    "host",
]
COMPARE_COLUMNS = [
    "system",
    "baseline",
    "baseline_hits",
    "agox_hits",
    "n",
    "fisher_p",
    "mann_whitney_u",
    "mann_whitney_p",
    "agox_inside_median_s",
    "agox_outside_median_s",
    "baseline_inside_median_s",
    "baseline_outside_median_s",
    "code_revision",
    "host",
]
SYSTEMS = ("lj98", "lj102", "lj103", "lj104", "lj150")


def seed_file(repo: Path) -> Path:
    return repo / "bench" / "ase-gofee-h2h" / "lj-large-seeds.csv"


def read_seeds(repo: Path) -> list[dict[str, str]]:
    path = seed_file(repo)
    if not path.is_file():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        rows.append(dict(zip(header, line.split(","))))
    return rows


def row_text(row: dict) -> str:
    calls = row.get("calls_to_hit") or ""
    if calls in ("", None):
        calls = ""
    else:
        calls = str(int(float(calls)))
    return ",".join(
        [
            row["system"],
            "paper" if row["arm"] == "paper" else row["arm"],
            str(int(row["seed"])),
            str(int(row["budget"])),
            str(int(row["hit"])),
            calls,
            str(int(row["charged"])),
            f"{float(row['wall_s']):.6f}",
            f"{float(row['inside_s']):.6f}",
            f"{float(row['outside_s']):.6f}",
            f"{float(row['best']):.8f}",
            row.get("stop") or ("hit" if int(row["hit"]) else "budget"),
            str(row.get("head") or row.get("code_revision") or ""),
            str(row.get("host") or "bench"),
        ]
    )


def append_rows(repo: Path, rows: list[dict]) -> None:
    """Append seed rows whose system, arm and seed are not already present."""
    path = seed_file(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    present = set()
    if path.is_file() and path.read_text().strip():
        text = path.read_text()
        for line in text.splitlines()[1:]:
            if line.strip() and not line.startswith("#"):
                system, arm, seed = line.split(",", 3)[:3]
                present.add((system, "anneal" if arm == "paper" else arm, seed))
    else:
        text = ",".join(SEED_COLUMNS) + "\n"
    extra = []
    for row in rows:
        arm = "anneal" if row["arm"] == "paper" else row["arm"]
        key = (row["system"], arm, str(int(row["seed"])))
        if key in present:
            continue
        stored = dict(row, arm=arm)
        extra.append(row_text(stored))
        present.add(key)
    if not extra:
        if not path.is_file():
            path.write_text(text)
        return
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + "\n".join(extra) + "\n")


def censored(rows: list[dict[str, str]]) -> bool:
    return any(row["stop"] in {"wall", "capped"} and row["hit"] == "0" for row in rows)


def calls(row: dict[str, str]) -> float:
    if row["calls_to_hit"]:
        return float(row["calls_to_hit"])
    return float(row["budget"])


def group(rows: list[dict[str, str]], system: str, arm: str) -> list[dict[str, str]]:
    found = [row for row in rows if row["system"] == system and row["arm"] == arm]
    found.sort(key=lambda row: int(row["seed"]))
    return found


def summary_line(system: str, arm: str, rows: list[dict[str, str]]) -> str:
    hits = [calls(row) for row in rows if row["hit"] == "1"]
    revisions = sorted({row["code_revision"] for row in rows if row["code_revision"]})
    hosts = sorted({row["host"] for row in rows if row["host"]})
    return ",".join(
        [
            system,
            arm,
            str(len(rows)),
            str(sum(row["hit"] == "1" for row in rows)),
            f"{median(hits):.6f}" if hits else "",
            f"{median([float(row['best']) for row in rows]):.8f}",
            f"{median([float(row['inside_s']) for row in rows]):.6f}",
            f"{median([float(row['outside_s']) for row in rows]):.6f}",
            f"{median([float(row['wall_s']) for row in rows]):.6f}",
            rows[0]["budget"],
            " ".join(revisions),
            hosts[0] if len(hosts) == 1 else " ".join(hosts),
        ]
    )


def compare_pair(system: str, baseline: str, base_rows: list[dict[str, str]], agox: list[dict[str, str]]) -> str:
    base_hits = sum(row["hit"] == "1" for row in base_rows)
    agox_hits = sum(row["hit"] == "1" for row in agox)
    fisher_p = f"{fisher(base_hits, agox_hits):.6g}" if len(base_rows) == 12 and len(agox) == 12 else ""
    if censored(base_rows) or censored(agox) or len(base_rows) != 12 or len(agox) != 12:
        u_text = mw_text = "not estimated"
    else:
        statistic, probability = mann_whitney([calls(row) for row in base_rows], [calls(row) for row in agox])
        u_text, mw_text = f"{statistic:.6g}", f"{probability:.6g}"
    revisions = sorted({row["code_revision"] for row in base_rows + agox if row["code_revision"]})
    return ",".join(
        [
            system,
            baseline,
            str(base_hits),
            str(agox_hits),
            "12",
            fisher_p,
            u_text,
            mw_text,
            f"{median([float(row['inside_s']) for row in agox]):.6f}",
            f"{median([float(row['outside_s']) for row in agox]):.6f}",
            f"{median([float(row['inside_s']) for row in base_rows]):.6f}" if base_rows and base_rows[0].get("inside_s") else "",
            f"{median([float(row['outside_s']) for row in base_rows]):.6f}" if base_rows and base_rows[0].get("outside_s") else "",
            " ".join(revisions),
            "bench",
        ]
    )


def rebuild(repo: Path) -> None:
    rows = read_seeds(repo)
    summary_path = repo / "bench" / "ase-gofee-h2h" / "lj-large.csv"
    previous = {}
    if summary_path.is_file():
        lines = [line for line in summary_path.read_text().splitlines() if line.strip()]
        header = lines[0].split(",")
        for line in lines[1:]:
            fields = dict(zip(header, line.split(",")))
            previous[(fields["system"], fields["arm"])] = line
    summary = [",".join(SUMMARY_COLUMNS)]
    compare = [",".join(COMPARE_COLUMNS)]
    for system in SYSTEMS:
        for arm in ("anneal", "gmin", "agox_gofee"):
            found = group(rows, system, arm)
            if len(found) == 12:
                summary.append(summary_line(system, arm, found))
            elif (system, arm) in previous:
                summary.append(previous[(system, arm)])
        agox = group(rows, system, "agox_gofee")
        anneal = group(rows, system, "anneal")
        if len(agox) == 12 and len(anneal) == 12:
            compare.append(compare_pair(system, "anneal", anneal, agox))
    summary_path.write_text("\n".join(summary) + "\n")
    if len(compare) > 1:
        (repo / "bench" / "ase-gofee-h2h" / "lj-large-compare.csv").write_text("\n".join(compare) + "\n")


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    rebuild(repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
