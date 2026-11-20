#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write summary.csv and README.md from the tuned-GMIN rows and the manifest.

Medians are Hyndman-Fan type 7 of ``evals_to_hit``. A miss is stored at
the budget, which is the call column of the printed table. The hit rule
is ``scripts/target_tol.py`` ``energy_hits`` with ``DEFAULT_HIT_TOL``
0.001: best <= reference + 0.001.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).resolve().parent / "manifest.csv"
ROWS = ROOT / "bench" / "paper-rerun" / "tuned-gmin" / "rows.csv"
SUMMARY = ROOT / "bench" / "paper-rerun" / "tuned-gmin" / "summary.csv"
README = ROOT / "bench" / "paper-rerun" / "tuned-gmin" / "README.md"

SUMMARY_FIELDS = [
    "cell",
    "system",
    "code",
    "status",
    "hits",
    "n",
    "calls_median",
    "printed_hits",
    "printed_n",
    "printed_calls_median",
    "budget",
    "seed_first",
    "seed_last",
    "code_revision",
]


def load_panel():
    path = ROOT / "scripts" / "gmin_panel" / "run_panel.py"
    spec = importlib.util.spec_from_file_location("gmin_panel_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def hits_of(group: list[dict[str, str]]) -> int:
    return sum(row["hit"] == "1" for row in group)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def median_calls(panel, group: list[dict[str, str]]) -> str:
    if not group:
        return ""
    return f"{panel.type7([float(row['evals_to_hit']) for row in group], 0.5):.10g}"


def summarise(panel, manifest: list[dict[str, str]], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["code"]), []).append(row)
    out = []
    for cell in manifest:
        group = grouped.get((cell["system"], cell["code"]), [])
        if cell["status"] == "run" and group:
            seeds = sorted(int(row["seed"]) for row in group)
            if seeds != list(range(int(cell["seed_first"]), int(cell["seed_last"]) + 1)):
                raise SystemExit(f"{cell['system']} {cell['code']}: stored seeds {seeds}")
            revisions = sorted({row["code_revision"] for row in group})
            code_revision = " ".join(revisions)
            hits = str(hits_of(group))
            count = str(len(group))
            calls = median_calls(panel, group)
        else:
            code_revision = ""
            hits = ""
            count = ""
            calls = ""
        out.append(
            {
                "cell": cell["cell"],
                "system": cell["system"],
                "code": cell["code"],
                "status": cell["status"],
                "hits": hits,
                "n": count,
                "calls_median": calls,
                "printed_hits": cell["printed_hits"],
                "printed_n": cell["printed_n"],
                "printed_calls_median": cell["printed_calls_median"],
                "budget": cell["budget"],
                "seed_first": cell["seed_first"],
                "seed_last": cell["seed_last"],
                "code_revision": code_revision,
            }
        )
    return out


def readme(panel, summary: list[dict[str, str]], rows: list[dict[str, str]]) -> str:
    hosts = sorted({row["host"] for row in rows})
    sessions = sorted({row["session"] for row in rows})
    blobs = sorted({row["harness_blob"] for row in rows})
    harnesses = sorted({row["harness_revision"] for row in rows})
    lines = [
        "# Tuned GMIN rerun",
        "",
        "Fresh measurement of the printed GMIN cells whose per-cluster settings",
        "are recorded in this tree and whose earlier harness commit is not.",
        "The base is 61fa67cf380e46876b569da45e7cc091647a775b.",
        "Budgets are PAPER_BUDGETS in scripts/gmin_panel/run_panel.py.",
        "Seeds of a run cell are 2000-2011. Arm order inside a seed is ABBA.",
        "",
        "A hit is energy_hits in scripts/target_tol.py: best <= reference + 0.001.",
        "DEFAULT_HIT_TOL is 0.001. A miss is stored at the budget.",
        "calls_median is the Hyndman-Fan type 7 median of evals_to_hit.",
        "",
        "GMIN is the build in scripts/gmin_panel/build_gmin.sh.",
        f"The pin is SVNREV {panel.GMIN_PIN}.",
        f"The tarball sha256 is {panel.GMIN_TARBALL_SHA256}.",
        "The script refuses a tarball whose SVNREV or sha256 differs.",
        "ALIGN, MPI and OpenMP are off. The link is the reference BLAS and LAPACK.",
        "Anneal code_revision is the base commit. GMIN code_revision is the pin.",
        "",
        f"host {', '.join(hosts) if hosts else 'none'}",
        f"session {', '.join(sessions) if sessions else 'none'}",
        f"harness_revision {', '.join(harnesses) if harnesses else 'none'}",
        f"harness_blob {', '.join(blobs) if blobs else 'none'}",
        "",
        "| cell | code | status | hits/n | calls median | printed hits/n | printed calls |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        measured = f"{row['hits']}/{row['n']}" if row["hits"] else ""
        printed = f"{row['printed_hits']}/{row['printed_n']}" if row["printed_n"] else row["printed_hits"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["cell"],
                    row["code"],
                    row["status"],
                    measured,
                    row["calls_median"],
                    printed,
                    row["printed_calls_median"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "status run was executed. status unrunnable has no recorded keyword",
            "file or start generator in this tree, so it was not guessed.",
            "status not_tuned is a printed GMIN count under one shared input,",
            "which the manuscript separates from a per-cluster input.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    panel = load_panel()
    manifest = read_csv(MANIFEST)
    rows = read_csv(ROWS)
    summary = summarise(panel, manifest, rows)
    write_csv(SUMMARY, SUMMARY_FIELDS, summary)
    README.write_text(readme(panel, summary, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
