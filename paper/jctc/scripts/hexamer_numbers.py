#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Numbers for the water-hexamer figure and table.

The values are read from the crest logs and from NUMBERS.tsv.
A missing log directory is an error. There is no fallback table.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import fetch_paper_data

# Polished gap is anneal minus CREST, in hartree.
# Label-free RMSD is in angstroms.
# Wall time is in seconds.
# old_counter is dynamics steps plus optimization steps.
# engrad is CREST's "Total number of energy+grad calls".

CPU = "Intel Xeon Processor, family 6, model 207"
VTIGHT_EH = 1e-7


def type7(values: list[float], probability: float) -> float:
    """Hyndman-Fan type-7 quantile."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    position = 1 + (n - 1) * probability
    index = position - 1
    low = math.floor(index)
    high = math.ceil(index)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def median_iqr(values: list[float]) -> tuple[float, float]:
    med = type7(values, 0.5)
    iqr = type7(values, 0.75) - type7(values, 0.25)
    return med, iqr


def _logs() -> Path:
    fetch_paper_data()
    raw = os.environ.get("WATER_LOGS")
    if raw:
        path = Path(raw)
    else:
        path = Path(__file__).resolve().parents[1] / "logs" / "water"
    if not (path / "NUMBERS.tsv").is_file():
        raise SystemExit(f"missing water logs: {path}")
    return path


def _engrad(log: Path) -> int:
    match = re.search(
        r"Total number of energy\+grad calls:\s+(\d+)",
        log.read_text(errors="replace"),
    )
    if match is None:
        raise SystemExit(f"missing energy+grad line in {log}")
    return int(match.group(1))


def _old_counter(log: Path) -> int:
    text = log.read_text(errors="replace")
    dynamics = re.search(r"^dynamics_steps=(\d+)$", text, re.M)
    optim = re.search(r"^opt_steps=(\d+)$", text, re.M)
    if dynamics is None or optim is None:
        raise SystemExit(f"missing step counters in {log}")
    return int(dynamics.group(1)) + int(optim.group(1))


def load() -> dict[int, dict[str, float]]:
    """Return one record per seed from the log directory."""
    logs = _logs()
    header = None
    by_seed_arm: dict[tuple[int, str], dict[str, str]] = {}
    for line in (logs / "NUMBERS.tsv").read_text().splitlines():
        if line.startswith("# seed\t") or line.startswith("seed\t"):
            header = line.lstrip("# ").split("\t")
            continue
        if line.startswith("#") or line.startswith("summary") or not line.strip():
            continue
        if header is None:
            continue
        parts = line.split("\t")
        record = dict(zip(header, parts))
        by_seed_arm[(int(record["seed"]), record["arm"])] = record
    rows: dict[int, dict[str, float]] = {}
    for seed in range(6):
        crest_log = logs / f"crest-hexamer-seed{seed}.log"
        if not crest_log.is_file():
            raise SystemExit(f"missing {crest_log}")
        crest = by_seed_arm[(seed, "crest")]
        short = by_seed_arm[(seed, "paper4000")]
        matched = by_seed_arm[(seed, "fair")]
        cpu = crest_log.read_text(errors="replace").splitlines()[3]
        if cpu != CPU:
            raise SystemExit(f"seed {seed} cpu {cpu!r}")
        rows[seed] = {
            "old": float(_old_counter(crest_log)),
            "engrad": float(_engrad(crest_log)),
            "gap_4000": float(short["signed_dE_polished_Eh"]),
            "gap_old": float(matched["signed_dE_polished_Eh"]),
            "rmsd_4000": float(short["rmsd_label_free_A"]),
            "rmsd_old": float(matched["rmsd_label_free_A"]),
            "wall_crest": float(crest["wall_s"]),
            "wall_4000": float(short["wall_s"]),
            "wall_old": float(matched["wall_s"]),
        }
    return rows


def walls(rows: dict[int, dict[str, float]]) -> dict[str, tuple[float, float]]:
    return {
        name: median_iqr([rows[seed][key] for seed in range(6)])
        for name, key in (
            ("crest", "wall_crest"),
            ("eval4000", "wall_4000"),
            ("old", "wall_old"),
        )
    }


def _sci(value: float) -> str:
    """Four digits after the decimal, matching the manuscript rows."""
    mantissa, exponent = f"{value:.4e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exponent)}}}$"


def table_lines(rows: dict[int, dict[str, float]]) -> list[str]:
    """LaTeX rows for the two hexamer tables."""
    lines = ["counters"]
    for seed in range(6):
        row = rows[seed]
        lines.append(f"{seed} & {int(row['old'])} & {int(row['engrad'])} \\\\")
    lines.append("gaps")
    for seed in range(6):
        row = rows[seed]
        lines.append(
            f"{seed} & {_sci(row['gap_4000'])} & {_sci(row['gap_old'])} "
            f"& {row['rmsd_4000']:.6f} & {row['rmsd_old']:.6f} \\\\"
        )
    return lines


def main() -> None:
    """Print the wall-time receipt, or the table rows with 'tables'."""
    rows = load()
    if len(sys.argv) > 1 and sys.argv[1] == "tables":
        sys.stdout.write("\n".join(table_lines(rows)) + "\n")
        return
    stats = walls(rows)
    lines = [
        "command python3 paper/jctc/scripts/hexamer_numbers.py",
        "source logs/water",
        "median_iqr type-7",
    ]
    for name in ("crest", "eval4000", "old"):
        median, iqr = stats[name]
        lines.append(f"{name} median {median:.10f} iqr {iqr:.10f}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
