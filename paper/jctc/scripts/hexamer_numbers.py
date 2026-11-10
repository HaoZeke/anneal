#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Numbers for the water-hexamer figure and table.

The values are read from the crest logs and from NUMBERS.tsv when those
files are present. The fallback below is the same parse, kept so a
checkout without the log tree still draws the committed figure.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

# Polished gap is anneal minus CREST, in hartree.
# Label-free RMSD is in angstroms.
# Wall time is in seconds.
# old_counter is dynamics steps plus optimization steps.
# engrad is CREST's "Total number of energy+grad calls".

FALLBACK = {
    0: {
        "old": 32112,
        "engrad": 151074,
        "gap_4000": -1.121399862e-08,
        "gap_old": -1.120999826e-08,
        "rmsd_4000": 0.001003747109,
        "rmsd_old": 0.0009865828747,
        "wall_crest": 552.804013,
        "wall_4000": 20.270713,
        "wall_old": 174.079992,
    },
    1: {
        "old": 50026,
        "engrad": 191132,
        "gap_4000": -1.350929999e-07,
        "gap_old": -1.350480012e-07,
        "rmsd_4000": 0.002022858719,
        "rmsd_old": 0.001964052149,
        "wall_crest": 506.756639,
        "wall_4000": 19.31865,
        "wall_old": 247.94321,
    },
    2: {
        "old": 32155,
        "engrad": 142110,
        "gap_4000": -2.54600252e-09,
        "gap_old": -2.585000658e-09,
        "rmsd_4000": 0.0005323654897,
        "rmsd_old": 0.00047715392,
        "wall_crest": 535.249075,
        "wall_4000": 18.38066,
        "wall_old": 165.63577,
    },
    3: {
        "old": 46062,
        "engrad": 172468,
        "gap_4000": -4.150997768e-09,
        "gap_old": -4.143000609e-09,
        "rmsd_4000": 0.0003219790088,
        "rmsd_old": 0.0003084858392,
        "wall_crest": 570.471693,
        "wall_4000": 17.494653,
        "wall_old": 216.896696,
    },
    4: {
        "old": 50063,
        "engrad": 157643,
        "gap_4000": -6.898000038e-08,
        "gap_old": -6.902200056e-08,
        "rmsd_4000": 0.002115530999,
        "rmsd_old": 0.002067989106,
        "wall_crest": 463.398034,
        "wall_4000": 19.96839,
        "wall_old": 280.951875,
    },
    5: {
        "old": 32170,
        "engrad": 209931,
        "gap_4000": -2.190002846e-09,
        "gap_old": -2.200000182e-09,
        "rmsd_4000": 0.0001568510743,
        "rmsd_old": 0.0001344492461,
        "wall_crest": 793.533117,
        "wall_4000": 17.499605,
        "wall_old": 145.888804,
    },
}

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


def _logs() -> Path | None:
    raw = os.environ.get("WATER_LOGS", "paper/jctc/logs/water")
    path = Path(raw)
    if (path / "NUMBERS.tsv").is_file():
        return path
    return None


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
    """Return one record per seed, checked against the logs when present."""
    rows = {seed: dict(values) for seed, values in FALLBACK.items()}
    logs = _logs()
    if logs is None:
        return rows
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
    for seed, row in rows.items():
        crest_log = logs / f"crest-hexamer-seed{seed}.log"
        engrad = _engrad(crest_log)
        old = _old_counter(crest_log)
        if engrad != row["engrad"] or old != row["old"]:
            raise SystemExit(f"seed {seed} counters {old}/{engrad} != fallback")
        crest = by_seed_arm[(seed, "crest")]
        short = by_seed_arm[(seed, "paper4000")]
        matched = by_seed_arm[(seed, "fair")]
        if abs(float(short["signed_dE_polished_Eh"]) - row["gap_4000"]) > 1e-18:
            raise SystemExit(f"seed {seed} 4000 gap drifted")
        if abs(float(matched["signed_dE_polished_Eh"]) - row["gap_old"]) > 1e-18:
            raise SystemExit(f"seed {seed} old-counter gap drifted")
        if abs(float(crest["wall_s"]) - row["wall_crest"]) > 1e-9:
            raise SystemExit(f"seed {seed} crest wall drifted")
        cpu = crest_log.read_text(errors="replace").splitlines()[3]
        if cpu != CPU:
            raise SystemExit(f"seed {seed} cpu {cpu!r}")
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
