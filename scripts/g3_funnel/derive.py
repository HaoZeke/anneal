#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Tables of the G3 funnel panel from the rows and the classified traces.

Usage: derive.py ROWS.csv MINIMA_DIR OUT_DIR

MINIMA_DIR holds ``<system>-<method>-<seed>.csv``, the output of
``classify.py`` for every trace. Writes to OUT_DIR:

``minima.csv``       every accepted minimum with its run, shares and funnel.
``occupancy.csv``    per method, system and funnel: accepted minima and force
                     calls. The calls from one accepted minimum's call index
                     to the next one's (to ``charged`` for the last) belong to
                     the first one's funnel; calls before the first accepted
                     minimum are ``before_first``.
``transitions.csv``  per method, system, move, from and to funnel: how many
                     consecutive accepted minima changed funnel with that move.
``misses.csv``       every run that is not a hit, with the lowest accepted
                     minimum of its trace (first on ties): funnel and energy.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

FUNNELS = ("icosahedral", "decahedral", "fcc", "other")
MINIMA_FIELDS = ["system", "method", "seed", "index", "call", "hop", "move", "energy",
                 "f421", "f422", "f555", "fother", "funnel"]


def writer(path: Path, fields: list[str]):
    handle = path.open("w")
    out = csv.writer(handle, lineterminator="\n")
    out.writerow(fields)
    return handle, out


def main() -> int:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    rows_path, minima_dir, out_dir = map(Path, sys.argv[1:])
    with rows_path.open() as handle:
        rows = list(csv.DictReader(handle))
    out_dir.mkdir(parents=True, exist_ok=True)
    minima_count: Counter = Counter()
    call_count: Counter = Counter()
    runs: Counter = Counter()
    charged_total: Counter = Counter()
    transitions: Counter = Counter()
    transition_total: Counter = Counter()
    minima_handle, minima_out = writer(out_dir / "minima.csv", MINIMA_FIELDS)
    miss_handle, miss_out = writer(
        out_dir / "misses.csv",
        ["system", "method", "seed", "status", "stop", "best", "charged", "minima",
         "terminal_index", "terminal_call", "terminal_energy", "terminal_funnel",
         "terminal_f421", "terminal_f422", "terminal_f555", "terminal_fother"],
    )
    for row in rows:
        system, method, seed = row["system"], row["method"], row["seed"]
        key = (method, system)
        runs[key] += 1
        charged = int(row["charged"]) if row["charged"] else 0
        charged_total[key] += charged
        path = minima_dir / f"{system}-{method}-{seed}.csv"
        minima = []
        if path.exists():
            with path.open() as handle:
                minima = list(csv.DictReader(handle))
        for m in minima:
            minima_out.writerow([system, method, seed] + [m[f] for f in MINIMA_FIELDS[3:]])
            minima_count[key + (m["funnel"],)] += 1
        calls = [int(m["call"]) for m in minima]
        if minima:
            call_count[key + ("before_first",)] += calls[0]
        for i, m in enumerate(minima):
            end = calls[i + 1] if i + 1 < len(minima) else max(charged, calls[i])
            call_count[key + (m["funnel"],)] += end - calls[i]
        for previous, current in zip(minima, minima[1:]):
            if previous["funnel"] != current["funnel"]:
                transitions[key + (current["move"], previous["funnel"], current["funnel"])] += 1
                transition_total[key] += 1
        if row["hit"] != "1":
            low = min(minima, key=lambda m: float(m["energy"])) if minima else None
            miss_out.writerow(
                [system, method, seed, row["status"], row["stop"], row["best"], row["charged"], len(minima)]
                + ([low["index"], low["call"], low["energy"], low["funnel"],
                    low["f421"], low["f422"], low["f555"], low["fother"]] if low else [""] * 8)
            )
    minima_handle.close()
    miss_handle.close()
    handle, out = writer(
        out_dir / "occupancy.csv",
        ["method", "system", "funnel", "runs", "minima", "minima_fraction", "calls", "calls_fraction",
         "transitions"],
    )
    for key in sorted(runs):
        total_minima = sum(minima_count[key + (f,)] for f in FUNNELS)
        total_calls = charged_total[key]
        for funnel in FUNNELS + ("before_first",):
            n = minima_count[key + (funnel,)]
            c = call_count[key + (funnel,)]
            out.writerow([
                key[0], key[1], funnel, runs[key],
                n if funnel != "before_first" else "",
                f"{n / total_minima:.4f}" if total_minima and funnel != "before_first" else "",
                c,
                f"{c / total_calls:.4f}" if total_calls else "",
                transition_total[key],
            ])
    handle.close()
    handle, out = writer(out_dir / "transitions.csv", ["method", "system", "move", "from", "to", "count"])
    for key, n in sorted(transitions.items(), key=lambda item: (item[0][0], item[0][1], -item[1], item[0][2:])):
        out.writerow(list(key) + [n])
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
