#!/usr/bin/env python3
"""Run the cluster half of the restack gate.

The stock driver requires flag order D, D2, G, P and would run that
order. Even seeds here run D, G, P, D2 and odd seeds run the reverse.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "generic-packings-r3-gate"
sys.path.insert(0, str(HARNESS))

import run_gate as gate  # noqa: E402

EVEN = ["D", "G", "P", "D2"]
ODD = ["D2", "P", "G", "D"]


def scheduled(arms: list[str], seed: int) -> list[str]:
    if list(arms) != ["D", "D2", "G", "P"]:
        raise SystemExit("flags must be D, D2, G, P")
    return list(EVEN if (seed - gate.SEEDS[0]) % 2 == 0 else ODD)


def main(argv: list[str]) -> int:
    if argv[1:] == ["--schedule-check"]:
        assert scheduled(["D", "D2", "G", "P"], 2000) == EVEN
        assert scheduled(["D", "D2", "G", "P"], 2001) == ODD
        assert gate.CELL["lj38"][2] == 400_000
        assert gate.CELL["lj75"][2] == 4_000_000
        assert gate.CELL["lj98"][2] == 4_000_000
        assert gate.CELL["m55r6"][2] == 1_000_000
        assert gate.CELL["lj55"][2] == 1_000_000
        assert gate.CELL["m38r14"][2] == 400_000
        assert gate.CELL["m38r6"][2] == 400_000
        assert gate.CELL["sc126-38"][2] == 400_000
        assert gate.CELL["sc126-55"][2] == 1_000_000
        assert gate.CELL["sc108-38"][2] == 400_000
        assert gate.CELL["sc96-38"][2] == 400_000
        assert gate.CELL["sc96-55"][2] == 1_000_000
        print("ok")
        return 0
    original = gate.append_csv

    def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
        original(path, fields, row)
        if "rss_anon_self_peak_kb" not in row:
            return
        journal = Path(path).with_suffix(".jsonl")
        payload = {key: row[key] for key in fields}
        with journal.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    gate.arm_order = scheduled
    gate.append_csv = append_csv
    return gate.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
