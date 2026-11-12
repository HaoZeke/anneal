# Copyright (c) anneal developers
"""Count energy and gradient calls.

Crest cost is the energy and gradient total in the crest log.
Anneal cost is the search ledger plus the polish ledger, plus the
reported-geometry evaluation when that line is present.
A missing total is left blank. This script does not print wall time.
"""

import re
import sys
from pathlib import Path

from crest_engrad import require_split

CALLS = re.compile(r"(?m)^energy_calls[:=]\s*(\d+)\s*$")
FINAL = re.compile(r"(?m)^final_best_evaluation[:=]\s*(\d+)\s*$")


def anneal_calls(text):
    calls = CALLS.findall(text)
    if not calls:
        return None
    final = FINAL.findall(text)
    extra = int(final[-1]) if final else 0
    return int(calls[-1]) + extra


def cell(value):
    return "nan" if value is None else str(value)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: count_cost.py LOG_DIR")
    root = Path(sys.argv[1])
    print("seed\tarm\tenergy_grad_calls")
    seeds = []
    for path in root.glob("crest-hexamer-seed*.log"):
        stem = path.name.removeprefix("crest-hexamer-seed").removesuffix(".log")
        if stem.isdigit():
            seeds.append(int(stem))
    for seed in sorted(seeds):
        crest = (root / f"crest-hexamer-seed{seed}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        try:
            _driver, total = require_split(crest)
        except ValueError:
            total = None
        print(f"{seed}\tcrest\t{cell(total)}")
        for arm, name in (
            ("driver-count budget", f"anneal-hexamer-seed{seed}.log"),
            ("paper4000", f"anneal-paper4000-seed{seed}.log"),
            ("engrad_matched", f"anneal-engrad-seed{seed}.log"),
        ):
            path = root / name
            if not path.is_file():
                print(f"{seed}\t{arm}\tnan")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            print(f"{seed}\t{arm}\t{cell(anneal_calls(text))}")


if __name__ == "__main__":
    main()
