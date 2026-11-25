#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Check that the retired GOFEE comparison tables remain absent.

The manuscript and Supporting Information no longer print these tables.
Their current comparison tables have dedicated generators.

    python3 paper/jctc/scripts/h2h_tables.py --check
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
SI = ROOT / "paper" / "jctc" / "si.tex"

RETIRED = {
    "gofee": (
        MANUSCRIPT,
        ("% gofee table: start", "% gofee table: end", "\\label{tab:gofee}"),
    ),
    "gofee-si": (
        SI,
        ("% gofee-si table: start", "% gofee-si table: end", "\\label{tab:gofee-si}"),
    ),
}


def main(_argv: list[str]) -> int:
    stale = False
    for name, (path, tokens) in RETIRED.items():
        text = path.read_text()
        if any(token in text for token in tokens):
            print(f"the retired {name} table remains in {path.name}", file=sys.stderr)
            stale = True
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
