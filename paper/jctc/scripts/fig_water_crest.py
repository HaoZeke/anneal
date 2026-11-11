#!/usr/bin/env python3
# Copyright (c) anneal developers
"""The 192-seed water-crest figure is not drawn.

water-v45b-perseed.csv has no raw stdout in this tree. Drawing the hit
shares or the McNemar p values would repeat numbers that are not in a run log.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(
        "water_crest.pdf is not drawn: the 192-seed raw logs are not in the record",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
