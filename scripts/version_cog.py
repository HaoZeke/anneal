#!/usr/bin/env python3
"""Run cogapp with one shared prologue.

The prologue imports ``anneal_version``. Cog blocks only print.
https://nedbatchelder.com/code/cog/
"""

import sys
from pathlib import Path

from cogapp import Cog


def main() -> int:
    scripts = Path(__file__).resolve().parent
    argv = [
        "cog",
        "-I",
        str(scripts),
        "-p",
        "import anneal_version",
        *sys.argv[1:],
    ]
    return Cog().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
