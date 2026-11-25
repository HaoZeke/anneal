#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fail unless ``anneal`` and ``anneal._core`` came from a virtual environment.

The release workflow installs the built sdist into ``.sdist-venv`` and runs
this file with that environment's interpreter. Importing the compiled
extension proves the sdist built. Its location proves the import was not
answered by a source tree, an editable install or another interpreter, which
would pass without the sdist being installed at all.

usage: check_installed_anneal.py VENV
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    venv = Path(argv[1]).resolve()

    import anneal
    import anneal._core

    for module in (anneal, anneal._core):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(venv) or "site-packages" not in path.parts:
            print(
                f"{module.__name__} was imported from {path}, "
                f"not from site-packages under {venv}",
                file=sys.stderr,
            )
            return 1
    print(anneal.__version__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
