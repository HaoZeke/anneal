#!/usr/bin/env python3
"""Build the in-process anonymous-RSS reporting probe."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-o",
            str(args.out),
            str(HERE / "rss_anon_self.c"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
