#!/usr/bin/env python3
"""Compile the in-process memory probe and the CuH2 call counter."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(source: str, output: Path, extra: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            *extra,
            "-o",
            str(output),
            str(HERE / source),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--counter", type=Path, required=True)
    args = parser.parse_args()
    build("rss_anon_probe.c", args.probe, ["-pthread"])
    build("cuh2_counter.c", args.counter, ["-ldl"])


if __name__ == "__main__":
    main()
