#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Record the tblite version from its pinned package metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
import urllib.request
from pathlib import Path


COMMIT = "fa8a4416e8fe093d0075bc10ac875494c2a449a9"
METADATA_URL = (
    "https://raw.githubusercontent.com/tblite/tblite/"
    f"{COMMIT}/fpm.toml"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data/water_v45b/rerun-9a30f319/toolchain.json"
)


def metadata_bytes(path: Path | None) -> bytes:
    if path is not None:
        return path.read_bytes()
    with urllib.request.urlopen(METADATA_URL, timeout=120) as response:
        return response.read()


def receipt(raw: bytes) -> dict[str, object]:
    package = tomllib.loads(raw.decode())
    if package.get("name") != "tblite":
        raise ValueError("package metadata is not for tblite")
    return {
        "schema": 1,
        "package": package["name"],
        "version": package["version"],
        "source_commit": COMMIT,
        "metadata": "fpm.toml",
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "metadata_url": METADATA_URL,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt(metadata_bytes(args.metadata)), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
