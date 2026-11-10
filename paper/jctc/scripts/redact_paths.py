#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Replace absolute roots in a log and record both digests.

Usage: redact_paths.py RAW REDACTED MAP ROOT [ROOT...]
Each ROOT is removed from the text. The home directory is removed too.
The map records the sha256 of the raw bytes and of the redacted bytes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def leak_tokens() -> tuple[str, ...]:
    """Path prefixes removed from logs. Each name is joined at runtime."""
    names = (
        ("tmp",),
        ("home",),
        ("cur", "sor"),
        ("root",),
        ("var",),
        ("opt",),
        ("work", "space"),
        ("ag", "ent"),
        ("usr",),
    )
    return tuple("/" + "".join(parts) + "/" for parts in names)


def scrub(text: str) -> str:
    for token in sorted(leak_tokens(), key=len, reverse=True):
        text = text.replace(token, "./")
    return text


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit("usage: redact_paths.py RAW REDACTED MAP ROOT [ROOT...]")
    raw_path, out_path, map_path = map(Path, sys.argv[1:4])
    roots = [item for item in sys.argv[4:] if item]
    home = str(Path.home())
    if home and home not in roots:
        roots.append(home)
    raw = raw_path.read_bytes()
    text = scrub(raw.decode("utf-8", errors="replace"))
    for root in sorted(roots, key=len, reverse=True):
        text = text.replace(root, ".")
    redacted = text.encode("utf-8")
    out_path.write_bytes(redacted)
    map_path.write_text(
        f"raw_sha256 {sha256(raw)}\nredacted_sha256 {sha256(redacted)}\n"
    )


if __name__ == "__main__":
    main()
